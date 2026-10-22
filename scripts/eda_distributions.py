import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import stats

df = pd.read_csv('data/raw/sberbank_train_raw.csv', parse_dates=['timestamp'])

# базовая чистка явного мусора перед анализом связей (не трогаем сырой файл!)
clean = df[(df['full_sq'] > 10) & (df['full_sq'] < 300) &
           (df['price_doc'] < 100_000_000)].copy()

numeric_cols = clean.select_dtypes(include=[np.number]).columns
numeric_cols = [c for c in numeric_cols if c not in ('price_doc', 'id')]

corrs = {}
for c in numeric_cols:
    sub = clean[[c, 'price_doc']].dropna()
    if len(sub) > 100:
        rho, _ = stats.spearmanr(sub[c], sub['price_doc'])
        corrs[c] = rho

corr_s = pd.Series(corrs).sort_values()
print("=== ТОП-15 отрицательных корреляций с price_doc (Spearman) ===")
print(corr_s.head(15).round(3))
print("\n=== ТОП-15 положительных корреляций с price_doc (Spearman) ===")
print(corr_s.tail(15).sort_values(ascending=False).round(3))

# графики
fig, axes = plt.subplots(2, 3, figsize=(16, 9))

axes[0,0].hist(clean['price_doc']/1e6, bins=60, color='#4C72B0')
axes[0,0].set_title('Распределение price_doc (млн ₽)')

axes[0,1].hist(np.log1p(clean['price_doc']), bins=60, color='#55A868')
axes[0,1].set_title('log(price_doc)')

axes[0,2].scatter(clean['full_sq'], clean['price_doc']/1e6, s=3, alpha=0.3)
axes[0,2].set_xlabel('full_sq'); axes[0,2].set_ylabel('price_doc, млн ₽')
axes[0,2].set_title('Площадь vs цена')

axes[1,0].scatter(clean['metro_min_walk'], clean['price_doc']/1e6, s=3, alpha=0.3, color='#C44E52')
axes[1,0].set_xlabel('метро, мин пешком'); axes[1,0].set_ylabel('price_doc, млн ₽')
axes[1,0].set_title('Удалённость метро vs цена')

axes[1,1].scatter(clean['sport_count_1500'], clean['price_doc']/1e6, s=3, alpha=0.3, color='#8172B2')
axes[1,1].set_xlabel('спортобъекты в 1500м'); axes[1,1].set_ylabel('price_doc, млн ₽')
axes[1,1].set_title('Спортинфраструктура vs цена')

clean.boxplot(column='price_doc', by='product_type', ax=axes[1,2])
axes[1,2].set_title('Цена по типу сделки'); plt.suptitle('')

plt.tight_layout()
plt.savefig('figures/distributions_overview.png', dpi=110)
print("\nГрафик сохранён: figures/distributions_overview.png")
