import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

df = pd.read_csv('data/raw/sberbank_train_raw.csv', parse_dates=['timestamp'])
clean = df[(df['full_sq'] > 10) & (df['full_sq'] < 300)].copy()

features = [
    'full_sq', 'life_sq', 'floor', 'num_room',
    'metro_min_walk', 'metro_km_avto', 'kremlin_km', 'sadovoe_km', 'ttk_km', 'bulvar_ring_km',
    'cafe_count_1500', 'cafe_count_2000', 'cafe_count_3000', 'cafe_count_5000',
    'trc_count_1500', 'trc_count_3000', 'sport_count_1500', 'office_km', 'workplaces_km',
]
corr = clean[features].corr(method='spearman')

print("=== Пары признаков с |корреляция| > 0.85 (возможное дублирование информации) ===")
pairs = []
for i in range(len(features)):
    for j in range(i+1, len(features)):
        r = corr.iloc[i, j]
        if abs(r) > 0.85:
            pairs.append((features[i], features[j], round(r, 3)))
for a, b, r in sorted(pairs, key=lambda x: -abs(x[2])):
    print(f"{a:20s} <-> {b:20s}  r = {r}")

fig, ax = plt.subplots(figsize=(11, 9))
im = ax.imshow(corr, cmap='coolwarm', vmin=-1, vmax=1)
ax.set_xticks(range(len(features))); ax.set_xticklabels(features, rotation=90, fontsize=8)
ax.set_yticks(range(len(features))); ax.set_yticklabels(features, fontsize=8)
plt.colorbar(im)
plt.title('Корреляции между признаками (Spearman)')
plt.tight_layout()
plt.savefig('figures/feature_correlation_heatmap.png', dpi=110)
print("\nГрафик сохранён: figures/feature_correlation_heatmap.png")
