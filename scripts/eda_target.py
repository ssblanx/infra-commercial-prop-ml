import pandas as pd
import numpy as np
from scipy import stats

df = pd.read_csv('data/raw/sberbank_train_raw.csv', parse_dates=['timestamp'])
p = df['price_doc']

print("Описательная статистика price_doc:")
print(p.describe())
print("\nSkewness:", round(p.skew(), 2))
print("Kurtosis:", round(p.kurtosis(), 2))

print("\nЛог-преобразование (log1p) — skewness:", round(np.log1p(p).skew(), 2))

print("\nТоп-10 самых частых значений цены:")
print(p.value_counts().head(10))

round_prices = p[p.isin([1000000, 2000000, 3000000])]
print(f"\nДоля 'круглых' цен (1/2/3 млн ровно): {len(round_prices)/len(p)*100:.2f}% ({len(round_prices)} из {len(p)})")

print("\nРаспределение по product_type:")
print(df['product_type'].value_counts(dropna=False))
print("\nМедианная цена по product_type:")
print(df.groupby('product_type')['price_doc'].median())

print("\nМинимум / максимум price_doc:", p.min(), "/", p.max())
print("Цена за кв.м (price_doc/full_sq), медиана:", round((p / df['full_sq']).median(), 1))
