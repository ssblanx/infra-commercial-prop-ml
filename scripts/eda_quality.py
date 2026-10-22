import pandas as pd
import numpy as np

df = pd.read_csv('data/raw/sberbank_train_raw.csv', parse_dates=['timestamp'])

print("=== ПРОПУСКИ (топ-15 колонок) ===")
na_pct = (df.isna().mean() * 100).sort_values(ascending=False)
print(na_pct.head(15).round(2))
print(f"\nВсего колонок с пропусками: {(na_pct > 0).sum()} из {df.shape[1]}")

print("\n=== ДУБЛИКАТЫ ===")
print("Полные дубликаты строк:", df.duplicated().sum())
print("Дубликаты по id объекта (id):", df['id'].duplicated().sum() if 'id' in df.columns else "нет колонки id")

print("\n=== НЕВОЗМОЖНЫЕ / ПОДОЗРИТЕЛЬНЫЕ ЗНАЧЕНИЯ ===")
print("full_sq == 0:", (df['full_sq'] == 0).sum())
print("full_sq < 10 (меньше 10 кв.м):", (df['full_sq'] < 10).sum())
print("full_sq > 300 (больше 300 кв.м):", (df['full_sq'] > 300).sum())
print("life_sq > full_sq (жилая больше общей):", (df['life_sq'] > df['full_sq']).sum())
print("kitch_sq > full_sq (кухня больше общей площади):", (df['kitch_sq'] > df['full_sq']).sum())
print("floor > max_floor (этаж выше этажности дома):", (df['floor'] > df['max_floor']).sum())
print("floor == 0:", (df['floor'] == 0).sum())
print("num_room == 0:", (df['num_room'] == 0).sum())

print("\nbuild_year — подозрительные значения (топ-15 редких/аномальных):")
by = df['build_year'].dropna()
print(by[(by < 1860) | (by > 2020)].value_counts().head(15))

print("\nstate — уникальные значения (должно быть 1-4):")
print(df['state'].value_counts(dropna=False))
