import pandas as pd

df = pd.read_csv('data/raw/sberbank_train_raw.csv', parse_dates=['timestamp'])

print("Колонки, содержащие 'price' (кроме самой price_doc):")
print([c for c in df.columns if 'price' in c.lower() and c != 'price_doc'])

print("\nКорреляция каждого признака с price_doc, отсортировано по модулю (топ-5 — проверка на явные дубликаты цели):")
num = df.select_dtypes('number').drop(columns=['price_doc', 'id'], errors='ignore')
corrs = num.corrwith(df['price_doc']).abs().sort_values(ascending=False)
print(corrs.head(5))
