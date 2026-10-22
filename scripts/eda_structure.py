import pandas as pd, collections, re

df = pd.read_csv('data/raw/sberbank_train_raw.csv', parse_dates=['timestamp'])
print("Shape:", df.shape)
print("\nТипы данных:")
print(df.dtypes.value_counts())

groups = collections.Counter()
for c in df.columns:
    if c in ('timestamp', 'price_doc', 'sub_area'):
        groups['служебные / целевая (timestamp, price_doc, sub_area)'] += 1
    elif c.startswith('ID_'):
        groups['ID_* — категориальные id зон/районов'] += 1
    elif c.endswith('_km'):
        groups['*_km — расстояние до объекта инфраструктуры'] += 1
    elif re.search(r'_(500|1000|1500|2000|3000|5000)$', c):
        groups['*_count_N — число объектов в радиусе N м'] += 1
    elif c.startswith('raion_') or c.endswith('_raion') or 'raion' in c:
        groups['raion_* — демография/статистика района'] += 1
    elif c in ['full_sq','life_sq','floor','max_floor','material',
               'build_year','num_room','kitch_sq','state','product_type']:
        groups['характеристики самого объекта недвижимости'] += 1
    else:
        groups['прочее (макро/район, по data_dictionary.txt)'] += 1

print("\nГруппы признаков:")
for g, n in groups.most_common():
    print(f"{n:4d}  {g}")

print("\nПервые и последние даты:", df['timestamp'].min(), "—", df['timestamp'].max())
print("Память:", round(df.memory_usage(deep=True).sum()/1e6, 1), "МБ")
