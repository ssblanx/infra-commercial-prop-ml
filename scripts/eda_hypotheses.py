import pandas as pd
from scipy import stats

df = pd.read_csv('data/raw/sberbank_train_raw.csv', parse_dates=['timestamp'])
clean = df[(df['full_sq'] > 10) & (df['full_sq'] < 300) & (df['price_doc'] < 100_000_000)].copy()

print("="*70)
print("H1: Удалённость от центра города (kremlin_km) отрицательно связана с ценой")
sub = clean[['kremlin_km', 'price_doc']].dropna()
rho, p = stats.spearmanr(sub['kremlin_km'], sub['price_doc'])
print(f"Spearman rho = {rho:.3f}, p-value = {p:.2e}  (n={len(sub)})")
print("Вывод:", "гипотеза подтверждается" if p < 0.05 and rho < 0 else "не подтверждается")

print("\n" + "="*70)
print("H2: Плотность кафе/магазинов в районе (cafe_count_3000) положительно связана с ценой")
sub = clean[['cafe_count_3000', 'price_doc']].dropna()
rho, p = stats.spearmanr(sub['cafe_count_3000'], sub['price_doc'])
print(f"Spearman rho = {rho:.3f}, p-value = {p:.2e}  (n={len(sub)})")
print("Вывод:", "гипотеза подтверждается" if p < 0.05 and rho > 0 else "не подтверждается")

print("\n" + "="*70)
print("H3: Цена отличается между инвестиционными сделками и сделками для собственного проживания")
inv = clean.loc[clean['product_type'] == 'Investment', 'price_doc'].dropna()
own = clean.loc[clean['product_type'] == 'OwnerOccupier', 'price_doc'].dropna()
u, p = stats.mannwhitneyu(inv, own, alternative='two-sided')
print(f"Investment: медиана {inv.median():,.0f} ₽ (n={len(inv)})")
print(f"OwnerOccupier: медиана {own.median():,.0f} ₽ (n={len(own)})")
print(f"Mann-Whitney U = {u:.0f}, p-value = {p:.2e}")
print("Вывод:", "различие статистически значимо" if p < 0.05 else "различие не значимо")

print("\n" + "="*70)
print("H4 (бонус): Состояние отделки объекта (state) связано с ценой")
sub = clean[['state', 'price_doc']].dropna()
groups = [g['price_doc'].values for _, g in sub[sub['state'].isin([1,2,3,4])].groupby('state')]
h, p = stats.kruskal(*groups)
print(f"Kruskal-Wallis H = {h:.2f}, p-value = {p:.2e}  (n={len(sub[sub['state'].isin([1,2,3,4])])})")
print("Вывод:", "различие между группами значимо" if p < 0.05 else "не значимо")
