import numpy as np
import pandas as pd
from itertools import product
from sklearn.model_selection import TimeSeriesSplit
from sklearn.linear_model import Ridge
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_squared_error
from features import clean, make_pipeline

df = pd.read_csv("../data/train_pr3.csv")
df = clean(df)
df = df.dropna(subset=["price_doc"]).reset_index(drop=True)

y = np.log1p(df["price_doc"].values)
X = df
tscv = TimeSeriesSplit(n_splits=5)

def rmse_cv(model):
    scores = []
    for tr_idx, val_idx in tscv.split(X):
        pipe = make_pipeline(model)
        pipe.fit(X.iloc[tr_idx], y[tr_idx])
        pred = pipe.predict(X.iloc[val_idx])
        scores.append(np.sqrt(mean_squared_error(y[val_idx], pred)))
    return np.mean(scores), np.std(scores)

print("=== ridge: подбор alpha ===")
best_ridge = None
for alpha in [5, 10, 20, 30, 50, 100]:
    m, s = rmse_cv(Ridge(alpha=alpha))
    print(f"alpha={alpha:<6} rmse {m:.4f} +- {s:.4f}")
    if best_ridge is None or m < best_ridge[1]:
        best_ridge = (alpha, m)
print("лучший ridge:", best_ridge)

print("\n=== градиентный бустинг: подбор сетки ===")
grid = list(product(
    [0.03, 0.05, 0.1],      # learning_rate
    [3, 5, None],           # max_depth
    [100, 200],             # max_iter
    [20, 50],               # min_samples_leaf
))
best_gb = None
for lr, depth, n_iter, leaf in grid:
    model = HistGradientBoostingRegressor(
        learning_rate=lr, max_depth=depth, max_iter=n_iter,
        min_samples_leaf=leaf, random_state=42,
    )
    m, s = rmse_cv(model)
    print(f"lr={lr:<6} depth={str(depth):<5} iter={n_iter:<4} leaf={leaf:<4} rmse {m:.4f} +- {s:.4f}")
    if best_gb is None or m < best_gb[1]:
        best_gb = ((lr, depth, n_iter, leaf), m)
print("лучший бустинг:", best_gb)
