import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error
from features import clean

df = pd.read_csv("../data/train_pr3.csv")
df = clean(df)
df = df.dropna(subset=["full_sq", "price_doc"]).reset_index(drop=True)

y = df["price_doc"].values
full_sq = df["full_sq"].values

tscv = TimeSeriesSplit(n_splits=5)

def rmse(a, b):
    return np.sqrt(mean_squared_error(a, b))

median_rmse, median_mae = [], []
rule_rmse, rule_mae = [], []

for train_idx, val_idx in tscv.split(df):
    y_train, y_val = y[train_idx], y[val_idx]
    sq_train, sq_val = full_sq[train_idx], full_sq[val_idx]

    # базовое решение 1 - медиана цены
    pred_median = np.full_like(y_val, fill_value=np.median(y_train), dtype=float)
    median_rmse.append(rmse(y_val, pred_median))
    median_mae.append(mean_absolute_error(y_val, pred_median))

    # базовое решение 2 - цена за м2 * площадь
    price_per_sqm = np.median(y_train / sq_train)
    pred_rule = sq_val * price_per_sqm
    rule_rmse.append(rmse(y_val, pred_rule))
    rule_mae.append(mean_absolute_error(y_val, pred_rule))

print("медиана цены   rmse %.0f +- %.0f   mae %.0f" % (np.mean(median_rmse), np.std(median_rmse), np.mean(median_mae)))
print("цена за м2     rmse %.0f +- %.0f   mae %.0f" % (np.mean(rule_rmse), np.std(rule_rmse), np.mean(rule_mae)))
