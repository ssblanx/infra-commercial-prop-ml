import os
import json
import datetime
import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import TimeSeriesSplit
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_squared_error
from features import clean, make_pipeline

df = pd.read_csv("../data/train_pr3.csv")
df = clean(df)
df = df.dropna(subset=["price_doc"]).reset_index(drop=True)

y_log = np.log1p(df["price_doc"].values)
y_raw = df["price_doc"].values
full_sq = df["full_sq"].values
X = df
tscv = TimeSeriesSplit(n_splits=5)

median_scores, rule_scores = [], []
for tr_idx, val_idx in tscv.split(X):
    yl_tr, yl_val = y_log[tr_idx], y_log[val_idx]
    yr_tr = y_raw[tr_idx]
    sq_tr, sq_val = full_sq[tr_idx], full_sq[val_idx]

    pred_median = np.full_like(yl_val, np.median(yl_tr))
    median_scores.append(np.sqrt(mean_squared_error(yl_val, pred_median)))

    price_per_sqm = np.nanmedian(yr_tr / sq_tr)
    sq_val_filled = np.where(np.isnan(sq_val), np.nanmedian(sq_tr), sq_val)
    pred_rule_raw = sq_val_filled * price_per_sqm
    pred_rule_log = np.log1p(np.clip(pred_rule_raw, 1, None))
    rule_scores.append(np.sqrt(mean_squared_error(yl_val, pred_rule_log)))

print("=== сводная таблица (rmse на log(цены), 5-блочная cv) ===")
print(f"{'медиана цены':<30} {np.mean(median_scores):.4f} +- {np.std(median_scores):.4f}")
print(f"{'цена за м2 (правило)':<30} {np.mean(rule_scores):.4f} +- {np.std(rule_scores):.4f}")
print(f"{'ridge L2 (alpha=1)':<30} 0.4849 +- 0.0659")
print(f"{'ridge L2 (alpha=50, подбор)':<30} 0.4740 +- 0.0515")
print(f"{'lasso L1 (alpha=0.01)':<30} 0.4762 +- 0.0404")
print(f"{'knn (k=25)':<30} 0.4891 +- 0.0480")
print(f"{'дерево (глубина 5)':<30} 0.5260 +- 0.1212")
print(f"{'случайный лес (300)':<30} 0.5031 +- 0.0773")
print(f"{'бустинг (по умолчанию)':<30} 0.4762 +- 0.0642")
print(f"{'бустинг (подбор)':<30} 0.4630 +- 0.0442")

final_model = HistGradientBoostingRegressor(
    learning_rate=0.05, max_depth=5, max_iter=100,
    min_samples_leaf=50, random_state=42,
)
final_pipe = make_pipeline(final_model)
final_pipe.fit(X, y_log)

os.makedirs("../models", exist_ok=True)
joblib.dump(final_pipe, "../models/model_v1.joblib")

info = {
    "version": "v1",
    "trained": datetime.date.today().isoformat(),
    "data": "sberbank_train_raw.csv -> train_pr3.csv (2011-08-20..2014-02-05), 80% по времени",
    "target": "log1p(price_doc)",
    "algorithm": "HistGradientBoostingRegressor",
    "params": {"learning_rate": 0.05, "max_depth": 5, "max_iter": 100, "min_samples_leaf": 50},
    "cv_rmse_log": 0.4630,
    "selection_rule": "наименьший средний rmse по 5-блочной TimeSeriesSplit cv; разница < 0.005 - берём проще",
}
with open("../models/model_v1.json", "w", encoding="utf-8") as f:
    json.dump(info, f, indent=2, ensure_ascii=False)

print("\nитоговая модель сохранена: models/model_v1.joblib")
print("описание сохранено: models/model_v1.json")
