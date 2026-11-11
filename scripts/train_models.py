import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import TimeSeriesSplit
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
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

print("=== модели разной природы ===")
knn_mean, knn_std = rmse_cv(KNeighborsRegressor(n_neighbors=25))
print(f"knn (k=25)               rmse {knn_mean:.4f} +- {knn_std:.4f}")

tree5_mean, tree5_std = rmse_cv(DecisionTreeRegressor(max_depth=5, random_state=42))
print(f"дерево (глубина 5)       rmse {tree5_mean:.4f} +- {tree5_std:.4f}")

tree_full_mean, tree_full_std = rmse_cv(DecisionTreeRegressor(random_state=42))
print(f"дерево (без ограничения) rmse {tree_full_mean:.4f} +- {tree_full_std:.4f}")

print("\n=== ансамбли ===")
rf_mean, rf_std = rmse_cv(RandomForestRegressor(n_estimators=300, random_state=42, n_jobs=-1))
print(f"случайный лес (300)      rmse {rf_mean:.4f} +- {rf_std:.4f}")

gb_mean, gb_std = rmse_cv(HistGradientBoostingRegressor(random_state=42))
print(f"градиентный бустинг      rmse {gb_mean:.4f} +- {gb_std:.4f}")

# зависимость качества дерева от глубины
depths = [1, 2, 3, 4, 5, 6, 7, 8, 10, 12, 15, None]
train_scores, cv_scores = [], []
print("\n=== глубина дерева ===")
for d in depths:
    cv_mean, _ = rmse_cv(DecisionTreeRegressor(max_depth=d, random_state=42))
    pipe = make_pipeline(DecisionTreeRegressor(max_depth=d, random_state=42))
    pipe.fit(X, y)
    pred_train = pipe.predict(X)
    train_rmse = np.sqrt(mean_squared_error(y, pred_train))
    train_scores.append(train_rmse)
    cv_scores.append(cv_mean)
    print(f"глубина {str(d):<5} train rmse {train_rmse:.4f}   cv rmse {cv_mean:.4f}")

depths_x = [d if d is not None else 20 for d in depths]
plt.figure(figsize=(8, 5))
plt.plot(depths_x, train_scores, marker="o", label="train")
plt.plot(depths_x, cv_scores, marker="s", label="cv (5 блоков)")
plt.xlabel("глубина дерева (20 = без ограничения)")
plt.ylabel("rmse на log(цены)")
plt.title("качество дерева решений на train и cv в зависимости от глубины")
plt.legend()
plt.tight_layout()
plt.savefig("../figures/tree_depth.png", dpi=150)
print("график глубины дерева сохранён")

# польза новых признаков расширенный набор против исходного (без add_features)
RAW_NUM = [
    "full_sq", "life_sq", "kitch_sq", "num_room",
    "floor", "max_floor", "build_year",
    "kremlin_km", "sadovoe_km", "ttk_km", "bulvar_ring_km",
    "cafe_count_1500", "cafe_count_2000", "cafe_count_3000", "cafe_count_5000",
    "metro_min_walk", "kindergarten_km", "school_km", "park_km", "green_zone_km",
    "shopping_centers_km", "public_transport_station_km", "office_km",
    "university_km", "workplaces_km", "catering_km",
]
RAW_CAT = ["material", "state", "sub_area", "product_type"]

df_raw = df.copy()
for col in RAW_CAT:
    df_raw[col] = df_raw[col].astype(str).replace("nan", "missing")

def make_pipeline_raw(model):
    numeric = Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())])
    categorical = Pipeline([("impute", SimpleImputer(strategy="constant", fill_value="missing")), ("encode", OneHotEncoder(handle_unknown="ignore", sparse_output=False))])
    prep = ColumnTransformer([("num", numeric, RAW_NUM), ("cat", categorical, RAW_CAT)])
    return Pipeline([("prep", prep), ("model", model)])

def rmse_cv_raw(model):
    scores = []
    for tr_idx, val_idx in tscv.split(df_raw):
        pipe = make_pipeline_raw(model)
        pipe.fit(df_raw.iloc[tr_idx], y[tr_idx])
        pred = pipe.predict(df_raw.iloc[val_idx])
        scores.append(np.sqrt(mean_squared_error(y[val_idx], pred)))
    return np.mean(scores), np.std(scores)

print("\n=== польза новых признаков ===")
ridge_new_mean, _ = rmse_cv(Ridge(alpha=10))
ridge_raw_mean, _ = rmse_cv_raw(Ridge(alpha=10))
print(f"ridge (alpha=10)   новые признаки {ridge_new_mean:.4f}   исходные признаки {ridge_raw_mean:.4f}")

gb_new_mean, _ = rmse_cv(HistGradientBoostingRegressor(random_state=42))
gb_raw_mean, _ = rmse_cv_raw(HistGradientBoostingRegressor(random_state=42))
print(f"бустинг            новые признаки {gb_new_mean:.4f}   исходные признаки {gb_raw_mean:.4f}")
