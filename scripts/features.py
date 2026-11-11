import sys
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, FunctionTransformer

NUM_COLS = [
    "full_sq", "life_sq", "kitch_sq", "num_room",
    "floor_ratio", "build_age", "dist_to_center", "cafe_density_3000", "is_investment",
    "metro_min_walk", "kindergarten_km", "school_km", "park_km", "green_zone_km",
    "shopping_centers_km", "public_transport_station_km", "office_km",
    "university_km", "workplaces_km", "catering_km",
]
CAT_COLS = ["material", "state", "sub_area"]

def clean(df):
    # убираем дубли и нереалистичные значения
    df = df.drop_duplicates(subset=["id"]).copy()
    df.loc[(df["full_sq"] < 10) | (df["full_sq"] > 300), "full_sq"] = np.nan
    df.loc[df["life_sq"] > df["full_sq"], "life_sq"] = np.nan
    df.loc[(df["kitch_sq"] > df["full_sq"]) | (df["kitch_sq"] > 100), "kitch_sq"] = np.nan
    df.loc[df["num_room"] > 10, "num_room"] = np.nan
    df.loc[(df["build_year"] < 1860) | (df["build_year"] > 2020), "build_year"] = np.nan
    df.loc[df["max_floor"] == 0, "max_floor"] = np.nan
    df.loc[df["floor"] > df["max_floor"], "floor"] = np.nan
    # флаг круглой цены
    if "price_doc" in df.columns:
        df["is_round_price"] = (df["price_doc"] % 1000000 == 0).astype(int)
    return df

def add_features(df):
    # считаем новые признаки по плану из пр2
    df = df.copy()
    df["dist_to_center"] = df[["kremlin_km", "sadovoe_km", "ttk_km", "bulvar_ring_km"]].mean(axis=1)
    df["cafe_density_3000"] = df[["cafe_count_1500", "cafe_count_2000", "cafe_count_3000", "cafe_count_5000"]].mean(axis=1)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df["build_age"] = df["timestamp"].dt.year - df["build_year"]
    df.loc[df["build_age"] < 0, "build_age"] = np.nan
    df["floor_ratio"] = df["floor"] / df["max_floor"]
    df["floor_ratio"] = df["floor_ratio"].replace([np.inf, -np.inf], np.nan)
    df["is_investment"] = (df["product_type"] == "Investment").astype(int)
    for col in ["material", "state", "sub_area"]:
        df[col] = df[col].astype(str).replace("nan", "missing")
    return df

def prepare_file(path):
    df = pd.read_csv(path)
    df = clean(df)
    df = add_features(df)
    return df

def make_pipeline(model):
    numeric = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
    ])
    categorical = Pipeline([
    ("impute", SimpleImputer(strategy="constant", fill_value="missing")),
    ("encode", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
])
    prep = ColumnTransformer([
        ("num", numeric, NUM_COLS),
        ("cat", categorical, CAT_COLS),
    ])
    return Pipeline([
        ("features", FunctionTransformer(add_features)),
        ("prep", prep),
        ("model", model),
    ])

if __name__ == "__main__":
    path = sys.argv[1]
    df1 = prepare_file(path)
    df2 = prepare_file(path)
    cols = NUM_COLS + CAT_COLS
    same = df1[cols].equals(df2[cols])
    print("воспроизводимость:", same)
    print(df1[cols].describe(include="all").T.head(15))
