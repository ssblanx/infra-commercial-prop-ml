import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import TimeSeriesSplit
from sklearn.linear_model import Ridge, Lasso
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

alphas = [0.001, 0.01, 0.1, 1, 10]
ridge_mean, ridge_std = [], []
lasso_mean, lasso_std, lasso_nonzero = [], [], []

print("alpha   ridge rmse(log)       lasso rmse(log)        lasso ненулевых")
for alpha in alphas:
    r_mean, r_std = rmse_cv(Ridge(alpha=alpha))
    l_mean, l_std = rmse_cv(Lasso(alpha=alpha, max_iter=5000))

    l_full = make_pipeline(Lasso(alpha=alpha, max_iter=5000))
    l_full.fit(X, y)
    n_nonzero = int(np.sum(l_full.named_steps["model"].coef_ != 0))

    ridge_mean.append(r_mean); ridge_std.append(r_std)
    lasso_mean.append(l_mean); lasso_std.append(l_std)
    lasso_nonzero.append(n_nonzero)

    print(f"{alpha:<8}{r_mean:.4f} +- {r_std:.4f}   {l_mean:.4f} +- {l_std:.4f}   {n_nonzero}")

# график: rmse(alpha) + число ненулевых коэффициентов lasso
fig, ax1 = plt.subplots(figsize=(8, 5))
ax1.errorbar(alphas, ridge_mean, yerr=ridge_std, marker="o", label="ridge rmse (log)")
ax1.errorbar(alphas, lasso_mean, yerr=lasso_std, marker="s", label="lasso rmse (log)")
ax1.set_xscale("log")
ax1.set_xlabel("alpha (сила регуляризации)")
ax1.set_ylabel("rmse на log(цены)")
ax1.legend(loc="upper left")

ax2 = ax1.twinx()
ax2.plot(alphas, lasso_nonzero, marker="^", color="gray", linestyle="--", label="ненулевых коэф. lasso")
ax2.set_ylabel("число ненулевых коэффициентов (lasso)")
ax2.legend(loc="upper right")

plt.title("влияние регуляризации на качество и число коэффициентов")
plt.tight_layout()
plt.savefig("../figures/regularization_effect.png", dpi=150)
print("график сохранён")

# анализ коэффициентов ridge alpha=1
best = make_pipeline(Ridge(alpha=1))
best.fit(X, y)
names = best.named_steps["prep"].get_feature_names_out()
coefs = best.named_steps["model"].coef_
order = np.argsort(-np.abs(coefs))[:15]
print("\nтоп коэффициентов ridge (alpha=1):")
for i in order:
    print(f"{names[i]:<30} {coefs[i]:+.4f}")