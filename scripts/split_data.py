import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW_PATH = ROOT / "data" / "raw" / "sberbank_train_raw.csv"
INCOMING_DIR = ROOT / "data" / "incoming"
INCOMING_DIR.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(RAW_PATH, parse_dates=["timestamp"])
df = df.sort_values("timestamp").reset_index(drop=True)

n_total = len(df)
cut = int(n_total * 0.6)
original = df.iloc[:cut].copy()
rest = df.iloc[cut:].copy()

# перезаписываем raw — там должно остаться только 60% "исходных" данных
original.to_csv(RAW_PATH, index=False)

# оставшиеся 40% делим на 4 порции по времени — "новые" поступления
n_batches = 4
edges = [int(i * len(rest) / n_batches) for i in range(n_batches + 1)]
for i in range(n_batches):
    part = rest.iloc[edges[i]:edges[i + 1]]
    part.to_csv(INCOMING_DIR / f"batch_{i+1:02d}.csv", index=False)
    print(f"batch_{i+1:02d}: {len(part)} rows, "
          f"{part['timestamp'].min().date()} — {part['timestamp'].max().date()}")

print(f"\noriginal (raw): {len(original)} rows, "
      f"{original['timestamp'].min().date()} — {original['timestamp'].max().date()}")
print(f"total: {n_total}")
