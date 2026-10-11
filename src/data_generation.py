from pathlib import Path
import numpy as np
import pandas as pd
from .config import SEED, N_RECORDS, TARGET_ANOMALY_RATE

def generate_transactions(n_records=N_RECORDS, anomaly_rate=TARGET_ANOMALY_RATE, seed=SEED):
    """Generate reproducible synthetic transactions; labels are evaluation-only."""
    if n_records < 1400:
        raise ValueError("n_records must be at least 1,400")
    if not 0 < anomaly_rate < 0.02:
        raise ValueError("anomaly_rate must be greater than 0 and below 2%")

    rng = np.random.default_rng(seed)
    timestamps = pd.date_range("2025-01-01", periods=n_records, freq="3h")
    vendors = np.array([f"V{v:03d}" for v in range(1, 31)])
    vendor = rng.choice(vendors, size=n_records, p=None)
    categories = np.array(["maintenance", "cleaning", "repairs", "utilities", "supplies"])
    category = rng.choice(categories, size=n_records, p=[.25, .20, .22, .13, .20])
    base = {"maintenance": 18000, "cleaning": 8500, "repairs": 24000,
            "utilities": 32000, "supplies": 6500}
    category_mean = np.array([base[c] for c in category], dtype=float)
    amount = rng.lognormal(mean=np.log(category_mean) - 0.5 * .45**2, sigma=.45)
    hour = timestamps.hour.to_numpy()
    after_hours = ((hour < 6) | (hour >= 22)).astype(int)
    weekend = (timestamps.dayofweek >= 5).astype(int)

    labels = np.zeros(n_records, dtype=int)
    n_anomalies = max(1, int(round(n_records * anomaly_rate)))
    # Spread events across the timeline to support temporal evaluation.
    anomaly_indices = rng.choice(np.arange(10, n_records), size=n_anomalies, replace=False)
    labels[anomaly_indices] = 1

    for i in anomaly_indices:
        kind = rng.choice(["amount_spike", "after_hours", "repeat_vendor"])
        if kind == "amount_spike":
            amount[i] *= rng.uniform(5, 12)
        elif kind == "after_hours":
            timestamps = timestamps.copy()
            timestamps = timestamps.to_series()
            timestamps.iloc[i] = timestamps.iloc[i].normalize() + pd.Timedelta(hours=int(rng.choice([1, 2, 3, 23])))
            timestamps = pd.DatetimeIndex(timestamps)
            hour = timestamps.hour.to_numpy()
            after_hours = ((hour < 6) | (hour >= 22)).astype(int)
        else:
            if i > 0:
                vendor[i] = vendor[i-1]
                amount[i] = amount[i-1] * rng.uniform(.95, 1.05)

    # Recompute calendar fields after synthetic timing changes.
    df = pd.DataFrame({
        "transaction_id": [f"TX{i+1:06d}" for i in range(n_records)],
        "timestamp": timestamps,
        "vendor_id": vendor,
        "category": category,
        "amount_pkr": np.round(amount, 2),
        "after_hours": after_hours,
        "weekend": weekend,
        "is_suspicious": labels,
    }).sort_values("timestamp", kind="stable").reset_index(drop=True)
    # IDs are retained as identifiers, not model inputs.
    return df

def save_dataset(df, path="data/transactions.csv"):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return path
