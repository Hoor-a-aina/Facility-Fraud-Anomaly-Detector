import numpy as np
import pandas as pd

FEATURE_COLUMNS = [
    "amount_log", "vendor_amount_deviation", "vendor_prior_count",
    "vendor_recent_velocity", "hour_sin", "hour_cos",
    "after_hours", "weekend", "category_frequency"
]

def add_behavioural_features(df):
    """Create leakage-aware features using strictly prior transactions."""
    data = df.copy()
    data["timestamp"] = pd.to_datetime(data["timestamp"])
    data = data.sort_values("timestamp", kind="stable").reset_index(drop=True)
    data["amount_log"] = np.log1p(data["amount_pkr"].clip(lower=0))

    grouped = data.groupby("vendor_id", sort=False)["amount_pkr"]
    prior_mean = grouped.transform(lambda s: s.shift(1).expanding(min_periods=1).mean())
    data["vendor_prior_count"] = data.groupby("vendor_id", sort=False).cumcount().astype(float)
    data["vendor_amount_deviation"] = (
        (data["amount_pkr"] - prior_mean).abs() / (prior_mean.abs() + 1.0)
    ).fillna(0.0)

    # Count only same-vendor transactions in the 24 hours before each row.
    # Positional indices avoid duplicate-timestamp alignment problems.
    timestamps = data["timestamp"].to_numpy()
    vendors = data["vendor_id"].to_numpy()
    velocity = np.zeros(len(data), dtype=float)
    for i in range(len(data)):
        lower = timestamps[i] - np.timedelta64(24, "h")
        velocity[i] = np.sum((vendors[:i] == vendors[i]) &
                             (timestamps[:i] < timestamps[i]) &
                             (timestamps[:i] >= lower))
    data["vendor_recent_velocity"] = velocity

    hours = data["timestamp"].dt.hour + data["timestamp"].dt.minute / 60.0
    data["hour_sin"] = np.sin(2 * np.pi * hours / 24)
    data["hour_cos"] = np.cos(2 * np.pi * hours / 24)
    data["after_hours"] = (((hours < 6) | (hours >= 22))).astype(int)
    data["weekend"] = (data["timestamp"].dt.dayofweek >= 5).astype(int)
    data["category_frequency"] = data.groupby("category", sort=False).cumcount().astype(float)
    return data.reset_index(drop=True)
