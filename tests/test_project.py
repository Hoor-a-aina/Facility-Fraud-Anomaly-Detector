import numpy as np
from src.data_generation import generate_transactions
from src.feature_engineering import add_behavioural_features, FEATURE_COLUMNS
from src.evaluation import precision_at_k, threshold_cost_table

def test_dataset_meets_size_and_imbalance_requirements():
    df = generate_transactions(n_records=2400, anomaly_rate=.015, seed=7)
    assert len(df) >= 1400
    assert 0 < df["is_suspicious"].mean() < .02
    assert df["timestamp"].is_monotonic_increasing

def test_features_finite_and_no_label_feature():
    raw = generate_transactions(seed=7)
    feat = add_behavioural_features(raw)
    assert "is_suspicious" not in FEATURE_COLUMNS
    assert np.isfinite(feat[FEATURE_COLUMNS].to_numpy(dtype=float)).all()

def test_precision_at_k_known_example():
    assert precision_at_k([1, 0, 1, 0], [0.9, 0.8, 0.7, 0.1], k=2) == .5

def test_cost_table_contains_expected_columns():
    table = threshold_cost_table([1, 0, 0], [.9, .8, .1], thresholds=[.5])
    assert {"tp", "fp", "fn", "alerts", "estimated_cost_pkr"}.issubset(table.columns)
    assert int(table.iloc[0]["tp"]) == 1
