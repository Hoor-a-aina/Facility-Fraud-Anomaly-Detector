import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve, average_precision_score

from .config import SEED, TRAIN_FRACTION, VALIDATION_FRACTION
from .data_generation import generate_transactions, save_dataset
from .feature_engineering import add_behavioural_features, FEATURE_COLUMNS
from .models import fit_score_models
from .evaluation import evaluate_model, threshold_cost_table, choose_validation_threshold

ROOT = Path(__file__).resolve().parents[1]

def main():
    (ROOT / "data").mkdir(exist_ok=True)
    (ROOT / "reports/results").mkdir(parents=True, exist_ok=True)
    (ROOT / "reports/figures").mkdir(parents=True, exist_ok=True)

    raw = generate_transactions()
    save_dataset(raw, ROOT / "data/transactions.csv")
    data = add_behavioural_features(raw)
    n = len(data)
    train_end = int(n * TRAIN_FRACTION)
    val_end = int(n * (TRAIN_FRACTION + VALIDATION_FRACTION))
    train, val, test = data.iloc[:train_end].copy(), data.iloc[train_end:val_end].copy(), data.iloc[val_end:].copy()

    if not (train.timestamp.max() < val.timestamp.min() and val.timestamp.max() < test.timestamp.min()):
        raise AssertionError("Chronological split order failed")
    if len(data) < 1400 or data.is_suspicious.mean() >= .02:
        raise AssertionError("Dataset size or imbalance requirement failed")

    # Exclude identifiers, categories, timestamps, and the label from features.
    X_train, X_val, X_test = train[FEATURE_COLUMNS], val[FEATURE_COLUMNS], test[FEATURE_COLUMNS]
    y_train, y_val, y_test = train["is_suspicious"].to_numpy(), val["is_suspicious"].to_numpy(), test["is_suspicious"].to_numpy()
    if "is_suspicious" in FEATURE_COLUMNS:
        raise AssertionError("Label leakage: label included as feature")
    if not np.isfinite(X_train.to_numpy(dtype=float)).all():
        raise AssertionError("Non-finite training features")

    scores = fit_score_models(X_train, X_val, X_test, y_train=y_train, seed=SEED)
    comparison = []
    cost_rows = []
    top_rows = []
    plt.figure(figsize=(8, 6))
    for model_name, (sv, st) in scores.items():
        comparison.append(evaluate_model(model_name, y_test, st))
        best_row, val_costs = choose_validation_threshold(y_val, sv)
        # Re-score threshold on held-out test without changing threshold.
        pred = (st >= float(best_row["threshold"])).astype(int)
        test_fn = int(((y_test == 1) & (pred == 0)).sum())
        test_fp = int(((y_test == 0) & (pred == 1)).sum())
        test_tp = int(((y_test == 1) & (pred == 1)).sum())
        alerts = int(pred.sum())
        cost_rows.append({
            "model": model_name, "validation_selected_threshold": float(best_row["threshold"]),
            "test_tp": test_tp, "test_fp": test_fp, "test_fn": test_fn,
            "test_alerts": alerts,
            "test_estimated_cost_pkr": test_fn * 20000 + test_fp * 500,
            "test_review_minutes": alerts * 5,
            "threshold_selected_on": "validation only"
        })
        ranked = test[["transaction_id", "timestamp", "vendor_id", "category", "amount_pkr", "is_suspicious"]].copy()
        ranked["model"] = model_name
        ranked["anomaly_score"] = st
        top_rows.append(ranked.sort_values("anomaly_score", ascending=False).head(50))

        precision, recall, _ = precision_recall_curve(y_test, st)
        ap = average_precision_score(y_test, st) if len(np.unique(y_test)) > 1 else float("nan")
        plt.plot(recall, precision, label=f"{model_name} (AP={ap:.3f})")

    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision–Recall Curves (chronological test period)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(ROOT / "reports/figures/pr_curves.png", dpi=160)
    plt.close()

    comparison_df = pd.DataFrame(comparison)
    comparison_df.to_csv(ROOT / "reports/results/model_comparison.csv", index=False)
    pd.DataFrame(cost_rows).to_csv(ROOT / "reports/results/cost_thresholds.csv", index=False)
    pd.concat(top_rows, ignore_index=True).to_csv(ROOT / "reports/results/top_50_alerts.csv", index=False)

    summary = {
        "records": int(len(data)),
        "suspicious_records": int(data.is_suspicious.sum()),
        "positive_rate": float(data.is_suspicious.mean()),
        "train_records": int(len(train)), "validation_records": int(len(val)), "test_records": int(len(test)),
        "train_period": [str(train.timestamp.min()), str(train.timestamp.max())],
        "validation_period": [str(val.timestamp.min()), str(val.timestamp.max())],
        "test_period": [str(test.timestamp.min()), str(test.timestamp.max())],
        "feature_columns": FEATURE_COLUMNS,
        "cost_assumptions_pkr": {"missed_event": 20000, "false_alarm": 500, "review_minutes_per_alert": 5},
        "budget_cap_pkr": 0
    }
    (ROOT / "reports/results/dataset_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    plt.figure(figsize=(6, 4))
    plt.bar(["Normal", "Suspicious"], [(data.is_suspicious == 0).sum(), (data.is_suspicious == 1).sum()])
    plt.title("Synthetic label distribution")
    plt.ylabel("Transactions")
    plt.tight_layout()
    plt.savefig(ROOT / "reports/figures/label_distribution.png", dpi=160)
    plt.close()

    costs = pd.DataFrame(cost_rows)
    plt.figure(figsize=(8, 4))
    plt.bar(costs["model"], costs["test_estimated_cost_pkr"])
    plt.title("Illustrative cost at validation-selected thresholds (test period)")
    plt.ylabel("Estimated PKR")
    plt.xticks(rotation=15, ha="right")
    plt.tight_layout()
    plt.savefig(ROOT / "reports/figures/threshold_costs.png", dpi=160)
    plt.close()

    print("\nDATA SUMMARY")
    print(json.dumps(summary, indent=2))
    print("\nMODEL COMPARISON")
    print(comparison_df.to_string(index=False))
    print("\nVALIDATION-SELECTED THRESHOLD / HELD-OUT TEST COST")
    print(costs.to_string(index=False))
    print("\nCharts and tables saved under reports/. Inspect the results before writing conclusions.")

if __name__ == "__main__":
    main()
