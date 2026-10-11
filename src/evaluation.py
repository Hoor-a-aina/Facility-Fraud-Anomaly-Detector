import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, precision_recall_curve, accuracy_score
from .config import TOP_K, MISSED_EVENT_COST_PKR, FALSE_ALARM_COST_PKR, REVIEWER_MINUTES_PER_ALERT

def precision_at_k(y_true, scores, k=TOP_K):
    y = np.asarray(y_true, dtype=int)
    s = np.asarray(scores, dtype=float)
    if len(y) == 0:
        return float("nan")
    k = min(k, len(y))
    idx = np.argsort(s)[::-1][:k]
    return float(y[idx].mean())

def threshold_cost_table(y_true, scores, thresholds=None):
    y = np.asarray(y_true, dtype=int)
    s = np.asarray(scores, dtype=float)
    if thresholds is None:
        thresholds = np.unique(np.quantile(s, np.linspace(0, 1, min(101, len(s) + 1))))
    rows = []
    for threshold in thresholds:
        pred = (s >= threshold).astype(int)
        fn = int(((y == 1) & (pred == 0)).sum())
        fp = int(((y == 0) & (pred == 1)).sum())
        tp = int(((y == 1) & (pred == 1)).sum())
        alerts = int(pred.sum())
        cost = fn * MISSED_EVENT_COST_PKR + fp * FALSE_ALARM_COST_PKR
        rows.append({"threshold": float(threshold), "tp": tp, "fp": fp, "fn": fn,
                     "alerts": alerts, "estimated_cost_pkr": int(cost),
                     "review_minutes": int(alerts * REVIEWER_MINUTES_PER_ALERT)})
    return pd.DataFrame(rows).sort_values(["estimated_cost_pkr", "alerts"]).reset_index(drop=True)

def evaluate_model(name, y_true, scores):
    y = np.asarray(y_true, dtype=int)
    s = np.asarray(scores, dtype=float)
    positives = int(y.sum())
    pr_auc = float(average_precision_score(y, s)) if len(np.unique(y)) > 1 else float("nan")
    p_at_k = precision_at_k(y, s, TOP_K)
    # Deliberately included only to show why accuracy can mislead.
    normal_only_accuracy = float((y == 0).mean())
    return {"model": name, "test_rows": len(y), "test_positives": positives,
            "test_positive_rate": float(y.mean()), "pr_auc_average_precision": pr_auc,
            f"precision_at_{TOP_K}": p_at_k,
            "all_normal_baseline_accuracy": normal_only_accuracy}

def choose_validation_threshold(y_val, scores_val):
    table = threshold_cost_table(y_val, scores_val)
    return table.iloc[0], table
