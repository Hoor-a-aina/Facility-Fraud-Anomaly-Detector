from pathlib import Path
import json

import pandas as pd
import streamlit as st

# --------------------------------------------------
# Configuration
# --------------------------------------------------
ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "reports" / "results"
FIGURES = ROOT / "reports" / "figures"

st.set_page_config(
    page_title="Facility Anomaly Detection",
    page_icon="🔎",
    layout="wide",
)

# --------------------------------------------------
# Helpers
# --------------------------------------------------
def load_csv(filename):
    path = RESULTS / filename
    if not path.exists():
        return None
    try:
        return pd.read_csv(path)
    except Exception as exc:
        st.error(f"Could not read {filename}: {exc}")
        return None

def show_download(df, filename, label):
    if df is not None:
        st.download_button(
            label=label,
            data=df.to_csv(index=False).encode("utf-8"),
            file_name=filename,
            mime="text/csv",
        )

def find_column(df, candidates):
    """Find a column without depending on capitalization."""
    lookup = {str(c).lower(): c for c in df.columns}
    for candidate in candidates:
        if candidate.lower() in lookup:
            return lookup[candidate.lower()]
    return None

# --------------------------------------------------
# Header and sidebar
# --------------------------------------------------
st.title("🔎 Facility Anomaly Detection")
st.caption(
    "Week 4 | Specialist Advanced ML | "
    "scikit-learn + PyTorch"
)

st.sidebar.title("Navigation")
page = st.sidebar.radio(
    "Explore the project",
    [
        "Overview",
        "Model Evaluation",
        "Cost Analysis",
        "Alert Review",
        "Project Information",
    ],
)

st.sidebar.info(
    "Budget cap: PKR 0\n\n"
    "Data: reproducible synthetic facility transactions"
)

st.warning(
    "Decision-support prototype only. An anomaly score is not "
    "proof of fraud. All alerts require human review."
)

# --------------------------------------------------
# Load actual pipeline outputs
# --------------------------------------------------
summary_path = RESULTS / "dataset_summary.json"
summary = None

if summary_path.exists():
    try:
        with open(summary_path, encoding="utf-8") as f:
            summary = json.load(f)
    except (OSError, json.JSONDecodeError):
        summary = None

metrics = load_csv("model_comparison.csv")
costs = load_csv("cost_thresholds.csv")
alerts = load_csv("top_50_alerts.csv")

transactions_path = ROOT / "data" / "transactions.csv"
transactions = None
if transactions_path.exists():
    try:
        transactions = pd.read_csv(transactions_path)
    except Exception:
        pass

# --------------------------------------------------
# Overview
# --------------------------------------------------
if page == "Overview":
    st.header("Project Overview")
    st.write(
        "This project prioritizes unusual maintenance, cleaning, "
        "repairs, utilities, and supplies transactions for review. "
        "It compares three unsupervised anomaly detection methods."
    )

    a, b, c, d = st.columns(4)

    if transactions is not None:
        total = len(transactions)
    else:
        total = None

    label_col = (
        find_column(transactions, ["label", "is_anomaly", "target", "is_suspicious"])
        if transactions is not None else None
    )

    positive_count = None
    positive_rate = None
    if transactions is not None and label_col is not None:
        labels = pd.to_numeric(transactions[label_col], errors="coerce")
        if labels.notna().any():
            positive_count = int((labels == 1).sum())
            positive_rate = float((labels == 1).mean() * 100)

    with a:
        st.metric("Transactions", f"{total:,}" if total is not None else "Unavailable")
    with b:
        st.metric(
            "Positive labels",
            f"{positive_count:,}" if positive_count is not None else "See summary",
        )
    with c:
        st.metric(
            "Positive rate",
            f"{positive_rate:.2f}%" if positive_rate is not None else "See summary",
        )
    with d:
        st.metric("Budget cap", "PKR 0")

    st.subheader("Detection methods")
    m1, m2, m3 = st.columns(3)

    with m1:
        st.markdown("### Isolation Forest")
        st.write("Detects unusual observations using random tree partitions.")

    with m2:
        st.markdown("### One-Class SVM")
        st.write("Learns a boundary around typical observations.")

    with m3:
        st.markdown("### PyTorch Autoencoder")
        st.write(
            "Learns to reconstruct normal observations; "
            "higher reconstruction error can indicate anomalies."
        )

    st.subheader("Dataset summary")
    if summary is not None:
        st.json(summary)
    elif transactions is not None:
        st.dataframe(transactions.head(10), use_container_width=True)
        st.caption("Showing the first 10 transaction rows.")
    else:
        st.info("Dataset summary is not available yet. Run the pipeline first.")

    st.subheader("Label distribution")
    label_chart = FIGURES / "label_distribution.png"
    if label_chart.exists():
        st.image(str(label_chart), use_container_width=True)
    else:
        st.info("label_distribution.png has not been generated.")

# --------------------------------------------------
# Model evaluation
# --------------------------------------------------
elif page == "Model Evaluation":
    st.header("Model Evaluation")
    st.write(
        "Evaluation should use the chronological held-out test period. "
        "PR-AUC and Precision@50 are more informative than accuracy alone "
        "when positive cases are rare."
    )

    if metrics is not None:
        st.dataframe(metrics, use_container_width=True)
        show_download(metrics, "model_comparison.csv", "Download model metrics")

        model_col = find_column(metrics, ["model", "model_name", "detector"])
        if model_col:
            st.subheader("Models in the results")
            st.write(", ".join(metrics[model_col].astype(str).tolist()))

        pr_col = find_column(
            metrics,
            ["pr_auc_average_precision", "pr_auc", "average_precision"],
        )
        p50_col = find_column(
            metrics,
            ["precision_at_50", "precision@50", "precision_at_k"],
        )

        if pr_col or p50_col:
            st.subheader("Key evaluation metrics")
            cols = st.columns(2)
            if pr_col:
                with cols[0]:
                    st.caption("PR-AUC / Average Precision")
                    st.dataframe(
                        metrics[[c for c in [model_col, pr_col] if c]],
                        use_container_width=True,
                    )
            if p50_col:
                with cols[1]:
                    st.caption("Precision at top 50")
                    st.dataframe(
                        metrics[[c for c in [model_col, p50_col] if c]],
                        use_container_width=True,
                    )
    else:
        st.warning("Model metrics are missing. Run python -m src.run_pipeline.")

    left, right = st.columns(2)

    with left:
        st.subheader("Precision–Recall Curves")
        path = FIGURES / "pr_curves.png"
        if path.exists():
            st.image(str(path), use_container_width=True)
        else:
            st.info("pr_curves.png not found.")

    with right:
        st.subheader("Threshold Cost Curve")
        path = FIGURES / "threshold_costs.png"
        if path.exists():
            st.image(str(path), use_container_width=True)
        else:
            st.info("threshold_costs.png not found.")

# --------------------------------------------------
# Cost analysis
# --------------------------------------------------
elif page == "Cost Analysis":
    st.header("Cost and Review Workload")
    st.write(
        "The following assumptions are illustrative and are not "
        "verified company losses or operational costs."
    )

    a, b, c = st.columns(3)
    a.metric("Missed suspicious event", "PKR 20,000")
    b.metric("False alarm", "PKR 500")
    c.metric("Budget cap", "PKR 0")

    st.caption(
        "Cost assumptions are defined in src/config.py. "
        "The displayed figures should match your current configuration."
    )

    if costs is not None:
        st.subheader("Validation threshold cost table")
        st.dataframe(costs, use_container_width=True)
        show_download(costs, "cost_thresholds.csv", "Download cost table")

        cost_col = find_column(
            costs, ["estimated_cost_pkr", "estimated_cost", "cost"]
        )
        review_col = find_column(
            costs, ["review_minutes", "review_time_minutes"]
        )
        alerts_col = find_column(costs, ["alerts", "alert_count"])

        if cost_col:
            numeric_cost = pd.to_numeric(costs[cost_col], errors="coerce")
            if numeric_cost.notna().any():
                st.metric(
                    "Lowest cost in displayed table",
                    f"PKR {numeric_cost.min():,.0f}",
                )
                st.caption(
                    "This is a modelled estimate under assumed costs, "
                    "not a verified business saving."
                )

        col1, col2 = st.columns(2)
        if alerts_col:
            with col1:
                st.metric(
                    "Alerts at first table row",
                    f"{costs.iloc[0][alerts_col]}",
                )
        if review_col:
            with col2:
                st.metric(
                    "Review minutes at first table row",
                    f"{costs.iloc[0][review_col]}",
                )

        st.caption(
            "The first row is the first row in the exported table. "
            "Confirm that the table was sorted by validation cost "
            "before interpreting it as the selected threshold."
        )
    else:
        st.warning("cost_thresholds.csv is missing.")

# --------------------------------------------------
# Alerts
# --------------------------------------------------
elif page == "Alert Review":
    st.header("Alerts for Human Review")
    st.write(
        "These are model-ranked candidates for investigation, "
        "not confirmed fraudulent transactions."
    )

    if alerts is not None:
        st.dataframe(alerts, use_container_width=True)
        show_download(alerts, "top_50_alerts.csv", "Download alert table")

        st.caption(f"Rows displayed: {len(alerts)}")
    else:
        st.warning("top_50_alerts.csv is missing.")

# --------------------------------------------------
# Project information
# --------------------------------------------------
elif page == "Project Information":
    st.header("Methodology and Limitations")

    st.subheader("Methodology")
    st.markdown(
        """
        - Synthetic facility-management transactions.
        - Chronological training, validation, and test periods.
        - Three unsupervised anomaly detectors.
        - Threshold selection using validation data and assumed costs.
        - Final evaluation using PR-AUC and Precision@50.
        """
    )

    st.subheader("Limitations")
    st.markdown(
        """
        - Synthetic data may not represent real vendor behaviour.
        - Cost assumptions are illustrative.
        - An anomaly does not establish fraud or misconduct.
        - Real deployment would require privacy controls, human review,
          drift monitoring, and evaluation on appropriately governed data.
        """
    )

    st.subheader("Available project files")
    for relative in [
        "data/transactions.csv",
        "reports/results/dataset_summary.json",
        "reports/results/model_comparison.csv",
        "reports/results/cost_thresholds.csv",
        "reports/results/top_50_alerts.csv",
        "reports/figures/pr_curves.png",
        "reports/figures/threshold_costs.png",
        "reports/figures/label_distribution.png",
    ]:
        path = ROOT / relative
        st.write(f"{'✅' if path.exists() else '❌'} `{relative}`")

st.divider()
st.caption(
    "Facility Anomaly Detection | Local CPU prototype | "
    "Synthetic data | Human review required"
)