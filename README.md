# Facility Management Anomaly Detection — Week 4

**Track:** Specialist Advanced ML  
**Methods:** Isolation Forest, One-Class SVM, PyTorch Autoencoder  
**Budget cap:** PKR 0 (local CPU execution; no paid APIs/cloud)  
**Data:** Reproducible synthetic facility-management transactions. This is not real company data.

## Business scope

A family-owned facility management company processes maintenance, cleaning, repairs, utilities, and supplies transactions. The prototype ranks unusual transactions for human review using behavioural features such as transaction amount, deviation from a vendor's historical amount, recent transaction velocity, and time of day.

The system is a **review-prioritisation aid**, not a fraud verdict. Synthetic labels are used for evaluation only and are never model input features.

## Requirements

- Python 3.11 recommended
- scikit-learn, PyTorch, pandas, NumPy, Matplotlib, SciPy, pytest

## Setup (Windows / Anaconda Prompt)

```powershell
conda activate facility-anomaly
python -m pip install -r requirements.txt
```

If the environment was created already and the verification command reported `ModuleNotFoundError: No module named 'torch'`, install PyTorch into the active environment:

```powershell
python -m pip install torch
```

For GPU-specific installation options, use the official PyTorch selector: https://pytorch.org/get-started/locally/

Verify:

```powershell
python -c "import sklearn, torch, pandas, matplotlib; print('sklearn', sklearn.__version__); print('torch', torch.__version__)"
```

## Run the complete pipeline

From the project root:

```powershell
python -m src.run_pipeline
```

It creates the synthetic dataset, trains three detectors using chronological training/validation/test periods, selects cost thresholds on validation data, evaluates on the held-out test period, and exports tables and charts.

Expected generated files:
- `data/transactions.csv`
- `reports/results/dataset_summary.json`
- `reports/results/model_comparison.csv`
- `reports/results/cost_thresholds.csv`
- `reports/results/top_50_alerts.csv`
- `reports/figures/pr_curves.png`
- `reports/figures/threshold_costs.png`
- `reports/figures/label_distribution.png`

Run tests:

```powershell
python -m pytest -q
```

## Methodology

1. Generate 2,400 timestamped synthetic transactions with a target suspicious-event rate of 1.5%.
2. Engineer behavioural features using transaction timestamps and prior transactions only. The current transaction is excluded from its own rolling history.
3. Split chronologically: earliest 60% training, next 20% validation, latest 20% test.
4. Fit unsupervised detectors on the training period only. The autoencoder is trained on the training period's normal transactions.
5. Use validation labels to select a threshold minimising assumed validation cost; do not tune on the test set.
6. Report PR-AUC and Precision@50 on the untouched test period. Accuracy is included only as a cautionary baseline.

## Models and scores

- **Isolation Forest:** anomaly score is the negative of `score_samples` (higher means more anomalous).
- **One-Class SVM:** anomaly score is the negative of `decision_function` (higher means more anomalous). Features are standardised using training data only.
- **Autoencoder:** reconstruction mean-squared error (higher means more anomalous). A small feed-forward network is trained with PyTorch on normal training rows.

## Cost assumptions (illustrative, not company-verified)

- Missed suspicious event: PKR 20,000
- False alarm reviewed: PKR 500
- Review capacity reference: top 50 alerts per evaluation period
- Weekly workload estimate: alert count in a 7-day test slice, scaled only if the slice spans a different number of days.

These assumptions are adjustable in `src/config.py`. They are not evidence of actual losses or review costs. A cost-minimising threshold is a modelling estimate and should be reviewed with the business owner before real use.

## Validation and limitations

The pipeline checks dataset size, class imbalance, chronological ordering, finite feature values, label exclusion from model inputs, and output creation. The test set is held out from threshold selection.

**Limitations:** synthetic data may not capture real vendor behaviour, seasonality, accounting workflows, or adversarial adaptation. Some synthetic anomaly patterns are deliberately designed and may make detection easier than in real operations. Cost values are assumptions. A production system would require access controls, privacy review, human investigation, drift monitoring, and evaluation on appropriately governed real data.

## Week 3 reflection 

- **Three things done well in Week 3:** 
1.Built robust chunking and retrieval pipelines using vector embeddings.
2.Structured clear domain-specific documentation tailored to a family-owned enterprise.
3.Implemented clean source attribution and verification steps.**
- **Two gaps from Week 3:**
1.Limited quantitative performance evaluation metrics for retrieval relevance (e.g., recall@k or precision@k).
2.Did not experiment with unsupervised neural representations or anomaly detection architectures.
- **New Week 4 skill:** 
Unsupervised anomaly detection on heavily imbalanced tabular/behavioral data using Isolation Forest (scikit-learn) and Deep Autoencoders (PyTorch), evaluated via business-cost matrices and ranking metrics (PR-AUC, Precision@k).

