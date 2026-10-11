# Project Summary & Technical Report: Fraud & Anomaly Detection on Imbalanced Data

**Project Title:** Fraud / Anomaly Detection on Imbalanced Data: Isolation Forest vs. Autoencoder for a Family-Owned Facility Management Company  
**Author:** Hoor Aina (Member 6, SDC Internship Program, SSUET)  
**Week:** 4 — Specialist Advanced Build  
**Primary Stack:** Python, scikit-learn, PyTorch, pandas, Streamlit  

---

## 1. Executive Summary & Scope Statement
The Family-Owned Facility Management Company manages commercial and residential properties, overseeing operational expenses, subcontractor payouts, equipment maintenance billing, and employee petty cash transactions. This project implements an automated unsupervised anomaly detection pipeline to flag rare suspicious transactions across operational records. 

By comparing tree-based partitioning (Isolation Forest), boundary learning (One-Class SVM), and neural reconstruction error (PyTorch Deep Autoencoder), the system optimizes alert triage to minimize financial loss from undetected fraud while keeping false-positive review workloads sustainable for the finance team.

* **Budget & Cost Adherence:** Maintained a strict **PKR 0** budget by utilizing 100% open-source local Python tools and CPU-based execution. All costs and resource usages are logged in the project repository.

---

## 2. Week 3 Reflection & Week 4 Transition
* **3 Things Done Well in Week 3 (Telecom RAG Assistant):**
  1. Built robust chunking and vector embedding retrieval pipelines.
  2. Structured clear domain documentation tailored to family-owned enterprise needs.
  3. Implemented clean source attribution and testing practices.
* **2 Gaps Identified in Week 3:**
  1. Lacked quantitative evaluation metrics for retrieval relevance (such as precision@k or recall curves).
  2. Did not explore deep unsupervised neural architectures or tabular anomaly detection.
* **New Skill Practiced in Week 4:** Advanced unsupervised anomaly detection on heavily imbalanced behavioral tabular data using **scikit-learn** and **PyTorch Autoencoders**, evaluated through ranking metrics (PR-AUC, Precision@k) and business-cost matrices.

---

## 3. Dataset & The Illusion of Accuracy
Standard accuracy is deceptive in rare-event anomaly detection. In our held-out test evaluation:
* **Total Test Rows:** 480 transactions
* **Actual Positives (Anomalies):** 8 instances (Positive rate: 1.67%)
* **All-Normal Baseline Accuracy:** **98.33%**

A naive model predicting "Normal" for every transaction achieves 98.33% accuracy while completely missing 100% of the anomalies. Therefore, evaluation relied strictly on ranking metrics (**PR-AUC** and **Precision@50**).

---

## 4. Model Training & Evaluation Results
Models were evaluated on the chronologically held-out test split (480 records). Below are the exact results extracted from `model_comparison.csv`:

| Model | Test Rows | Test Positives | PR-AUC (Average Precision) | Precision @ Top 50 | Baseline Accuracy |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **PyTorch Autoencoder** | 480 | 8 | **0.2390** | **0.06** | 98.33% |
| **Isolation Forest** | 480 | 8 | **0.2059** | **0.08** | 98.33% |
| **One-Class SVM** | 480 | 8 | **0.1441** | **0.02** | 98.33% |

*Key Insight:* The **PyTorch Autoencoder** achieved the highest PR-AUC (**0.2390**), demonstrating superior non-linear pattern capture through reconstruction error, while **Isolation Forest** achieved the highest top-tier precision (**0.08** Precision@50) for immediate alert triage.

---

## 5. Cost Analysis & Threshold Optimization
To translate anomaly scores into actionable business value, an explicit cost matrix was established for the facility management company:
* **Cost of Undetected Suspicious Event (False Negative):** PKR 20,000 per missed incident.
* **Cost of False Alarm Investigation (False Positive):** PKR 500 per reviewed false alert.

Threshold optimization via `cost_thresholds.csv` confirms that reviewing the top alerts per cycle minimizes overall business risk and keeps the weekly review workload sustainable for the compliance team.

---

## 6. Limitations & Future Improvements
* **Limitations:**
  1. Synthetic data, while distributionally realistic, cannot fully encapsulate complex, evolving human fraud tactics.
  2. Deep autoencoders required careful tuning of hidden layer dimensions and learning rates to prevent overfitting on normal training data.
* **Future Improvements:**
  1. Integrate sequence-based deep learning (such as LSTM Autoencoders) to capture multi-step temporal anomalies across extended vendor billing cycles.
  2. Implement an active learning feedback loop where manual finance team triage labels directly update detector weights.

---
*Submitted as the official Week 4 Specialist Advanced Deliverable for the SDC Internship Program.*