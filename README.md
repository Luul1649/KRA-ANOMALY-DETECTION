# 🏦 Financial Ledger Anomaly Detection: A Framework for Revenue Assurance (KRA Proxy)

An end-to-end Machine Learning and risk profiling console built to demonstrate how digital financial ledgers can be automatically audited for revenue leakage. Using a high-fidelity **Kaggle Mobile Money Dataset** as a financial proxy, this application represents a **Proof of Concept (PoC)** designed to align with the automated audit selection and compliance tracking objectives of modern revenue bodies like the **Kenya Revenue Authority (KRA)**.

## 📊 Project Performance & Architecture
* **Source Dataset:** Kaggle PaySim Synthetic Mobile Money Transaction Logs
* **Algorithm:** Random Forest Classifier (Ensemble Decision Trees)
* **Interface Deployment:** Streamlit Community Cloud
* **Key Features:** Balance Discrepancy Errors (`sender_balance_error`, `receiver_balance_error`), Channel One-Hot Coding, and High-Value Transaction Isolation.

## ⚙️ Core Engineering Pipeline
1. **Dynamic Schema Mapping:** Built a normalization layer using `.str.lower()` and `.str.strip()` to dynamically parse incoming data streams, correcting case mismatches or hidden trailing spaces in the ledger.
2. **Tax Feature Engineering:** Created internal ledger accounting discrepancy metrics. In a valid transfer, account changes must match the transfer value perfectly; deviations are engineered as feature inputs to train the AI on anomalies.
3. **Data Imbalance Safeguards:** Handled highly imbalanced financial fraud classes by isolating minority outliers without generating false positives on standard personal peer-to-peer usage profiles.
4. **Structural Fail-Safe:** Integrated a programmatic data generation fallback layer. If the primary CSV stream fails to load or corrupts, the pipeline generates valid structural matrices on the fly to guarantee 100% app uptime.

## 🚀 Deployment & Local Replication

### 1. Requirements (`requirements.txt`)
```text
streamlit
pandas
numpy
scikit-learn
matplotlib
seaborn
```

### 2. Execution
```bash
streamlit run app.py
```
