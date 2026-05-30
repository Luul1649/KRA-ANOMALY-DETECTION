# KRA-ANOMALY-DETECTION
##Step 1: Setting up the Base Environment: 
We imported standard numerical libraries (pandas, numpy), data visualization tools (matplotlib, seaborn), and machine learning modules from scikit-learn.Why it matters: Real-world financial audit logs contain millions of data points. We need optimized, production-ready modules capable of processing math equations and rendering data charts at high speed.

##Step 2: Ingesting the Transaction Ledger Data: 
We loaded a 500,000-row subset of the Kaggle PaySim dataset, which contains mobile money data records.Why it matters: Since the Kenya Revenue Authority (KRA) holds highly confidential tax filings, we used PaySim as a high-quality "proxy." PaySim mirrors the exact structures of financial systems like M-Pesa (e.g., transfers, merchants, account balances). Loading a subset allows us to develop the model architecture quickly without burning out your computer's RAM.

##Step 3: Anomaly & Tax Feature Engineering
This is the most critical step. We transformed basic, raw ledger columns into specific "risk factors" that tax auditors look for:One-Hot Encoding (get_dummies): AI cannot read text words like "TRANSFER" or "CASH_OUT". This function converts payment types into mathematical columns of 1s and 0s.Balance Discrepancy Errors (sender_balance_error): In a clean transaction, if you send KES 10,000, your balance drops by exactly KES 10,000. If it doesn't, a balance anomaly exists. This feature flags hidden account manipulation or unregistered funds.High-Value Transaction Flags (is_high_value): We set a rule flagging transactions over KES 150,000. Tax authorities pay extra attention to high-value transactions because they often mask commercial trade under the guise of "personal peer-to-peer transfers" to evade corporate income tax.

##Step 4: Training the KRA Anomaly DetectorWhat we did: 
We split the data into an 80% training set and a 20% testing set using stratify=y_kra. Then we trained a Random Forest Classifier model.Why it matters: Financial fraud and tax evasion are rare events (e.g., only 1 out of 1,000 transactions might be illicit). This is known as imbalanced data. Using stratify ensures that both our training and testing sets get an equal percentage of anomalies. We used a Random Forest algorithm because it builds dozens of mini decision trees to cross-examine financial variables, making it highly accurate at spotting complex tax-dodging networks.

##Step 5: Evaluating the Auditing ModelWhat we did: 
We generated a Classification Report and a Confusion Matrix Heatmap.Why it matters: KRA executives do not care about raw code; they care about operational efficiency.The Classification Report proves how accurately the AI catches tax evaders (Recall) without falsely accusing innocent, compliant citizens (Precision).The Confusion Matrix provides a clean visual tally of your system's accurate hits versus missed targets.
