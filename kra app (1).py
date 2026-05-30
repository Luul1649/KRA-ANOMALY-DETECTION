import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
import seaborn as sns
import os

# 1. Page Configuration (KRA Auditing Theme)
st.set_page_config(page_title="Kaggle Tax Anomaly Console", page_icon="🏦", layout="wide")

st.title("🏦 AI Tax Anomaly & Compliance Engine (Financial Proxy)")
st.markdown("""
This production-grade interface demonstrates how revenue authorities can automate audit selection. 
Using financial data, this Proof of Concept maps financial anomalies to identify hidden commercial 
transactions and potential tax evasion frameworks.
""")

# Setup clean tabs
tab1, tab2 = st.tabs(["🔍 Auditor Risk Console", "📖 Data Framework Details"])

# 2. Cache Model and Data Loading
@st.cache_data
def load_and_train_kra():
    data_file = 'paysim_small.csv'
    
    # Check if file exists and has data
    if os.path.exists(data_file):
        df = pd.read_csv(data_file, nrows=50000)
    else:
        # Fallback dataset with exact matching column names
        np.random.seed(42)
        n_rows = 5000
        df = pd.DataFrame({
            'amount': np.random.exponential(scale=30000, size=n_rows),
            'oldbalanceOrg': np.random.exponential(scale=100000, size=n_rows),
            'oldbalanceDest': np.random.exponential(scale=200000, size=n_rows),
            'type': np.random.choice(['TRANSFER', 'CASH_OUT', 'CASH_IN', 'DEBIT', 'PAYMENT'], size=n_rows),
            'isFraud': np.random.choice([0, 1], size=n_rows, p=[0.99, 0.01])
        })
        df['newbalanceOrig'] = np.maximum(0, df['oldbalanceOrg'] - df['amount'])
        df['newbalanceDest'] = df['oldbalanceDest'] + df['amount']

    # Clean up column names (remove leading/trailing spaces)
    df.columns = df.columns.str.strip()

    # Safety: If 'type' is missing, add a dummy type column
    if 'type' not in df.columns:
        df['type'] = 'TRANSFER'

    # Process categorical features cleanly
    df = pd.get_dummies(df, columns=['type'], drop_first=False)
    
    # Force standard one-hot columns to exist so the app never throws feature errors
    required_types = ['type_CASH_OUT', 'type_DEBIT', 'type_PAYMENT', 'type_TRANSFER', 'type_CASH_IN']
    for t_col in required_types:
        if t_col not in df.columns:
            df[t_col] = 0
            
    # Feature Engineering (Discrepancy Indicators)
    df['sender_balance_error'] = df['newbalanceOrig'] + df['amount'] - df['oldbalanceOrg']
    df['receiver_balance_error'] = df['oldbalanceDest'] + df['amount'] - df['newbalanceDest']
    df['is_high_value'] = (df['amount'] > 150000).astype(int)
    
    # Explicitly separate inputs and targets
    drop_cols = ['nameOrig', 'nameDest', 'isFlaggedFraud', 'isFraud']
    features = [
        'amount', 'oldbalanceOrg', 'newbalanceOrig', 'oldbalanceDest', 'newbalanceDest',
        'type_CASH_OUT', 'type_DEBIT', 'type_PAYMENT', 'type_TRANSFER', 'type_CASH_IN',
        'sender_balance_error', 'receiver_balance_error', 'is_high_value'
    ]
    
    # Intersect features to make sure they exist in the dataframe
    features = [f for f in features if f in df.columns]
    
    X = df[features].astype(float)
    y = df['isFraud'].astype(int) if 'isFraud' in df.columns else np.zeros(len(df))
    
    # Train the Random Forest Classifier
    model = RandomForestClassifier(n_estimators=20, random_state=42, n_jobs=-1)
    model.fit(X, y)
    
    return model, features, df

with st.spinner("⏳ Training financial audit intelligence model..."):
    kra_model, kra_features, raw_df = load_and_train_kra()

with tab1:
    # 3. Sidebar inputs for transaction auditing simulation
    st.sidebar.header("🧾 Transaction Audit Parameters")
    
    tx_amount = st.sidebar.number_input("Transaction Amount (KES/Units)", value=25000.0, step=1000.0)
    tx_type = st.sidebar.selectbox("Transaction Channel Type", ["TRANSFER", "CASH_OUT", "CASH_IN", "DEBIT", "PAYMENT"])
    
    old_orig = st.sidebar.number_input("Sender Initial Balance", value=50000.0)
    new_orig = st.sidebar.number_input("Sender Post-Transaction Balance", value=25000.0)
    
    old_dest = st.sidebar.number_input("Recipient Initial Balance", value=10000.0)
    new_dest = st.sidebar.number_input("Recipient Post-Transaction Balance", value=35000.0)

    # 4. Process Inputs to Match Model Features
    sender_err = new_orig + tx_amount - old_orig
    receiver_err = old_dest + tx_amount - new_dest
    high_val = 1 if tx_amount > 150000 else 0
    
    # Build input dictionary mapping exactly to model features
    input_dict = {feat: 0.0 for feat in kra_features}
    if 'amount' in input_dict: input_dict['amount'] = float(tx_amount)
    if 'oldbalanceOrg' in input_dict: input_dict['oldbalanceOrg'] = float(old_orig)
    if 'newbalanceOrig' in input_dict: input_dict['newbalanceOrig'] = float(new_orig)
    if 'oldbalanceDest' in input_dict: input_dict['oldbalanceDest'] = float(old_dest)
    if 'newbalanceDest' in input_dict: input_dict['newbalanceDest'] = float(new_dest)
    if 'sender_balance_error' in input_dict: input_dict['sender_balance_error'] = float(sender_err)
    if 'receiver_balance_error' in input_dict: input_dict['receiver_balance_error'] = float(receiver_err)
    if 'is_high_value' in input_dict: input_dict['is_high_value'] = float(high_val)
    
    # Set selected category flag
    type_col = f"type_{tx_type}"
    if type_col in input_dict:
        input_dict[type_col] = 1.0
        
    input_df = pd.DataFrame([input_dict])[kra_features]
    
    # Run predictions
    risk_prediction = kra_model.predict(input_df)
    risk_proba = kra_model.predict_proba(input_df)[0][1] * 100

    # Layout Outputs
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🔮 Transaction Risk Assessment")
        st.metric(label="Anomalous Tax Evasion Probability", value=f"{risk_proba:.2f}%")
        
        if risk_prediction == 1 or risk_proba > 50:
            st.error("🚨 **HIGH RISK OUTLIER AUDIT FLAG:** Transaction exhibits significant variance. Recommended for automated eTIMS commercial reconciliation audit.")
        else:
            st.success("🟢 **COMPLIANT PATTERN:** Transaction falls within standard usage profiles. Low audit priority.")

    with col2:
        st.subheader("📊 System Risk Diagnostics")
        st.markdown(f"""
        * **Computed Sender Balance Discrepancy:** {sender_err:,.2f}
        * **Computed Recipient Balance Discrepancy:** {receiver_err:,.2f}
        * **High Value Classification (>150k):** {'Yes' if high_val == 1 else 'No'}
        """)

    st.markdown("---")
    st.subheader("📈 Ledger Profile Distributions")
    fig, ax = plt.subplots(figsize=(10, 3))
    sns.boxplot(x=raw_df['amount'], color='#2b5c8f', ax=ax)
    ax.set_title("Distribution of Global Ledger Transaction Amounts")
    st.pyplot(fig)

with tab2:
    st.subheader("📋 Methodology & Transparency")
    st.markdown("""
    ### Project Intent
    This system models how digital ledgers can be automatically audited for revenue leakage. 
    By looking at internal discrepancies within account balances rather than just transaction sizes, 
    the engine uncovers structural tax evasion footprints.
    
    ### Data Source Transparency
    * **Proxy Source:** Kaggle PaySim Dataset / Local System Simulation.
    * **Compliance Baseline:** Engineered to align with digital revenue assurance objectives tracked globally by modern revenue authorities like the Kenya Revenue Authority (KRA).
    """)
