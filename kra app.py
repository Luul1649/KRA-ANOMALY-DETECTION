import streamlit as st
pip install scikit-learn
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Page Configuration (KRA Auditing Theme)
st.set_page_config(page_title="Kaggle Tax Anomaly Console", page_icon="🏦", layout="wide")

st.title("🏦 AI Tax Anomaly & Compliance Engine (Kaggle Financial Proxy)")
st.markdown("""
This interface demonstrates how revenue authorities can automate audit selection. 
Using a **Kaggle Mobile Money Dataset**, this Proof of Concept maps financial anomalies 
to identify hidden commercial transactions and potential tax evasion frameworks.
""")

# Setup clean tabs
tab1, tab2 = st.tabs(["🔍 Auditor Risk Console", "📖 Data Framework Details"])

# 2. Cache Model and Data Loading
@st.cache_data
def load_and_train_kra():
    # Load dataset subset safely
    df = pd.read_csv('PS_20174392719_1491204439457_log.csv', nrows=200000)
    
    # Process types
    df = pd.get_dummies(df, columns=['type'], drop_first=True)
    
    # Engineering Features
    df['sender_balance_error'] = df['newbalanceOrig'] + df['amount'] - df['oldbalanceOrg']
    df['receiver_balance_error'] = df['oldbalanceDest'] + df['amount'] - df['newbalanceDest']
    df['is_high_value'] = (df['amount'] > 150000).astype(int)
    
    # Drop labels
    drop_cols = ['nameOrig', 'nameDest', 'isFlaggedFraud', 'isFraud']
    features = [col for col in df.columns if col not in drop_cols]
    
    X = df[features]
    y = df['isFraud']
    
    # Build light Random Forest for speed in Streamlit
    model = RandomForestClassifier(n_estimators=20, random_state=42, n_jobs=-1)
    model.fit(X, y)
    
    return model, features, df

with st.spinner("⏳ Training financial audit intelligence model..."):
    kra_model, kra_features, raw_df = load_and_train_kra()

with tab1:
    # 3. Sidebar inputs for transaction auditing
    st.sidebar.header("🧾 Transaction Audit Parameters")
    
    tx_amount = st.sidebar.number_input("Transaction Amount (KES/Units)", value=25000.0, step=1000.0)
    tx_type = st.sidebar.selectbox("Transaction Channel Type", ["TRANSFER", "CASH_OUT", "CASH_IN", "DEBIT", "PAYMENT"])
    
    old_orig = st.sidebar.number_input("Sender Initial Balance", value=50000.0)
    new_orig = st.sidebar.number_input("Sender Post-Transaction Balance", value=25000.0)
    
    old_dest = st.sidebar.number_input("Recipient Initial Balance", value=10000.0)
    new_dest = st.sidebar.number_input("Recipient Post-Transaction Balance", value=35000.0)

    # 4. Process Inputs to Match Model Features
    # Recreate the engineered features from user input
    sender_err = new_orig + tx_amount - old_orig
    receiver_err = old_dest + tx_amount - new_dest
    high_val = 1 if tx_amount > 150000 else 0
    
    # Build input dictionary with baseline zeroes for one-hot encoding columns
    input_dict = {feat: 0.0 for feat in kra_features}
    input_dict['amount'] = tx_amount
    input_dict['oldbalanceOrg'] = old_orig
    input_dict['newbalanceOrig'] = new_orig
    input_dict['oldbalanceDest'] = old_dest
    input_dict['newbalanceDest'] = new_dest
    input_dict['sender_balance_error'] = sender_err
    input_dict['receiver_balance_error'] = receiver_err
    input_dict['is_high_value'] = high_val
    
    # Set the selected type flag to 1
    type_col = f"type_{tx_type}"
    if type_col in input_dict:
        input_dict[type_col] = 1.0
        
    input_df = pd.DataFrame([input_dict])[kra_features]
    
    # Run prediction
    risk_prediction = kra_model.predict(input_df)[0]
    risk_proba = kra_model.predict_proba(input_df)[0][1] * 100

    # Layout Outputs
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🔮 Transaction Risk Assessment")
        st.metric(label="Anomalous Tax Evasion Probability", value=f"{risk_proba:.2f}%")
        
        if risk_prediction == 1 or risk_proba > 50:
            st.error("🚨 **HIGH RISK OUTLIER AUDIT FLAG:** Transaction exhibits significant balance variance. Recommended for automated eTIMS commercial reconciliation audit.")
        else:
            st.success("🟢 **COMPLIANT PATTERN:** Transaction falls within standard personal peer-to-peer usage parameters. Low audit priority.")

    with col2:
        st.subheader("📊 System Risk Diagnostics")
        st.markdown(f"""
        * **Computed Sender Balance Discrepancy:** {sender_err:,.2f}
        * **Computed Recipient Balance Discrepancy:** {receiver_err:,.2f}
        * **High Value Classification (>150k):** {'Yes' if high_val == 1 else 'No'}
        """)

    st.markdown("---")
    st.subheader("📈 Ledger Profile Distributions")
    # Show a simple preview of the transaction amounts processed
    fig, ax = plt.subplots(figsize=(10, 3))
    sns.boxplot(x=raw_df['amount'], color='#2b5c8f', ax=ax)
    ax.set_title("Distribution of Global Ledger Transaction Amounts")
    st.pyplot(fig)

with tab2:
    st.subheader("📋 Methodology & Transparency")
    st.markdown("""
    ### Project Intent
    This system models how digital ledgers can be automatically scrubbed for systemic revenue leakage. 
    By looking at internal discrepancies within account balances rather than just transaction sizes, 
    the engine uncovers structural tax evasion footprints.
    
    ### Data Source Transparency
    * **Proxy Source:** Kaggle PaySim Mobile Money Dataset.
    * **Compliance Baseline:** Engineered to align with digital revenue assurance objectives tracked globally by modern revenue authorities.
    """)
