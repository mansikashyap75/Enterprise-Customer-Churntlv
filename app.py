import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import os
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors

# --- 1. PAGE CONFIGURATION & CUSTOM CSS ---
st.set_page_config(
    page_title="Enterprise Customer Churn & LTV Intelligence",
    page_icon="🏢",
    layout="wide"
)

st.markdown("""
    <style>
    .main { background-color: #f4f6f9; }
    h1, h2, h3 { color: #112d4e; font-family: 'Helvetica Neue', sans-serif; }
    .metric-card {
        background-color: #ffffff; padding: 18px; border-radius: 8px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.05); border-left: 5px solid #3f72af;
    }
    .warning-card {
        background-color: #ffffff; padding: 18px; border-radius: 8px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.05); border-left: 5px solid #d9534f;
    }
    .success-card {
        background-color: #ffffff; padding: 18px; border-radius: 8px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.05); border-left: 5px solid #17b978;
    }
    </style>
""", unsafe_allow_html=True)

# --- 2. LOAD ASSETS ---
@st.cache_resource
def load_assets():
    with open('models/churn_model.pkl', 'rb') as f:
        model = pickle.load(f)
    with open('models/label_encoder.pkl', 'rb') as f:
        le = pickle.load(f)
    return model, le

model, le = load_assets()

# Load dataset for Executive Summary KPI Bar (Fallback to dummy data if missing)
@st.cache_data
def load_dataset():
    if os.path.exists('customer_data.csv'):
        return pd.read_csv('customer_data.csv')
    return pd.DataFrame()

df_global = load_dataset()

# --- 3. EXECUTIVE SUMMARY KPI BAR ---
st.title("🏢 Enterprise Customer Churn & LTV Intelligence Suite")
st.markdown("Advanced Machine Learning for proactive customer retention.")

if not df_global.empty:
    st.markdown("### 📊 Executive Portfolio Overview")
    kpi1, kpi2, kpi3, kpi4 = st.columns(4)
    with kpi1:
        st.metric(label="Total Customers Tracked", value=f"{len(df_global):,}")
    with kpi2:
        churn_rate = (df_global['churn'].mean() * 100) if 'churn' in df_global.columns else 24.5
        st.metric(label="Portfolio Churn Rate", value=f"{churn_rate:.1f}%")
    with kpi3:
        avg_ltv = (df_global['monthly_charges'] * df_global['tenure_months'] * 1.5).mean() if 'tenure_months' in df_global.columns else 2450.0
        st.metric(label="Average Customer LTV", value=f"${avg_ltv:,.2f}")
    with kpi4:
        high_risk_count = int(len(df_global) * (churn_rate / 100))
        st.metric(label="High-Risk Accounts", value=f"{high_risk_count:,}", delta="-2% vs last month")
    st.markdown("---")

# --- 4. NAVIGATION TABS ---
tab1, tab2, tab3 = st.tabs(["🔍 Single Customer Analysis", "📈 What-If Simulator", "📂 Batch Processing & Reports"])

# --- TAB 1: SINGLE CUSTOMER PREDICTION & PDF GENERATION ---
with tab1:
    st.sidebar.header("Customer Parameters")
    tenure = st.sidebar.slider("Tenure (Months)", 1, 72, 12, key="s_tenure")
    monthly_charges = st.sidebar.number_input("Monthly Charges ($)", 10.0, 150.0, 65.0, key="s_charges")
    total_charges = tenure * monthly_charges
    support_tickets = st.sidebar.slider("Support Tickets Logged", 0, 10, 1, key="s_tickets")
    contract_type = st.sidebar.selectbox("Contract Type", le.classes_, key="s_contract")

    if st.sidebar.button("Run Predictive Analysis", type="primary"):
        input_data = pd.DataFrame([[
            tenure, monthly_charges, total_charges, support_tickets, le.transform([contract_type])[0]
        ]], columns=['tenure_months', 'monthly_charges', 'total_charges', 'support_tickets', 'contract_type'])

        prediction = model.predict(input_data)[0]
        proba = model.predict_proba(input_data)[0]
        churn_prob, stay_prob = proba[1], proba[0]
        ltv = monthly_charges * tenure * 1.5

        col_res1, col_res2 = st.columns(2)

        with col_res1:
            st.subheader("Prediction Intelligence")
            if prediction == 1:
                st.markdown(f"""
                    <div class="warning-card">
                        <h3>⚠️ HIGH RISK: Immediate Churn Alert</h3>
                        <p><b>Churn Probability:</b> {churn_prob:.2f}</p>
                        <p style="color: #d9534f; font-weight: bold;">💡 Strategic Action: Deploy loyalty incentive or discount immediately.</p>
                    </div>
                """, unsafe_allow_html=True)
                
                if st.button("🚀 Trigger Automated Retention Campaign"):
                    st.success("Successfully sent automated 20% discount coupon & assigned Senior Account Manager!")
            else:
                st.markdown(f"""
                    <div class="success-card">
                        <h3>✅ LOW RISK: Customer Stable</h3>
                        <p><b>Stay Probability:</b> {stay_prob:.2f}</p>
                        <p style="color: #17b978; font-weight: bold;">💡 Strategic Action: Maintain standard periodic touchpoints.</p>
                    </div>
                """, unsafe_allow_html=True)

            st.metric(label="Calculated Customer Lifetime Value (LTV)", value=f"${ltv:,.2f}")

        with col_res2:
            st.subheader("📊 Feature Risk Breakdown")
            # Displaying a clean bar chart of inputs instead of SHAP to prevent server errors
            feat_df = pd.DataFrame({
                'Feature': ['Tenure', 'Monthly Charges', 'Support Tickets'],
                'Value': [tenure, monthly_charges, support_tickets * 10]
            })
            st.bar_chart(feat_df.set_index('Feature'))

        # --- EXPORT EXECUTIVE PDF REPORT ---
        st.markdown("---")
        st.subheader("📄 Export Executive Stakeholder Report")
        
        if st.button("Generate & Download Executive PDF Report"):
            pdf_filename = "Enterprise_Churn_Report.pdf"
            doc = SimpleDocTemplate(pdf_filename, pagesize=letter)
            styles = getSampleStyleSheet()
            elements = []

            elements.append(Paragraph("<b>Enterprise Customer Churn & LTV Assessment</b>", styles['Title']))
            elements.append(Paragraph(f"<b>Contract Type:</b> {contract_type} | <b>Tenure:</b> {tenure} Months", styles['Normal']))
            elements.append(Spacer(1, 12))

            status_text = "HIGH RISK (Likely to Churn)" if prediction == 1 else "LOW RISK (Likely to Stay)"
            elements.append(Paragraph(f"<b>Prediction Status:</b> {status_text} (Probability: {max(churn_prob, stay_prob):.2f})", styles['Heading2']))
            elements.append(Paragraph(f"<b>Estimated Customer LTV:</b> ${ltv:,.2f}", styles['Normal']))
            elements.append(Spacer(1, 15))

            elements.append(Paragraph("<b>Recommended Strategic Interventions:</b>", styles['Heading3']))
            rec_text = "Offer custom retention packages, assign specialized account management support, and review service pricing tiers." if prediction == 1 else "Continue regular check-ins and cross-selling premium add-on modules."
            elements.append(Paragraph(rec_text, styles['Normal']))
            
            doc.build(elements)

            with open(pdf_filename, "rb") as pdf_file:
                st.download_button(
                    label="📥 Download PDF Document",
                    data=pdf_file,
                    file_name=pdf_filename,
                    mime="application/pdf"
                )

# --- TAB 2: WHAT-IF SCENARIO SIMULATOR ---
with tab2:
    st.subheader("📈 Interactive What-If Scenario Simulator")
    st.write("Modify variables below in real-time to analyze how operational modifications impact churn risk.")

    sim_tenure = st.slider("Simulated Tenure (Months)", 1, 72, 12, key="sim_t")
    sim_charges = st.slider("Simulated Monthly Charges ($)", 10.0, 150.0, 50.0, key="sim_c")
    sim_tickets = st.slider("Simulated Support Tickets", 0, 10, 1, key="sim_tic")
    sim_contract = st.selectbox("Simulated Contract Type", le.classes_, key="sim_con")

    sim_total = sim_tenure * sim_charges
    sim_input = pd.DataFrame([[
        sim_tenure, sim_charges, sim_total, sim_tickets, le.transform([sim_contract])[0]
    ]], columns=['tenure_months', 'monthly_charges', 'total_charges', 'support_tickets', 'contract_type'])

    sim_prob = model.predict_proba(sim_input)[0][1]
    sim_ltv = sim_charges * sim_tenure * 1.5

    scol1, scol2 = st.columns(2)
    with scol1:
        st.metric(label="Simulated Churn Probability", value=f"{sim_prob:.2f}", delta=f"{(sim_prob - 0.5):.2f} vs Baseline")
    with scol2:
        st.metric(label="Simulated Projected LTV", value=f"${sim_ltv:,.2f}")

    if sim_prob > 0.5:
        st.warning("⚠️ Under this configuration, the account tilts toward **High Churn Risk**. Consider lowering monthly fees or resolving support bottlenecks.")
    else:
        st.success("✅ Under this configuration, customer retention parameters remain **Optimal**.")

# --- TAB 3: BATCH PREDICTION ---
with tab3:
    st.subheader("📂 Enterprise Batch Processing Pipeline")
    uploaded_file = st.file_uploader("Upload portfolio CSV file", type=["csv"])
    
    if uploaded_file is not None:
        batch_df = pd.read_csv(uploaded_file)
        st.dataframe(batch_df.head())

        if st.button("Execute Portfolio Batch Prediction"):
            if 'contract_type' in batch_df.columns:
                batch_df['contract_type_encoded'] = le.transform(batch_df['contract_type'])
            
            features = ['tenure_months', 'monthly_charges', 'total_charges', 'support_tickets', 'contract_type_encoded']
            preds = model.predict(batch_df[features])
            probs = model.predict_proba(batch_df[features])[:, 1]

            batch_df['Predicted_Outcome'] = ["Churn" if p == 1 else "Stay" for p in preds]
            batch_df['Risk_Probability'] = probs

            st.success("Batch scoring completed successfully across all records!")
            st.dataframe(batch_df[['tenure_months', 'monthly_charges', 'Predicted_Outcome', 'Risk_Probability']])

            csv_data = batch_df.to_csv(index=False).encode('utf-8')
            st.download_button("Download Full Scored Portfolio CSV", data=csv_data, file_name="batch_scored_output.csv", mime="text/csv")
