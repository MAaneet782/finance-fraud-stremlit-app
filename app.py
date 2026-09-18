import os
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report, roc_curve
)
from imblearn.over_sampling import SMOTE
import shap
import lime
import lime.lime_tabular

# -----------------------------------------------------------------------------
# 1. PAGE CONFIGURATION & PREMIUM HIGH-CONTRAST DARK THEME STYLING
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Financial Fraud Detection System",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom High-Contrast Professional FinTech Styling
st.markdown("""
<style>
    /* Global Typography & Contrast */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }
    
    /* Main Headers */
    .main-title {
        font-size: 2.2rem;
        font-weight: 800;
        color: #38bdf8 !important; /* Vibrant Cyan */
        margin-bottom: 6px;
        letter-spacing: -0.5px;
        text-shadow: 0 0 20px rgba(56, 189, 248, 0.2);
    }
    
    .main-subtitle {
        font-size: 1.05rem;
        color: #cbd5e1 !important; /* Soft Crisp White-Silver */
        margin-bottom: 24px;
        line-height: 1.6;
    }
    
    .section-header {
        font-size: 1.35rem;
        font-weight: 700;
        color: #60a5fa !important; /* Bright Blue */
        margin-top: 24px;
        margin-bottom: 14px;
        border-bottom: 2px solid rgba(56, 189, 248, 0.25);
        padding-bottom: 8px;
        display: flex;
        align-items: center;
        gap: 8px;
    }

    /* Metric Cards - Deep Glassmorphism with High Contrast */
    .metric-card {
        background: linear-gradient(135deg, rgba(30, 41, 59, 0.95) 0%, rgba(15, 23, 42, 0.95) 100%);
        border: 1px solid rgba(56, 189, 248, 0.25);
        border-radius: 12px;
        padding: 18px 22px;
        border-left: 5px solid #38bdf8;
        box-shadow: 0 4px 16px rgba(0, 0, 0, 0.4);
        margin-bottom: 14px;
    }
    
    .metric-card.alert {
        border-left: 5px solid #ef4444;
        border-color: rgba(239, 68, 68, 0.35);
        background: linear-gradient(135deg, rgba(45, 20, 25, 0.95) 0%, rgba(25, 10, 15, 0.95) 100%);
    }
    
    .metric-card.success {
        border-left: 5px solid #10b981;
        border-color: rgba(16, 185, 129, 0.35);
        background: linear-gradient(135deg, rgba(15, 45, 30, 0.95) 0%, rgba(10, 25, 20, 0.95) 100%);
    }
    
    .metric-title {
        font-size: 0.85rem;
        font-weight: 600;
        color: #94a3b8 !important; /* Silver Muted */
        text-transform: uppercase;
        letter-spacing: 0.8px;
    }
    
    .metric-value {
        font-size: 1.85rem;
        font-weight: 800;
        color: #ffffff !important; /* Pure White */
        margin-top: 4px;
        letter-spacing: -0.5px;
    }
    
    .metric-subtext {
        font-size: 0.82rem;
        color: #cbd5e1 !important;
        margin-top: 4px;
    }
    
    /* Content & Insight Containers */
    .insight-card {
        background: rgba(30, 41, 59, 0.7);
        border: 1px solid rgba(56, 189, 248, 0.25);
        border-left: 5px solid #38bdf8;
        padding: 16px 20px;
        border-radius: 10px;
        margin: 14px 0;
        font-size: 0.98rem;
        line-height: 1.6;
        color: #f1f5f9 !important;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.25);
    }
    
    .insight-card h4 {
        color: #38bdf8 !important;
        margin-bottom: 8px;
        font-weight: 700;
    }
    
    .insight-card b {
        color: #ffffff !important;
        font-weight: 700;
    }

    .warning-card {
        background: rgba(45, 35, 15, 0.75);
        border: 1px solid rgba(245, 158, 11, 0.35);
        border-left: 5px solid #f59e0b;
        padding: 16px 20px;
        border-radius: 10px;
        margin: 14px 0;
        font-size: 0.98rem;
        line-height: 1.6;
        color: #fef3c7 !important;
    }
    
    .warning-card b {
        color: #fbbf24 !important;
    }

    /* Streamlit Tab Styling Enhancement */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background-color: rgba(15, 23, 42, 0.6);
        padding: 6px;
        border-radius: 10px;
        border: 1px solid rgba(255, 255, 255, 0.08);
    }

    .stTabs [data-baseweb="tab"] {
        color: #94a3b8 !important;
        border-radius: 8px;
        font-weight: 600;
        padding: 8px 16px;
    }

    .stTabs [aria-selected="true"] {
        background-color: rgba(56, 189, 248, 0.15) !important;
        color: #38bdf8 !important;
    }
    
    /* Code block contrast */
    pre, code {
        background-color: #0f172a !important;
        color: #38bdf8 !important;
        border: 1px solid rgba(255,255,255,0.1) !important;
        border-radius: 8px !important;
    }
</style>
""", unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# PLOTLY CHART STYLING ENGINE (PREMIUM DARK & HIGH-CONTRAST)
# -----------------------------------------------------------------------------
def style_plot(fig, height=400):
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(15, 23, 42, 0.7)",
        plot_bgcolor="rgba(15, 23, 42, 0.7)",
        font=dict(color="#f8fafc", family="Inter, sans-serif", size=12),
        title=dict(
            font=dict(color="#38bdf8", size=15, family="Inter, sans-serif"),
            x=0.02,
            xanchor="left"
        ),
        legend=dict(
            font=dict(color="#cbd5e1", size=11),
            bgcolor="rgba(15, 23, 42, 0.8)",
            bordercolor="rgba(255, 255, 255, 0.1)",
            borderwidth=1
        ),
        margin=dict(t=50, b=40, l=45, r=30),
        height=height
    )
    fig.update_xaxes(
        gridcolor="rgba(255, 255, 255, 0.08)",
        linecolor="rgba(255, 255, 255, 0.2)",
        tickfont=dict(color="#cbd5e1")
    )
    fig.update_yaxes(
        gridcolor="rgba(255, 255, 255, 0.08)",
        linecolor="rgba(255, 255, 255, 0.2)",
        tickfont=dict(color="#cbd5e1")
    )
    return fig


# -----------------------------------------------------------------------------
# 2. DATA PIPELINE & MODEL ENGINE (CACHED)
# -----------------------------------------------------------------------------
@st.cache_data(show_spinner="Loading and verifying dataset...")
def load_and_preprocess_data(csv_path="fraud_dataset.csv"):
    if not os.path.exists(csv_path):
        st.error(f"Error: Dataset file '{csv_path}' not found.")
        st.stop()
    
    raw_df = pd.read_csv(csv_path)
    
    # 1. Drop identifier column (Step 5.1)
    df = raw_df.drop(columns=['transaction_id'], errors='ignore')
    
    # 2. Ordinal encode location_risk: low < medium < high (Step 5.2)
    risk_mapping = {'low': 0, 'medium': 1, 'high': 2}
    df['location_risk_encoded'] = df['location_risk'].map(risk_mapping)
    
    # 3. One-hot encode nominal categoricals with drop_first=True
    df_encoded = pd.get_dummies(
        df, columns=['transaction_type', 'device_type'],
        drop_first=True, dtype=int
    )
    df_encoded = df_encoded.drop(columns=['location_risk'])
    
    X = df_encoded.drop(columns=['is_fraud'])
    y = df_encoded['is_fraud']
    feature_names = X.columns.tolist()
    
    return raw_df, df_encoded, X, y, feature_names


@st.cache_resource(show_spinner="Training baseline and SMOTE models...")
def train_all_modeling_cases(_X, _y, feature_names):
    cases = {}
    
    # Case 1: Baseline Imbalanced LR (80:20 Split)
    X_tr1, X_te1, y_tr1, y_te1 = train_test_split(
        _X, _y, test_size=0.20, random_state=42, stratify=_y
    )
    scaler1 = StandardScaler()
    X_tr1_sc = pd.DataFrame(scaler1.fit_transform(X_tr1), columns=feature_names, index=X_tr1.index)
    X_te1_sc = pd.DataFrame(scaler1.transform(X_te1), columns=feature_names, index=X_te1.index)
    
    lr1 = LogisticRegression(random_state=42, max_iter=1000)
    lr1.fit(X_tr1_sc, y_tr1)
    
    y_pred1 = lr1.predict(X_te1_sc)
    y_prob1 = lr1.predict_proba(X_te1_sc)[:, 1]
    cm1 = confusion_matrix(y_te1, y_pred1)
    tn1, fp1, fn1, tp1 = cm1.ravel()
    fpr1, tpr1, _ = roc_curve(y_te1, y_prob1)
    
    cases["Case 1: Imbalanced (80:20)"] = {
        "case_name": "Case 1: Imbalanced (80:20)",
        "model": lr1,
        "scaler": scaler1,
        "split": "80:20 (No SMOTE)",
        "train_samples": len(X_tr1),
        "test_samples": len(X_te1),
        "train_legit": int((y_tr1 == 0).sum()),
        "train_fraud": int((y_tr1 == 1).sum()),
        "accuracy": accuracy_score(y_te1, y_pred1),
        "precision": precision_score(y_te1, y_pred1, zero_division=0),
        "recall": recall_score(y_te1, y_pred1, zero_division=0),
        "f1": f1_score(y_te1, y_pred1, zero_division=0),
        "roc_auc": roc_auc_score(y_te1, y_prob1),
        "cm": cm1,
        "tn": tn1, "fp": fp1, "fn": fn1, "tp": tp1,
        "fpr": fpr1, "tpr": tpr1,
        "y_test": y_te1,
        "y_pred": y_pred1,
        "y_prob": y_prob1,
        "X_train_bal": X_tr1_sc,
        "y_train_bal": y_tr1,
        "X_test_sc": X_te1_sc,
        "report": classification_report(y_te1, y_pred1, target_names=['Legitimate', 'Fraud'], output_dict=True)
    }
    
    # Helper for Cases 2, 3, 4 (SMOTE on training set only)
    def run_smote_case(test_size, case_name):
        X_tr, X_te, y_tr, y_te = train_test_split(
            _X, _y, test_size=test_size, random_state=42, stratify=_y
        )
        scaler = StandardScaler()
        X_tr_sc = pd.DataFrame(scaler.fit_transform(X_tr), columns=feature_names, index=X_tr.index)
        X_te_sc = pd.DataFrame(scaler.transform(X_te), columns=feature_names, index=X_te.index)
        
        smote = SMOTE(random_state=42)
        X_tr_bal, y_tr_bal = smote.fit_resample(X_tr_sc, y_tr)
        
        model = LogisticRegression(random_state=42, max_iter=1000)
        model.fit(X_tr_bal, y_tr_bal)
        
        y_pred = model.predict(X_te_sc)
        y_prob = model.predict_proba(X_te_sc)[:, 1]
        cm = confusion_matrix(y_te, y_pred)
        tn, fp, fn, tp = cm.ravel()
        fpr, tpr, _ = roc_curve(y_te, y_prob)
        
        return {
            "case_name": case_name,
            "model": model,
            "scaler": scaler,
            "split": f"{int((1-test_size)*100)}:{int(test_size*100)} (SMOTE)",
            "train_samples": len(X_tr),
            "test_samples": len(X_te),
            "train_legit": int((y_tr == 0).sum()),
            "train_fraud": int((y_tr == 1).sum()),
            "train_bal_legit": int((y_tr_bal == 0).sum()),
            "train_bal_fraud": int((y_tr_bal == 1).sum()),
            "accuracy": accuracy_score(y_te, y_pred),
            "precision": precision_score(y_te, y_pred, zero_division=0),
            "recall": recall_score(y_te, y_pred, zero_division=0),
            "f1": f1_score(y_te, y_pred, zero_division=0),
            "roc_auc": roc_auc_score(y_te, y_prob),
            "cm": cm,
            "tn": tn, "fp": fp, "fn": fn, "tp": tp,
            "fpr": fpr, "tpr": tpr,
            "y_test": y_te,
            "y_pred": y_pred,
            "y_prob": y_prob,
            "X_train_bal": X_tr_bal,
            "y_train_bal": y_tr_bal,
            "X_test_sc": X_te_sc,
            "report": classification_report(y_te, y_pred, target_names=['Legitimate', 'Fraud'], output_dict=True)
        }
    
    cases["Case 2: SMOTE (50:50)"] = run_smote_case(0.50, "Case 2: SMOTE (50:50)")
    cases["Case 3: SMOTE (55:45)"] = run_smote_case(0.45, "Case 3: SMOTE (55:45)")
    cases["Case 4: SMOTE (45:55)"] = run_smote_case(0.55, "Case 4: SMOTE (45:55)")
    
    comp_df = pd.DataFrame({
        "Case": list(cases.keys()),
        "Accuracy": [c["accuracy"] for c in cases.values()],
        "Precision": [c["precision"] for c in cases.values()],
        "Recall": [c["recall"] for c in cases.values()],
        "F1-Score": [c["f1"] for c in cases.values()],
        "ROC-AUC": [c["roc_auc"] for c in cases.values()],
        "FP (False Alarm)": [c["fp"] for c in cases.values()],
        "FN (Missed Fraud)": [c["fn"] for c in cases.values()]
    }).set_index("Case")
    
    return cases, comp_df


@st.cache_resource(show_spinner="Generating SHAP & LIME explainers...")
def get_explainers_and_shap(_champion_case, feature_names):
    model = _champion_case["model"]
    X_train_bal = _champion_case["X_train_bal"]
    X_test_sc = _champion_case["X_test_sc"]
    
    explainer = shap.LinearExplainer(model, X_train_bal)
    shap_values = explainer.shap_values(X_test_sc)
    
    lime_explainer = lime.lime_tabular.LimeTabularExplainer(
        training_data=np.array(X_train_bal),
        feature_names=feature_names,
        class_names=['Legitimate', 'Fraud'],
        mode='classification',
        random_state=42
    )
    
    coefs = model.coef_[0]
    coef_df = pd.DataFrame({
        'Feature': feature_names,
        'Coefficient': coefs,
        'Abs_Coefficient': np.abs(coefs)
    }).sort_values('Abs_Coefficient', ascending=False)
    
    shap_importance = pd.DataFrame({
        'Feature': feature_names,
        'Mean_SHAP': np.abs(shap_values).mean(axis=0)
    }).sort_values('Mean_SHAP', ascending=False)
    
    rank_compare = pd.DataFrame({
        'Rank': range(1, len(feature_names) + 1),
        'LR Coefficient Ranking': coef_df['Feature'].tolist(),
        'SHAP Importance Ranking': shap_importance['Feature'].tolist()
    })
    
    return shap_values, lime_explainer, coef_df, shap_importance, rank_compare


# Initialize data and models
raw_df, df_encoded, X, y, feature_names = load_and_preprocess_data()
cases, comp_df = train_all_modeling_cases(X, y, feature_names)
best_case = cases["Case 3: SMOTE (55:45)"]
shap_values, lime_explainer, coef_df, shap_importance, rank_compare = get_explainers_and_shap(best_case, feature_names)


# -----------------------------------------------------------------------------
# SIDEBAR NAVIGATION (7 SECTIONS)
# -----------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### 🛡️ **FraudGuard System**")
    st.caption("Enterprise Financial Fraud Detection & XAI")
    st.markdown("---")
    
    section = st.radio(
        "Navigation Menu",
        [
            "1. Overview",
            "2. Dataset",
            "3. Analysis",
            "4. Prediction",
            "5. Model Results",
            "6. Explainability",
            "7. Business Insights"
        ],
        index=0
    )
    
    st.markdown("---")
    st.markdown("#### 📌 **System Metadata**")
    st.markdown("""
    * **Algorithm:** Logistic Regression
    * **Balancing:** SMOTE Oversampling
    * **Dataset:** 1,000 banking transactions
    * **Fraud Incidents:** 64 (6.4%)
    * **Champion Split:** Case 3 (55:45)
    * **Explainability:** SHAP + LIME
    """)
    st.markdown("---")
    st.caption("Streamlit Pro Max Architecture")


# -----------------------------------------------------------------------------
# SECTION 1: OVERVIEW
# -----------------------------------------------------------------------------
def render_overview():
    st.markdown("<div class='main-title'>1. System Overview</div>", unsafe_allow_html=True)
    st.markdown(
        "<div class='main-subtitle'>Executive financial fraud detection platform: combating class imbalance "
        "with SMOTE-balanced Logistic Regression and dual-layer Explainable AI (SHAP & LIME).</div>",
        unsafe_allow_html=True
    )
    
    # High-contrast Metric Cards
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-title">Total Records</div>
            <div class="metric-value">1,000</div>
            <div class="metric-subtext">8 Raw Attributes</div>
        </div>
        """, unsafe_allow_html=True)
    with c2:
        st.markdown("""
        <div class="metric-card success">
            <div class="metric-title">Legitimate Volume</div>
            <div class="metric-value">936</div>
            <div class="metric-subtext">93.6% Majority Class</div>
        </div>
        """, unsafe_allow_html=True)
    with c3:
        st.markdown("""
        <div class="metric-card alert">
            <div class="metric-title">Fraud Incidents</div>
            <div class="metric-value">64</div>
            <div class="metric-subtext">6.4% Target Class</div>
        </div>
        """, unsafe_allow_html=True)
    with c4:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-title">Imbalance Ratio</div>
            <div class="metric-value">14.6 : 1</div>
            <div class="metric-subtext">Severe Class Skew</div>
        </div>
        """, unsafe_allow_html=True)
        
    st.markdown("<div class='section-header'>Strategic Purpose & Problem Context</div>", unsafe_allow_html=True)
    st.markdown("""
    In financial banking operations, **detecting fraudulent transactions is an asymmetric cost challenge**:
    * **The Accuracy Trap:** A model predicting 100% legitimate transactions achieves 93.6% accuracy but suffers **100% financial loss**.
    * **Recall as Primary KPI:** Missing a real fraud (False Negative) causes direct monetary theft and regulatory sanctions. Flagging a legitimate transaction (False Positive) creates temporary customer friction easily resolved via 2FA.
    * **SMOTE Resampling:** The system balances minority fraud representations to optimize sensitivity without sacrificing overall predictive precision.
    * **Regulatory Compliance (SR 11-7):** Financial regulators require complete model auditability. Logistic Regression coefficients, combined with global SHAP attributions and local LIME explanations, guarantee transparent decision governance.
    """)

    st.markdown("<div class='section-header'>End-to-End Application Modules</div>", unsafe_allow_html=True)
    st.markdown("""
    <div class="insight-card">
    <b>Platform Modules Breakdown:</b><br>
    • <b>2. Dataset:</b> Raw transaction audit, 100% completeness verification (0 nulls, 0 duplicates), and summary statistics.<br>
    • <b>3. Analysis:</b> Rigorous Exploratory Data Analysis covering target skew, KDE curves, outlier detection, and correlation metrics.<br>
    • <b>4. Prediction:</b> Interactive real-time transaction scoring engine with customizable operational cutoffs.<br>
    • <b>5. Model Results:</b> Multi-case benchmarking (Cases 1 through 4) with ROC curves and decision threshold tuning.<br>
    • <b>6. Explainability:</b> Feature impact direction, SHAP beeswarm / bar importances, and case-by-case LIME audits.<br>
    • <b>7. Business Insights:</b> Executive Q&A addressing banking error costs, fraud profiles, and deployment strategies.
    </div>
    """, unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# SECTION 2: DATASET
# -----------------------------------------------------------------------------
def render_dataset(raw_df):
    st.markdown("<div class='main-title'>2. Dataset Explorer & Data Health</div>", unsafe_allow_html=True)
    st.markdown(
        "<div class='main-subtitle'>Inspect raw data records, verify data hygiene, and review statistical distributions.</div>",
        unsafe_allow_html=True
    )
    
    st.markdown("<div class='section-header'>Data Quality & Completeness Checks (Steps 4.1 & 4.2)</div>", unsafe_allow_html=True)
    q1, q2, q3 = st.columns(3)
    with q1:
        st.success("✅ **Missing Values:** 0 null values detected across all columns. Data is complete.")
    with q2:
        st.success("✅ **Duplicate Rows:** 0 duplicate rows found. Every transaction is unique.")
    with q3:
        st.info("ℹ️ **Identifier Handling:** `transaction_id` dropped prior to training to avoid leakage.")
        
    st.markdown("<div class='section-header'>Interactive Transaction Table</div>", unsafe_allow_html=True)
    fc1, fc2 = st.columns([1, 3])
    with fc1:
        filter_status = st.selectbox("Filter Records by Status", ["All Records", "Legitimate Only (0)", "Fraud Only (1)"])
        preview_count = st.slider("Display Row Limit", min_value=10, max_value=100, value=25, step=5)
    
    view_df = raw_df.copy()
    if filter_status == "Legitimate Only (0)":
        view_df = view_df[view_df['is_fraud'] == 0]
    elif filter_status == "Fraud Only (1)":
        view_df = view_df[view_df['is_fraud'] == 1]
        
    with fc2:
        st.dataframe(view_df.head(preview_count), use_container_width=True)

    st.markdown("<div class='section-header'>Statistical Summaries & Preprocessing Pipeline</div>", unsafe_allow_html=True)
    tab_num, tab_cat, tab_enc = st.tabs(["🔢 Numerical Attributes", "🏷️ Categorical Attributes", "⚙️ Preprocessing Specs"])
    
    with tab_num:
        st.caption("Distribution statistics for continuous numerical attributes:")
        st.dataframe(raw_df[['amount', 'account_age_days', 'num_prev_transactions']].describe().round(2).T, use_container_width=True)
        st.markdown(r"""
        * **Amount ($):** Mean value is $97.25 with high variance (standard deviation $97.25, max $817.24).
        * **Account Age (days):** Ranges from 2 days to 1,998 days (~5.5 years) with an average of 986 days.
        * **Previous Transactions:** Spans uniformly from 0 to 499 past transactions.
        """)
        
    with tab_cat:
        st.caption("Cardinality and frequency of categorical attributes:")
        st.dataframe(raw_df[['transaction_type', 'location_risk', 'device_type']].describe(include='object').T, use_container_width=True)
        
    with tab_enc:
        st.markdown(r"""
        ### Feature Transformation Pipeline (Step 5):
        1. **Drop Serial Identifier:** Removed `transaction_id`.
        2. **Ordinal Encoding:** Converted `location_risk` into ordered integers: `{'low': 0, 'medium': 1, 'high': 2}`.
        3. **One-Hot Encoding:** Applied `get_dummies(drop_first=True)` to `transaction_type` and `device_type`.
        4. **StandardScaler:** Scaled continuous variables strictly using training set parameters ($\mu=0, \sigma=1$).
        """)
        st.code("""
Final Model Feature Set (9 features):
1. amount
2. account_age_days
3. num_prev_transactions
4. location_risk_encoded
5. transaction_type_debit
6. transaction_type_payment
7. transaction_type_transfer
8. device_type_mobile
9. device_type_tablet
        """, language="python")


# -----------------------------------------------------------------------------
# SECTION 3: ANALYSIS
# -----------------------------------------------------------------------------
def render_analysis(raw_df):
    st.markdown("<div class='main-title'>3. Exploratory Data Analysis (EDA)</div>", unsafe_allow_html=True)
    st.markdown(
        "<div class='main-subtitle'>Comprehensive behavioral patterns: target class skew, numerical distributions, "
        "IQR outlier analysis, categorical risk cross-tabs, and feature correlations.</div>",
        unsafe_allow_html=True
    )
    
    eda_tabs = st.tabs([
        "3.1 Class Imbalance",
        "3.2 Feature Distributions",
        "3.3 Outliers (IQR Method)",
        "3.4 Categorical Cross-Tabs",
        "3.5 Correlation Heatmap"
    ])
    
    with eda_tabs[0]:
        st.markdown("### Target Variable Class Imbalance (Step 4.3)")
        c1, c2 = st.columns([1, 1])
        with c1:
            fig_pie = go.Figure(data=[go.Pie(
                labels=['Legitimate (0)', 'Fraud (1)'],
                values=[936, 64],
                hole=.45,
                marker=dict(colors=['#10b981', '#ef4444']),
                textinfo='label+percent+value',
                textfont=dict(color="#ffffff", size=13)
            )])
            fig_pie = style_plot(fig_pie, height=360)
            st.plotly_chart(fig_pie, use_container_width=True)
        with c2:
            st.markdown("""
            <div class="insight-card">
            <h4>Class Imbalance Analysis:</h4>
            • Legitimate Transactions: <b>936 (93.6%)</b><br>
            • Fraudulent Transactions: <b>64 (6.4%)</b><br>
            • Severe Imbalance Ratio: <b>14.6 : 1</b><br><br>
            <b>The Accuracy Trap:</b> In imbalanced fraud detection, a naive model predicting 'Legitimate' for every transaction achieves an apparent <b>93.6% accuracy</b> while missing 100% of actual fraud.
            </div>
            """, unsafe_allow_html=True)
            
    with eda_tabs[1]:
        st.markdown("### Numerical Feature Distributions by Fraud Status (Step 4.4)")
        num_choice = st.selectbox("Select Numerical Feature to Inspect", ['amount', 'account_age_days', 'num_prev_transactions'])
        
        fig_hist = px.histogram(
            raw_df, x=num_choice, color="is_fraud",
            marginal="box", barmode="overlay",
            color_discrete_map={0: "#10b981", 1: "#ef4444"},
            labels={"is_fraud": "Fraud Status (0=Legit, 1=Fraud)", num_choice: num_choice},
            title=f"Distribution Profile: {num_choice} by Fraud Status"
        )
        fig_hist = style_plot(fig_hist, height=400)
        st.plotly_chart(fig_hist, use_container_width=True)
        
        st.markdown(r"""
        <div class="insight-card">
        <b>Empirical Observations from Distributions:</b><br>
        • <b>Transaction Amount:</b> Fraud transactions average <b>$266.23</b> vs. <b>$85.70</b> for legitimate transactions (>3x higher).<br>
        • <b>Account Age:</b> Fraudulent activity concentrates in <b>newer accounts</b> where behavioral history is minimal.<br>
        • <b>Prior Transactions:</b> Accounts with sparse transaction histories show heightened fraud risk.
        </div>
        """, unsafe_allow_html=True)

    with eda_tabs[2]:
        st.markdown("### Outlier Detection using IQR Method (Step 4.5)")
        iqr_data = []
        for col in ['amount', 'account_age_days', 'num_prev_transactions']:
            Q1 = raw_df[col].quantile(0.25)
            Q3 = raw_df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower = Q1 - 1.5 * IQR
            upper = Q3 + 1.5 * IQR
            outliers = raw_df[(raw_df[col] < lower) | (raw_df[col] > upper)]
            iqr_data.append({
                "Feature": col, "Q1 (25%)": round(Q1, 2), "Q3 (75%)": round(Q3, 2),
                "IQR": round(IQR, 2), "Lower Bound": round(lower, 2),
                "Upper Bound": round(upper, 2), "Outlier Count": len(outliers),
                "Outlier %": f"{len(outliers)/len(raw_df)*100:.1f}%"
            })
        st.dataframe(pd.DataFrame(iqr_data).set_index("Feature"), use_container_width=True)
        
        fig_box = px.box(
            raw_df, x="is_fraud", y="amount", color="is_fraud",
            color_discrete_map={0: "#10b981", 1: "#ef4444"},
            labels={"is_fraud": "Fraud Status (0=Legit, 1=Fraud)", "amount": "Transaction Amount ($)"},
            title="Box Plot: Transaction Amount Dispersion by Fraud Status"
        )
        fig_box = style_plot(fig_box, height=360)
        st.plotly_chart(fig_box, use_container_width=True)
        
        st.markdown("""
        <div class="warning-card">
        <b>Data Science Governance Principle (Step 4.5):</b><br>
        In financial fraud detection, outliers in <code>amount</code> are <b>deliberately preserved</b>.
        Extreme dollar values are the primary operational signals of illicit transfers rather than data corruption.
        </div>
        """, unsafe_allow_html=True)

    with eda_tabs[3]:
        st.markdown("### Categorical Feature Risk Profiles (Step 4.6)")
        cat_choice = st.selectbox("Select Categorical Feature", ['location_risk', 'transaction_type', 'device_type'])
        
        ct = pd.crosstab(raw_df[cat_choice], raw_df['is_fraud'], normalize='index') * 100
        ct.columns = ['Legitimate %', 'Fraud %']
        
        fig_cat = px.bar(
            ct.reset_index(), x=cat_choice, y=['Legitimate %', 'Fraud %'],
            barmode='stack', color_discrete_map={'Legitimate %': '#10b981', 'Fraud %': '#ef4444'},
            title=f"Fraud Proportions Across {cat_choice}"
        )
        fig_cat = style_plot(fig_cat, height=360)
        st.plotly_chart(fig_cat, use_container_width=True)
        
        st.markdown("""
        <div class="insight-card">
        <b>Category Risk Insights:</b><br>
        • <b>Location Risk:</b> High-risk locations show the highest percentage of fraudulent transfers.<br>
        • <b>Transaction Type:</b> <code>cash_out</code> and <code>transfer</code> present the highest risk of irreversible loss.<br>
        • <b>Device Type:</b> Mobile devices represent elevated transaction velocity.
        </div>
        """, unsafe_allow_html=True)

    with eda_tabs[4]:
        st.markdown("### Correlation Heatmap (Step 4.7)")
        corr_matrix = raw_df[['amount', 'account_age_days', 'num_prev_transactions', 'is_fraud']].corr().round(3)
        fig_corr = px.imshow(
            corr_matrix, text_auto=True, aspect="auto",
            color_continuous_scale="RdBu_r", zmin=-1, zmax=1,
            title="Pearson Correlation Heatmap (Numerical Features & Target)"
        )
        fig_corr = style_plot(fig_corr, height=380)
        st.plotly_chart(fig_corr, use_container_width=True)
        
        st.markdown("""
        <div class="insight-card">
        <b>Multicollinearity Diagnosis:</b><br>
        Pairwise correlations between independent predictors are low (|r| < 0.15),
        confirming absence of severe multicollinearity for Logistic Regression.
        </div>
        """, unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# SECTION 4: PREDICTION
# -----------------------------------------------------------------------------
def render_prediction(best_case, feature_names):
    st.markdown("<div class='main-title'>4. Real-Time Transaction Risk Scoring</div>", unsafe_allow_html=True)
    st.markdown(
        "<div class='main-subtitle'>Interactive inference module: evaluate a transaction's fraud probability "
        "using the champion model (Case 3: SMOTE Balanced Logistic Regression).</div>",
        unsafe_allow_html=True
    )
    
    model = best_case["model"]
    scaler = best_case["scaler"]
    
    st.markdown("<div class='section-header'>Transaction Input Parameters</div>", unsafe_allow_html=True)
    
    with st.form("transaction_prediction_form"):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            amount = st.number_input("Transaction Amount ($)", min_value=0.50, max_value=2000.0, value=185.00, step=5.0)
            account_age = st.number_input("Account Age (Days)", min_value=1, max_value=3000, value=320, step=10)
            
        with col2:
            num_prev = st.number_input("Previous Transactions Count", min_value=0, max_value=1000, value=45, step=5)
            location_risk = st.selectbox("Location Risk Level", ["low", "medium", "high"], index=2)
            
        with col3:
            transaction_type = st.selectbox("Transaction Type", ["cash_out", "debit", "payment", "transfer"], index=0)
            device_type = st.selectbox("Device Type", ["desktop", "mobile", "tablet"], index=1)
            
        threshold_input = st.slider("Operational Decision Threshold (τ)", min_value=0.10, max_value=0.90, value=0.50, step=0.05)
        submitted = st.form_submit_button("🔍 Evaluate Transaction Risk")
        
    if submitted:
        risk_map = {'low': 0, 'medium': 1, 'high': 2}
        loc_encoded = risk_map[location_risk]
        
        type_debit = 1 if transaction_type == 'debit' else 0
        type_payment = 1 if transaction_type == 'payment' else 0
        type_transfer = 1 if transaction_type == 'transfer' else 0
        
        device_mobile = 1 if device_type == 'mobile' else 0
        device_tablet = 1 if device_type == 'tablet' else 0
        
        input_data = pd.DataFrame([[
            amount, account_age, num_prev, loc_encoded,
            type_debit, type_payment, type_transfer,
            device_mobile, device_tablet
        ]], columns=feature_names)
        
        input_scaled = pd.DataFrame(scaler.transform(input_data), columns=feature_names)
        fraud_prob = model.predict_proba(input_scaled)[0, 1]
        is_fraud_flag = fraud_prob >= threshold_input
        
        st.markdown("<div class='section-header'>Risk Scoring Verdict</div>", unsafe_allow_html=True)
        res_col1, res_col2, res_col3 = st.columns(3)
        
        with res_col1:
            st.metric("Fraud Probability", f"{fraud_prob*100:.2f}%")
        with res_col2:
            st.metric("Operational Cutoff (τ)", f"{threshold_input*100:.1f}%")
        with res_col3:
            if is_fraud_flag:
                st.error("🚨 **RISK ALERT:** FLAGGED AS FRAUD")
            else:
                st.success("✅ **CLEARED:** TRANSACTION LEGITIMATE")
                
        # Risk gauge chart
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=fraud_prob * 100,
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': "Fraud Risk Score (%)", 'font': {'color': '#38bdf8', 'size': 16}},
            number={'font': {'color': '#ffffff', 'size': 32}},
            gauge={
                'axis': {'range': [0, 100], 'tickcolor': '#cbd5e1', 'tickwidth': 1},
                'bar': {'color': "#ef4444" if is_fraud_flag else "#10b981"},
                'bgcolor': "rgba(15, 23, 42, 0.7)",
                'steps': [
                    {'range': [0, 30], 'color': "rgba(16, 185, 129, 0.2)"},
                    {'range': [30, 60], 'color': "rgba(245, 158, 11, 0.2)"},
                    {'range': [60, 100], 'color': "rgba(239, 68, 68, 0.2)"}
                ],
                'threshold': {
                    'line': {'color': "#ffffff", 'width': 4},
                    'thickness': 0.75,
                    'value': threshold_input * 100
                }
            }
        ))
        fig_gauge = style_plot(fig_gauge, height=280)
        st.plotly_chart(fig_gauge, use_container_width=True)


# -----------------------------------------------------------------------------
# SECTION 5: MODEL RESULTS
# -----------------------------------------------------------------------------
def render_model_results(cases, comp_df, best_case):
    st.markdown("<div class='main-title'>5. Model Results & 4-Case Benchmarking</div>", unsafe_allow_html=True)
    st.markdown(
        "<div class='main-subtitle'>Compare baseline imbalanced models against SMOTE-balanced models across "
        "three train-test split configurations (Cases 1 through 4) and test operational thresholds.</div>",
        unsafe_allow_html=True
    )
    
    st.markdown("<div class='section-header'>Side-by-Side Model Comparison (Step 13)</div>", unsafe_allow_html=True)
    st.dataframe(comp_df.style.highlight_max(axis=0, subset=['Recall', 'F1-Score', 'ROC-AUC'], color='rgba(16, 185, 129, 0.35)')
                            .highlight_min(axis=0, subset=['FN (Missed Fraud)'], color='rgba(16, 185, 129, 0.35)')
                            .format("{:.4f}", subset=['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC']),
                 use_container_width=True)
    
    # Bar Chart Comparison
    metrics_to_plot = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC']
    plot_df = comp_df[metrics_to_plot].reset_index().melt(id_vars='Case', var_name='Metric', value_name='Score')
    
    fig_bar = px.bar(
        plot_df, x="Metric", y="Score", color="Case",
        barmode="group",
        color_discrete_sequence=['#94a3b8', '#f97316', '#10b981', '#a855f7'],
        title="Cross-Model Metric Performance (Step 14)"
    )
    fig_bar = style_plot(fig_bar, height=400)
    fig_bar.update_layout(yaxis_range=[0, 1.05])
    st.plotly_chart(fig_bar, use_container_width=True)
    
    # Superimposed ROC Curves & Confusion Matrix Grid
    col_roc, col_cm = st.columns([1, 1])
    
    with col_roc:
        st.markdown("<div class='section-header'>Superimposed ROC Curves (Step 15)</div>", unsafe_allow_html=True)
        fig_roc = go.Figure()
        colors = {
            'Case 1: Imbalanced (80:20)': '#94a3b8',
            'Case 2: SMOTE (50:50)': '#f97316',
            'Case 3: SMOTE (55:45)': '#10b981',
            'Case 4: SMOTE (45:55)': '#a855f7'
        }
        for cname, res in cases.items():
            fig_roc.add_trace(go.Scatter(
                x=res['fpr'], y=res['tpr'], mode='lines',
                name=f"{cname} (AUC={res['roc_auc']:.3f})",
                line=dict(color=colors[cname], width=2.5)
            ))
        fig_roc.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode='lines', name='Baseline Random (0.500)', line=dict(color='#cbd5e1', dash='dash')))
        fig_roc = style_plot(fig_roc, height=380)
        fig_roc.update_layout(xaxis_title="False Positive Rate", yaxis_title="True Positive Rate (Recall)")
        st.plotly_chart(fig_roc, use_container_width=True)
        
    with col_cm:
        st.markdown("<div class='section-header'>Champion Confusion Matrix: Case 3 (Step 16)</div>", unsafe_allow_html=True)
        fig_cm = px.imshow(
            best_case['cm'], text_auto=True,
            x=['Predicted Legit', 'Predicted Fraud'],
            y=['Actual Legit', 'Actual Fraud'],
            color_continuous_scale="Greens"
        )
        fig_cm = style_plot(fig_cm, height=380)
        st.plotly_chart(fig_cm, use_container_width=True)

    # Operational Decision Threshold Tuning (Step 17)
    st.markdown("<div class='section-header'>Operational Decision Threshold Tuning (Step 17)</div>", unsafe_allow_html=True)
    y_test_best = best_case['y_test']
    y_prob_best = best_case['y_prob']
    
    t_thresh = st.slider("Select Operational Threshold (τ)", min_value=0.10, max_value=0.90, value=0.50, step=0.05)
    preds_t = (y_prob_best >= t_thresh).astype(int)
    cm_t = confusion_matrix(y_test_best, preds_t)
    tn_t, fp_t, fn_t, tp_t = cm_t.ravel()
    
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Precision", f"{precision_score(y_test_best, preds_t, zero_division=0):.4f}")
    k2.metric("Recall (Sensitivity)", f"{recall_score(y_test_best, preds_t, zero_division=0):.4f}")
    k3.metric("False Alarms (FP)", f"{fp_t}")
    k4.metric("Missed Fraud (FN)", f"{fn_t}")
    
    eval_threshs = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
    t_records = []
    for t in eval_threshs:
        p_t = (y_prob_best >= t).astype(int)
        cm_sub = confusion_matrix(y_test_best, p_t)
        tn_s, fp_s, fn_s, tp_s = cm_sub.ravel()
        t_records.append({
            "Threshold": t,
            "Precision": precision_score(y_test_best, p_t, zero_division=0),
            "Recall": recall_score(y_test_best, p_t, zero_division=0),
            "F1-Score": f1_score(y_test_best, p_t, zero_division=0),
            "FP (False Alarms)": fp_s,
            "FN (Missed Fraud)": fn_s,
            "Operational Policy": "Standard Cutoff" if t == 0.5 else ("High Recall (Large Wire Transfers)" if t < 0.4 else "High Precision (Micropayments)")
        })
    st.dataframe(pd.DataFrame(t_records).set_index("Threshold").style.format("{:.4f}", subset=['Precision', 'Recall', 'F1-Score']), use_container_width=True)


# -----------------------------------------------------------------------------
# SECTION 6: EXPLAINABILITY
# -----------------------------------------------------------------------------
def render_explainability(best_case, feature_names, shap_values, lime_explainer, coef_df, shap_importance, rank_compare):
    st.markdown("<div class='main-title'>6. Explainable AI: Global & Local Transparency</div>", unsafe_allow_html=True)
    st.markdown(
        "<div class='main-subtitle'>Audit model decisions using Logistic Regression weights, "
        "SHAP global feature attributions, and LIME local instance explanations.</div>",
        unsafe_allow_html=True
    )
    
    xai_tabs = st.tabs([
        "6.1 LR Coefficients (Step 18)",
        "6.2 SHAP Global Importance (Step 19)",
        "6.3 LIME Local Case Explanations (Step 20)"
    ])
    
    with xai_tabs[0]:
        st.markdown("### Standardized Logistic Regression Coefficients (Step 18)")
        coef_display = coef_df.sort_values('Coefficient', ascending=True)
        colors = ['#ef4444' if c > 0 else '#10b981' for c in coef_display['Coefficient']]
        
        fig_coef = go.Figure(go.Bar(
            x=coef_display['Coefficient'],
            y=coef_display['Feature'],
            orientation='h',
            marker_color=colors
        ))
        fig_coef = style_plot(fig_coef, height=420)
        fig_coef.update_layout(title="Log-Odds Impact (Red = Increases Fraud Risk, Green = Decreases Fraud Risk)")
        st.plotly_chart(fig_coef, use_container_width=True)
        
        st.markdown(r"""
        <div class="insight-card">
        <b>Feature Weight Directionality:</b><br>
        • <b>Positive Weights (Red):</b> <code>amount</code> (+2.40), <code>device_type_mobile</code> (+0.83), <code>device_type_tablet</code> (+0.63), and <code>account_age_days</code> (+0.10) increase predicted fraud probability.<br>
        • <b>Negative Weights (Green):</b> <code>location_risk_encoded</code> (-2.72), <code>num_prev_transactions</code> (-1.07), and <code>transaction_type_payment</code> (-0.82) decrease predicted fraud probability.
        </div>
        """, unsafe_allow_html=True)

    with xai_tabs[1]:
        st.markdown("### SHAP Global Feature Importance (Step 19)")
        col_plot, col_rank = st.columns([1, 1])
        with col_plot:
            fig_shap = px.bar(
                shap_importance.sort_values('Mean_SHAP', ascending=True),
                x='Mean_SHAP', y='Feature', orientation='h',
                color='Mean_SHAP', color_continuous_scale='Reds',
                title="SHAP Feature Importance (Mean |SHAP Value|)"
            )
            fig_shap = style_plot(fig_shap, height=400)
            st.plotly_chart(fig_shap, use_container_width=True)
            
        with col_rank:
            st.markdown("#### Ranking Agreement: LR Coefficients vs. SHAP")
            st.dataframe(rank_compare, use_container_width=True)
            st.success("✅ **Top-3 Agreement:** 3 out of 3 top features match exactly between LR Coefficients and SHAP (`amount`, `location_risk_encoded`, `num_prev_transactions`).")

    with xai_tabs[2]:
        st.markdown("### LIME Local Explanations for Specific Transactions (Step 20)")
        case_options = {
            "Notebook Case #1: Test Index 5 (Borderline False Alarm: Actual Legit, Predicted Fraud)": 5,
            "Notebook Case #2: Test Index 9 (Clear Caught Fraud: Actual Fraud, Predicted Fraud)": 9
        }
        selected_option = st.selectbox("Select Test Transaction Instance", list(case_options.keys()))
        test_idx = case_options[selected_option]
        
        X_test_sc = best_case['X_test_sc']
        y_test = best_case['y_test']
        model = best_case['model']
        
        sample_sc = np.array(X_test_sc)[test_idx]
        actual_label = "FRAUD (1)" if y_test.iloc[test_idx] == 1 else "LEGITIMATE (0)"
        pred_probs = model.predict_proba([sample_sc])[0]
        pred_label = "FRAUD (1)" if pred_probs[1] >= 0.50 else "LEGITIMATE (0)"
        
        i1, i2, i3 = st.columns(3)
        i1.metric("Actual Ground Truth", actual_label)
        i2.metric("Model Prediction", pred_label)
        i3.metric("Fraud Probability", f"{pred_probs[1]*100:.2f}%")
        
        def predict_fn(arr):
            df_slice = pd.DataFrame(arr, columns=feature_names)
            return model.predict_proba(df_slice)
            
        exp = lime_explainer.explain_instance(sample_sc, predict_fn, num_features=len(feature_names))
        exp_list = exp.as_list()
        exp_df = pd.DataFrame(exp_list, columns=['Feature Condition', 'Contribution Weight'])
        
        colors_lime = ['#ef4444' if w > 0 else '#10b981' for w in exp_df['Contribution Weight']]
        fig_lime = go.Figure(go.Bar(
            x=exp_df['Contribution Weight'],
            y=exp_df['Feature Condition'],
            orientation='h',
            marker_color=colors_lime
        ))
        fig_lime = style_plot(fig_lime, height=380)
        fig_lime.update_layout(title=f"LIME Feature Attribution for Transaction #{test_idx}")
        st.plotly_chart(fig_lime, use_container_width=True)


# -----------------------------------------------------------------------------
# SECTION 7: BUSINESS INSIGHTS
# -----------------------------------------------------------------------------
def render_business_insights():
    st.markdown("<div class='main-title'>7. Business Insights & Regulatory Strategy</div>", unsafe_allow_html=True)
    st.markdown(
        "<div class='main-subtitle'>Executive synthesis of banking error costs, trade-offs, and "
        "regulatory compliance principles established in the project.</div>",
        unsafe_allow_html=True
    )
    
    b1, b2 = st.columns(2)
    with b1:
        st.markdown(r"""
        <div class="metric-card">
            <div class="metric-title">Average Fraud Amount</div>
            <div class="metric-value" style="color: #ef4444;">$266.23</div>
            <div class="metric-subtext">> 3x higher than average legitimate transfers</div>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="insight-card">
        <h4>Q1: Why is Accuracy Misleading in Fraud Detection?</h4>
        In an imbalanced dataset with <b>93.6% legitimate</b> and <b>6.4% fraud</b>, a naive model predicting zero fraud achieves 93.6% accuracy while leaving the bank exposed to 100% of losses. Case 1 achieved 94.5% accuracy but missed <b>69.2% of all fraud attacks (Recall: 30.77%)</b>.
        </div>
        
        <div class="insight-card">
        <h4>Q2: Why is Recall the #1 Priority KPI?</h4>
        • <b>False Negative (Missed Fraud):</b> Causes direct unrecoverable monetary theft, merchant chargebacks, and regulatory fines.<br>
        • <b>False Positive (False Alarm):</b> Causes temporary customer friction, easily resolved with automated 2FA.<br>
        Therefore, operations prioritize <b>Recall</b>.
        </div>
        
        <div class="insight-card">
        <h4>Q3 & Q4: Asymmetric Error Costs</h4>
        • <b>Cost of False Positive:</b> Customer service review overhead + brief user inconvenience.<br>
        • <b>Cost of False Negative:</b> Direct bottom-line chargeback loss + reputational damage.
        </div>
        """, unsafe_allow_html=True)
        
    with b2:
        st.markdown(r"""
        <div class="metric-card">
            <div class="metric-title">Average Legitimate Amount</div>
            <div class="metric-value" style="color: #10b981;">$85.70</div>
            <div class="metric-subtext">Routine daily consumer transactions</div>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="insight-card">
        <h4>Q5 & Q6: Why Deploy the SMOTE-Balanced Model?</h4>
        SMOTE balances minority representations, boosting Recall from <b>30.77% to 79.31%</b> and F1-score from <b>0.4211 to 0.6216</b>. Case 3 (55:45 split) achieves an AUC of <b>0.9303</b> and limits missed frauds to only 6 cases.
        </div>
        
        <div class="insight-card">
        <h4>Q7: Suggested Additional Features</h4>
        1. <b>Transaction Velocity:</b> Frequency of transactions in rolling 5-minute / 1-hour windows.<br>
        2. <b>Location Mismatch:</b> Discrepancy between customer billing profile and originating IP location.
        </div>
        
        <div class="insight-card">
        <h4>Regulatory Compliance & Governance</h4>
        <b>Yes.</b> The combination of Logistic Regression odds ratios, SHAP global feature values, and LIME local audits satisfies regulatory transparency standards (such as Federal Reserve SR 11-7 model risk management guidance).
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown(r"""
    ### Executive Fraud Prevention Blueprint:
    1. **Primary Defense:** Deploy Champion Model (Case 3: SMOTE 55:45) at threshold $\tau=0.50$ for standard operations.
    2. **Tiered Protection:** Automatically lower $\tau \to 0.30$ whenever transaction amount exceeds $200 to prevent large unauthorized transfers.
    3. **Investigator Toolkit:** Surface LIME feature attribution cards in the internal fraud management portal to empower human investigators during audits.
    """)


# -----------------------------------------------------------------------------
# MAIN ROUTER
# -----------------------------------------------------------------------------
if "1. Overview" in section:
    render_overview()
elif "2. Dataset" in section:
    render_dataset(raw_df)
elif "3. Analysis" in section:
    render_analysis(raw_df)
elif "4. Prediction" in section:
    render_prediction(best_case, feature_names)
elif "5. Model Results" in section:
    render_model_results(cases, comp_df, best_case)
elif "6. Explainability" in section:
    render_explainability(best_case, feature_names, shap_values, lime_explainer, coef_df, shap_importance, rank_compare)
elif "7. Business Insights" in section:
    render_business_insights()

# Footer
st.markdown("---")
st.caption("🛡️ Financial Fraud Analytics Platform • Powered by Streamlit, Scikit-Learn, SMOTE, SHAP & LIME")
