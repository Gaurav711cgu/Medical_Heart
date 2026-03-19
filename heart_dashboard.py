# ============================================================
#   HEART DISEASE PREDICTION — STREAMLIT DASHBOARD
#   Run with: streamlit run heart_dashboard.py
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import (accuracy_score, confusion_matrix,
                             classification_report)

# ── Page Config ─────────────────────────────────────────────
st.set_page_config(
    page_title="Heart Disease Prediction",
    page_icon="❤️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Custom CSS ───────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;700&display=swap');
    html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }

    /* ── Dark background ── */
    .main, .stApp, .block-container { background: #0D1117 !important; }

    /* ── Global text — only apply to elements that don't have explicit color ── */
    body, p, .stMarkdown p, .stText,
    h1, h2, h3, h4, h5, h6,
    .stMarkdown h1, .stMarkdown h2, .stMarkdown h3 {
        color: #E6EDF3 !important;
    }

    /* ── Metric cards ── */
    .metric-card {
        background: #161B22;
        border-radius: 14px;
        padding: 20px 24px;
        box-shadow: 0 2px 16px rgba(0,0,0,0.4);
        border-left: 5px solid #E53935;
        margin-bottom: 10px;
    }
    .metric-value { font-size: 2.2rem; font-weight: 700; color: #7EB3FF !important; }
    .metric-label { font-size: 0.85rem; color: #8B949E !important; font-weight: 500;
                    letter-spacing: 0.5px; text-transform: uppercase; }

    /* ── Section title ── */
    .section-title {
        font-size: 1.4rem; font-weight: 700; color: #7EB3FF !important;
        border-bottom: 3px solid #E53935;
        padding-bottom: 8px; margin-bottom: 20px;
    }

    /* ── Final verdict boxes ── */
    .result-yes {
        background: #2D1515 !important; border: 2px solid #E53935;
        border-radius: 12px; padding: 20px; text-align: center;
        font-size: 1.3rem; font-weight: 700; color: #FF6B6B !important;
    }
    .result-no {
        background: #0D2016 !important; border: 2px solid #2E7D32;
        border-radius: 12px; padding: 20px; text-align: center;
        font-size: 1.3rem; font-weight: 700; color: #56D364 !important;
    }

    /* ── Predict / Submit button ── */
    .stButton > button,
    .stFormSubmitButton > button {
        background: linear-gradient(135deg, #E53935, #B71C1C) !important;
        color: #FFFFFF !important;
        border: none !important;
        border-radius: 10px !important;
        padding: 12px 30px !important;
        font-size: 1rem !important;
        font-weight: 700 !important;
        width: 100% !important;
        cursor: pointer !important;
        box-shadow: 0 4px 15px rgba(229,57,53,0.4) !important;
    }
    .stButton > button:hover,
    .stFormSubmitButton > button:hover {
        background: linear-gradient(135deg, #FF5252, #E53935) !important;
        box-shadow: 0 6px 20px rgba(229,57,53,0.6) !important;
    }

    /* ── Sidebar ── */
    div[data-testid="stSidebar"] { background: #161B22 !important; }
    div[data-testid="stSidebar"] * { color: #E6EDF3 !important; }

    /* ── Number +/- step buttons ── */
    button[data-testid="stNumberInputStepDown"],
    button[data-testid="stNumberInputStepUp"],
    div[data-baseweb="base-input"] button {
        background: #2D333B !important;
        color: #E6EDF3 !important;
        border: 1px solid #444C56 !important;
        border-radius: 6px !important;
        font-size: 1rem !important;
        font-weight: 700 !important;
    }
    button[data-testid="stNumberInputStepDown"]:hover,
    button[data-testid="stNumberInputStepUp"]:hover {
        background: #E53935 !important;
        color: #FFFFFF !important;
    }

    /* ── Input fields ── */
    input[type="number"], input[type="text"],
    div[data-baseweb="input"] input,
    div[data-baseweb="base-input"] input {
        background-color: #21262D !important;
        color: #E6EDF3 !important;
        font-weight: 600 !important;
        font-size: 1rem !important;
        border: 1.5px solid #444C56 !important;
        border-radius: 8px !important;
        white-space: nowrap !important;
        overflow: visible !important;
    }
    div[data-baseweb="base-input"] {
        background-color: #21262D !important;
        border: 1.5px solid #444C56 !important;
        border-radius: 8px !important;
    }

    /* ── Select / Dropdown ── */
    div[data-baseweb="select"] > div {
        background-color: #21262D !important;
        color: #E6EDF3 !important;
        border: 1.5px solid #444C56 !important;
        border-radius: 8px !important;
        font-weight: 600 !important;
        white-space: nowrap !important;
        overflow: visible !important;
        text-overflow: unset !important;
    }
    div[data-baseweb="select"] span { color: #E6EDF3 !important; }
    div[data-baseweb="select"] svg  { fill: #8B949E !important; }

    /* ── Dropdown list ── */
    ul[data-baseweb="menu"],
    div[data-baseweb="popover"] ul {
        background-color: #21262D !important;
        border: 1.5px solid #444C56 !important;
        border-radius: 8px !important;
    }
    li[role="option"] {
        background-color: #21262D !important;
        color: #E6EDF3 !important;
        font-weight: 500 !important;
        white-space: nowrap !important;
    }
    li[role="option"]:hover, li[aria-selected="true"] {
        background-color: #2D333B !important;
        color: #7EB3FF !important;
    }

    /* ── Labels ── */
    label, .stSelectbox label, .stNumberInput label,
    div[data-testid="stForm"] label,
    div[data-testid="column"] label {
        color: #8B949E !important;
        font-weight: 600 !important;
        font-size: 0.88rem !important;
    }

    /* ── Form container ── */
    div[data-testid="stForm"] {
        background: #161B22 !important;
        border-radius: 16px !important;
        padding: 24px !important;
        box-shadow: 0 4px 24px rgba(0,0,0,0.5) !important;
        border: 1px solid #30363D !important;
    }

    /* ── Tabs ── */
    .stTabs [data-baseweb="tab-list"] { background: #161B22 !important; border-radius: 8px; }
    .stTabs [data-baseweb="tab"] { color: #8B949E !important; }
    .stTabs [aria-selected="true"] { color: #7EB3FF !important; border-bottom: 2px solid #E53935 !important; }

    /* ── Divider ── */
    hr { border-color: #30363D !important; }

    /* ── Alert / info boxes ── */
    .stAlert { background: #1C2128 !important; color: #E6EDF3 !important; border-color: #444C56 !important; }
</style>
""", unsafe_allow_html=True)


# ── Load & Train ─────────────────────────────────────────────
@st.cache_data
def load_and_train():
    df = pd.read_csv("heart.csv")
    X = df.drop('target', axis=1)
    y = df['target']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    X_tr = scaler.fit_transform(X_train)
    X_te = scaler.transform(X_test)
    knn = KNeighborsClassifier(n_neighbors=5); knn.fit(X_tr, y_train)
    lr  = LogisticRegression(max_iter=1000, random_state=42); lr.fit(X_tr, y_train)
    nb  = GaussianNB(); nb.fit(X_tr, y_train)
    return df, X, y, X_test, y_test, scaler, knn, lr, nb

df, X, y, X_test, y_test, scaler, knn, lr, nb = load_and_train()

y_pred_knn = knn.predict(scaler.transform(X_test))
y_pred_lr  = lr.predict(scaler.transform(X_test))
y_pred_nb  = nb.predict(scaler.transform(X_test))

acc_knn = round(accuracy_score(y_test, y_pred_knn) * 100, 2)
acc_lr  = round(accuracy_score(y_test, y_pred_lr)  * 100, 2)
acc_nb  = round(accuracy_score(y_test, y_pred_nb)  * 100, 2)


# ── Sidebar Navigation ───────────────────────────────────────
st.sidebar.markdown("## ❤️ Heart Disease\n### Prediction Dashboard")
st.sidebar.markdown("---")
page = st.sidebar.radio("Navigate", [
    "🏠  Overview",
    "📊  Data Analysis",
    "🤖  Model Performance",
    "🔮  Live Prediction",
    "📋  Clinical Summary"
])
st.sidebar.markdown("---")
st.sidebar.markdown(f"**Dataset:** `heart.csv`")
st.sidebar.markdown(f"**Rows:** {df.shape[0]}  |  **Cols:** {df.shape[1]}")
st.sidebar.markdown(f"**Best Model:** KNN ({acc_knn}%)")


# ════════════════════════════════════════════════════════════
# PAGE 1 — OVERVIEW
# ════════════════════════════════════════════════════════════
if page == "🏠  Overview":
    st.markdown("# ❤️ Heart Disease Prediction Dashboard")
    st.markdown("**KNN · Logistic Regression · Naive Bayes** — trained on UCI Heart Disease Dataset")
    st.markdown("---")

    c1, c2, c3, c4, c5 = st.columns(5)
    for col, val, label in zip(
        [c1, c2, c3, c4, c5],
        [df.shape[0], df.shape[1]-1, df['target'].sum(), (df['target']==0).sum(), f"{acc_knn}%"],
        ["Total Patients","Features","With Disease","No Disease","Best Accuracy"]
    ):
        with col:
            st.markdown(f"""<div class="metric-card">
                <div class="metric-value">{val}</div>
                <div class="metric-label">{label}</div></div>""", unsafe_allow_html=True)

    st.markdown("---")
    st.markdown('<div class="section-title">Model Accuracy Summary</div>', unsafe_allow_html=True)
    m1, m2, m3 = st.columns(3)
    for col, name, acc, color, emoji in zip(
        [m1, m2, m3],
        ["K-Nearest Neighbors","Logistic Regression","Naive Bayes"],
        [acc_knn, acc_lr, acc_nb],
        ["#1565C0","#2E7D32","#E65100"],
        ["🔵","🟢","🟠"]
    ):
        with col:
            st.markdown(f"""
            <div style="background:#161B22;border-radius:14px;padding:22px;
                        box-shadow:0 2px 16px rgba(0,0,0,0.4);border-top:5px solid {color}">
                <div style="font-size:1rem;font-weight:700;color:#8B949E !important">{emoji} {name}</div>
                <div style="font-size:2.5rem;font-weight:800;color:{color} !important;margin:8px 0">{acc}%</div>
                <div style="background:#21262D;border-radius:6px;height:10px;overflow:hidden">
                    <div style="background:{color};width:{int(acc)}%;height:100%;border-radius:6px"></div>
                </div>
            </div>""", unsafe_allow_html=True)

    st.markdown("---")
    st.markdown('<div class="section-title">Dataset Preview</div>', unsafe_allow_html=True)
    st.dataframe(df.head(10), use_container_width=True)
    st.markdown(f"*Showing 10 of {df.shape[0]} rows · {df.isnull().sum().sum()} missing values*")


# ════════════════════════════════════════════════════════════
# PAGE 2 — DATA ANALYSIS
# ════════════════════════════════════════════════════════════
elif page == "📊  Data Analysis":
    st.markdown("# 📊 Data Analysis")
    st.markdown("Explore distributions, correlations, and patterns in the dataset.")
    st.markdown("---")

    tab1, tab2, tab3 = st.tabs(["📈 Distributions", "🔥 Correlation", "🔍 Feature Deep Dive"])

    with tab1:
        c1, c2 = st.columns(2)
        with c1:
            fig, ax = plt.subplots(figsize=(6, 4))
            counts = df['target'].value_counts()
            ax.bar(['No Disease (0)','Has Disease (1)'],[counts[0],counts[1]],
                   color=['#2196F3','#F44336'],edgecolor='white',linewidth=2)
            ax.set_title('Target Distribution',fontweight='bold',fontsize=13)
            ax.set_ylabel('Count')
            for i, v in enumerate([counts[0],counts[1]]):
                ax.text(i, v+5, str(v), ha='center', fontweight='bold')
            fig.tight_layout(); st.pyplot(fig); plt.close()

        with c2:
            fig, ax = plt.subplots(figsize=(6, 4))
            ax.hist(df['age'],bins=20,color='#1565C0',edgecolor='white',linewidth=0.8)
            ax.set_title('Age Distribution',fontweight='bold',fontsize=13)
            ax.set_xlabel('Age'); ax.set_ylabel('Count')
            fig.tight_layout(); st.pyplot(fig); plt.close()

        c3, c4 = st.columns(2)
        with c3:
            fig, ax = plt.subplots(figsize=(6, 4))
            ax.hist(df['chol'],bins=25,color='#E65100',edgecolor='white',linewidth=0.8)
            ax.set_title('Cholesterol Distribution',fontweight='bold',fontsize=13)
            ax.set_xlabel('mg/dl'); ax.set_ylabel('Count')
            fig.tight_layout(); st.pyplot(fig); plt.close()

        with c4:
            fig, ax = plt.subplots(figsize=(6, 4))
            ax.hist(df['thalach'],bins=20,color='#2E7D32',edgecolor='white',linewidth=0.8)
            ax.set_title('Max Heart Rate Distribution',fontweight='bold',fontsize=13)
            ax.set_xlabel('BPM'); ax.set_ylabel('Count')
            fig.tight_layout(); st.pyplot(fig); plt.close()

        st.markdown("---")
        c5, c6 = st.columns(2)
        w = 0.35
        with c5:
            fig, ax = plt.subplots(figsize=(6, 4))
            sex_d = df.groupby(['sex','target']).size().unstack(fill_value=0)
            x = np.arange(2)
            b1 = ax.bar(x-w/2, sex_d[1], w, color='#F44336', label='Heart Disease', edgecolor='white')
            b2 = ax.bar(x+w/2, sex_d[0], w, color='#37474F', label='No Disease',    edgecolor='white')
            for b in list(b1)+list(b2):
                ax.annotate(str(int(b.get_height())),
                            (b.get_x()+b.get_width()/2, b.get_height()+3),
                            ha='center', fontsize=9, fontweight='bold')
            ax.set_xticks(x); ax.set_xticklabels(['Male','Female'])
            ax.set_title('5.2 Sex vs Disease Status',fontweight='bold',fontsize=12)
            ax.set_ylabel('Count'); ax.legend()
            fig.tight_layout(); st.pyplot(fig); plt.close()

        with c6:
            fig, ax = plt.subplots(figsize=(6, 4))
            cp_d = df.groupby(['cp','target']).size().unstack(fill_value=0)
            x = np.arange(len(cp_d))
            b1 = ax.bar(x-w/2, cp_d[1], w, color='#F44336', label='Heart Disease', edgecolor='white')
            b2 = ax.bar(x+w/2, cp_d[0], w, color='#37474F', label='No Disease',    edgecolor='white')
            for b in list(b1)+list(b2):
                ax.annotate(str(int(b.get_height())),
                            (b.get_x()+b.get_width()/2, b.get_height()+2),
                            ha='center', fontsize=9, fontweight='bold')
            ax.set_xticks(x)
            ax.set_xticklabels(['Typical\n(0)','Atypical\n(1)','Non-ang\n(2)','Asymp\n(3)'])
            ax.set_title('5.3 Chest Pain Type vs Disease',fontweight='bold',fontsize=12)
            ax.set_ylabel('Count'); ax.legend()
            fig.tight_layout(); st.pyplot(fig); plt.close()

        c7, c8 = st.columns(2)
        with c7:
            fig, ax = plt.subplots(figsize=(6, 4))
            sl_d = df.groupby(['slope','target']).size().unstack(fill_value=0)
            x = np.arange(len(sl_d))
            b1 = ax.bar(x-w/2, sl_d[1], w, color='#F44336', label='Heart Disease', edgecolor='white')
            b2 = ax.bar(x+w/2, sl_d[0], w, color='#37474F', label='No Disease',    edgecolor='white')
            for b in list(b1)+list(b2):
                ax.annotate(str(int(b.get_height())),
                            (b.get_x()+b.get_width()/2, b.get_height()+2),
                            ha='center', fontsize=9, fontweight='bold')
            ax.set_xticks(x); ax.set_xticklabels(['Up(0)','Flat(1)','Down(2)'])
            ax.set_title('5.7 ST Slope vs Disease',fontweight='bold',fontsize=12)
            ax.set_ylabel('Count'); ax.legend()
            fig.tight_layout(); st.pyplot(fig); plt.close()

        with c8:
            fig, ax = plt.subplots(figsize=(6, 4))
            ca_d = df.groupby(['ca','target']).size().unstack(fill_value=0)
            x = np.arange(len(ca_d))
            b1 = ax.bar(x-w/2, ca_d.get(1, pd.Series(0,index=ca_d.index)), w,
                        color='#F44336', label='Heart Disease', edgecolor='white')
            b2 = ax.bar(x+w/2, ca_d.get(0, pd.Series(0,index=ca_d.index)), w,
                        color='#37474F', label='No Disease',    edgecolor='white')
            for b in list(b1)+list(b2):
                h = int(b.get_height())
                if h > 0:
                    ax.annotate(str(h),(b.get_x()+b.get_width()/2, h+2),
                                ha='center', fontsize=9, fontweight='bold')
            ax.set_xticks(x); ax.set_xticklabels([str(i) for i in ca_d.index])
            ax.set_title('5.8 Major Vessels (ca) vs Disease',fontweight='bold',fontsize=12)
            ax.set_ylabel('Count'); ax.legend()
            fig.tight_layout(); st.pyplot(fig); plt.close()

    with tab2:
        fig, ax = plt.subplots(figsize=(10, 7))
        mask = np.triu(np.ones_like(df.corr(), dtype=bool))
        sns.heatmap(df.corr(), annot=True, fmt='.2f', cmap='RdYlGn',
                    ax=ax, mask=mask, linewidths=0.5, annot_kws={'size': 9})
        ax.set_title('Feature Correlation Matrix', fontweight='bold', fontsize=14)
        fig.tight_layout(); st.pyplot(fig); plt.close()

        st.markdown("#### Top Features Correlated with Target")
        corr_target = df.corr()['target'].drop('target').sort_values(key=abs, ascending=False)
        fig, ax = plt.subplots(figsize=(8, 4))
        clrs = ['#E53935' if v > 0 else '#1565C0' for v in corr_target.values]
        ax.barh(corr_target.index, corr_target.values, color=clrs, edgecolor='white')
        ax.axvline(0, color='black', linewidth=0.8)
        ax.set_title('Correlation with Target (Red=Positive / Blue=Negative)',
                     fontweight='bold', fontsize=12)
        fig.tight_layout(); st.pyplot(fig); plt.close()

        st.markdown("---")
        st.markdown('<div class="section-title">6.1 Feature–Target Correlation Direction Summary</div>',
                    unsafe_allow_html=True)
        corr_summary = pd.DataFrame({
            "Feature":   ["cp","thalach","thal","ca","oldpeak","exang","age","sex",
                          "trestbps","chol","fbs","restecg","slope"],
            "Direction": ["Negative","Negative","Negative","Negative","Negative",
                          "Positive","Positive","Negative","Positive","Positive",
                          "Positive","Negative","Negative"],
            "Strength":  ["Strong","Strong","Strong","Strong","Strong",
                          "Moderate","Moderate","Moderate","Weak","Very Weak",
                          "Very Weak","Weak","Moderate"],
            "Key Insight": [
                "Typical angina (type 0) strongly predicts disease",
                "Disease patients avg 139 bpm vs 158 bpm — impaired cardiac reserve",
                "Reversible defect (3) = strongest thal predictor of ischaemia",
                "More vessels coloured = fewer disease labels (coding-specific)",
                "Higher ST depression → likely NO disease (inverse coding in dataset)",
                "Exercise-induced angina directly linked to ischaemia under stress",
                "Older patients carry higher cardiovascular risk",
                "Females show proportionally higher disease rate in this sample",
                "Slightly elevated resting BP in disease group",
                "Minimal difference — poor standalone predictor of disease",
                "Slightly more diabetics in disease group",
                "Normal ECG paradoxically more common in disease group here",
                "Flat/upsloping more common in disease cases"
            ]
        })
        st.dataframe(corr_summary, use_container_width=True)

        st.markdown("---")
        st.markdown('<div class="section-title">3.2 ST Depression (oldpeak) — Clinical Thresholds</div>',
                    unsafe_allow_html=True)
        m1, m2, m3 = st.columns(3)
        with m1:
            st.markdown("""<div class="metric-card">
                <div class="metric-value">1.07 mm</div>
                <div class="metric-label">Overall Mean oldpeak</div></div>""", unsafe_allow_html=True)
        with m2:
            st.markdown("""<div class="metric-card" style="border-left-color:#E53935">
                <div class="metric-value" style="color:#FF6B6B !important">0.57 mm</div>
                <div class="metric-label">Mean — Heart Disease Group</div></div>""", unsafe_allow_html=True)
        with m3:
            st.markdown("""<div class="metric-card" style="border-left-color:#2E7D32">
                <div class="metric-value" style="color:#56D364 !important">1.60 mm</div>
                <div class="metric-label">Mean — No Disease Group</div></div>""", unsafe_allow_html=True)

        st.dataframe(pd.DataFrame({
            "Range (mm)":     ["0.0 – 0.5","0.5 – 1.0","≥ 1.0","≥ 2.0"],
            "Classification": ["Normal","Borderline","Significant","Severe"],
            "Clinical Significance": [
                "No clinically significant depression. Typical in healthy individuals.",
                "Equivocal — may indicate mild ischaemia, especially with other risk factors.",
                "Horizontal or downsloping ≥1 mm is a strong indicator of myocardial ischaemia.",
                "High specificity for obstructive CAD. Associated with multi-vessel disease."
            ]
        }), use_container_width=True)

        st.markdown("---")
        st.markdown("#### 3.3 ST Slope Distribution")
        st.dataframe(pd.DataFrame({
            "Value":       ["0 — Upsloping","1 — Flat","2 — Downsloping"],
            "Description": ["ST rises during exercise","ST remains level","ST falls during exercise"],
            "Clinical Interpretation": [
                "Generally benign; less predictive of ischaemia",
                "Borderline; combined with depression, suggests ischaemia",
                "Most concerning — strongest predictor of CAD"
            ],
            "Count": [74, 482, 469]
        }), use_container_width=True)

    with tab3:
        feature = st.selectbox("Select a feature to explore:",
                               [c for c in df.columns if c != 'target'])
        c1, c2 = st.columns(2)
        with c1:
            fig, ax = plt.subplots(figsize=(6, 4))
            for t, color, label in [(0,'#2196F3','No Disease'),(1,'#F44336','Has Disease')]:
                ax.hist(df[df['target']==t][feature], bins=20,
                        alpha=0.6, color=color, label=label, edgecolor='white')
            ax.set_title(f'{feature} by Target', fontweight='bold')
            ax.legend(); fig.tight_layout(); st.pyplot(fig); plt.close()
        with c2:
            fig, ax = plt.subplots(figsize=(6, 4))
            df.boxplot(column=feature, by='target', ax=ax,
                       boxprops=dict(color='#1A237E'),
                       medianprops=dict(color='#E53935', linewidth=2))
            ax.set_title(f'{feature} Boxplot by Target', fontweight='bold')
            ax.set_xlabel('Target (0=No Disease, 1=Has Disease)')
            plt.suptitle('')
            fig.tight_layout(); st.pyplot(fig); plt.close()

        st.markdown(f"**Statistics for `{feature}`:**")
        st.dataframe(df.groupby('target')[feature].describe().round(2), use_container_width=True)


# ════════════════════════════════════════════════════════════
# PAGE 3 — MODEL PERFORMANCE
# ════════════════════════════════════════════════════════════
elif page == "🤖  Model Performance":
    st.markdown("# 🤖 Model Performance")
    st.markdown("Evaluate all 3 trained models side by side.")
    st.markdown("---")

    st.markdown('<div class="section-title">Accuracy Comparison</div>', unsafe_allow_html=True)
    fig, ax = plt.subplots(figsize=(8, 4))
    bars = ax.bar(['KNN','Logistic Regression','Naive Bayes'],
                  [acc_knn, acc_lr, acc_nb],
                  color=['#1565C0','#2E7D32','#E65100'],
                  width=0.45, edgecolor='white', linewidth=2)
    for bar, acc in zip(bars, [acc_knn, acc_lr, acc_nb]):
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.3,
                f'{acc:.2f}%', ha='center', fontweight='bold', fontsize=12)
    ax.set_ylim(60, 105)
    ax.axhline(y=80, color='gray', linestyle='--', alpha=0.4, label='80% baseline')
    ax.set_ylabel('Accuracy (%)'); ax.legend()
    ax.set_title('Model Accuracy', fontweight='bold')
    fig.tight_layout(); st.pyplot(fig); plt.close()

    st.markdown("---")
    st.markdown('<div class="section-title">Confusion Matrices</div>', unsafe_allow_html=True)
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    for ax, (name, y_pred, acc, color) in zip(axes, [
        ("KNN",                y_pred_knn, acc_knn, '#1565C0'),
        ("Logistic Regression",y_pred_lr,  acc_lr,  '#2E7D32'),
        ("Naive Bayes",        y_pred_nb,  acc_nb,  '#E65100')
    ]):
        cm = confusion_matrix(y_test, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                    xticklabels=['No Disease','Has Disease'],
                    yticklabels=['No Disease','Has Disease'])
        ax.set_title(f"{name}\nAccuracy: {acc}%", color=color, fontweight='bold')
        ax.set_ylabel('Actual'); ax.set_xlabel('Predicted')
    fig.tight_layout(); st.pyplot(fig); plt.close()

    st.markdown("---")
    st.markdown('<div class="section-title">Detailed Classification Reports</div>', unsafe_allow_html=True)
    t1, t2, t3 = st.tabs(["🔵 KNN","🟢 Logistic Regression","🟠 Naive Bayes"])
    for tab, y_pred, name in zip([t1,t2,t3],
                                  [y_pred_knn,y_pred_lr,y_pred_nb],
                                  ["KNN","Logistic Regression","Naive Bayes"]):
        with tab:
            report_df = pd.DataFrame(
                classification_report(y_test, y_pred,
                                      target_names=['No Disease','Has Disease'],
                                      output_dict=True)
            ).transpose().round(2)
            st.dataframe(report_df, use_container_width=True)


# ════════════════════════════════════════════════════════════
# PAGE 4 — LIVE PREDICTION
# ════════════════════════════════════════════════════════════
elif page == "🔮  Live Prediction":
    st.markdown("# 🔮 Live Patient Prediction")
    st.markdown("Fill in the patient details below and click **Predict** to get results from all 3 models.")
    st.markdown("---")

    with st.form("prediction_form"):
        st.markdown('<div class="section-title">Patient Details</div>', unsafe_allow_html=True)

        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown("**👤 Personal Info**")
            age = st.number_input("Age (years)", min_value=1, max_value=120, value=52)
            sex = st.selectbox("Sex", options=[1,0],
                               format_func=lambda x: "Male" if x==1 else "Female")
            cp  = st.selectbox("Chest Pain Type", options=[0,1,2,3],
                               format_func=lambda x: {0:"Typical Angina",1:"Atypical Angina",
                                                       2:"Non-Anginal Pain",3:"Asymptomatic"}[x])
        with c2:
            st.markdown("**🩸 Blood Metrics**")
            trestbps = st.number_input("Resting Blood Pressure (mm Hg)", min_value=80, max_value=220, value=125)
            chol     = st.number_input("Cholesterol (mg/dl)", min_value=100, max_value=600, value=212)
            fbs      = st.selectbox("Fasting Blood Sugar", options=[0,1],
                                    format_func=lambda x: ">120 mg/dl (High)" if x==1 else "≤120 mg/dl (Normal)")
        with c3:
            st.markdown("**💓 Heart Metrics**")
            restecg = st.selectbox("Resting ECG", options=[0,1,2],
                                   format_func=lambda x: {0:"Normal",1:"ST-T Abnormality",2:"LV Hypertrophy"}[x])
            thalach = st.number_input("Max Heart Rate Achieved", min_value=60, max_value=220, value=168)
            exang   = st.selectbox("Exercise Induced Angina", options=[0,1],
                                   format_func=lambda x: "Yes" if x==1 else "No")

        st.markdown("---")
        c4, c5, c6 = st.columns(3)
        with c4:
            oldpeak = st.number_input("ST Depression (oldpeak)", min_value=0.0, max_value=10.0,
                                      value=1.0, step=0.1)
        with c5:
            slope = st.selectbox("ST Slope", options=[0,1,2],
                                 format_func=lambda x: {0:"Upsloping",1:"Flat",2:"Downsloping"}[x])
        with c6:
            ca = st.selectbox("Major Vessels (0-4)", options=[0,1,2,3,4])

        thal = st.selectbox("Thalassemia", options=[0,1,2,3],
                            format_func=lambda x: {0:"None",1:"Normal",
                                                    2:"Fixed Defect",3:"Reversible Defect"}[x])
        st.markdown("---")
        submitted = st.form_submit_button("🔮 Predict Heart Disease", use_container_width=True)

    if submitted:
        patient = pd.DataFrame(
            [[age, sex, cp, trestbps, chol, fbs,
              restecg, thalach, exang, oldpeak, slope, ca, thal]],
            columns=X.columns
        )
        patient_scaled = scaler.transform(patient)

        pred_knn = knn.predict(patient_scaled)[0]
        pred_lr  = lr.predict(patient_scaled)[0]
        pred_nb  = nb.predict(patient_scaled)[0]

        prob_knn = knn.predict_proba(patient_scaled)[0][1] * 100
        prob_lr  = lr.predict_proba(patient_scaled)[0][1] * 100
        prob_nb  = nb.predict_proba(patient_scaled)[0][1] * 100

        votes = pred_knn + pred_lr + pred_nb

        st.markdown("---")
        st.markdown("## 🔍 Prediction Results")

        r1, r2, r3 = st.columns(3)
        for col, name, pred, prob, border_color, emoji in zip(
            [r1, r2, r3],
            ["KNN","Logistic Regression","Naive Bayes"],
            [pred_knn, pred_lr, pred_nb],
            [prob_knn, prob_lr, prob_nb],
            ["#1565C0","#2E7D32","#E65100"],
            ["🔵","🟢","🟠"]
        ):
            with col:
                # ── fully dark cards — no light background so global CSS won't hide text ──
                if pred == 1:
                    card_bg      = "#2D1515"
                    verdict_text = "❤️ Has Disease"
                    verdict_col  = "#FF6B6B"
                else:
                    card_bg      = "#0D2016"
                    verdict_text = "💚 No Disease"
                    verdict_col  = "#56D364"

                st.markdown(f"""
                <div style="background:{card_bg} !important;
                            border:2px solid {border_color};
                            border-radius:14px;padding:22px;text-align:center;
                            box-shadow:0 4px 16px rgba(0,0,0,0.5)">
                    <div style="font-size:1rem;font-weight:700;
                                color:{border_color} !important;margin-bottom:8px">
                        {emoji} {name}
                    </div>
                    <div style="font-size:1.7rem;font-weight:800;
                                color:{verdict_col} !important;margin:10px 0">
                        {verdict_text}
                    </div>
                    <div style="background:#30363D;border-radius:6px;
                                height:10px;overflow:hidden;margin:10px 0">
                        <div style="background:{border_color};
                                    width:{prob:.0f}%;height:100%;border-radius:6px"></div>
                    </div>
                    <div style="font-size:1rem;font-weight:700;
                                color:#E6EDF3 !important;margin-top:6px">
                        Disease probability: <span style="color:{verdict_col} !important">
                        {prob:.1f}%</span>
                    </div>
                </div>""", unsafe_allow_html=True)

        st.markdown("---")

        avg_prob = (prob_knn + prob_lr + prob_nb) / 3
        if votes >= 2:
            st.markdown(f"""
            <div style="background:#2D1515 !important;border:2px solid #E53935;
                        border-radius:12px;padding:24px;text-align:center;
                        box-shadow:0 4px 20px rgba(229,57,53,0.3)">
                <div style="font-size:1.5rem;font-weight:800;
                            color:#FF6B6B !important;margin-bottom:8px">
                    ❤️ FINAL VERDICT: LIKELY HAS HEART DISEASE
                </div>
                <div style="font-size:1rem;font-weight:500;color:#C9D1D9 !important">
                    {votes}/3 models agree &nbsp;·&nbsp;
                    Average disease probability:
                    <span style="color:#FF6B6B !important;font-weight:700">
                    {avg_prob:.1f}%</span>
                </div>
            </div>""", unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div style="background:#0D2016 !important;border:2px solid #2E7D32;
                        border-radius:12px;padding:24px;text-align:center;
                        box-shadow:0 4px 20px rgba(46,125,50,0.3)">
                <div style="font-size:1.5rem;font-weight:800;
                            color:#56D364 !important;margin-bottom:8px">
                    💚 FINAL VERDICT: LIKELY NO HEART DISEASE
                </div>
                <div style="font-size:1rem;font-weight:500;color:#C9D1D9 !important">
                    {3-votes}/3 models agree &nbsp;·&nbsp;
                    Average disease probability:
                    <span style="color:#56D364 !important;font-weight:700">
                    {avg_prob:.1f}%</span>
                </div>
            </div>""", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        st.info("⚠️ This is an educational ML tool only. Always consult a qualified doctor for medical advice.")


# ════════════════════════════════════════════════════════════
# PAGE 5 — CLINICAL SUMMARY
# ════════════════════════════════════════════════════════════
elif page == "📋  Clinical Summary":
    st.markdown("# 📋 Clinical Observations & Conclusions")
    st.markdown("Based on the Heart Disease Dataset — UCI / Kaggle (johnsmith88). 1,025 patient records.")
    st.markdown("---")

    st.markdown('<div class="section-title">7.1 Key Clinical Observations</div>', unsafe_allow_html=True)
    observations = [
        ("⚖️ Balanced Dataset",
         "51.3% disease (n=526) vs 48.7% no disease (n=499) — suitable for ML without aggressive resampling."),
        ("👥 Sex Distribution",
         "Males = 69.6% of dataset (n=713), yet females show proportionally higher disease rate in this sample."),
        ("💤 Silent CAD Paradox",
         "Asymptomatic chest pain (cp=0) is the most common type at ~48.5% — highlighting the silent nature of CAD."),
        ("🩸 Diabetes Prevalence",
         "Only 14.9% of patients have elevated fasting blood sugar (>120 mg/dL) — predominantly non-diabetic sample."),
        ("📉 ST Depression (oldpeak)",
         "Mean oldpeak: 1.07 mm overall — 0.57 mm in disease group vs 1.60 mm in no-disease group (1.03 mm gap). Note inverse coding."),
        ("💓 Max Heart Rate",
         "Disease patients average 139 bpm vs 158 bpm in healthy patients — 19 bpm gap reflecting impaired cardiac reserve."),
        ("🔬 Thalassemia Findings",
         "Reversible defect (thal=3) appears in 40.0% of all patients — strongly linked to a positive disease diagnosis."),
        ("🩺 Vessel Blockage",
         "56.4% of patients have zero major vessels coloured by fluoroscopy, reflecting early-stage disease cases."),
    ]
    for title, text in observations:
        st.markdown(f"""
        <div style="background:#161B22 !important;border-left:5px solid #E53935;
                    border-radius:10px;padding:16px 20px;margin-bottom:12px;
                    box-shadow:0 2px 8px rgba(0,0,0,0.3)">
            <div style="font-size:1rem;font-weight:700;
                        color:#7EB3FF !important;margin-bottom:6px">{title}</div>
            <div style="font-size:0.92rem;color:#C9D1D9 !important;line-height:1.6">{text}</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("---")
    st.markdown('<div class="section-title">6.2 Top Predictive Features</div>', unsafe_allow_html=True)
    features = [
        ("cp — Chest Pain Type",    "#1565C0",
         "Strong categorical predictor. Typical angina (type 0) and asymptomatic (type 3) show distinct disease associations."),
        ("thalach — Max Heart Rate","#2E7D32",
         "Patients with disease average 139 bpm vs 158 bpm without — impaired stress response is a hallmark of CAD."),
        ("thal — Thalassemia",      "#E53935",
         "Reversible defect on nuclear scan is the single strongest non-invasive indicator of ischaemia."),
        ("ca — Fluoroscopy Vessels","#7B1FA2",
         "Number of visible blocked vessels directly reflects anatomical disease burden."),
        ("oldpeak — ST Depression", "#E65100",
         "Magnitude of exercise-induced ST depression is a critical continuous marker of myocardial ischaemia."),
        ("exang — Exercise Angina", "#00695C",
         "Presence of exercise-induced chest pain is a direct symptom of ischaemia under physical stress."),
    ]
    f1, f2 = st.columns(2)
    for i, (fname, color, desc) in enumerate(features):
        col = f1 if i % 2 == 0 else f2
        with col:
            st.markdown(f"""
            <div style="background:#161B22 !important;border-top:4px solid {color};
                        border-radius:10px;padding:14px 18px;margin-bottom:12px;
                        box-shadow:0 2px 8px rgba(0,0,0,0.3)">
                <div style="font-size:0.95rem;font-weight:700;
                            color:{color} !important;margin-bottom:6px">{fname}</div>
                <div style="font-size:0.88rem;color:#C9D1D9 !important;line-height:1.5">{desc}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown("---")
    st.markdown('<div class="section-title">7.2 ST Depression (oldpeak) — Final Summary</div>',
                unsafe_allow_html=True)
    st.markdown("""
    <div style="background:#1C2128 !important;border:1px solid #444C56;
                border-radius:12px;padding:20px 24px;line-height:1.8">
        <span style="color:#7EB3FF !important;font-weight:700">oldpeak</span>
        <span style="color:#C9D1D9 !important"> measures exercise-induced ST segment depression in millimetres.
        A higher value traditionally indicates greater myocardial ischaemia. In this dataset's coding however,
        it is </span>
        <span style="color:#FF6B6B !important;font-weight:700">inversely correlated</span>
        <span style="color:#C9D1D9 !important"> with the disease label — suggesting disease patients had less severe
        depression at testing. Combined with </span>
        <span style="color:#7EB3FF !important;font-weight:700">slope</span>
        <span style="color:#C9D1D9 !important">, it remains one of the most clinically interpretable features in any CAD risk model.</span>
    </div>""", unsafe_allow_html=True)

    st.markdown("---")
    st.markdown('<div class="section-title">7.3 Dataset Limitations</div>', unsafe_allow_html=True)
    limitations = [
        ("🔀 Multi-source Merging",
         "Records merged from Cleveland, Hungary, Switzerland, and Long Beach VA — potential batch effects."),
        ("❓ Thal Coding Anomaly",
         "thal=0 appears for 7 patients, possibly representing missing or erroneous data."),
        ("⚠️ CA Value Out of Range",
         "CA values of 4 appear in 18 records; the original dataset caps at 3 — possible data-entry issues."),
        ("📊 Binary Target Simplification",
         "Binary target (0/1) simplifies a spectrum of disease severity — nuance of mild vs severe CAD is lost."),
        ("📅 Single Visit Snapshot",
         "All features are from a single clinical visit; longitudinal patient dynamics are not captured."),
    ]
    for title, text in limitations:
        st.markdown(f"""
        <div style="background:#161B22 !important;border-left:5px solid #444C56;
                    border-radius:10px;padding:14px 18px;margin-bottom:10px">
            <div style="font-size:0.95rem;font-weight:700;
                        color:#8B949E !important;margin-bottom:4px">{title}</div>
            <div style="font-size:0.88rem;color:#C9D1D9 !important;line-height:1.5">{text}</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("---")
    st.info("⚠️ Educational & academic purposes only. Not intended as clinical medical advice. Always consult a qualified cardiologist.")
