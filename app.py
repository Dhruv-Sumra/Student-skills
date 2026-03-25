

import os, warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
import pickle
import streamlit as st
warnings.filterwarnings("ignore")

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

# ── Page Config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="SkillRec · Student Recommendation",
    page_icon="🎓",
    layout="wide"
)

# ── Global CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@400;700;900&display=swap');

/* ── Variables ── */
:root {
    --ink:      #0a0e1a;
    --navy:     #112244;
    --navy-mid: #1a3566;
    --crimson:  #9b1d2a;
    --gold:     #c9841a;
    --gold-lt:  #e8a830;
    --teal:     #1a6b72;
    --sage:     #2d6a4f;
    --paper:    #f7f3ec;
    --card:     #ffffff;
    --border:   #c8bfa8;
    --muted:    #5c5446;
    --rule:     #b0a080;
}

html, body, [class*="css"] {
    font-family: "Times New Roman", "Georgia", Times, serif !important;
    background-color: var(--paper) !important;
    color: var(--ink) !important;
}

#MainMenu, footer, header { visibility: hidden; }

.main .block-container {
    padding: 2rem 3rem 5rem 3rem !important;
    max-width: 1260px;
}

/* ── Masthead ── */
.masthead {
    background: var(--navy);
    padding: 2rem 2.5rem 1.6rem;
    margin-bottom: 2.5rem;
    border-left: 6px solid var(--gold);
    position: relative;
}
.masthead-rule {
    height: 2px;
    background: linear-gradient(90deg, var(--gold), var(--gold-lt) 40%, transparent);
    margin: 1.2rem 0 0 0;
}
.masthead .kicker {
    font-size: 0.72rem;
    letter-spacing: 0.2em;
    text-transform: uppercase;
    color: var(--gold-lt);
    margin-bottom: 0.5rem;
}
.masthead h1 {
    font-family: "Playfair Display", "Times New Roman", serif !important;
    font-size: 2.2rem !important;
    font-weight: 900 !important;
    color: #ffffff !important;
    margin: 0 0 0.3rem 0 !important;
    line-height: 1.2;
}
.masthead .tagline {
    font-style: italic;
    color: #c8d4e8 !important;
    font-size: 1rem;
    margin: 0 !important;
}

/* ── Section & sub headings ── */
.section-title {
    font-family: "Playfair Display", "Times New Roman", serif !important;
    font-size: 1.65rem;
    font-weight: 700;
    color: var(--navy);
    border-bottom: 3px solid var(--gold);
    padding-bottom: 0.45rem;
    margin: 1.5rem 0 1.2rem 0;
}
.sub-title {
    font-family: "Times New Roman", Times, serif !important;
    font-size: 1.05rem;
    font-weight: 700;
    color: var(--navy-mid);
    text-transform: uppercase;
    letter-spacing: 0.06em;
    margin: 1.4rem 0 0.55rem 0;
    padding-bottom: 0.2rem;
    border-bottom: 1px solid var(--border);
}

/* ── Chart label strip — sits above each chart ── */
.chart-label {
    background: var(--navy);
    color: #ffffff;
    padding: 0.5rem 1rem;
    font-size: 0.82rem;
    font-weight: 700;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    margin-bottom: 0;
    border-left: 4px solid var(--gold);
}

/* ── Sidebar ── */
section[data-testid="stSidebar"] {
    background: var(--ink) !important;
    border-right: 4px solid var(--gold) !important;
}
section[data-testid="stSidebar"] * {
    color: #e8e2d4 !important;
    font-family: "Times New Roman", Times, serif !important;
}
section[data-testid="stSidebar"] h2 {
    font-size: 0.75rem !important;
    letter-spacing: 0.18em;
    text-transform: uppercase;
    color: var(--gold-lt) !important;
    border-bottom: 1px solid #3a3020;
    padding-bottom: 0.4rem;
    margin-bottom: 0.9rem !important;
}
section[data-testid="stSidebar"] hr {
    border-color: #3a3020 !important;
}
section[data-testid="stSidebar"] .stRadio label {
    font-size: 0.93rem !important;
}
section[data-testid="stSidebar"] [data-testid="stRadio"] > div > label {
    padding: 0.3rem 0.2rem !important;
}

/* ── Sidebar model badge ── */
.model-badge {
    background: var(--gold);
    color: var(--ink);
    padding: 1rem;
    text-align: center;
    margin-top: 0.8rem;
}
.model-badge .badge-acc {
    font-size: 2rem;
    font-weight: 900;
    display: block;
    line-height: 1.1;
}
.model-badge .badge-lbl {
    font-size: 0.68rem;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    display: block;
    margin-top: 0.15rem;
}

/* ── Metric cards ── */
[data-testid="metric-container"] {
    background: var(--card) !important;
    border: 1px solid var(--border) !important;
    border-top: 4px solid var(--navy) !important;
    border-radius: 0 !important;
    padding: 1rem 1.2rem !important;
    box-shadow: 0 2px 8px rgba(0,0,0,0.08) !important;
}
[data-testid="metric-container"] label {
    font-size: 0.73rem !important;
    text-transform: uppercase;
    letter-spacing: 0.12em;
    color: var(--muted) !important;
}
[data-testid="metric-container"] [data-testid="stMetricValue"] {
    font-size: 2.1rem !important;
    font-weight: 700 !important;
    color: var(--navy) !important;
}

/* ── Tabs ── */
.stTabs [data-baseweb="tab-list"] {
    border-bottom: 3px solid var(--navy) !important;
    gap: 0 !important;
}
.stTabs [data-baseweb="tab"] {
    background: #ede8de !important;
    border: 1px solid var(--border) !important;
    border-bottom: none !important;
    border-radius: 0 !important;
    padding: 0.45rem 1.3rem !important;
    font-family: "Times New Roman", Times, serif !important;
    font-size: 0.87rem !important;
    letter-spacing: 0.04em;
    color: var(--muted) !important;
    margin-right: 3px;
}
.stTabs [aria-selected="true"] {
    background: var(--navy) !important;
    color: #ffffff !important;
    border-color: var(--navy) !important;
}

/* ── Dataframe ── */
.stDataFrame thead th {
    background: var(--navy) !important;
    color: #ffffff !important;
    font-family: "Times New Roman", Times, serif !important;
    font-size: 0.82rem !important;
    text-transform: uppercase;
    letter-spacing: 0.06em;
}

/* ── Alerts ── */
.stAlert {
    border-radius: 0 !important;
    font-family: "Times New Roman", Times, serif !important;
}

/* ── Sliders ── */
.stSlider label {
    font-family: "Times New Roman", Times, serif !important;
    font-size: 0.92rem !important;
    color: var(--ink) !important;
}

/* ── Buttons ── */
.stButton button[kind="primary"] {
    background: var(--navy) !important;
    color: #ffffff !important;
    border: 2px solid var(--gold) !important;
    border-radius: 0 !important;
    font-family: "Times New Roman", Times, serif !important;
    font-size: 0.95rem !important;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    padding: 0.6rem 2.5rem !important;
    transition: all 0.2s;
}
.stButton button[kind="primary"]:hover {
    background: var(--gold) !important;
    color: var(--ink) !important;
}
.stButton button {
    border-radius: 0 !important;
    font-family: "Times New Roman", Times, serif !important;
}

/* ── Callout / pull-quote ── */
.callout {
    border-left: 4px solid var(--gold);
    background: #f0ebe0;
    padding: 0.9rem 1.3rem;
    font-style: italic;
    font-size: 0.97rem;
    color: var(--ink);
    margin: 0.9rem 0;
}
.callout b, .callout strong { font-style: normal; color: var(--navy); }

/* ── Gold rule ── */
.gold-rule {
    border: none;
    border-top: 1px solid var(--rule);
    margin: 1.6rem 0;
}

/* ── Explanation expander (only for chart explanations) ── */
.streamlit-expanderHeader {
    background: #eee8da !important;
    border: 1px solid var(--border) !important;
    border-left: 4px solid var(--teal) !important;
    border-radius: 0 !important;
    font-family: "Times New Roman", Times, serif !important;
    font-size: 0.88rem !important;
    font-weight: 700 !important;
    color: var(--teal) !important;
    letter-spacing: 0.04em;
    padding: 0.55rem 1rem !important;
    text-transform: uppercase;
}
.streamlit-expanderContent {
    border: 1px solid var(--border) !important;
    border-top: none !important;
    border-left: 4px solid var(--teal) !important;
    border-radius: 0 !important;
    background: var(--card) !important;
    padding: 1rem 1.3rem !important;
    font-size: 0.95rem;
    line-height: 1.7;
}

/* ── Result prediction card ── */
.result-card {
    background: var(--navy);
    border-left: 6px solid var(--gold);
    padding: 2.4rem 2rem;
    text-align: center;
    margin: 1.5rem 0;
}
.result-card .rc-kicker {
    font-size: 0.72rem;
    letter-spacing: 0.22em;
    text-transform: uppercase;
    color: var(--gold-lt);
    margin-bottom: 0.6rem;
}
.result-card h1 {
    font-family: "Playfair Display", "Times New Roman", serif !important;
    font-size: 3rem !important;
    color: #ffffff !important;
    margin: 0 0 0.2rem 0 !important;
    font-weight: 900 !important;
}
.result-card .rc-sub {
    font-style: italic;
    color: #c8d4e8;
    font-size: 1.05rem;
    margin-bottom: 1.2rem;
}
.result-card .rc-conf {
    font-size: 1.2rem;
    color: var(--gold-lt);
    border-top: 1px solid #2a4070;
    padding-top: 1rem;
    margin-top: 0.5rem;
}

/* ── Strength rows ── */
.str-row {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 0.5rem 0;
    border-bottom: 1px dashed #d8d0c0;
    font-size: 0.93rem;
}
.str-pill {
    background: var(--navy);
    color: #ffffff;
    padding: 0.18rem 0.6rem;
    font-size: 0.8rem;
    letter-spacing: 0.05em;
    white-space: nowrap;
}
</style>
""", unsafe_allow_html=True)

# ── Constants ──────────────────────────────────────────────────────────────────
SKILL_COLS   = ["Skill1", "Skill2", "Skill3", "Skill4"]
SUBJECT_COLS = ["Subject1", "Subject2", "Subject3", "Subject4", "Subject5"]

SKILL_NAMES = {
    "Skill1": "Logical Reasoning",
    "Skill2": "Analytical Thinking",
    "Skill3": "Problem Solving",
    "Skill4": "Creative Thinking"
}
SUBJECT_NAMES = {
    "Subject1": "Mathematics",
    "Subject2": "Science",
    "Subject3": "Technology",
    "Subject4": "Arts",
    "Subject5": "Business"
}

BASE = os.path.dirname(os.path.abspath(__file__))

# ── Chart color palette ────────────────────────────────────────────────────────
# High-contrast, distinct colours for charts
C_NAVY    = "#112244"
C_CRIMSON = "#9b1d2a"
C_TEAL    = "#1a6b72"
C_AMBER   = "#c9841a"
C_SAGE    = "#2d6a4f"
C_VIOLET  = "#5a3472"
C_PAPER   = "#ffffff"      # pure white chart bg for max contrast

CHART_PALETTE = [C_NAVY, C_CRIMSON, C_TEAL, C_AMBER, C_SAGE, C_VIOLET]

def apply_chart_style(ax, title="", xlabel="", ylabel=""):
    """Clean white background, minimal grid, serif labels"""
    ax.set_facecolor(C_PAPER)
    ax.figure.patch.set_facecolor("#f7f3ec")
    for spine in ax.spines.values():
        spine.set_color("#9a8e7a"); spine.set_linewidth(0.7)
    ax.tick_params(colors="#0a0e1a", labelsize=9)
    ax.xaxis.label.set_color("#2a2010")
    ax.yaxis.label.set_color("#2a2010")
    if title:
        ax.set_title(title, fontsize=11, fontweight='bold', color=C_NAVY,
                     pad=10, fontfamily='serif')
    if xlabel: ax.set_xlabel(xlabel, fontsize=9, fontfamily='serif')
    if ylabel: ax.set_ylabel(ylabel, fontsize=9, fontfamily='serif')
    ax.grid(axis='y', color="#ddd8cc", linestyle='--', alpha=0.55, linewidth=0.55)
    ax.set_axisbelow(True)

# ── Data generation (logic untouched) ─────────────────────────────────────────
def generate_sample_data():
    expert_path = os.path.join(BASE, "ExpertSkillMap.csv")
    if not os.path.exists(expert_path):
        pd.DataFrame({
            "Subject": ["Mathematics", "Science", "Technology", "Arts", "Business"],
            "Skill1": [1, 2, 3, 4, 2], "Skill2": [2, 1, 4, 3, 3],
            "Skill3": [3, 4, 1, 2, 4], "Skill4": [4, 3, 2, 1, 1]
        }).to_csv(expert_path, index=False)

    cog_path = os.path.join(BASE, "StudentCognitiveSkill.csv")
    if not os.path.exists(cog_path):
        np.random.seed(42)
        pd.DataFrame({
            "Student": [f"S{i}" for i in range(1, 501)],
            "Skill1": np.random.randint(1, 5, 500), "Skill2": np.random.randint(1, 5, 500),
            "Skill3": np.random.randint(1, 5, 500), "Skill4": np.random.randint(1, 5, 500)
        }).to_csv(cog_path, index=False)

    sub_path = os.path.join(BASE, "StudentSubjectKnowledge.csv")
    if not os.path.exists(sub_path):
        np.random.seed(42)
        pd.DataFrame({
            "Student": [f"S{i}" for i in range(1, 501)],
            "Subject1": np.random.randint(1, 6, 500), "Subject2": np.random.randint(1, 6, 500),
            "Subject3": np.random.randint(1, 6, 500), "Subject4": np.random.randint(1, 6, 500),
            "Subject5": np.random.randint(1, 6, 500)
        }).to_csv(sub_path, index=False)

@st.cache_data
def load_data():
    generate_sample_data()
    expert    = pd.read_csv(os.path.join(BASE, "ExpertSkillMap.csv"))
    cognitive = pd.read_csv(os.path.join(BASE, "StudentCognitiveSkill.csv"))
    subject   = pd.read_csv(os.path.join(BASE, "StudentSubjectKnowledge.csv"))
    if 'Subject1' not in subject.columns:
        for i, col in enumerate(subject.columns):
            if col != 'Student' and i <= 5:
                subject.rename(columns={col: f'Subject{i}'}, inplace=True)
    return expert, cognitive, subject

def create_features(df):
    df = df.copy()
    sk = [c for c in SKILL_COLS   if c in df.columns]
    su = [c for c in SUBJECT_COLS if c in df.columns]
    if sk:
        df['avg_skill']       = df[sk].mean(axis=1).round(2)
        df['total_skill']     = df[sk].sum(axis=1)
        df['strongest_skill'] = df[sk].idxmax(axis=1)
        df['weakest_skill']   = df[sk].idxmin(axis=1)
        df['skill_range']     = df[sk].max(axis=1) - df[sk].min(axis=1)
    if su:
        df['avg_subject']   = df[su].mean(axis=1).round(2)
        df['total_subject'] = df[su].sum(axis=1)
        df['best_subject']  = df[su].idxmax(axis=1)
        df['worst_subject'] = df[su].idxmin(axis=1)
    if 'avg_skill' in df.columns and 'avg_subject' in df.columns:
        df['overall_score'] = (df['avg_skill'] * 0.6 + df['avg_subject'] * 0.4).round(2)
    elif 'avg_skill'    in df.columns: df['overall_score'] = df['avg_skill'].round(2)
    elif 'avg_subject'  in df.columns: df['overall_score'] = df['avg_subject'].round(2)
    return df

def build_target(df, expert):
    targets, sk = [], [c for c in SKILL_COLS if c in df.columns]
    for _, row in df.iterrows():
        sv = np.array([row[c] for c in sk])
        best, best_sc = None, -1
        for _, er in expert.iterrows():
            sc = np.sum(np.abs(sv - np.array([er[c] for c in sk])))
            if best is None or sc < best_sc: best_sc = sc; best = er['Subject']
        targets.append(best)
    return targets

@st.cache_resource
def train_models():
    expert, cognitive, subject = load_data()
    df = pd.merge(cognitive, subject, on='Student')
    df['Target'] = build_target(df, expert)
    df = create_features(df)
    sk = [c for c in SKILL_COLS   if c in df.columns]
    su = [c for c in SUBJECT_COLS if c in df.columns]
    feat_cols = sk + su + [c for c in
        ['avg_skill','total_skill','avg_subject','total_subject','skill_range','overall_score']
        if c in df.columns]
    X, y = df[feat_cols], df['Target']
    le = LabelEncoder(); y_enc = le.fit_transform(y)
    X_tr, X_te, y_tr, y_te = train_test_split(X, y_enc, test_size=0.2, random_state=42)
    sc = StandardScaler(); Xtr = sc.fit_transform(X_tr); Xte = sc.transform(X_te)
    models = {
        'Random Forest':       RandomForestClassifier(n_estimators=100, random_state=42),
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
        'Decision Tree':       DecisionTreeClassifier(max_depth=5, random_state=42),
        'KNN':                 KNeighborsClassifier(n_neighbors=5),
        'SVM':                 SVC(kernel='rbf', probability=True, random_state=42)
    }
    results = {}; best_m, best_sc_val, best_nm = None, 0, ""
    for nm, m in models.items():
        m.fit(Xtr, y_tr); yp = m.predict(Xte)
        acc = accuracy_score(y_te, yp); f1 = f1_score(y_te, yp, average='weighted')
        results[nm] = {'model': m, 'accuracy': acc, 'f1_score': f1}
        if acc > best_sc_val: best_sc_val = acc; best_m = m; best_nm = nm
    bundle = {'model': best_m, 'model_name': best_nm, 'scaler': sc,
              'label_encoder': le, 'feature_cols': feat_cols, 'classes': le.classes_}
    return bundle, df, results

# ── EDA Plot Functions ─────────────────────────────────────────────────────────

def plot_skill_distribution(df):
    avail = [c for c in SKILL_COLS if c in df.columns]
    if not avail: return None
    n = len(avail)
    fig, axes = plt.subplots(1, n, figsize=(4.2 * n, 4.2))
    if n == 1: axes = [axes]
    for i, col in enumerate(avail):
        counts = df[col].value_counts().sort_index()
        bars = axes[i].bar(counts.index, counts.values,
                           color=C_NAVY, edgecolor="#ffffff",
                           linewidth=1.2, width=0.6)
        for b, v in zip(bars, counts.values):
            axes[i].text(b.get_x() + b.get_width()/2,
                         b.get_height() + 1.8, str(v),
                         ha='center', fontsize=8.5, fontweight='bold', color=C_NAVY)
        apply_chart_style(axes[i], title=SKILL_NAMES.get(col, col),
                          xlabel='Rating (1–4)', ylabel='Students')
        axes[i].set_xticks([1, 2, 3, 4])
    fig.patch.set_facecolor("#f7f3ec")
    plt.tight_layout(pad=2)
    return fig

def plot_subject_distribution(df):
    avail = [c for c in SUBJECT_COLS if c in df.columns]
    if not avail: return None
    n = len(avail); ncols = min(3, n); nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(5.2 * ncols, 4.2 * nrows))
    axes = np.array(axes).flatten() if n > 1 else [axes]
    for i, col in enumerate(avail):
        counts = df[col].value_counts().sort_index()
        bars = axes[i].bar(counts.index, counts.values,
                           color=C_CRIMSON, edgecolor="#ffffff",
                           linewidth=1.2, width=0.65)
        for b, v in zip(bars, counts.values):
            axes[i].text(b.get_x() + b.get_width()/2,
                         b.get_height() + 1.8, str(v),
                         ha='center', fontsize=8.5, fontweight='bold', color=C_CRIMSON)
        apply_chart_style(axes[i], title=SUBJECT_NAMES.get(col, col),
                          xlabel='Rating (1–5)', ylabel='Students')
        axes[i].set_xticks([1, 2, 3, 4, 5])
    for j in range(n, len(axes)): axes[j].set_visible(False)
    fig.patch.set_facecolor("#f7f3ec")
    plt.tight_layout(pad=2)
    return fig

def plot_target_distribution(df):
    if 'Target' not in df.columns: return None
    counts = df['Target'].value_counts()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    colors = CHART_PALETTE[:len(counts)]

    bars = ax1.bar(counts.index, counts.values, color=colors,
                   edgecolor='#ffffff', linewidth=1.2, width=0.6)
    for b, v in zip(bars, counts.values):
        ax1.text(b.get_x() + b.get_width()/2, b.get_height() + 2, str(v),
                 ha='center', fontweight='bold', fontsize=9, color=C_NAVY)
    ax1.set_xticklabels(counts.index, rotation=28, ha='right', fontsize=9)
    apply_chart_style(ax1, title="Students per Recommended Subject",
                      xlabel="", ylabel="Number of Students")

    wedges, texts, autotexts = ax2.pie(
        counts.values, labels=counts.index, autopct='%1.1f%%',
        colors=colors, startangle=90,
        wedgeprops={'linewidth': 2, 'edgecolor': '#f7f3ec'},
        textprops={'fontsize': 9, 'fontfamily': 'serif', 'color': '#0a0e1a'})
    for at in autotexts:
        at.set_fontsize(8.5); at.set_fontweight('bold'); at.set_color('#ffffff')
    ax2.set_title("Proportion of Recommendations", fontsize=11,
                  fontweight='bold', color=C_NAVY, fontfamily='serif', pad=8)
    ax2.set_facecolor(C_PAPER); fig.patch.set_facecolor("#f7f3ec")
    plt.tight_layout(pad=2)
    return fig

def plot_skill_by_target(df):
    if 'Target' not in df.columns: return None
    avail = [c for c in SKILL_COLS if c in df.columns]
    if not avail: return None
    n = len(avail); nrows = (n + 1) // 2
    fig, axes = plt.subplots(nrows, 2, figsize=(14, 5 * nrows))
    axes = axes.flatten()
    targets = sorted(df['Target'].unique())
    palette = CHART_PALETTE[:len(targets)]
    for i, col in enumerate(avail):
        data = [df[df['Target'] == t][col].values for t in targets]
        bp = axes[i].boxplot(data, labels=targets, patch_artist=True,
                             medianprops=dict(color=C_AMBER, linewidth=2.5),
                             whiskerprops=dict(color='#444444', linewidth=1),
                             capprops=dict(color='#444444', linewidth=1.5),
                             flierprops=dict(marker='o', markerfacecolor=C_AMBER,
                                            markeredgecolor='white', markersize=4))
        for patch, color in zip(bp['boxes'], palette):
            patch.set_facecolor(color); patch.set_alpha(0.82)
        apply_chart_style(axes[i], title=SKILL_NAMES.get(col, col),
                          xlabel="Recommended Subject", ylabel="Skill Rating (1–4)")
        axes[i].set_xticklabels(targets, rotation=28, ha='right', fontsize=8.5)
        axes[i].set_yticks([1, 2, 3, 4])
    for j in range(n, len(axes)): axes[j].set_visible(False)
    fig.suptitle("Skill Profiles by Recommended Subject",
                 fontsize=13, fontweight='bold', color=C_NAVY, fontfamily='serif', y=1.01)
    fig.patch.set_facecolor("#f7f3ec")
    plt.tight_layout(pad=2)
    return fig

def plot_correlation_heatmap(df):
    avail = [c for c in SKILL_COLS + SUBJECT_COLS if c in df.columns]
    if len(avail) < 2: return None
    corr = df[avail].corr()
    readable = {c: SKILL_NAMES.get(c, SUBJECT_NAMES.get(c, c)) for c in avail}
    corr = corr.rename(index=readable, columns=readable)
    fig, ax = plt.subplots(figsize=(11, 7))
    # High-contrast diverging: deep crimson → white → deep teal
    cmap = mpl.colors.LinearSegmentedColormap.from_list(
        "cr_teal", [C_CRIMSON, "#f7f3ec", C_TEAL], N=256)
    sns.heatmap(corr, ax=ax, annot=True, fmt=".2f", cmap=cmap,
                linewidths=0.6, linecolor="#e0d8cc",
                vmin=-1, vmax=1,
                annot_kws={"size": 9, "weight": "bold", "color": "#ffffff"},
                cbar_kws={"shrink": 0.75, "label": "Pearson r"})
    # Fix annotation colors for readability near center
    for text in ax.texts:
        val = float(text.get_text())
        text.set_color("#0a0e1a" if abs(val) < 0.35 else "#ffffff")
    ax.set_title("Correlation Matrix — Skills & Subject Knowledge",
                 fontsize=12, fontweight='bold', color=C_NAVY, fontfamily='serif', pad=14)
    ax.set_facecolor(C_PAPER); fig.patch.set_facecolor("#f7f3ec")
    plt.xticks(rotation=30, ha='right', fontsize=9)
    plt.yticks(rotation=0, fontsize=9)
    plt.tight_layout(pad=2)
    return fig

def plot_avg_score_by_target(df):
    if 'Target' not in df.columns or 'overall_score' not in df.columns: return None
    avg = df.groupby('Target')['overall_score'].mean().sort_values(ascending=False)
    fig, ax = plt.subplots(figsize=(9, 4.5))
    colors = CHART_PALETTE[:len(avg)]
    bars = ax.bar(avg.index, avg.values, color=colors,
                  edgecolor='#ffffff', linewidth=1.2, width=0.55)
    for b, v in zip(bars, avg.values):
        ax.text(b.get_x() + b.get_width()/2, b.get_height() + 0.02,
                f"{v:.2f}", ha='center', fontsize=9.5, fontweight='bold', color=C_NAVY)
    apply_chart_style(ax, title="Average Overall Score per Recommended Subject",
                      xlabel="", ylabel="Mean Overall Score")
    ax.set_xticklabels(avg.index, rotation=22, ha='right', fontsize=9)
    fig.patch.set_facecolor("#f7f3ec")
    plt.tight_layout(pad=2)
    return fig

def plot_violin_skills(df):
    """Violin plot: skill value distributions by subject — richer than boxplot"""
    if 'Target' not in df.columns: return None
    avail = [c for c in SKILL_COLS if c in df.columns]
    if not avail: return None
    targets = sorted(df['Target'].unique())
    n = len(avail); nrows = (n + 1) // 2
    fig, axes = plt.subplots(nrows, 2, figsize=(14, 5 * nrows))
    axes = axes.flatten()
    palette = {t: CHART_PALETTE[i % len(CHART_PALETTE)] for i, t in enumerate(targets)}
    for i, col in enumerate(avail):
        melt = df[['Target', col]].rename(columns={col: 'rating'})
        # manual violin using seaborn
        sns.violinplot(data=melt, x='Target', y='rating',
                       order=targets, palette=palette,
                       inner='box', linewidth=1.2, ax=axes[i])
        axes[i].set_xticklabels(targets, rotation=28, ha='right', fontsize=8.5)
        axes[i].set_yticks([1, 2, 3, 4])
        apply_chart_style(axes[i], title=SKILL_NAMES.get(col, col),
                          xlabel="Recommended Subject", ylabel="Skill Rating")
    for j in range(n, len(axes)): axes[j].set_visible(False)
    fig.suptitle("Skill Rating Distributions per Subject (Violin)",
                 fontsize=13, fontweight='bold', color=C_NAVY, fontfamily='serif', y=1.01)
    fig.patch.set_facecolor("#f7f3ec")
    plt.tight_layout(pad=2)
    return fig

# ── Helper: chart section wrapper ─────────────────────────────────────────────
def chart_section(label, fig, explanation_title, explanation_md, key_suffix):
    """Show chart always-visible; explanation in a teal expander below it."""
    st.markdown(f"<div class='chart-label'>{label}</div>", unsafe_allow_html=True)
    if fig:
        st.pyplot(fig, use_container_width=True); plt.close()
    else:
        st.info("Data unavailable for this chart.")
    with st.expander(f"📖  {explanation_title}"):
        st.markdown(explanation_md)
    st.markdown("<hr class='gold-rule'>", unsafe_allow_html=True)

# ── Main ───────────────────────────────────────────────────────────────────────
def main():

    # ── Masthead ──
    st.markdown("""
    <div class='masthead'>
        <p class='kicker'>Machine Learning · Academic Guidance System · Vol. I</p>
        <h1>Student Skill-Based Course Recommendation</h1>
        <p class='tagline'>Discover the subject that best aligns with your cognitive profile and academic strengths.</p>
        <div class='masthead-rule'></div>
    </div>
    """, unsafe_allow_html=True)

    with st.spinner("Initialising recommendation engine…"):
        try:
            bundle, df, results = train_models()
        except Exception as e:
            st.error(f"System error: {e}")
            st.info("Verify that CSV files exist and are correctly formatted.")
            return

    # ── Sidebar ──
    st.sidebar.markdown("## Navigation")
    page = st.sidebar.radio("", [
        "Problem Statement",
        "Dataset Overview",
        "Exploratory Analysis",
        "Feature Engineering",
        "Model Performance",
        "Get Recommendation"
    ])
    st.sidebar.markdown("---")
    st.sidebar.markdown("### Champion Model")
    best_nm  = bundle['model_name']
    best_acc = results[best_nm]['accuracy'] * 100
    best_f1  = results[best_nm]['f1_score']  * 100
    st.sidebar.markdown(f"""
    <div class='model-badge'>
        <strong style='font-size:0.9rem;'>{best_nm}</strong>
        <span class='badge-acc'>{best_acc:.1f}%</span>
        <span class='badge-lbl'>Accuracy</span>
        <hr style='border-color:#8a6010;margin:0.5rem 0;'>
        <span style='font-size:0.88rem;'>F1 Score: <b>{best_f1:.1f}%</b></span>
    </div>
    """, unsafe_allow_html=True)

    # ════════════════════════════════════════════════════════
    # PAGE 1 · PROBLEM STATEMENT
    # ════════════════════════════════════════════════════════
    if page == "Problem Statement":
        st.markdown("<div class='section-title'>The Problem We Solve</div>", unsafe_allow_html=True)

        col1, col2 = st.columns(2, gap="large")
        with col1:
            st.markdown("<div class='sub-title'>Why students choose poorly</div>", unsafe_allow_html=True)
            st.markdown("""
            Many students make subject choices based on peer pressure, parental expectation,
            or simple unfamiliarity with their own strengths — leading to poor academic
            performance and reduced career satisfaction.

            **Common root causes:**
            - Inability to objectively assess personal strengths
            - Lack of data-driven guidance tools
            - Over-reliance on anecdotal or social advice
            """)
            st.markdown("<div class='sub-title'>Our Approach</div>", unsafe_allow_html=True)
            st.markdown("""
            We collect two types of data:
            - **Cognitive Skills** — four measurable intellectual traits rated 1–4
            - **Subject Knowledge** — proficiency across five academic domains rated 1–5

            A trained ML classifier maps this profile to the most suitable subject.
            """)

        with col2:
            st.markdown("<div class='sub-title'>System Workflow</div>", unsafe_allow_html=True)
            st.markdown("""
            | Step | Action |
            |------|--------|
            | 1 | Student rates their skills and subject knowledge |
            | 2 | System derives statistical features from the input |
            | 3 | Trained model predicts the best-fit subject |
            | 4 | Recommendation returned with a confidence score |
            """)
            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown("""
            <div class='callout'>
            "A data-driven recommendation — even an imperfect one — is more reliable than
            intuition alone, particularly for consequential academic decisions."
            </div>
            """, unsafe_allow_html=True)
            st.markdown("""
            **Rating scales:**
            - Skills: 1 = Weak · 2 = Developing · 3 = Proficient · 4 = Expert
            - Subjects: 1 = Beginner · 2 = Elementary · 3 = Intermediate · 4 = Advanced · 5 = Expert
            """)

    # ════════════════════════════════════════════════════════
    # PAGE 2 · DATASET OVERVIEW
    # ════════════════════════════════════════════════════════
    elif page == "Dataset Overview":
        st.markdown("<div class='section-title'>Dataset Overview</div>", unsafe_allow_html=True)

        c1, c2, c3, c4 = st.columns(4)
        with c1: st.metric("Total Students", len(df))
        with c2: st.metric("Skill Dimensions", len([c for c in SKILL_COLS if c in df.columns]))
        with c3: st.metric("Subject Domains", len([c for c in SUBJECT_COLS if c in df.columns]))
        with c4: st.metric("Target Classes", df['Target'].nunique() if 'Target' in df.columns else "–")

        st.markdown("<hr class='gold-rule'>", unsafe_allow_html=True)
        tab1, tab2, tab3 = st.tabs(["  Skill Ratings  ", "  Subject Knowledge  ", "  Target Labels  "])
        sk = [c for c in SKILL_COLS   if c in df.columns]
        su = [c for c in SUBJECT_COLS if c in df.columns]

        with tab1:
            if sk:
                d = df[['Student'] + sk].head(12).copy()
                d.columns = ['Student'] + [SKILL_NAMES[c] for c in sk]
                st.dataframe(d, use_container_width=True)
                st.caption("First 12 records · 1 = Low, 4 = High")
            else: st.warning("No skill data found.")

        with tab2:
            if su:
                d = df[['Student'] + su].head(12).copy()
                d.columns = ['Student'] + [SUBJECT_NAMES[c] for c in su]
                st.dataframe(d, use_container_width=True)
                st.caption("First 12 records · 1 = Beginner, 5 = Expert")
            else: st.warning("No subject data found.")

        with tab3:
            if 'Target' in df.columns:
                cts = df['Target'].value_counts().reset_index()
                cts.columns = ['Recommended Subject', 'Count']
                cts['Share (%)'] = (cts['Count'] / len(df) * 100).round(1)
                st.dataframe(cts, use_container_width=True)
            else: st.warning("No target labels available.")

    # ════════════════════════════════════════════════════════
    # PAGE 3 · EXPLORATORY ANALYSIS
    # Charts always visible. Only explanation text in expanders.
    # ════════════════════════════════════════════════════════
    elif page == "Exploratory Analysis":
        st.markdown("<div class='section-title'>Exploratory Data Analysis</div>", unsafe_allow_html=True)
        st.markdown(
            "All charts are displayed below. Click **📖 Read explanation** under any chart "
            "to understand what it shows and how to interpret it.",
        )
        st.markdown("<hr class='gold-rule'>", unsafe_allow_html=True)

        # ── Chart 1 ───────────────────────────────────────
        chart_section(
            label="① Cognitive Skill Rating Distributions",
            fig=plot_skill_distribution(df),
            explanation_title="What this chart shows & how to read it",
            explanation_md="""
**What it shows:** How many students received each rating (1–4) across the four
cognitive skill dimensions — Logical Reasoning, Analytical Thinking, Problem Solving,
and Creative Thinking.

**How to read it:** Each bar represents a rating level. A taller bar means more
students share that rating. If most bars are clustered at 1–2, the cohort is
generally weaker in that skill. A relatively even spread across all four ratings
suggests a well-balanced group.

**Why it matters:** Knowing the skill distribution helps us check whether the training
data is representative. A heavily skewed distribution could bias the model towards
recommending subjects that suit the majority of students.
            """,
            key_suffix="skill_dist"
        )

        # ── Chart 2 ───────────────────────────────────────
        chart_section(
            label="② Subject Knowledge Distributions",
            fig=plot_subject_distribution(df),
            explanation_title="What this chart shows & how to read it",
            explanation_md="""
**What it shows:** How student knowledge is spread across five subject domains
(Mathematics, Science, Technology, Arts, Business) on a 1–5 scale.

**How to read it:** A bar that peaks at rating 1 means most students are beginners
in that subject. A peak at 4–5 indicates strong existing knowledge. Comparing the
shape of each subject's distribution reveals where the cohort's strengths lie.

**Why it matters:** Subjects with very lopsided distributions may have fewer training
examples at the extremes, which can affect how confidently the model recommends them.
            """,
            key_suffix="subj_dist"
        )

        # ── Chart 3 ───────────────────────────────────────
        chart_section(
            label="③ Recommendation Label Distribution",
            fig=plot_target_distribution(df),
            explanation_title="What this chart shows & how to read it",
            explanation_md="""
**What it shows:** How the five recommended subjects are distributed across the entire
student population — shown as both a bar chart (absolute counts) and a pie chart
(percentage share).

**How to read it:** The bar chart makes it easy to compare raw numbers; the pie chart
highlights relative proportions at a glance. A roughly equal split (≈20% each) is
ideal for balanced model training.

**Why it matters for ML:** Severe class imbalance — for example if 60% of students
are recommended Arts — would make the model biased. It would learn to "play safe"
and over-recommend the dominant class, producing poor results for the others.
            """,
            key_suffix="target_dist"
        )

        # ── Chart 4 ───────────────────────────────────────
        chart_section(
            label="④ Skill Profiles by Recommended Subject — Box Plots",
            fig=plot_skill_by_target(df),
            explanation_title="What this chart shows & how to read it",
            explanation_md="""
**What it shows:** Box plots of each skill rating grouped by the recommended subject.
Each box summarises the distribution of a skill for all students assigned to that subject.

**How to read a box plot:**
- The **central line** (amber/gold) = median rating
- The **box edges** = 25th and 75th percentiles (middle 50% of students)
- The **whiskers** = the full range, excluding outliers
- **Dots** outside whiskers = individual outlier students

**What to look for:** If the boxes for a skill differ noticeably between subjects
(e.g., Technology students score much higher in Logical Reasoning than Arts students),
that skill is a strong predictive signal. Heavily overlapping boxes suggest the skill
alone cannot separate the two subjects.
            """,
            key_suffix="box_skill"
        )

        # ── Chart 5 ───────────────────────────────────────
        chart_section(
            label="⑤ Skill Distributions per Subject — Violin Plots",
            fig=plot_violin_skills(df),
            explanation_title="What this chart shows & how to read it",
            explanation_md="""
**What it shows:** Violin plots combine a box plot and a density estimate.
The wider the violin at a particular rating, the more students hold that rating
for that subject.

**How to read it:** A violin that is wide at rating 3–4 for a given subject means
many students assigned to that subject are proficient or expert in that skill.
A narrow violin at the top indicates few students reach high skill levels for
that subject recommendation.

**Advantage over box plots:** Violin plots reveal whether the data is unimodal
(one cluster), bimodal (two clusters), or skewed — patterns that box plots hide.
            """,
            key_suffix="violin_skill"
        )

        # ── Chart 6 ───────────────────────────────────────
        chart_section(
            label="⑥ Correlation Matrix — Skills & Subject Knowledge",
            fig=plot_correlation_heatmap(df),
            explanation_title="What this chart shows & how to read it",
            explanation_md="""
**What it shows:** Pearson correlation coefficients between every pair of skill and
subject columns. Values range from −1 to +1.

**How to read the colours:**
- **Deep crimson** (−1) = strong negative relationship: as one rises, the other falls
- **Near-white / cream** (≈0) = no meaningful linear relationship
- **Deep teal** (+1) = strong positive relationship: both rise together
- The diagonal is always **1.0** (a variable correlated with itself)

**Why it matters:** High correlation between two features means they carry overlapping
information. If Logical Reasoning and Analytical Thinking are highly correlated,
the model may not need both — or may treat them redundantly. Understanding correlations
also flags potential data quality issues.
            """,
            key_suffix="heatmap"
        )

        # ── Chart 7 ───────────────────────────────────────
        chart_section(
            label="⑦ Average Overall Score by Recommended Subject",
            fig=plot_avg_score_by_target(df),
            explanation_title="What this chart shows & how to read it",
            explanation_md="""
**What it shows:** The mean *overall score* for students assigned to each recommended
subject. The overall score is a weighted composite: 60% average skill + 40% average
subject knowledge.

**How to read it:** A taller bar means the typical student in that recommendation group
has a higher combined cognitive-and-knowledge profile. A short bar means students
recommended to that subject tend to have lower overall ratings — not necessarily bad,
as subject fit depends on the *pattern* of skills, not just the total.

**What to watch for:** If one subject consistently attracts the highest overall scores,
it might be the "catch-all" for strong students rather than a meaningful differentiation.
            """,
            key_suffix="avg_score"
        )

        # ── Chart 8 · Descriptive stats ───────────────────
        st.markdown("<div class='chart-label'>⑧ Descriptive Statistics — All Numeric Columns</div>",
                    unsafe_allow_html=True)
        avail = [c for c in SKILL_COLS + SUBJECT_COLS if c in df.columns]
        if 'overall_score' in df.columns: avail.append('overall_score')
        st.dataframe(df[avail].describe().round(3), use_container_width=True)
        with st.expander("📖  What this table shows & how to read it"):
            st.markdown("""
**What it shows:** Standard descriptive statistics for every numeric column:
count, mean, standard deviation (std), minimum, 25th percentile (Q1), median (50%),
75th percentile (Q3), and maximum.

**How to read it:**
- **Mean** tells you the average rating; compare it to the midpoint of the scale
  (2.5 for skills, 3.0 for subjects) to see whether the cohort skews high or low.
- **Std** measures spread — a high standard deviation means ratings are widely varied.
- **Min / Max** confirm that all values are within the expected range (1–4 for skills,
  1–5 for subjects). Values outside the range would indicate data quality issues.
- **Q1 and Q3** together define the interquartile range: where the middle 50% of
  students lie. A narrow IQR means most students are clustered near the median.
            """)
        st.markdown("<hr class='gold-rule'>", unsafe_allow_html=True)

    # ════════════════════════════════════════════════════════
    # PAGE 4 · FEATURE ENGINEERING
    # ════════════════════════════════════════════════════════
    elif page == "Feature Engineering":
        st.markdown("<div class='section-title'>Feature Engineering</div>", unsafe_allow_html=True)
        st.markdown(
            "Raw ratings are the foundation, but derived features expose deeper patterns "
            "and improve model accuracy. Below are the additional features constructed "
            "before model training."
        )
        st.markdown("<hr class='gold-rule'>", unsafe_allow_html=True)

        col1, col2 = st.columns(2, gap="large")
        with col1:
            st.markdown("<div class='sub-title'>Skill-Derived Features</div>", unsafe_allow_html=True)
            for name, desc in [
                ("Average Skill Score",
                 "Mean of all four skill ratings. Captures overall cognitive level in one number."),
                ("Total Skill Score",
                 "Sum of all four ratings. Equivalent to average on a larger scale — sometimes preferred by tree-based models."),
                ("Strongest Skill",
                 "The column with the highest rating — a student's primary cognitive edge."),
                ("Weakest Skill",
                 "The column with the lowest rating — highlights where development is needed."),
                ("Skill Range",
                 "Max minus min skill. Small range = balanced student. Large range = highly specialised."),
            ]:
                st.markdown(f"""<div class='callout'>
                    <strong>{name}</strong><br>
                    <span style='font-style:normal;font-size:0.93rem;'>{desc}</span>
                </div>""", unsafe_allow_html=True)

        with col2:
            st.markdown("<div class='sub-title'>Subject-Derived Features</div>", unsafe_allow_html=True)
            for name, desc in [
                ("Average Subject Score", "Mean knowledge across all five domains."),
                ("Total Subject Score",   "Sum of subject scores — a broad measure of academic exposure."),
                ("Best Subject",          "Domain with the highest self-reported knowledge."),
                ("Worst Subject",         "Domain most in need of development."),
            ]:
                st.markdown(f"""<div class='callout'>
                    <strong>{name}</strong><br>
                    <span style='font-style:normal;font-size:0.93rem;'>{desc}</span>
                </div>""", unsafe_allow_html=True)

            st.markdown("<div class='sub-title'>Composite Feature</div>", unsafe_allow_html=True)
            st.markdown("""<div class='callout'>
                <strong>Overall Score</strong><br>
                <span style='font-style:normal;font-size:0.93rem;'>
                Weighted blend: <em>60% average skill + 40% average subject knowledge.</em>
                Cognitive ability is weighted higher because the expert mapping is
                defined purely in skill space. This single value summarises a student's
                all-round preparedness.
                </span>
            </div>""", unsafe_allow_html=True)

        st.markdown("<hr class='gold-rule'>", unsafe_allow_html=True)
        st.markdown("<div class='sub-title'>Sample Profiles with Engineered Features</div>",
                    unsafe_allow_html=True)
        show_cols = ['Student'] + [c for c in
            ['avg_skill','strongest_skill','skill_range',
             'avg_subject','best_subject','overall_score','Target']
            if c in df.columns]
        if len(show_cols) > 1:
            sample = df[show_cols].head(8).rename(columns={
                'avg_skill': 'Avg Skill', 'strongest_skill': 'Strongest Skill',
                'skill_range': 'Skill Range', 'avg_subject': 'Avg Subject',
                'best_subject': 'Best Subject', 'overall_score': 'Overall Score',
                'Target': 'Recommended Subject'
            })
            st.dataframe(sample, use_container_width=True)

    # ════════════════════════════════════════════════════════
    # PAGE 5 · MODEL PERFORMANCE
    # ════════════════════════════════════════════════════════
    elif page == "Model Performance":
        st.markdown("<div class='section-title'>Model Performance</div>", unsafe_allow_html=True)

        res_df = pd.DataFrame([
            {'Model': n, 'Accuracy (%)': f"{r['accuracy']*100:.1f}",
             'F1 Score (%)': f"{r['f1_score']*100:.1f}"}
            for n, r in results.items()
        ]).sort_values('Accuracy (%)', ascending=False).reset_index(drop=True)
        res_df.index += 1
        st.dataframe(res_df, use_container_width=True)

        best = results[bundle['model_name']]
        st.success(
            f"**Champion: {bundle['model_name']}** — "
            f"Accuracy {best['accuracy']*100:.1f}%  ·  F1 Score {best['f1_score']*100:.1f}%"
        )
        st.markdown("<hr class='gold-rule'>", unsafe_allow_html=True)

        if hasattr(bundle['model'], 'feature_importances_') and bundle['feature_cols']:
            st.markdown("<div class='sub-title'>Feature Importance</div>", unsafe_allow_html=True)
            st.markdown("Which input signals carry the most weight in the model's decisions?")
            importances = bundle['model'].feature_importances_
            feat_names  = bundle['feature_cols']
            readable = [SKILL_NAMES.get(f, SUBJECT_NAMES.get(f, f.replace('_', ' ').title()))
                        for f in feat_names]
            n_show  = min(10, len(importances))
            indices = np.argsort(importances)[-n_show:]

            fig, ax = plt.subplots(figsize=(10, 5.5))
            max_idx = indices[-1]
            bar_colors = [C_AMBER if i == max_idx else C_NAVY for i in indices]
            ax.barh([readable[i] for i in indices], importances[indices],
                    color=bar_colors, edgecolor='white', linewidth=0.8)
            ax.invert_yaxis()
            for i_b, idx in enumerate(indices):
                ax.text(importances[idx] + 0.002, i_b,
                        f"{importances[idx]:.3f}",
                        va='center', fontsize=8.5, fontweight='bold', color=C_NAVY)
            apply_chart_style(ax, title=f"Top {n_show} Features by Importance Score",
                              xlabel="Importance Score", ylabel="")
            fig.patch.set_facecolor("#f7f3ec")
            plt.tight_layout(pad=2)
            st.pyplot(fig, use_container_width=True); plt.close()

        st.markdown("<hr class='gold-rule'>", unsafe_allow_html=True)
        st.markdown("<div class='sub-title'>Interpreting the Metrics</div>", unsafe_allow_html=True)
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("""
            **Accuracy**
            The percentage of test-set students whose recommended subject was correctly predicted.
            Random chance for 5 classes is 20%, so anything significantly above that is meaningful.
            """)
        with c2:
            st.markdown("""
            **Weighted F1 Score**
            Balances precision (correct positive predictions) and recall (coverage of true positives),
            weighted by each class's frequency. A better overall measure when classes differ in size.
            """)

    # ════════════════════════════════════════════════════════
    # PAGE 6 · GET RECOMMENDATION
    # ════════════════════════════════════════════════════════
    else:
        st.markdown("<div class='section-title'>Get Your Subject Recommendation</div>",
                    unsafe_allow_html=True)
        st.markdown(
            "Rate your abilities honestly. There are no right or wrong answers — "
            "the more accurate your self-assessment, the better the recommendation."
        )
        st.markdown("<hr class='gold-rule'>", unsafe_allow_html=True)

        col1, col2 = st.columns(2, gap="large")
        with col1:
            st.markdown("<div class='sub-title'>Cognitive Skills  (1 = Low · 4 = High)</div>",
                        unsafe_allow_html=True)
            skill1 = st.slider("Logical Reasoning",   1, 4, 2,
                                help="Puzzles, sequential thinking, if-then logic")
            skill2 = st.slider("Analytical Thinking", 1, 4, 2,
                                help="Pattern recognition, breaking problems into parts")
            skill3 = st.slider("Problem Solving",     1, 4, 2,
                                help="Finding solutions to novel, complex challenges")
            skill4 = st.slider("Creative Thinking",   1, 4, 2,
                                help="Generating original ideas, lateral thinking")

        with col2:
            st.markdown("<div class='sub-title'>Subject Knowledge  (1 = Beginner · 5 = Expert)</div>",
                        unsafe_allow_html=True)
            subj1 = st.slider("Mathematics", 1, 5, 3)
            subj2 = st.slider("Science",     1, 5, 3)
            subj3 = st.slider("Technology",  1, 5, 3)
            subj4 = st.slider("Arts",        1, 5, 3)
            subj5 = st.slider("Business",    1, 5, 3)

        st.markdown("<br>", unsafe_allow_html=True)

        if st.button("Generate Recommendation", type="primary"):
            try:
                features = np.array([
                    skill1, skill2, skill3, skill4,
                    subj1, subj2, subj3, subj4, subj5,
                    (skill1+skill2+skill3+skill4)/4,
                    skill1+skill2+skill3+skill4,
                    (subj1+subj2+subj3+subj4+subj5)/5,
                    subj1+subj2+subj3+subj4+subj5,
                    max(skill1,skill2,skill3,skill4) - min(skill1,skill2,skill3,skill4),
                    ((skill1+skill2+skill3+skill4)/4)*0.6 + ((subj1+subj2+subj3+subj4+subj5)/5)*0.4
                ]).reshape(1, -1)

                fs        = bundle['scaler'].transform(features)
                pred      = bundle['model'].predict(fs)[0]
                probs     = bundle['model'].predict_proba(fs)[0]
                pred_subj = bundle['label_encoder'].inverse_transform([pred])[0]
                conf      = probs[pred] * 100

                st.markdown(f"""
                <div class='result-card'>
                    <p class='rc-kicker'>Your Recommended Subject</p>
                    <h1>{pred_subj}</h1>
                    <p class='rc-sub'>{SUBJECT_NAMES.get(pred_subj, '')}</p>
                    <p class='rc-conf'>Model Confidence: <strong>{conf:.1f}%</strong></p>
                </div>
                """, unsafe_allow_html=True)

                st.markdown("<hr class='gold-rule'>", unsafe_allow_html=True)
                st.markdown("<div class='sub-title'>All Subject Probabilities</div>",
                            unsafe_allow_html=True)

                prob_df = pd.DataFrame({
                    'Subject': bundle['label_encoder'].classes_,
                    'Probability (%)': probs * 100
                }).sort_values('Probability (%)', ascending=True)

                fig, ax = plt.subplots(figsize=(9, 4.5))
                bar_cols = [C_AMBER if s == pred_subj else C_NAVY for s in prob_df['Subject']]
                ax.barh(prob_df['Subject'], prob_df['Probability (%)'],
                        color=bar_cols, edgecolor='white', linewidth=0.8)
                for _, row in prob_df.iterrows():
                    ax.text(row['Probability (%)'] + 0.4,
                            list(prob_df['Subject']).index(row['Subject']),
                            f"{row['Probability (%)']:.1f}%",
                            va='center', fontsize=9, fontweight='bold', color=C_NAVY)
                apply_chart_style(ax, title="Recommendation Probability Breakdown",
                                  xlabel="Probability (%)", ylabel="")
                ax.set_xlim(0, 108)
                fig.patch.set_facecolor("#f7f3ec")
                plt.tight_layout(pad=2)
                st.pyplot(fig, use_container_width=True); plt.close()

                st.markdown("<hr class='gold-rule'>", unsafe_allow_html=True)
                st.markdown("<div class='sub-title'>Your Cognitive Profile</div>",
                            unsafe_allow_html=True)

                sk_d = {'Logical Reasoning': skill1, 'Analytical Thinking': skill2,
                        'Problem Solving': skill3,   'Creative Thinking': skill4}
                su_d = {'Mathematics': subj1, 'Science': subj2,
                        'Technology': subj3,  'Arts': subj4, 'Business': subj5}
                top_sk = sorted(sk_d.items(), key=lambda x: x[1], reverse=True)
                top_su = sorted(su_d.items(), key=lambda x: x[1], reverse=True)

                c1, c2 = st.columns(2)
                with c1:
                    st.markdown("**Skill Ranking**")
                    for skill, rating in top_sk:
                        filled = "★" * rating + "☆" * (4 - rating)
                        st.markdown(
                            f"<div class='str-row'><span>{skill}</span>"
                            f"<span class='str-pill'>{filled} {rating}/4</span></div>",
                            unsafe_allow_html=True)
                with c2:
                    st.markdown("**Subject Ranking**")
                    for subj, rating in top_su:
                        filled = "★" * rating + "☆" * (5 - rating)
                        st.markdown(
                            f"<div class='str-row'><span>{subj}</span>"
                            f"<span class='str-pill'>{filled} {rating}/5</span></div>",
                            unsafe_allow_html=True)

                st.markdown("<br>", unsafe_allow_html=True)
                st.info(
                    f"**Why {pred_subj}?**  Your strongest cognitive skill is "
                    f"*{top_sk[0][0]}* ({top_sk[0][1]}/4) and your highest subject "
                    f"knowledge is in *{top_su[0][0]}* ({top_su[0][1]}/5). "
                    f"The expert mapping associates this profile most closely with **{pred_subj}**."
                )

            except Exception as e:
                st.error(f"Prediction error: {e}")
                st.info("Ensure the model is fully trained before running predictions.")


if __name__ == "__main__":
    main()
