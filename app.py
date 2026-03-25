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
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
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
    layout="wide",
    initial_sidebar_state="expanded"
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

/* ── Chart label strip ── */
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

/* ════════════════════════════════════════════════════
   SIDEBAR — FIXED
   ════════════════════════════════════════════════════ */

/* Main sidebar container */
[data-testid="stSidebar"] {
    background: var(--navy) !important;
    border-right: 6px solid var(--gold) !important;
    min-width: 280px !important;
    display: block !important;
    visibility: visible !important;
    opacity: 1 !important;
}
[data-testid="stSidebar"] > div {
    background: var(--navy) !important;
}

/* All text inside sidebar */
[data-testid="stSidebar"] * {
    color: #e8e2d4 !important;
    font-family: "Times New Roman", "Georgia", Times, serif !important;
}

/* Sidebar headings */
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3 {
    font-size: 0.75rem !important;
    letter-spacing: 0.18em !important;
    text-transform: uppercase !important;
    color: var(--gold-lt) !important;
    border-bottom: 1px solid #3a3020 !important;
    padding-bottom: 0.4rem !important;
    margin-bottom: 0.9rem !important;
}

/* Sidebar markdown text */
[data-testid="stSidebar"] p,
[data-testid="stSidebar"] span,
[data-testid="stSidebar"] label {
    color: #e8e2d4 !important;
}

/* Sidebar divider */
[data-testid="stSidebar"] hr {
    border-color: #3a3020 !important;
}

/* Radio labels */
[data-testid="stSidebar"] .stRadio label {
    font-size: 0.93rem !important;
    font-weight: 500 !important;
    color: #e8e2d4 !important;
}

/* Radio option hover */
[data-testid="stSidebar"] [data-testid="stRadio"] div label {
    padding: 0.5rem 0.8rem !important;
    border-radius: 0 !important;
    transition: all 0.2s ease !important;
}
[data-testid="stSidebar"] [data-testid="stRadio"] div label:hover {
    background: rgba(201, 132, 26, 0.15) !important;
    border-left: 3px solid var(--gold) !important;
}

/* ── Sidebar COLLAPSE button (the ❮ arrow on the sidebar edge) ── */
/* This targets the button that appears ON the sidebar to collapse it */
[data-testid="stSidebar"] [data-testid="stBaseButton-header"],
[data-testid="stSidebar"] button[data-testid="stBaseButton-header"] {
    background: rgba(201, 132, 26, 0.2) !important;
    border: 1px solid var(--gold) !important;
    border-radius: 50% !important;
    color: var(--gold-lt) !important;
}
[data-testid="stSidebar"] [data-testid="stBaseButton-header"]:hover,
[data-testid="stSidebar"] button[data-testid="stBaseButton-header"]:hover {
    background: var(--gold) !important;
}
[data-testid="stSidebar"] [data-testid="stBaseButton-header"] svg,
[data-testid="stSidebar"] button[data-testid="stBaseButton-header"] svg {
    fill: var(--gold-lt) !important;
    stroke: var(--gold-lt) !important;
}

/* ── EXPAND button (the ▶ arrow shown when sidebar is collapsed) ── */
[data-testid="stBaseButton-headerNoPadding"],
button[data-testid="stBaseButton-headerNoPadding"] {
    background: var(--navy) !important;
    border: 2px solid var(--gold) !important;
    border-radius: 0 !important;
    color: var(--gold-lt) !important;
    width: 2rem !important;
    height: 2.5rem !important;
}
[data-testid="stBaseButton-headerNoPadding"]:hover,
button[data-testid="stBaseButton-headerNoPadding"]:hover {
    background: var(--gold) !important;
    border-color: var(--gold-lt) !important;
}
[data-testid="stBaseButton-headerNoPadding"] svg,
button[data-testid="stBaseButton-headerNoPadding"] svg {
    fill: var(--gold-lt) !important;
    stroke: var(--gold-lt) !important;
    color: var(--gold-lt) !important;
}

/* Older Streamlit versions use collapsedControl */
[data-testid="collapsedControl"] {
    background: var(--navy) !important;
    border: 2px solid var(--gold) !important;
    border-radius: 0 !important;
    color: var(--gold-lt) !important;
}
[data-testid="collapsedControl"]:hover {
    background: var(--gold) !important;
}
[data-testid="collapsedControl"] svg {
    fill: var(--gold-lt) !important;
    stroke: var(--gold-lt) !important;
}

/* ── Sidebar model badge ── */
.model-badge {
    background: var(--gold);
    color: var(--ink);
    padding: 1rem;
    text-align: center;
    margin-top: 0.8rem;
    border-radius: 0;
    box-shadow: none;
    border: none;
}
.model-badge .badge-acc {
    font-size: 2rem;
    font-weight: 900;
    display: block;
    line-height: 1.1;
    color: var(--ink);
}
.model-badge .badge-lbl {
    font-size: 0.68rem;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    display: block;
    margin-top: 0.15rem;
    color: var(--ink);
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

/* ── Explanation expander ── */
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
SKILL_COLS   = ["Skill1", "Skill2", "Skill3", "Skill4", "Skill5", "Skill6"]
SUBJECT_COLS = ["Subject1", "Subject2", "Subject3", "Subject4", "Subject5", 
                "Subject6", "Subject7", "Subject8"]

SKILL_NAMES = {
    "Skill1": "Logical Reasoning",
    "Skill2": "Analytical Thinking",
    "Skill3": "Problem Solving",
    "Skill4": "Creative Thinking",
    "Skill5": "Communication Skills",
    "Skill6": "Practical Application"
}

SUBJECT_NAMES = {
    "Subject1": "Mathematics",
    "Subject2": "Science",
    "Subject3": "Technology",
    "Subject4": "Arts",
    "Subject5": "Business",
    "Subject6": "Engineering",
    "Subject7": "Social Sciences",
    "Subject8": "Healthcare"
}

BASE = os.path.dirname(os.path.abspath(__file__))

# ── Chart color palette ────────────────────────────────────────────────────────
C_NAVY    = "#112244"
C_CRIMSON = "#9b1d2a"
C_TEAL    = "#1a6b72"
C_AMBER   = "#c9841a"
C_SAGE    = "#2d6a4f"
C_VIOLET  = "#5a3472"
C_PAPER   = "#ffffff"

CHART_PALETTE = [C_NAVY, C_CRIMSON, C_TEAL, C_AMBER, C_SAGE, C_VIOLET]

PRIMARY = "#2563eb"
SECONDARY = "#7c3aed"
ACCENT = "#f59e0b"
SUCCESS = "#059669"
WARNING = "#f97316"
DANGER = "#dc2626"
INFO = "#0891b2"
CHART_COLORS = [PRIMARY, SECONDARY, ACCENT, SUCCESS, WARNING, INFO, DANGER]

def apply_chart_style(ax, title="", xlabel="", ylabel=""):
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

# ── Data generation ──────────────────────────────────────────────────────────
def generate_sample_data():
    expert_path = os.path.join(BASE, "ExpertSkillMap.csv")
    if not os.path.exists(expert_path):
        expert_data = {
            "Subject": ["Mathematics", "Science", "Technology", "Arts", "Business", 
                       "Engineering", "Social Sciences", "Healthcare"],
            "Skill1": [4, 3, 4, 2, 3, 4, 3, 3],
            "Skill2": [4, 4, 3, 2, 3, 4, 3, 4],
            "Skill3": [3, 3, 4, 3, 4, 4, 3, 3],
            "Skill4": [2, 3, 3, 4, 3, 3, 3, 2],
            "Skill5": [2, 3, 3, 4, 4, 3, 4, 4],
            "Skill6": [3, 4, 4, 3, 3, 4, 2, 4]
        }
        pd.DataFrame(expert_data).to_csv(expert_path, index=False)

    cog_path = os.path.join(BASE, "StudentCognitiveSkill.csv")
    if not os.path.exists(cog_path):
        np.random.seed(42)
        n_students = 2000
        students = []
        for i in range(1, n_students + 1):
            student_type = np.random.choice([
                'analytical', 'creative', 'balanced', 'technical', 'practical',
                'communicator', 'researcher', 'hands_on', 'struggling', 'gifted'
            ], p=[0.12, 0.10, 0.20, 0.12, 0.10, 0.10, 0.08, 0.08, 0.06, 0.04])
            self_assessment_bias = np.random.choice([-1, 0, 1], p=[0.15, 0.70, 0.15])
            if student_type == 'analytical':
                skill1 = np.clip(np.random.choice([2, 3, 4], p=[0.2, 0.4, 0.4]) + self_assessment_bias, 1, 4)
                skill2 = np.clip(np.random.choice([2, 3, 4], p=[0.1, 0.4, 0.5]) + self_assessment_bias, 1, 4)
                skill3 = np.clip(np.random.choice([2, 3, 4], p=[0.3, 0.4, 0.3]) + self_assessment_bias, 1, 4)
                skill4 = np.clip(np.random.choice([1, 2, 3], p=[0.3, 0.5, 0.2]) + self_assessment_bias, 1, 4)
                skill5 = np.clip(np.random.choice([1, 2, 3], p=[0.2, 0.6, 0.2]) + self_assessment_bias, 1, 4)
                skill6 = np.clip(np.random.choice([2, 3], p=[0.6, 0.4]) + self_assessment_bias, 1, 4)
            elif student_type == 'creative':
                skill1 = np.clip(np.random.choice([1, 2, 3], p=[0.2, 0.6, 0.2]) + self_assessment_bias, 1, 4)
                skill2 = np.clip(np.random.choice([2, 3], p=[0.6, 0.4]) + self_assessment_bias, 1, 4)
                skill3 = np.clip(np.random.choice([2, 3, 4], p=[0.3, 0.5, 0.2]) + self_assessment_bias, 1, 4)
                skill4 = np.clip(np.random.choice([3, 4], p=[0.4, 0.6]) + self_assessment_bias, 1, 4)
                skill5 = np.clip(np.random.choice([2, 3, 4], p=[0.2, 0.5, 0.3]) + self_assessment_bias, 1, 4)
                skill6 = np.clip(np.random.choice([2, 3], p=[0.5, 0.5]) + self_assessment_bias, 1, 4)
            elif student_type == 'technical':
                skill1 = np.clip(np.random.choice([2, 3, 4], p=[0.3, 0.5, 0.2]) + self_assessment_bias, 1, 4)
                skill2 = np.clip(np.random.choice([2, 3, 4], p=[0.3, 0.5, 0.2]) + self_assessment_bias, 1, 4)
                skill3 = np.clip(np.random.choice([3, 4], p=[0.5, 0.5]) + self_assessment_bias, 1, 4)
                skill4 = np.clip(np.random.choice([1, 2, 3], p=[0.3, 0.5, 0.2]) + self_assessment_bias, 1, 4)
                skill5 = np.clip(np.random.choice([1, 2, 3], p=[0.3, 0.5, 0.2]) + self_assessment_bias, 1, 4)
                skill6 = np.clip(np.random.choice([3, 4], p=[0.6, 0.4]) + self_assessment_bias, 1, 4)
            elif student_type == 'struggling':
                skill1 = np.clip(np.random.choice([1, 2], p=[0.6, 0.4]) + self_assessment_bias, 1, 4)
                skill2 = np.clip(np.random.choice([1, 2], p=[0.5, 0.5]) + self_assessment_bias, 1, 4)
                skill3 = np.clip(np.random.choice([1, 2, 3], p=[0.4, 0.4, 0.2]) + self_assessment_bias, 1, 4)
                skill4 = np.clip(np.random.choice([1, 2], p=[0.5, 0.5]) + self_assessment_bias, 1, 4)
                skill5 = np.clip(np.random.choice([1, 2], p=[0.6, 0.4]) + self_assessment_bias, 1, 4)
                skill6 = np.clip(np.random.choice([1, 2], p=[0.5, 0.5]) + self_assessment_bias, 1, 4)
            elif student_type == 'gifted':
                skill1 = np.clip(np.random.choice([3, 4], p=[0.3, 0.7]) + self_assessment_bias, 1, 4)
                skill2 = np.clip(np.random.choice([3, 4], p=[0.3, 0.7]) + self_assessment_bias, 1, 4)
                skill3 = np.clip(np.random.choice([3, 4], p=[0.4, 0.6]) + self_assessment_bias, 1, 4)
                skill4 = np.clip(np.random.choice([3, 4], p=[0.5, 0.5]) + self_assessment_bias, 1, 4)
                skill5 = np.clip(np.random.choice([3, 4], p=[0.4, 0.6]) + self_assessment_bias, 1, 4)
                skill6 = np.clip(np.random.choice([3, 4], p=[0.5, 0.5]) + self_assessment_bias, 1, 4)
            else:
                skill1 = np.clip(np.random.choice([1, 2, 3, 4], p=[0.1, 0.4, 0.4, 0.1]) + self_assessment_bias, 1, 4)
                skill2 = np.clip(np.random.choice([1, 2, 3, 4], p=[0.1, 0.4, 0.4, 0.1]) + self_assessment_bias, 1, 4)
                skill3 = np.clip(np.random.choice([1, 2, 3, 4], p=[0.1, 0.3, 0.5, 0.1]) + self_assessment_bias, 1, 4)
                skill4 = np.clip(np.random.choice([1, 2, 3, 4], p=[0.1, 0.4, 0.4, 0.1]) + self_assessment_bias, 1, 4)
                skill5 = np.clip(np.random.choice([1, 2, 3, 4], p=[0.1, 0.4, 0.4, 0.1]) + self_assessment_bias, 1, 4)
                skill6 = np.clip(np.random.choice([1, 2, 3, 4], p=[0.1, 0.4, 0.4, 0.1]) + self_assessment_bias, 1, 4)
            students.append({
                "Student": f"S{i}",
                "Skill1": int(skill1), "Skill2": int(skill2),
                "Skill3": int(skill3), "Skill4": int(skill4),
                "Skill5": int(skill5), "Skill6": int(skill6)
            })
        pd.DataFrame(students).to_csv(cog_path, index=False)

    sub_path = os.path.join(BASE, "StudentSubjectKnowledge.csv")
    if not os.path.exists(sub_path):
        np.random.seed(42)
        cog_df = pd.read_csv(cog_path)
        subjects = []
        for idx, row in cog_df.iterrows():
            noise_factor = np.random.uniform(0.5, 1.2)
            math       = int(np.clip((row['Skill1'] + row['Skill2']) / 2 * noise_factor + np.random.normal(0, 1.0), 1, 5))
            science    = int(np.clip((row['Skill2'] + row['Skill3']) / 2 * noise_factor + np.random.normal(0, 1.0), 1, 5))
            tech       = int(np.clip((row['Skill3'] + row['Skill1']) / 2 * noise_factor + np.random.normal(0, 1.0), 1, 5))
            arts       = int(np.clip((row['Skill4'] + row['Skill5']) / 2 * noise_factor + np.random.normal(0, 1.2), 1, 5))
            business   = int(np.clip((row['Skill5'] + row['Skill3']) / 2 * noise_factor + np.random.normal(0, 1.0), 1, 5))
            engineering= int(np.clip((row['Skill3'] + row['Skill6']) / 2 * noise_factor + np.random.normal(0, 1.0), 1, 5))
            social     = int(np.clip((row['Skill5'] + row['Skill2']) / 2 * noise_factor + np.random.normal(0, 1.2), 1, 5))
            healthcare = int(np.clip((row['Skill6'] + row['Skill2'] + row['Skill5']) / 3 * noise_factor + np.random.normal(0, 1.0), 1, 5))
            subjects.append({
                "Student": row['Student'],
                "Subject1": math,    "Subject2": science,
                "Subject3": tech,    "Subject4": arts,
                "Subject5": business,"Subject6": engineering,
                "Subject7": social,  "Subject8": healthcare
            })
        pd.DataFrame(subjects).to_csv(sub_path, index=False)

@st.cache_data
def load_data():
    generate_sample_data()
    expert    = pd.read_csv(os.path.join(BASE, "ExpertSkillMap.csv"))
    cognitive = pd.read_csv(os.path.join(BASE, "StudentCognitiveSkill.csv"))
    subject   = pd.read_csv(os.path.join(BASE, "StudentSubjectKnowledge.csv"))
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
    elif 'avg_skill'   in df.columns: df['overall_score'] = df['avg_skill'].round(2)
    elif 'avg_subject' in df.columns: df['overall_score'] = df['avg_subject'].round(2)
    return df

def build_target(df, expert):
    targets, sk = [], [c for c in SKILL_COLS if c in df.columns]
    np.random.seed(42)
    for _, row in df.iterrows():
        sv = np.array([row[c] for c in sk])
        subject_scores = []
        for _, er in expert.iterrows():
            sc = np.sum(np.abs(sv - np.array([er[c] for c in sk])))
            subject_scores.append((er['Subject'], sc))
        subject_scores.sort(key=lambda x: x[1])
        rand_val = np.random.random()
        if rand_val < 0.15 and len(subject_scores) > 1:
            best = subject_scores[1][0] if rand_val < 0.10 else (
                subject_scores[2][0] if len(subject_scores) > 2 else subject_scores[1][0])
        else:
            best = subject_scores[0][0]
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
    le = LabelEncoder()
    y_enc = le.fit_transform(y)
    X_tr, X_te, y_tr, y_te = train_test_split(X, y_enc, test_size=0.2, random_state=42, stratify=y_enc)
    sc = StandardScaler()
    Xtr = sc.fit_transform(X_tr)
    Xte = sc.transform(X_te)
    models = {
        'Random Forest': RandomForestClassifier(n_estimators=100, max_depth=10, min_samples_split=10, min_samples_leaf=5, random_state=42),
        'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, learning_rate=0.05, max_depth=4, min_samples_split=10, min_samples_leaf=5, random_state=42),
        'Logistic Regression': LogisticRegression(max_iter=2000, C=0.5, solver='lbfgs', random_state=42),
        'Decision Tree': DecisionTreeClassifier(max_depth=8, min_samples_split=15, min_samples_leaf=6, criterion='gini', random_state=42),
        'KNN': KNeighborsClassifier(n_neighbors=9, weights='distance', metric='minkowski'),
        'SVM': SVC(kernel='rbf', C=5.0, gamma='scale', probability=True, random_state=42)
    }
    results = {}
    best_m, best_sc_val, best_nm = None, 0, ""
    for nm, m in models.items():
        m.fit(Xtr, y_tr)
        yp = m.predict(Xte)
        acc = accuracy_score(y_te, yp)
        f1  = f1_score(y_te, yp, average='weighted')
        cv_scores = cross_val_score(m, Xtr, y_tr, cv=5, scoring='accuracy')
        results[nm] = {'model': m, 'accuracy': acc, 'f1_score': f1,
                       'cv_mean': cv_scores.mean(), 'cv_std': cv_scores.std()}
        if acc > best_sc_val:
            best_sc_val = acc; best_m = m; best_nm = nm
    bundle = {'model': best_m, 'model_name': best_nm, 'scaler': sc,
              'label_encoder': le, 'feature_cols': feat_cols, 'classes': le.classes_}
    return bundle, df, results

# ── EDA Plot Functions ──────────────────────────────────────────────────────

def plot_skill_distribution(df):
    available_skills = [col for col in SKILL_COLS if col in df.columns]
    if not available_skills: return None
    fig, axes = plt.subplots(2, 3, figsize=(18, 11))
    axes = axes.flatten()
    skill_labels = {1: 'Weak', 2: 'Developing', 3: 'Proficient', 4: 'Expert'}
    for i, skill in enumerate(available_skills):
        counts = df[skill].value_counts().sort_index()
        percentages = (counts / len(df) * 100).round(1)
        bars = axes[i].bar(counts.index, counts.values, color=CHART_COLORS[i % len(CHART_COLORS)],
                           edgecolor='white', linewidth=3, width=0.6, alpha=0.9)
        for bar, val, pct in zip(bars, counts.values, percentages.values):
            height = bar.get_height()
            axes[i].text(bar.get_x() + bar.get_width()/2, height + max(counts.values)*0.04,
                         f'{val} students\n({pct}%)', ha='center', va='bottom',
                         fontsize=11, color='#0f172a', fontweight='700', linespacing=1.1)
        axes[i].set_title(f"📊 {SKILL_NAMES.get(skill, skill)}", fontsize=15, fontweight='700', color='#0f172a', pad=18)
        axes[i].set_xlabel('Skill Level', fontsize=12, fontweight='600', color='#475569', labelpad=10)
        axes[i].set_ylabel('Number of Students', fontsize=12, fontweight='600', color='#475569', labelpad=10)
        axes[i].set_xticks([1, 2, 3, 4])
        axes[i].set_xticklabels([f'{skill_labels[j]}\n({j})' for j in [1,2,3,4]], fontsize=11, color='#475569', fontweight='500')
        axes[i].set_ylim(0, max(counts.values) * 1.25)
        axes[i].grid(axis='y', alpha=0.25, linestyle='--', linewidth=1.2, color='#cbd5e1')
        axes[i].set_axisbelow(True)
        axes[i].spines['top'].set_visible(False); axes[i].spines['right'].set_visible(False)
        axes[i].spines['left'].set_linewidth(1.5); axes[i].spines['bottom'].set_linewidth(1.5)
        axes[i].spines['left'].set_color('#cbd5e1'); axes[i].spines['bottom'].set_color('#cbd5e1')
        axes[i].set_facecolor('#f8fafc'); axes[i].tick_params(colors='#475569', width=1.5)
    fig.patch.set_facecolor('#f8fafc')
    fig.suptitle("Student Cognitive Skill Ratings Distribution", fontsize=19, fontweight='700', color='#0f172a', y=0.995)
    plt.tight_layout()
    return fig

def plot_subject_distribution(df):
    available_subjects = [col for col in SUBJECT_COLS if col in df.columns]
    if not available_subjects: return None
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    axes = axes.flatten()
    subject_labels = {1: 'Beginner', 2: 'Elementary', 3: 'Intermediate', 4: 'Advanced', 5: 'Expert'}
    for i, subj in enumerate(available_subjects):
        counts = df[subj].value_counts().sort_index()
        percentages = (counts / len(df) * 100).round(1)
        bars = axes[i].bar(counts.index, counts.values, color=CHART_COLORS[i % len(CHART_COLORS)],
                           edgecolor='white', linewidth=3, width=0.65, alpha=0.9)
        for bar, val, pct in zip(bars, counts.values, percentages.values):
            height = bar.get_height()
            axes[i].text(bar.get_x() + bar.get_width()/2, height + max(counts.values)*0.04,
                         f'{val}\n({pct}%)', ha='center', va='bottom', fontsize=10,
                         color='#0f172a', fontweight='700', linespacing=1.1)
        axes[i].set_title(f"📚 {SUBJECT_NAMES.get(subj, subj)}", fontsize=15, fontweight='700', color='#0f172a', pad=18)
        axes[i].set_xlabel('Knowledge Level', fontsize=12, fontweight='600', color='#475569', labelpad=10)
        axes[i].set_ylabel('Number of Students', fontsize=12, fontweight='600', color='#475569', labelpad=10)
        axes[i].set_xticks([1, 2, 3, 4, 5])
        axes[i].set_xticklabels([f'{subject_labels[j]}\n({j})' for j in [1,2,3,4,5]], fontsize=9.5, color='#475569', fontweight='500')
        axes[i].set_ylim(0, max(counts.values) * 1.25)
        axes[i].grid(axis='y', alpha=0.25, linestyle='--', linewidth=1.2, color='#cbd5e1')
        axes[i].set_axisbelow(True)
        axes[i].spines['top'].set_visible(False); axes[i].spines['right'].set_visible(False)
        axes[i].spines['left'].set_linewidth(1.5); axes[i].spines['bottom'].set_linewidth(1.5)
        axes[i].spines['left'].set_color('#cbd5e1'); axes[i].spines['bottom'].set_color('#cbd5e1')
        axes[i].set_facecolor('#f8fafc'); axes[i].tick_params(colors='#475569', width=1.5)
    fig.patch.set_facecolor('#f8fafc')
    fig.suptitle("Student Subject Knowledge Distribution", fontsize=19, fontweight='700', color='#0f172a', y=0.995)
    plt.tight_layout()
    return fig

def plot_target_distribution(df):
    if 'Target' not in df.columns: return None
    counts = df['Target'].value_counts().sort_values(ascending=False)
    percentages = (counts / len(df) * 100).round(1)
    fig = plt.figure(figsize=(16, 6))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.2, 1])
    ax1 = fig.add_subplot(gs[0]); ax2 = fig.add_subplot(gs[1])
    colors = CHART_COLORS[:len(counts)]
    bars = ax1.barh(counts.index, counts.values, color=colors, edgecolor='white', linewidth=3, alpha=0.9)
    for bar, val, pct in zip(bars, counts.values, percentages.values):
        width = bar.get_width()
        ax1.text(width + max(counts.values)*0.02, bar.get_y() + bar.get_height()/2,
                 f'{val} students ({pct}%)', va='center', ha='left', fontsize=12,
                 color='#0f172a', fontweight='700')
    ax1.set_title("🎯 Number of Students per Recommended Subject", fontsize=15, fontweight='700', color='#0f172a', pad=18)
    ax1.set_xlabel('Number of Students', fontsize=12, fontweight='600', color='#475569', labelpad=10)
    ax1.set_xlim(0, max(counts.values) * 1.3)
    ax1.grid(axis='x', alpha=0.25, linestyle='--', linewidth=1.2, color='#cbd5e1')
    ax1.set_axisbelow(True)
    ax1.spines['top'].set_visible(False); ax1.spines['right'].set_visible(False)
    ax1.spines['left'].set_linewidth(1.5); ax1.spines['bottom'].set_linewidth(1.5)
    ax1.spines['left'].set_color('#cbd5e1'); ax1.spines['bottom'].set_color('#cbd5e1')
    ax1.set_facecolor('#f8fafc'); ax1.tick_params(colors='#475569', labelsize=11, width=1.5)
    wedges, texts, autotexts = ax2.pie(counts.values, labels=counts.index, autopct='%1.1f%%',
                                        colors=colors, startangle=90,
                                        wedgeprops={'linewidth': 3, 'edgecolor': 'white', 'width': 0.6},
                                        textprops={'fontsize': 11, 'color': '#0f172a', 'fontweight': '600'},
                                        pctdistance=0.75)
    for autotext in autotexts:
        autotext.set_color('white'); autotext.set_fontsize(11); autotext.set_fontweight('700')
    ax2.set_title("📊 Percentage Distribution", fontsize=15, fontweight='700', color='#0f172a', pad=18)
    ax2.set_facecolor('#f8fafc')
    fig.patch.set_facecolor('#f8fafc')
    fig.suptitle("Recommended Subject Distribution", fontsize=19, fontweight='700', color='#0f172a', y=0.98)
    plt.tight_layout()
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
    fig.suptitle("Skill Profiles by Recommended Subject", fontsize=13, fontweight='bold', color=C_NAVY, fontfamily='serif', y=1.01)
    fig.patch.set_facecolor("#f7f3ec"); plt.tight_layout(pad=2)
    return fig

def plot_correlation_heatmap(df):
    avail = [c for c in SKILL_COLS + SUBJECT_COLS if c in df.columns]
    if len(avail) < 2: return None
    corr = df[avail].corr()
    readable = {c: SKILL_NAMES.get(c, SUBJECT_NAMES.get(c, c)) for c in avail}
    corr = corr.rename(index=readable, columns=readable)
    fig, ax = plt.subplots(figsize=(11, 7))
    cmap = mpl.colors.LinearSegmentedColormap.from_list("cr_teal", [C_CRIMSON, "#f7f3ec", C_TEAL], N=256)
    sns.heatmap(corr, ax=ax, annot=True, fmt=".2f", cmap=cmap, linewidths=0.6,
                linecolor="#e0d8cc", vmin=-1, vmax=1,
                annot_kws={"size": 9, "weight": "bold", "color": "#ffffff"},
                cbar_kws={"shrink": 0.75, "label": "Pearson r"})
    for text in ax.texts:
        val = float(text.get_text())
        text.set_color("#0a0e1a" if abs(val) < 0.35 else "#ffffff")
    ax.set_title("Correlation Matrix — Skills & Subject Knowledge", fontsize=12, fontweight='bold', color=C_NAVY, fontfamily='serif', pad=14)
    ax.set_facecolor(C_PAPER); fig.patch.set_facecolor("#f7f3ec")
    plt.xticks(rotation=30, ha='right', fontsize=9); plt.yticks(rotation=0, fontsize=9)
    plt.tight_layout(pad=2)
    return fig

def plot_avg_score_by_target(df):
    if 'Target' not in df.columns or 'overall_score' not in df.columns: return None
    avg = df.groupby('Target')['overall_score'].mean().sort_values(ascending=False)
    fig, ax = plt.subplots(figsize=(9, 4.5))
    colors = CHART_PALETTE[:len(avg)]
    bars = ax.bar(avg.index, avg.values, color=colors, edgecolor='#ffffff', linewidth=1.2, width=0.55)
    for b, v in zip(bars, avg.values):
        ax.text(b.get_x() + b.get_width()/2, b.get_height() + 0.02,
                f"{v:.2f}", ha='center', fontsize=9.5, fontweight='bold', color=C_NAVY)
    apply_chart_style(ax, title="Average Overall Score per Recommended Subject", xlabel="", ylabel="Mean Overall Score")
    ax.set_xticklabels(avg.index, rotation=22, ha='right', fontsize=9)
    fig.patch.set_facecolor("#f7f3ec"); plt.tight_layout(pad=2)
    return fig

def chart_section(label, fig, explanation_title, explanation_md, key_suffix):
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
            return

    # ── Sidebar ──
    st.sidebar.markdown("## 📍 Navigation")
    page = st.sidebar.radio("", [
        "🏠 Problem Statement",
        "📊 Dataset Overview",
        "📈 Exploratory Analysis",
        "🔧 Feature Engineering",
        "🎯 Model Performance",
        "🔮 Get Recommendation"
    ])
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 🏆 Champion Model")
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
    st.sidebar.markdown("---")
    st.sidebar.markdown(f"""
    <div style='text-align:center;padding:1rem 0.5rem;font-size:0.85rem;color:#e8e2d4;'>
        <p style='margin:0.3rem 0;'>📚 <b>{len(df)}</b> Students</p>
        <p style='margin:0.3rem 0;'>🎓 <b>8</b> Subjects</p>
        <p style='margin:0.3rem 0;'>🧠 <b>6</b> Skills</p>
        <p style='margin:0.3rem 0;'>🤖 <b>6</b> ML Models</p>
    </div>
    """, unsafe_allow_html=True)

    # ════════════════════════════════════════════════════════
    if page == "🏠 Problem Statement":
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
- **Cognitive Skills** — six measurable intellectual traits rated 1–4
- **Subject Knowledge** — proficiency across eight academic domains rated 1–5

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

    elif page == "📊 Dataset Overview":
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
        with tab2:
            if su:
                d = df[['Student'] + su].head(12).copy()
                d.columns = ['Student'] + [SUBJECT_NAMES[c] for c in su]
                st.dataframe(d, use_container_width=True)
                st.caption("First 12 records · 1 = Beginner, 5 = Expert")
        with tab3:
            if 'Target' in df.columns:
                cts = df['Target'].value_counts().reset_index()
                cts.columns = ['Recommended Subject', 'Count']
                cts['Share (%)'] = (cts['Count'] / len(df) * 100).round(1)
                st.dataframe(cts, use_container_width=True)

    elif page == "📈 Exploratory Analysis":
        st.markdown("<div class='section-title'>Exploratory Data Analysis</div>", unsafe_allow_html=True)
        st.markdown("All charts are displayed below. Click **📖 Read explanation** under any chart to understand what it shows.")
        st.markdown("<hr class='gold-rule'>", unsafe_allow_html=True)
        chart_section("① Cognitive Skill Rating Distributions", plot_skill_distribution(df),
            "What this chart shows & how to read it", """
**What it shows:** How many students received each rating (1–4) across the six cognitive skill dimensions.

**How to read it:** Each bar represents a rating level. A taller bar means more students share that rating.

**Why it matters:** Knowing the skill distribution helps check whether the training data is representative.
            """, "skill_dist")
        chart_section("② Subject Knowledge Distributions", plot_subject_distribution(df),
            "What this chart shows & how to read it", """
**What it shows:** How student knowledge is spread across eight subject domains on a 1–5 scale.

**How to read it:** A bar peaking at rating 1 means most students are beginners. A peak at 4–5 indicates strong knowledge.
            """, "subj_dist")
        chart_section("③ Recommendation Label Distribution", plot_target_distribution(df),
            "What this chart shows & how to read it", """
**What it shows:** How recommended subjects are distributed across the student population.

**Why it matters for ML:** Severe class imbalance would bias the model toward over-recommending dominant classes.
            """, "target_dist")
        chart_section("④ Skill Profiles by Recommended Subject — Box Plots", plot_skill_by_target(df),
            "What this chart shows & how to read it", """
**What it shows:** Box plots of each skill rating grouped by recommended subject.

**How to read a box plot:** The central line = median. Box edges = 25th/75th percentiles. Dots = outliers.
            """, "box_skill")
        chart_section("⑤ Correlation Matrix — Skills & Subject Knowledge", plot_correlation_heatmap(df),
            "What this chart shows & how to read it", """
**What it shows:** Pearson correlation coefficients between every pair of skill and subject columns.

**How to read the colours:** Deep crimson = strong negative. Near-white = no relationship. Deep teal = strong positive.
            """, "heatmap")
        chart_section("⑥ Average Overall Score by Recommended Subject", plot_avg_score_by_target(df),
            "What this chart shows & how to read it", """
**What it shows:** Mean overall score (60% avg skill + 40% avg subject knowledge) per recommended subject.
            """, "avg_score")
        st.markdown("<div class='chart-label'>⑦ Descriptive Statistics — All Numeric Columns</div>", unsafe_allow_html=True)
        avail = [c for c in SKILL_COLS + SUBJECT_COLS if c in df.columns]
        if 'overall_score' in df.columns: avail.append('overall_score')
        st.dataframe(df[avail].describe().round(3), use_container_width=True)
        st.markdown("<hr class='gold-rule'>", unsafe_allow_html=True)

    elif page == "🔧 Feature Engineering":
        st.markdown("<div class='section-title'>Feature Engineering</div>", unsafe_allow_html=True)
        st.markdown("Raw ratings are the foundation, but derived features expose deeper patterns and improve model accuracy.")
        st.markdown("<hr class='gold-rule'>", unsafe_allow_html=True)
        col1, col2 = st.columns(2, gap="large")
        with col1:
            st.markdown("<div class='sub-title'>Skill-Derived Features</div>", unsafe_allow_html=True)
            for name, desc in [
                ("Average Skill Score", "Mean of all six skill ratings. Captures overall cognitive level in one number."),
                ("Total Skill Score", "Sum of all six ratings. Sometimes preferred by tree-based models."),
                ("Strongest Skill", "The column with the highest rating — a student's primary cognitive edge."),
                ("Weakest Skill", "The column with the lowest rating — highlights where development is needed."),
                ("Skill Range", "Max minus min skill. Small range = balanced. Large range = highly specialised."),
            ]:
                st.markdown(f"""<div class='callout'><strong>{name}</strong><br><span style='font-style:normal;font-size:0.93rem;'>{desc}</span></div>""", unsafe_allow_html=True)
        with col2:
            st.markdown("<div class='sub-title'>Subject-Derived Features</div>", unsafe_allow_html=True)
            for name, desc in [
                ("Average Subject Score", "Mean knowledge across all eight domains."),
                ("Total Subject Score", "Sum of subject scores — a broad measure of academic exposure."),
                ("Best Subject", "Domain with the highest self-reported knowledge."),
                ("Worst Subject", "Domain most in need of development."),
            ]:
                st.markdown(f"""<div class='callout'><strong>{name}</strong><br><span style='font-style:normal;font-size:0.93rem;'>{desc}</span></div>""", unsafe_allow_html=True)
            st.markdown("<div class='sub-title'>Composite Feature</div>", unsafe_allow_html=True)
            st.markdown("""<div class='callout'><strong>Overall Score</strong><br><span style='font-style:normal;font-size:0.93rem;'>Weighted blend: <em>60% average skill + 40% average subject knowledge.</em></span></div>""", unsafe_allow_html=True)
        st.markdown("<hr class='gold-rule'>", unsafe_allow_html=True)
        st.markdown("<div class='sub-title'>Sample Profiles with Engineered Features</div>", unsafe_allow_html=True)
        show_cols = ['Student'] + [c for c in ['avg_skill','strongest_skill','skill_range','avg_subject','best_subject','overall_score','Target'] if c in df.columns]
        if len(show_cols) > 1:
            sample = df[show_cols].head(8).rename(columns={
                'avg_skill': 'Avg Skill', 'strongest_skill': 'Strongest Skill',
                'skill_range': 'Skill Range', 'avg_subject': 'Avg Subject',
                'best_subject': 'Best Subject', 'overall_score': 'Overall Score', 'Target': 'Recommended Subject'
            })
            st.dataframe(sample, use_container_width=True)

    elif page == "🎯 Model Performance":
        st.markdown("<div class='section-title'>Model Performance</div>", unsafe_allow_html=True)
        res_df = pd.DataFrame([
            {'Model': n, 'Test Accuracy (%)': f"{r['accuracy']*100:.1f}",
             'F1 Score (%)': f"{r['f1_score']*100:.1f}",
             'CV Accuracy (%)': f"{r['cv_mean']*100:.1f} ± {r['cv_std']*100:.1f}"}
            for n, r in results.items()
        ]).sort_values('Test Accuracy (%)', ascending=False).reset_index(drop=True)
        res_df.index += 1
        st.dataframe(res_df, use_container_width=True)
        best = results[bundle['model_name']]
        st.success(f"🏆 **Champion: {bundle['model_name']}** — Test Accuracy {best['accuracy']*100:.1f}% · F1 Score {best['f1_score']*100:.1f}% · CV Accuracy {best['cv_mean']*100:.1f}% (±{best['cv_std']*100:.1f}%)")
        st.info(f"📊 **Model trained on {len(df)} students** with {len(bundle['feature_cols'])} features. Cross-validation (5-fold) ensures generalization.")
        st.markdown("<hr class='gold-rule'>", unsafe_allow_html=True)
        if hasattr(bundle['model'], 'feature_importances_') and bundle['feature_cols']:
            st.markdown("<div class='sub-title'>Feature Importance</div>", unsafe_allow_html=True)
            importances = bundle['model'].feature_importances_
            feat_names  = bundle['feature_cols']
            readable = [SKILL_NAMES.get(f, SUBJECT_NAMES.get(f, f.replace('_', ' ').title())) for f in feat_names]
            n_show  = min(10, len(importances))
            indices = np.argsort(importances)[-n_show:]
            fig, ax = plt.subplots(figsize=(10, 5.5))
            max_idx = indices[-1]
            bar_colors = [C_AMBER if i == max_idx else C_NAVY for i in indices]
            ax.barh([readable[i] for i in indices], importances[indices], color=bar_colors, edgecolor='white', linewidth=0.8)
            ax.invert_yaxis()
            for i_b, idx in enumerate(indices):
                ax.text(importances[idx] + 0.002, i_b, f"{importances[idx]:.3f}", va='center', fontsize=8.5, fontweight='bold', color=C_NAVY)
            apply_chart_style(ax, title=f"Top {n_show} Features by Importance Score", xlabel="Importance Score", ylabel="")
            fig.patch.set_facecolor("#f7f3ec"); plt.tight_layout(pad=2)
            st.pyplot(fig, use_container_width=True); plt.close()
        st.markdown("<hr class='gold-rule'>", unsafe_allow_html=True)
        st.markdown("<div class='sub-title'>Understanding the Metrics</div>", unsafe_allow_html=True)
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown("**Test Accuracy**\n\nPercentage of correct predictions on held-out test data (20%). Random chance for 8 classes = 12.5%.")
        with col2:
            st.markdown("**F1 Score**\n\nHarmonic mean of precision and recall, weighted by class frequency. Better than accuracy for imbalanced datasets.")
        with col3:
            st.markdown("**CV Accuracy**\n\n5-fold cross-validation score with standard deviation. Shows model consistency across different data splits.")

    else:  # Get Recommendation
        st.markdown("<div class='section-title'>Get Your Subject Recommendation</div>", unsafe_allow_html=True)
        st.markdown("Rate your abilities honestly across 6 cognitive skills and 8 subject areas.")
        st.markdown("<hr class='gold-rule'>", unsafe_allow_html=True)
        col1, col2 = st.columns(2, gap="large")
        with col1:
            st.markdown("<div class='sub-title'>🧠 Cognitive Skills (1=Low, 4=High)</div>", unsafe_allow_html=True)
            skill1 = st.slider("💡 Logical Reasoning", 1, 4, 2, help="Puzzles, sequential thinking, if-then logic")
            skill2 = st.slider("🔍 Analytical Thinking", 1, 4, 2, help="Pattern recognition, breaking problems into parts")
            skill3 = st.slider("🎯 Problem Solving", 1, 4, 2, help="Finding solutions to novel, complex challenges")
            skill4 = st.slider("🎨 Creative Thinking", 1, 4, 2, help="Generating original ideas, lateral thinking")
            skill5 = st.slider("💬 Communication Skills", 1, 4, 2, help="Expressing ideas clearly, teamwork, presentation")
            skill6 = st.slider("🔧 Practical Application", 1, 4, 2, help="Hands-on work, applying theory to practice")
        with col2:
            st.markdown("<div class='sub-title'>📚 Subject Knowledge (1=Beginner, 5=Expert)</div>", unsafe_allow_html=True)
            subj1 = st.slider("📐 Mathematics", 1, 5, 3)
            subj2 = st.slider("🔬 Science", 1, 5, 3)
            subj3 = st.slider("💻 Technology", 1, 5, 3)
            subj4 = st.slider("🎭 Arts", 1, 5, 3)
            subj5 = st.slider("💼 Business", 1, 5, 3)
            subj6 = st.slider("⚙️ Engineering", 1, 5, 3)
            subj7 = st.slider("🌍 Social Sciences", 1, 5, 3)
            subj8 = st.slider("🏥 Healthcare", 1, 5, 3)
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("🚀 Generate Recommendation", type="primary"):
            try:
                all_skills   = [skill1, skill2, skill3, skill4, skill5, skill6]
                all_subjects = [subj1, subj2, subj3, subj4, subj5, subj6, subj7, subj8]
                features = np.array(
                    all_skills + all_subjects + [
                        np.mean(all_skills), np.sum(all_skills),
                        np.mean(all_subjects), np.sum(all_subjects),
                        np.max(all_skills) - np.min(all_skills),
                        np.mean(all_skills) * 0.6 + np.mean(all_subjects) * 0.4
                    ]
                ).reshape(1, -1)
                fs        = bundle['scaler'].transform(features)
                pred      = bundle['model'].predict(fs)[0]
                probs     = bundle['model'].predict_proba(fs)[0]
                pred_subj = bundle['label_encoder'].inverse_transform([pred])[0]
                conf      = probs[pred] * 100
                st.markdown(f"""
                <div class='result-card'>
                    <p class='rc-kicker'>Your Recommended Subject</p>
                    <h1>🎓 {pred_subj}</h1>
                    <p class='rc-sub'>Best match for your profile</p>
                    <p class='rc-conf'>Model Confidence: <strong>{conf:.1f}%</strong></p>
                </div>
                """, unsafe_allow_html=True)
                st.markdown("<hr class='gold-rule'>", unsafe_allow_html=True)
                st.markdown("<div class='sub-title'>📊 All Subject Probabilities</div>", unsafe_allow_html=True)
                prob_df = pd.DataFrame({'Subject': bundle['label_encoder'].classes_, 'Probability (%)': probs * 100}).sort_values('Probability (%)', ascending=True)
                fig, ax = plt.subplots(figsize=(9, 4.5))
                bar_cols = [C_AMBER if s == pred_subj else C_NAVY for s in prob_df['Subject']]
                ax.barh(prob_df['Subject'], prob_df['Probability (%)'], color=bar_cols, edgecolor='white', linewidth=0.8)
                for _, row in prob_df.iterrows():
                    ax.text(row['Probability (%)'] + 0.4, list(prob_df['Subject']).index(row['Subject']),
                            f"{row['Probability (%)']:.1f}%", va='center', fontsize=9, fontweight='bold', color=C_NAVY)
                apply_chart_style(ax, title="Recommendation Probability Breakdown", xlabel="Probability (%)", ylabel="")
                ax.set_xlim(0, 108); fig.patch.set_facecolor("#f7f3ec"); plt.tight_layout(pad=2)
                st.pyplot(fig, use_container_width=True); plt.close()
                st.markdown("<hr class='gold-rule'>", unsafe_allow_html=True)
                st.markdown("<div class='sub-title'>Your Cognitive Profile</div>", unsafe_allow_html=True)
                sk_d = {'Logical Reasoning': skill1, 'Analytical Thinking': skill2, 'Problem Solving': skill3,
                        'Creative Thinking': skill4, 'Communication': skill5, 'Practical Application': skill6}
                su_d = {'Mathematics': subj1, 'Science': subj2, 'Technology': subj3, 'Arts': subj4,
                        'Business': subj5, 'Engineering': subj6, 'Social Sciences': subj7, 'Healthcare': subj8}
                top_sk = sorted(sk_d.items(), key=lambda x: x[1], reverse=True)
                top_su = sorted(su_d.items(), key=lambda x: x[1], reverse=True)
                c1, c2 = st.columns(2)
                with c1:
                    st.markdown("**Skill Ranking**")
                    for skill, rating in top_sk:
                        filled = "★" * rating + "☆" * (4 - rating)
                        st.markdown(f"<div class='str-row'><span>{skill}</span><span class='str-pill'>{filled} {rating}/4</span></div>", unsafe_allow_html=True)
                with c2:
                    st.markdown("**Subject Ranking**")
                    for subj, rating in top_su:
                        filled = "★" * rating + "☆" * (5 - rating)
                        st.markdown(f"<div class='str-row'><span>{subj}</span><span class='str-pill'>{filled} {rating}/5</span></div>", unsafe_allow_html=True)
                st.markdown("<br>", unsafe_allow_html=True)
                st.info(f"**Why {pred_subj}?** Your strongest skill is *{top_sk[0][0]}* ({top_sk[0][1]}/4) and highest subject knowledge is *{top_su[0][0]}* ({top_su[0][1]}/5). The expert mapping associates this profile most closely with **{pred_subj}**.")
            except Exception as e:
                st.error(f"Prediction error: {e}")

if __name__ == "__main__":
    main()


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
