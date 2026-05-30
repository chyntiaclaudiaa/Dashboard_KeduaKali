import datetime
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.ticker as mticker
import matplotlib.gridspec as gridspec
import matplotlib.dates as mdates
from matplotlib.lines import Line2D
import seaborn as sns
import warnings
import requests
import time
warnings.filterwarnings('ignore')

AI_SERVICE_URL = "http://localhost:8000/predict"

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="KeduaKali Dashboard",
    page_icon="♻️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─────────────────────────────────────────────
# CUSTOM CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Sora:wght@600;700;800&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

.stApp { background: #F8F9FA; color: #1A1A2E; }

[data-testid="stSidebar"] {
    background: #FFFFFF !important;
    border-right: 1px solid #E0E4EC;
}

.brand-header {
    background: linear-gradient(135deg, #EBF5FB 0%, #FDFEFE 100%);
    border: 1px solid #AED6F1;
    border-radius: 16px;
    padding: 28px 32px;
    margin-bottom: 24px;
    position: relative;
    overflow: hidden;
}
.brand-header::before {
    content: '';
    position: absolute;
    top: -40px; right: -40px;
    width: 180px; height: 180px;
    background: radial-gradient(circle, rgba(52,152,219,0.12) 0%, transparent 70%);
    border-radius: 50%;
}
.brand-title {
    font-family: 'Sora', sans-serif;
    font-size: 2.2rem;
    font-weight: 800;
    color: #1A5276;
    margin: 0; padding: 0;
}
.brand-sub {
    font-size: 0.85rem;
    color: #5D6D7E;
    margin-top: 4px;
    font-weight: 400;
    letter-spacing: 0.3px;
}

.metric-card {
    background: #FFFFFF;
    border: 1px solid #D5E8F3;
    border-radius: 12px;
    padding: 20px 22px;
    text-align: center;
    transition: border-color 0.2s, box-shadow 0.2s;
    box-shadow: 0 2px 6px rgba(52,152,219,0.06);
}
.metric-card:hover {
    border-color: #2E86C1;
    box-shadow: 0 4px 14px rgba(52,152,219,0.12);
}
.metric-value {
    font-family: 'Sora', sans-serif;
    font-size: 1.2rem;
    font-weight: 700;
    color: #1A5276;
    line-height: 1.1;
}
.metric-label {
    font-size: 0.55rem;
    color: #7F8C8D;
    margin-top: 4px;
    text-transform: uppercase;
    letter-spacing: 0.8px;
    font-weight: 500;
}

.section-header {
    font-family: 'Sora', sans-serif;
    font-size: 1.1rem;
    font-weight: 700;
    color: #1A1A2E;
    border-left: 3px solid #2E86C1;
    padding-left: 12px;
    margin: 28px 0 16px 0;
}

.insight-box {
    background: #EBF5FB;
    border: 1px solid #AED6F1;
    border-left: 3px solid #2E86C1;
    border-radius: 10px;
    padding: 16px 20px;
    margin-top: 12px;
    font-size: 0.85rem;
    line-height: 1.7;
    color: #2C3E50;
}
.insight-box strong { color: #1A5276; }

.filter-info-box {
    background: #F0F4F8;
    border: 1px solid #CBD5E0;
    border-radius: 8px;
    padding: 10px 14px;
    font-size: 0.8rem;
    color: #4A5568;
    margin-bottom: 16px;
    font-style: italic;
}

[data-testid="stSidebar"] label,
[data-testid="stSidebar"] .stMultiSelect label,
[data-testid="stSidebar"] .stDateInput label,
[data-testid="stSidebar"] p,
[data-testid="stSidebar"] span,
[data-testid="stSidebar"] div {
    color: #1A1A2E !important;
}

[data-testid="stTab"] { font-family: 'Inter', sans-serif; font-weight: 600; color: #1A1A2E !important; }
button[data-baseweb="tab"] { color: #1A1A2E !important; }
button[data-baseweb="tab"] p { color: #1A1A2E !important; font-weight: 600; }
button[data-baseweb="tab"][aria-selected="true"] p { color: #1A5276 !important; font-weight: 700; }

[data-testid="stSidebar"] .stMultiSelect span[data-baseweb="tag"] {
    background-color: #D5E8F3 !important;
    color: #1A5276 !important;
}

.custom-divider { border: none; border-top: 1px solid #D5E8F3; margin: 28px 0; }
.badge-high { background: #FDEDEC; color: #C0392B; padding: 2px 10px; border-radius: 20px; font-size: 0.75rem; font-weight: 600; }
.badge-low  { background: #EAFAF1; color: #1E8449; padding: 2px 10px; border-radius: 20px; font-size: 0.75rem; font-weight: 600; }

/* Tab 2 styles */
.tab2-main-title { font-family: 'Sora', sans-serif !important; font-weight: 800; color: #1A5276; margin-bottom: 4px; }
div[data-testid="stForm"] label p { font-family: 'Inter', sans-serif !important; color: #0f172a !important; font-weight: 600 !important; font-size: 0.85rem !important; }
div[data-testid="stForm"] input, div[data-baseweb="select"] {
    font-family: 'Inter', sans-serif !important;
    border-radius: 8px !important;
    border: 1px solid #CBD5E1 !important;
    background-color: #FFFFFF !important;
    color: #1e293b !important;
    transition: all 0.2s ease-in-out;
}
div[data-baseweb="select"] > div { background-color: transparent !important; border: none !important; }
div[data-testid="stForm"] input:focus, div[data-baseweb="select"]:focus-within {
    border-color: #1A5276 !important;
    box-shadow: 0 0 0 1px #1A5276 !important;
}
.modern-header {
    font-family: 'Sora', sans-serif !important;
    font-size: 1.05rem;
    font-weight: 700;
    color: #1A1A2E;
    margin-top: 1.5rem;
    margin-bottom: 1rem;
    padding-bottom: 0.5rem;
    border-bottom: 1px solid #e2e8f0;
    border-left: 3px solid #2E86C1;
    padding-left: 12px;
}
div[data-testid="stToggle"] {
    background-color: #F1F5F9 !important;
    padding: 12px 16px !important;
    border-radius: 10px !important;
    border: 1px solid #E2E8F0 !important;
    margin-bottom: 10px !important;
    transition: all 0.2s ease;
}
div[data-testid="stToggle"]:hover { border-color: #CBD5E1 !important; background-color: #E2E8F0 !important; }
[data-testid="stFormSubmitButton"] button {
    font-family: 'Inter', sans-serif !important;
    background-color: #1A5276 !important;
    color: white !important;
    border-radius: 10px !important;
    font-weight: 600 !important;
    font-size: 1rem !important;
    padding: 0.6rem 1rem !important;
    border: none !important;
    transition: all 0.3s ease !important;
    box-shadow: 0 4px 6px -1px rgba(26, 82, 118, 0.2) !important;
}
[data-testid="stFormSubmitButton"] button:hover {
    background-color: #154360 !important;
    box-shadow: 0 6px 8px -1px rgba(26, 82, 118, 0.3) !important;
    transform: translateY(-1px);
}
.result-card {
    background: #ffffff;
    border: 1px solid #e2e8f0;
    border-radius: 12px;
    padding: 24px;
    box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05);
    height: 100%;
}
.result-title { font-family: 'Sora', sans-serif !important; color: #7F8C8D; font-size: 0.85rem; font-weight: 600; text-transform: uppercase; letter-spacing: 0.5px; margin-bottom: 8px; }
.result-value { font-family: 'Sora', sans-serif !important; color: #1A5276; font-size: 2.5rem; font-weight: 800; margin-bottom: 12px; line-height: 1.2; }
.badge-danger { font-family: 'Inter', sans-serif !important; background: #fef2f2; color: #ef4444; padding: 4px 10px; border-radius: 6px; font-size: 0.75rem; font-weight: 700; }
.badge-safe   { font-family: 'Inter', sans-serif !important; background: #f0fdf4; color: #22c55e; padding: 4px 10px; border-radius: 6px; font-size: 0.75rem; font-weight: 700; }
.badge-warn   { font-family: 'Inter', sans-serif !important; background: #fffbeb; color: #f59e0b; padding: 4px 10px; border-radius: 6px; font-size: 0.75rem; font-weight: 700; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# MATPLOTLIB LIGHT THEME
# ─────────────────────────────────────────────
plt.rcParams.update({
    'figure.facecolor': 'white',
    'axes.facecolor': 'white',
    'axes.edgecolor': '#BDC3C7',
    'axes.labelcolor': '#2C3E50',
    'xtick.color': '#5D6D7E',
    'ytick.color': '#5D6D7E',
    'text.color': '#2C3E50',
    'grid.color': '#ECF0F1',
    'grid.linewidth': 0.8,
    'legend.facecolor': 'white',
    'legend.edgecolor': '#BDC3C7',
    'font.family': 'DejaVu Sans',
    'axes.spines.top': False,
    'axes.spines.right': False,
})

# Color palette
C_HIGH      = '#D84315'
C_LOW       = '#0277BD'
C_NORMAL    = '#CFD8DC'
C_AVG_LINE  = '#546E7A'
C_RED       = '#E24B4A'
C_BLUE      = '#2980B9'
C_GREEN     = '#2ECC71'
C_ORANGE    = '#E67E22'
C_NAVY      = '#1D3557'
C_TEAL      = '#A8DADC'
C_LIGHTBLUE = '#D6EAF8'
COLOR_PEAK    = '#002347'
COLOR_NEUTRAL = '#3498DB'
COLOR_SURPLUS = '#D6EAF8'

# Label maps (LabelEncoder — alfabetis)
REST_TYPE_MAP = {0: 'Cafe', 1: 'Casual Dining', 2: 'Fine Dining', 3: 'Food Stall', 4: 'Kopitiam'}
MEAL_TYPE_MAP = {0: 'Breakfast', 1: 'Dinner', 2: 'Lunch'}
WEATHER_MAP   = {0: 'Cloudy', 1: 'Rainy', 2: 'Sunny'}

# ─────────────────────────────────────────────
# DATA LOADING
# ─────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def load_data():
    RESTAURANT_CSV     = 'restaurant_sales_cleaned.csv'
    FOOD_WASTAGE_CSV   = 'food_wastage_cleaned.csv'
    GLOBAL_WASTAGE_CSV = 'global_food_wastage_dataset.csv'

    rest = pd.read_csv(RESTAURANT_CSV)
    fw   = pd.read_csv(FOOD_WASTAGE_CSV)
    gfw  = pd.read_csv(GLOBAL_WASTAGE_CSV)

    # restaurant_sales_cleaned
    n = len(rest)
    rest['date'] = pd.date_range('2024-01-01', periods=n, freq='h')[:n]
    rest['month_year'] = rest['date'].dt.strftime('%Y-%m')
    rest['restaurant_type_label'] = rest['restaurant_type'].map(REST_TYPE_MAP)
    rest['meal_type_label']       = rest['meal_type'].map(MEAL_TYPE_MAP)
    rest['weather_label']         = rest['weather_condition'].map(WEATHER_MAP)
    rest['has_promotion']         = rest['has_promotion'].astype(int)

    # food_wastage_cleaned
    fw['Timestamp'] = pd.to_datetime(fw['Timestamp'])

    # global_food_wastage_dataset
    gfw['Year'] = gfw['Year_int'].astype(int)

    return rest, fw, gfw


# ─────────────────────────────────────────────
# HELPER: build filter description string
# ─────────────────────────────────────────────
def build_filter_desc(selected_rest, selected_weather,
                      selected_meal, rest_types_all,
                      weather_all, meal_types_all):

    parts = []

    if set(selected_rest) != set(rest_types_all):
        parts.append(f"Tipe Restoran: {', '.join(selected_rest)}")
    else:
        parts.append("Semua Tipe Restoran")

    if set(selected_weather) != set(weather_all):
        parts.append(f"Cuaca: {', '.join(selected_weather)}")
    else:
        parts.append("Semua Kondisi Cuaca")

    if set(selected_meal) != set(meal_types_all):
        parts.append(f"Meal Type: {', '.join(selected_meal)}")
    else:
        parts.append("Semua Waktu Sajian")

    return " · ".join(parts)


# ─────────────────────────────────────────────
# HELPER: filter rest dataframe (single source of truth)
# ─────────────────────────────────────────────
def apply_global_filter(rest, selected_rest, selected_weather, selected_meal):
    mask = (
        (rest['restaurant_type_label'].isin(selected_rest)) &
        (rest['weather_label'].isin(selected_weather)) &
        (rest['meal_type_label'].isin(selected_meal))
    )
    return rest[mask].copy()

# ─────────────────────────────────────────────
# DYNAMIC ANALYSIS TABLES (depend on filtered df)
# ─────────────────────────────────────────────
def compute_dynamic_tables(df_filtered):
    """Compute all pivot tables from the already-filtered dataframe."""

    # BQ4: Pivot qty per tipe restoran per bulan
    if df_filtered.empty:
        pivot_qty  = pd.DataFrame()
        pivot_left = pd.DataFrame()
        pivot_wx   = pd.DataFrame()
        df_promo   = pd.DataFrame()
        return pivot_qty, pivot_left, pivot_wx, df_promo

    pivot_qty = df_filtered.pivot_table(
        index='month_year', columns='restaurant_type_label',
        values='quantity_sold', aggfunc='mean'
    )
    pivot_qty.index = pd.PeriodIndex(pivot_qty.index, freq='M')
    pivot_qty = pivot_qty.sort_index()
    pivot_qty.index = pivot_qty.index.strftime('%Y-%m').tolist()
    pivot_qty['TOTAL'] = pivot_qty.sum(axis=1)

    pivot_left = df_filtered.pivot_table(
        index='month_year', columns='restaurant_type_label',
        values='leftover', aggfunc='mean'
    )
    pivot_left.index = pd.PeriodIndex(pivot_left.index, freq='M')
    pivot_left = pivot_left.sort_index()
    pivot_left.index = pivot_left.index.strftime('%Y-%m').tolist()

    pivot_wx = df_filtered.pivot_table(
        index='weather_label', columns='restaurant_type_label',
        values='quantity_sold', aggfunc='mean'
    )

    df_promo = df_filtered.copy()
    df_promo['Promo'] = df_promo['has_promotion'].apply(lambda x: 'Ada Promo' if x == 1 else 'Tanpa Promo')

    return pivot_qty, pivot_left, pivot_wx, df_promo


# Static global tables (BQ1/2/3 — not affected by restaurant filters)
@st.cache_data(show_spinner=False)
def compute_global_tables(gfw_df):
    category_comparison = gfw_df.groupby('Food Category').agg(
        Economic_Loss_Sum=('Economic Loss (Million $)', 'sum'),
        Total_Waste_Sum=('Total Waste (Tons)', 'sum')
    ).reset_index()

    prepared      = gfw_df[gfw_df['Food Category'] == 'Prepared Food']
    global_vs_indo = prepared.groupby(['Country', 'Year']).agg(
        Total_Waste=('Total Waste (Tons)', 'sum')
    ).reset_index()

    indo_prepared = gfw_df[
        (gfw_df['Country'] == 'Indonesia') & (gfw_df['Food Category'] == 'Prepared Food')
    ]
    indo_loss = indo_prepared.groupby('Year').agg(
        Economic_Loss=('Economic Loss (Million $)', 'sum'),
        Total_Waste=('Total Waste (Tons)', 'sum')
    ).reset_index().sort_values('Year')

    return category_comparison, global_vs_indo, indo_loss


# ═══════════════════════════════════════════════════════════════════
# SIDEBAR — GLOBAL FILTERS (single source of truth for ALL tabs)
# ═══════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("""
    <div style="text-align:center; padding: 20px 0 28px 0;">
        <div style="font-size:2.5rem;">♻️</div>
        <div style="font-family:'Sora',sans-serif; font-size:1.3rem; font-weight:800; color:#1A5276;">
            KeduaKali
        </div>
        <div style="font-size:0.72rem; color:#7F8C8D; margin-top:4px;">
            Platform Cerdas Penyelamat Makanan
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown(
        '<div style="font-size:0.7rem; color:#7F8C8D; text-transform:uppercase; '
        'letter-spacing:1px; margin-bottom:8px; font-weight:600;">Filter Global</div>',
        unsafe_allow_html=True
    )

    with st.spinner("Loading data..."):
        rest, fw, gfw = load_data()

    # ── Restaurant type ──
    rest_types_all = sorted(rest['restaurant_type_label'].dropna().unique().tolist())
    selected_rest  = st.multiselect("Tipe Restoran", rest_types_all, default=rest_types_all)

    if len(selected_rest) == 0:
        st.warning("Minimal pilih 1 tipe restoran.")
        selected_rest = rest_types_all

    # ── Weather ──
    weather_all      = sorted(rest['weather_label'].dropna().unique().tolist())
    selected_weather = st.multiselect("Kondisi Cuaca", weather_all, default=weather_all)

    if len(selected_weather) == 0:
        st.warning("Minimal pilih 1 tipe cuaca.")
        selected_weather = weather_all

    # ── Meal type (NEW global filter) ──
    meal_types_all  = sorted(rest['meal_type_label'].dropna().unique().tolist())
    selected_meal   = st.multiselect("Waktu Sajian (Meal Type)", meal_types_all, default=meal_types_all)

    if len(selected_meal) == 0:
        st.warning("Minimal pilih 1 waktu sajian.")
        selected_meal = meal_types_all

    st.markdown("<hr style='border:1px solid #D5E8F3; margin:20px 0;'>", unsafe_allow_html=True)
    
    # ── APPLY GLOBAL FILTER (single filtered df for all tabs) ──
    rest_f = apply_global_filter(
    rest,
    selected_rest,
    selected_weather,
    selected_meal
)
    
    # Data info panel
    st.markdown(f"""
    <div style="background:#EBF5FB; border:1px solid #AED6F1; border-radius:10px; padding:14px; font-size:0.78rem; color:#5D6D7E;">
        <div>📊 <b style="color:#1A5276;">Data Aktif</b></div>
        <div style="margin-top:6px;">🏪 {len(rest_f):,} transaksi restoran</div>
        <div>🍽️ {len(fw):,} event food wastage</div>
        <div>🌍 {gfw['Country'].nunique()} negara, {gfw['Year'].nunique()} tahun</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown(
        '<div style="font-size:0.68rem; color:#AEB6BF; text-align:center;">CC26-PSU226 · Capstone Project 2026</div>',
        unsafe_allow_html=True
    )

# ─────────────────────────────────────────────
# COMPUTE ALL TABLES FROM FILTERED DATA
# ─────────────────────────────────────────────
pivot_surplus_qty, pivot_surplus_left, pivot_weather, df_promo = compute_dynamic_tables(rest_f)
category_comparison, global_vs_indo, indo_loss = compute_global_tables(gfw)

# ─────────────────────────────────────────────
# MAIN HEADER
# ─────────────────────────────────────────────
st.markdown("""
<div class="brand-header">
    <div class="brand-title">♻️ KeduaKali Dashboard</div>
    <div class="brand-sub">Platform Cerdas Penyelamat Makanan Leftover & Barang Imperfect · Sustainable & Responsible Consumption</div>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# ACTIVE FILTER INDICATOR (shown below header)
# ─────────────────────────────────────────────
filter_desc = build_filter_desc(
    selected_rest,
    selected_weather,
    selected_meal,
    rest_types_all,
    weather_all,
    meal_types_all
)
st.markdown(
    f'<div class="filter-info-box">🔍 Filter Aktif: {filter_desc}</div>',
    unsafe_allow_html=True
)

# ─────────────────────────────────────────────
# TOP KPI METRICS — dari data filtered
# ─────────────────────────────────────────────
k1, k2, k3, k4, k5 = st.columns(5)

if rest_f.empty:
    total_qty = total_leftover = waste_rate = promo_lift = 0
else:
    total_leftover = rest_f['leftover'].sum()
    total_qty      = rest_f['quantity_sold'].sum()
    waste_rate     = (total_leftover / (total_leftover + total_qty) * 100) if (total_leftover + total_qty) > 0 else 0
    promo_val_on   = rest_f[rest_f['has_promotion'] == 1]['quantity_sold'].mean()
    promo_val_off  = rest_f[rest_f['has_promotion'] == 0]['quantity_sold'].mean()
    promo_lift     = (promo_val_on - promo_val_off) if not (np.isnan(promo_val_on) or np.isnan(promo_val_off)) else 0

n_countries = gfw['Country'].nunique()

for col, val, lbl in [
    (k1, f"{total_qty:,.0f}",     "Total Qty Terjual (Unit)"),
    (k2, f"{total_leftover:,.0f}", "Total Leftover (Unit)"),
    (k3, f"{waste_rate:.1f}%",    "Waste Rate Relatif"),
    (k4, f"{promo_lift:+.1f}",    "Lift Promo (Unit)"),
    (k5, f"{n_countries}",         "Negara Tercakup"),
]:
    col.markdown(f"""
    <div class="metric-card">
        <div class="metric-value">{val}</div>
        <div class="metric-label">{lbl}</div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# TABS
# ─────────────────────────────────────────────
tab1, tab2 = st.tabs(["📊  Business Insights", "🤖  AI Prediction"])


# ═══════════════════════════════════════════════════════════════════
# TAB 1 — BUSINESS INSIGHTS (fully dynamic, uses rest_f)
# ═══════════════════════════════════════════════════════════════════
with tab1:

    if rest_f.empty:
        st.warning("⚠️ Tidak ada data yang sesuai dengan filter yang dipilih. Silakan sesuaikan filter di sidebar.")
        st.stop()

    period_label = "Seluruh Periode Data"

    # ──────────────────────────────────────────────────────
    # BQ1: Economic Loss per Food Category (Global — static context)
    # ──────────────────────────────────────────────────────
    st.markdown(
        '<div class="section-header">BQ1 · Kerugian Ekonomi per Kategori Pangan Global</div>',
        unsafe_allow_html=True
    )

    plot_data_bq1 = category_comparison.sort_values('Economic_Loss_Sum', ascending=True).reset_index(drop=True)
    categories = plot_data_bq1['Food Category'].values
    values_bq1 = plot_data_bq1['Economic_Loss_Sum'].values
    colors_bq1 = ['#F2A9A9' if i != len(values_bq1) - 1 else '#E24B4A' for i in range(len(values_bq1))]

    fig1, ax1 = plt.subplots(figsize=(14, 7), facecolor='white')
    bars = ax1.bar(categories, values_bq1, color=colors_bq1, edgecolor='none', width=0.6)
    for bar, val in zip(bars, values_bq1):
        ax1.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + values_bq1.max() * 0.01,
            f'${val:,.0f}M', ha='center', va='bottom',
            fontsize=10, fontweight='bold',
            color='#E24B4A' if val == values_bq1.max() else '#2C2C2A'
        )
    ax1.set_title('Kerugian Ekonomi per Kategori Pangan (Data Global)', fontsize=15, pad=16, fontweight='bold', color='#2C2C2A')
    ax1.set_ylabel('Total Kerugian Ekonomi (Million $)', fontsize=12, fontweight='bold', color='#2C2C2A')
    ax1.set_xlabel('Kategori Pangan', fontsize=12, fontweight='bold', color='#2C2C2A')
    ax1.tick_params(axis='x', rotation=20)
    ax1.tick_params(colors='#2C2C2A')
    ax1.tick_params(axis='y', length=6, width=1.2)
    ax1.yaxis.grid(False); ax1.xaxis.grid(False)
    ax1.set_axisbelow(True)
    ax1.spines[['top', 'right']].set_visible(False)
    ax1.spines['left'].set_color('#D3D1C7')
    ax1.spines['bottom'].set_color('#D3D1C7')
    red_patch  = mpatches.Patch(color='#E24B4A', label='Tertinggi')
    pink_patch = mpatches.Patch(color='#F2A9A9', label='Kategori Lainnya')
    ax1.legend(handles=[red_patch, pink_patch], loc='upper left', frameon=False)
    plt.tight_layout()
    st.pyplot(fig1); plt.close()

    st.markdown("""
    <div class="insight-box">
        <strong>Insight:</strong> <strong>Prepared Food</strong> mendominasi kerugian ekonomi global akibat food waste.
        Hal ini memvalidasi fokus platform <strong>KeduaKali</strong> pada segmen makanan siap saji sebagai prioritas utama
        distribusi real-time untuk meminimalisir <em>economic loss</em>.
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<hr class='custom-divider'>", unsafe_allow_html=True)

    # ──────────────────────────────────────────────────────
    # BQ2: Tren Volume Surplus Prepared Food — Indonesia vs Global
    # ──────────────────────────────────────────────────────
    st.markdown(
        '<div class="section-header">BQ2 · Tren Surplus Prepared Food: Indonesia vs Negara Lain</div>',
        unsafe_allow_html=True
    )

    df_plot_bq2    = global_vs_indo.copy()
    df_plot_bq2    = df_plot_bq2[df_plot_bq2['Year'] >= 2018].sort_values(['Country', 'Year'])
    top_countries  = df_plot_bq2.groupby('Country')['Total_Waste'].sum().nlargest(6).index.tolist()
    if 'Indonesia' not in top_countries:
        top_countries.append('Indonesia')

    fig2, ax2 = plt.subplots(figsize=(12, 6), facecolor='white')
    others        = [c for c in top_countries if c != 'Indonesia']
    last_positions = []
    for country in others:
        d = df_plot_bq2[df_plot_bq2['Country'] == country]
        ax2.plot(d['Year'], d['Total_Waste'], color='#BDC3C7', linewidth=2, alpha=0.5, zorder=2)
        d_last = d[d['Year'] == d['Year'].max()]
        if not d_last.empty:
            last_positions.append({'country': country, 'val': d_last['Total_Waste'].values[0]})

    last_positions.sort(key=lambda x: x['val'], reverse=True)
    min_dist = max(df_plot_bq2['Total_Waste'].max() * 0.03, 500)
    for i, item in enumerate(last_positions):
        if i > 0:
            prev_adj = last_positions[i - 1].get('val_adjusted', last_positions[i - 1]['val'])
            diff = prev_adj - item['val']
            item['val_adjusted'] = prev_adj - min_dist if diff < min_dist else item['val']
        else:
            item['val_adjusted'] = item['val']
        ax2.text(df_plot_bq2['Year'].max() + 0.15, item['val_adjusted'],
                 item['country'], color='#7F8C8D', fontsize=10, fontweight='bold', va='center')

    indo = df_plot_bq2[df_plot_bq2['Country'] == 'Indonesia']
    ax2.plot(indo['Year'], indo['Total_Waste'], color='#FF0000', linewidth=5,
             marker='o', markersize=10, markerfacecolor='#FF0000',
             markeredgecolor='white', markeredgewidth=2, zorder=20, label='Indonesia')

    ind_last = indo[indo['Year'] == indo['Year'].max()]
    if not ind_last.empty:
        ax2.text(df_plot_bq2['Year'].max() + 0.15, ind_last['Total_Waste'].values[0],
                 'Indonesia', color='#FF0000', fontsize=12, fontweight='black', va='center')

    y_values = indo['Total_Waste'].tolist()
    x_values = indo['Year'].tolist()
    for i, (x, y) in enumerate(zip(x_values, y_values)):
        if i == 0 or y >= y_values[i - 1]:
            va_pos, offset_y = 'bottom', 12
        else:
            va_pos, offset_y = 'top', -12
        ax2.annotate(f"{y:,.0f}", (x, y), xytext=(0, offset_y), textcoords='offset points',
                     ha='center', va=va_pos, fontsize=10, fontweight='bold', color='#D32F2F',
                     bbox=dict(facecolor='white', edgecolor='none', alpha=0.8, boxstyle='round,pad=0.1'))

    ax2.set_title('Tren Surplus Makanan Siap Saji dengan Negara Lain', fontsize=15, fontweight='bold', pad=20, loc='left')
    ax2.grid(False, axis='x')
    ax2.grid(True, axis='y', linestyle='--', color='#BDBDBD', alpha=0.7)
    ax2.set_xlim(df_plot_bq2['Year'].min() - 0.5, df_plot_bq2['Year'].max() + 0.6)
    ax2.set_xticks(range(int(df_plot_bq2['Year'].min()), int(df_plot_bq2['Year'].max()) + 1))
    ax2.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, p: f'{x:,.0f}'))
    ax2.set_xlabel('Tahun', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Volume (Ton)', fontsize=12, fontweight='bold')
    sns.despine(ax=ax2)
    plt.tight_layout()
    st.pyplot(fig2); plt.close()

    st.markdown("""
    <div class="insight-box">
        <strong>Insight:</strong> Indonesia menunjukkan lonjakan surplus yang sangat ekstrem — kenaikan hampir 5x lipat dalam setahun (2022→2023).
        Volatilitas ini mengindikasikan urgensi infrastruktur digital seperti KeduaKali untuk monitoring real-time.
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<hr class='custom-divider'>", unsafe_allow_html=True)

    # ──────────────────────────────────────────────────────
    # BQ3: Tren Economic Loss Indonesia — Dual Panel
    # ──────────────────────────────────────────────────────
    st.markdown(
        '<div class="section-header">BQ3 · Tren Kerugian Ekonomi Indonesia — Prepared Food</div>',
        unsafe_allow_html=True
    )

    df_indo   = indo_loss.copy()
    years     = df_indo['Year'].values
    losses    = df_indo['Economic_Loss'].values
    waste_arr = df_indo['Total_Waste'].values

    loss_min  = losses.min()
    loss_max  = losses.max()
    loss_mean = losses.mean()

    C_TITLE    = '#1A237E'
    C_SUB      = '#37474F'
    C_TICK_BQ3 = '#607D8B'
    C_LABEL    = '#37474F'
    C_LINE_BQ3 = '#2E7D32'

    fig3 = plt.figure(figsize=(13, 8.5), facecolor='white')
    gs3  = fig3.add_gridspec(2, 1, height_ratios=[0.58, 0.42],
                              hspace=0.38, top=0.87, bottom=0.09, left=0.09, right=0.93)
    ax3a = fig3.add_subplot(gs3[0])
    ax3b = fig3.add_subplot(gs3[1])

    fig3.text(0.07, 0.965, 'Dampak Ekonomi Limbah Makanan Siap Saji di Indonesia',
              fontsize=15, fontweight='bold', color=C_TITLE, va='top')
    fig3.text(0.07, 0.928, 'Tren Kerugian Finansial & Volume Limbah  ·  Prepared Food',
              fontsize=10, color=C_SUB, va='top')

    def bar_color_bq3(v):
        if v == loss_max: return C_HIGH
        if v == loss_min: return C_LOW
        return C_NORMAL

    bar_colors3 = [bar_color_bq3(v) for v in losses]
    ax3a.bar(years, losses, color=bar_colors3, width=0.55, zorder=3, linewidth=0)
    ax3a.set_ylim(0, loss_max * 1.38)

    for bar, val in zip(ax3a.patches, losses):
        x      = bar.get_x() + bar.get_width() / 2
        is_max = val == loss_max
        is_min = val == loss_min
        if is_max or is_min:
            color_lbl = C_HIGH if is_max else C_LOW
            bg        = '#FBE9E7' if is_max else '#E1F5FE'
            badge_txt = '▲ Tertinggi' if is_max else '▼ Terendah'
            y_val   = val + loss_max * 0.015
            ax3a.text(x, y_val, f'${val:,.0f}M', ha='center', va='bottom',
                      fontsize=10, fontweight='bold', color=color_lbl, zorder=7)
            y_badge = y_val + loss_max * 0.085
            ax3a.text(x, y_badge, badge_txt, ha='center', va='bottom', fontsize=7.5,
                      color=color_lbl, zorder=5,
                      bbox=dict(boxstyle='round,pad=0.35', facecolor=bg, edgecolor=color_lbl, linewidth=0.8))
        else:
            ax3a.text(x, val + loss_max * 0.015, f'${val:,.0f}M', ha='center', va='bottom',
                      fontsize=8.5, color=C_LABEL, zorder=5)

    ax3a.axhline(loss_mean, color=C_AVG_LINE, linewidth=1.3, linestyle='--', alpha=0.75, zorder=2)
    ax3a.text(years[0] - 0.38, loss_mean, f'Rata-rata: ${loss_mean:,.0f}M',
              fontsize=8.5, color=C_AVG_LINE, va='center', ha='left', zorder=6,
              bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='none'))
    ax3a.set_ylabel('Kerugian Ekonomi (Juta USD)', fontsize=10, color=C_SUB, labelpad=10)
    ax3a.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'${x/1000:,.0f}K'))
    ax3a.set_xticks(years)
    ax3a.set_xticklabels(years, fontsize=10, color=C_TICK_BQ3)
    ax3a.tick_params(axis='x', length=0, pad=5)
    ax3a.tick_params(axis='y', labelsize=9, colors=C_TICK_BQ3, length=0)
    ax3a.yaxis.grid(True, linestyle='--', linewidth=0.5, color='#ECEFF1', zorder=0)
    ax3a.set_axisbelow(True)
    for sp in ax3a.spines.values(): sp.set_visible(False)

    legend_bars3 = [
        mpatches.Patch(facecolor=C_NORMAL, label='Kerugian ekonomi'),
        mpatches.Patch(facecolor=C_HIGH, label='Tertinggi'),
        mpatches.Patch(facecolor=C_LOW, label='Terendah'),
        plt.Line2D([0],[0], color=C_AVG_LINE, linewidth=1.5, linestyle='--', label='Rata-rata'),
    ]
    ax3a.legend(handles=legend_bars3, fontsize=8.5, frameon=False, ncol=4, loc='upper left', handlelength=1.6)

    ax3b.plot(years, waste_arr, color=C_LINE_BQ3, marker='D', linewidth=2.3, markersize=7,
              markerfacecolor='white', markeredgecolor=C_LINE_BQ3, markeredgewidth=2, zorder=3)
    ax3b.fill_between(years, waste_arr, alpha=0.15, color=C_LINE_BQ3, zorder=1)

    waste_max = waste_arr.max()
    waste_min = waste_arr.min()
    for yr, wv in zip(years, waste_arr):
        if wv == waste_max:
            ax3b.text(yr, wv + waste_max * 0.06, f'{wv:,.0f}', ha='center', va='bottom',
                      fontsize=8.5, fontweight='bold', color=C_LINE_BQ3,
                      bbox=dict(boxstyle='round,pad=0.3', facecolor='#E8F5E9', edgecolor=C_LINE_BQ3, linewidth=0.8))
        elif wv == waste_min:
            ax3b.text(yr, wv - waste_max * 0.09, f'{wv:,.0f}', ha='center', va='top',
                      fontsize=8.5, fontweight='bold', color=C_LINE_BQ3,
                      bbox=dict(boxstyle='round,pad=0.3', facecolor='#E8F5E9', edgecolor=C_LINE_BQ3, linewidth=0.8))
        else:
            idx  = list(years).index(yr)
            flip = waste_max * 0.05 if idx % 2 == 0 else -waste_max * 0.08
            ax3b.text(yr, wv + flip, f'{wv:,.0f}', ha='center', va='bottom', fontsize=7.5, color=C_TICK_BQ3)

    ax3b.set_ylabel('Volume Limbah (Ton)', fontsize=10, color=C_LINE_BQ3, labelpad=10)
    ax3b.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'{x/1000:,.0f}K'))
    ax3b.set_xticks(years)
    ax3b.set_xticklabels(years, fontsize=10, color=C_TICK_BQ3)
    ax3b.tick_params(axis='x', length=0, pad=5)
    ax3b.tick_params(axis='y', labelsize=9, colors=C_LINE_BQ3, length=0)
    ax3b.yaxis.grid(True, linestyle='--', linewidth=0.5, color='#ECEFF1', zorder=0)
    ax3b.set_axisbelow(True)
    ax3b.set_ylim(0, waste_max * 1.22)
    for sp in ax3b.spines.values(): sp.set_visible(False)

    legend_line3 = [plt.Line2D([0],[0], color=C_LINE_BQ3, marker='D', linewidth=2,
                               markerfacecolor='white', markeredgecolor=C_LINE_BQ3,
                               markeredgewidth=2, label='Volume limbah')]
    ax3b.legend(handles=legend_line3, fontsize=8.5, frameon=False, loc='upper left')
    fig3.text(0.50, 0.02, 'Tahun', ha='center', fontsize=10, color=C_SUB)
    st.pyplot(fig3); plt.close()

    st.markdown("""
    <div class="insight-box">
        <strong>Insight:</strong> Tahun <strong>2023 menjadi titik puncak</strong> dengan kerugian ekonomi tertinggi berbanding lurus
        dengan lonjakan volume limbah. Tanpa intervensi distribusi real-time, risiko kerugian tetap kritis.
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<hr class='custom-divider'>", unsafe_allow_html=True)

    # ──────────────────────────────────────────────────────
    # BQ4: Small Multiples + Heatmap + Line — DYNAMIC
    # ──────────────────────────────────────────────────────
    st.markdown(
        f'<div class="section-header">BQ4 · Penjualan per Tipe Restoran ({period_label})</div>',
        unsafe_allow_html=True
    )

    if pivot_surplus_qty.empty or pivot_surplus_qty.shape[0] == 0:
        st.info("Tidak cukup data untuk BQ4 pada filter yang dipilih.")
    else:
        df_plot_bq4      = pivot_surplus_qty.copy()
        restaurant_cols_bq4 = [c for c in df_plot_bq4.columns if c != 'TOTAL']
        months_bq4       = df_plot_bq4.index.tolist()
        n_types          = len(restaurant_cols_bq4)

        if n_types > 0:
            ncols    = 2
            nrows_sm = int(np.ceil(n_types / ncols))
            fig4a, axes4a = plt.subplots(nrows_sm, ncols, figsize=(18, nrows_sm * 5.5), facecolor='white')
            axes_flat = np.array(axes4a).flatten()
            x_sm     = np.arange(len(months_bq4))

            for i, rtype in enumerate(restaurant_cols_bq4):
                ax      = axes_flat[i]
                vals_sm = df_plot_bq4[rtype].fillna(0).values
                max_val = vals_sm.max()
                non_zero = vals_sm[vals_sm > 0]
                min_val  = non_zero.min() if len(non_zero) > 0 else 0

                bc = []
                for val in vals_sm:
                    if val == max_val and val > 0: bc.append(COLOR_PEAK)
                    elif val == min_val and val > 0: bc.append(COLOR_SURPLUS)
                    else: bc.append(COLOR_NEUTRAL)

                ax.bar(x_sm, vals_sm, width=0.75, color=bc, edgecolor='#BDC3C7', linewidth=0.6, zorder=3)
                for j, val in enumerate(vals_sm):
                    if val == max_val or val == min_val:
                        color_t = COLOR_PEAK if val == max_val else '#2980B9'
                        ax.text(j, val + (max_val * 0.02 if max_val != 0 else 0.02),
                                f'{val:,.1f}', ha='center', va='bottom', fontsize=11, fontweight='bold', color=color_t)

                ax.set_title(rtype.upper(), fontsize=15, fontweight='black', color='#2C3E50', loc='left', pad=20)
                ax.set_xticks(x_sm)
                ax.set_xticklabels(months_bq4, rotation=45, ha='right', fontsize=9, color='#34495E')
                ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, p: f'{x:,.1f}'))
                ax.tick_params(axis='y', labelsize=9, colors='#7F8C8D')
                for s in ['top', 'right', 'left']: ax.spines[s].set_visible(False)
                ax.spines['bottom'].set_color('#7F8C8D')
                ax.spines['bottom'].set_linewidth(1.5)
                ax.yaxis.grid(True, linestyle='--', alpha=0.3, zorder=0)

            for j in range(n_types, len(axes_flat)):
                axes_flat[j].axis('off')

            plt.suptitle('ANALISIS TREN PENJUALAN UNIT RESTORAN',
                         fontsize=24, fontweight='black', color='#1A242F', x=0.05, ha='left', y=0.98)
            fig4a.text(0.05, 0.955,
                       f'Periode Data: {period_label} | Biru Tua (Puncak) vs Biru Muda (Terendah)',
                       fontsize=13, color='#5D6D7E', ha='left')
            legend_sm = [
                mpatches.Patch(color=COLOR_PEAK,    label='Penjualan Tertinggi (Peak)'),
                mpatches.Patch(color=COLOR_NEUTRAL, label='Penjualan Bulanan Normal'),
                mpatches.Patch(color=COLOR_SURPLUS, label='Penjualan Terendah (Potensi Surplus)'),
            ]
            fig4a.legend(handles=legend_sm, loc='lower center', ncol=3,
                         bbox_to_anchor=(0.5, -0.04), frameon=True, facecolor='#FDFEFE', fontsize=12)
            plt.tight_layout(rect=[0, 0, 1, 0.93])
            plt.subplots_adjust(hspace=0.7)
            st.pyplot(fig4a); plt.close()

            # Heatmap + Line
            fig4b = plt.figure(figsize=(18, 14), facecolor='white')
            gs4b  = gridspec.GridSpec(2, 1, height_ratios=[1.2, 1], figure=fig4b)
            plt.subplots_adjust(hspace=0.6, top=0.88, bottom=0.1, left=0.15)

            ax_heat = fig4b.add_subplot(gs4b[0])
            ax_line = fig4b.add_subplot(gs4b[1])

            heat_data = df_plot_bq4[restaurant_cols_bq4].T
            sns.heatmap(heat_data, ax=ax_heat, cmap='Blues', annot=True, fmt='.1f',
                        annot_kws={'size': 10, 'weight': 'bold'},
                        linewidths=1.2, linecolor='white',
                        cbar_kws={'label': 'Unit Terjual', 'shrink': 0.8})
            ax_heat.set_ylabel('TIPE RESTORAN', fontweight='bold', fontsize=12, labelpad=15)
            ax_heat.set_xlabel('BULAN', fontweight='bold', fontsize=12, labelpad=10)
            ax_heat.set_title('Distribusi Penjualan per Tipe Restoran', fontsize=15, fontweight='bold', pad=20, loc='left')
            ax_heat.set_xticklabels(months_bq4, rotation=0, ha='center', fontsize=9)

            x_ln     = np.arange(len(months_bq4))
            total_ln = df_plot_bq4['TOTAL'].values
            peak_idx  = int(np.argmax(total_ln))
            peak_val  = total_ln[peak_idx]
            peak_month = months_bq4[peak_idx]

            ax_line.fill_between(x_ln, total_ln, alpha=0.1, color='#2980B9')
            ax_line.plot(x_ln, total_ln, color='#2980B9', lw=4, marker='o', mfc='white', mew=2, markersize=8)
            ax_line.scatter([peak_idx], [peak_val], color='#E74C3C', s=180, zorder=6, edgecolors='white')
            ax_line.text(peak_idx + 0.15, peak_val + abs(total_ln.max()) * 0.02,
                         f'PUNCAK: {peak_val:,.1f}\n({peak_month})',
                         ha='left', va='bottom', color='#E74C3C', fontweight='bold', fontsize=11)
            for xi, yi in zip(x_ln, total_ln):
                if xi != peak_idx:
                    ax_line.text(xi, yi + abs(total_ln.max()) * 0.015, f'{yi:,.1f}',
                                 ha='center', fontsize=9, color='#2C3E50', fontweight='bold',
                                 bbox=dict(facecolor='white', alpha=0.6, edgecolor='none', pad=1))

            ax_line.set_title('Tren Akumulasi Penjualan Seluruh Restoran', fontsize=14, fontweight='bold', pad=20, loc='left')
            ax_line.set_xlabel('BULAN', fontweight='bold', fontsize=12, labelpad=10)
            ax_line.set_xticks(x_ln)
            ax_line.set_xticklabels(months_bq4, rotation=0, ha='center', fontsize=9)
            sns.despine(ax=ax_line)

            fig4b.text(0.09, 0.96, 'LAPORAN PERFORMA PENJUALAN RESTORAN', fontsize=22, fontweight='black', color='#2C3E50')
            fig4b.text(0.09, 0.93, f'Analisis Heatmap Detail & Tren Agregat Bulanan ({period_label})', fontsize=13, color='#7F8C8D')
            st.pyplot(fig4b); plt.close()

        # Dynamic insight
        if not df_plot_bq4.empty and len(restaurant_cols_bq4) > 0:
            avg_by_type    = df_plot_bq4[restaurant_cols_bq4].mean()
            lowest_type    = avg_by_type.idxmin()
            highest_type   = avg_by_type.idxmax()
            peak_m_label   = months_bq4[int(np.argmax(df_plot_bq4['TOTAL'].values))] if 'TOTAL' in df_plot_bq4.columns else "-"
            st.markdown(f"""
            <div class="insight-box">
                <strong>Insight ({period_label}):</strong>
                Berdasarkan filter aktif, <strong>{lowest_type}</strong> memiliki rata-rata penjualan terendah — menjadi prioritas
                fitur revenue recovery. <strong>{highest_type}</strong> mencatat penjualan tertinggi.
                Bulan dengan akumulasi penjualan tertinggi: <strong>{peak_m_label}</strong>.
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<hr class='custom-divider'>", unsafe_allow_html=True)

    # ──────────────────────────────────────────────────────
    # BQ5: Heatmap Cuaca × Tipe Restoran — DYNAMIC
    # ──────────────────────────────────────────────────────
    st.markdown(
        f'<div class="section-header">BQ5 · Pengaruh Kondisi Cuaca terhadap Penjualan per Tipe Restoran ({period_label})</div>',
        unsafe_allow_html=True
    )

    if pivot_weather.empty or pivot_weather.shape[0] == 0:
        st.info("Tidak cukup data untuk BQ5 pada filter yang dipilih.")
    else:
        fig5, ax5 = plt.subplots(figsize=(12, max(4, pivot_weather.shape[0] * 1.5)), facecolor='white')
        sns.heatmap(pivot_weather, annot=True, fmt='.2f', cmap='YlGnBu',
                    linewidths=0.5, ax=ax5, cbar_kws={'label': 'Rata-rata Unit Terjual'})
        ax5.set_title(f'Pengaruh Cuaca terhadap Rata-rata Penjualan per Tipe Restoran\n(Periode: {period_label})',
                      fontsize=13, fontweight='bold', pad=20)
        ax5.set_xlabel('Tipe Restoran', fontsize=12, fontweight='bold')
        ax5.set_ylabel('Kondisi Cuaca', fontsize=12, fontweight='bold')
        plt.tight_layout()
        st.pyplot(fig5); plt.close()

        # Dynamic insight for BQ5
        try:
            max_loc = pivot_weather.stack().idxmax()
            min_loc = pivot_weather.stack().idxmin()
            st.markdown(f"""
            <div class="insight-box">
                <strong>Insight ({period_label}):</strong>
                Kombinasi cuaca–tipe restoran dengan penjualan tertinggi: <strong>{max_loc[0]} × {max_loc[1]}</strong>.
                Kombinasi terendah (potensi surplus tertinggi): <strong>{min_loc[0]} × {min_loc[1]}</strong> —
                jadikan prioritas penyelamatan stok saat kondisi tersebut.
            </div>
            """, unsafe_allow_html=True)
        except Exception:
            st.markdown("""
            <div class="insight-box">
                <strong>Insight:</strong> Cuaca memengaruhi tipe restoran secara berbeda. Sesuaikan filter untuk melihat pola spesifik.
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<hr class='custom-divider'>", unsafe_allow_html=True)

    # ──────────────────────────────────────────────────────
    # BQ6: Regplot + Dual Subplot (food_wastage — independent data)
    # ──────────────────────────────────────────────────────
    st.markdown(
        '<div class="section-header">BQ6 · Rasio Makanan/Tamu vs Tingkat Pemborosan</div>',
        unsafe_allow_html=True
    )

    sns.set_theme(style='whitegrid')
    fig6a, ax6a = plt.subplots(figsize=(12, 7), facecolor='white')
    sns.regplot(data=fw, x='food_per_guest_ratio', y='Wastage Food Amount',
                scatter_kws={'alpha': 0.2, 's': 20, 'color': '#2ECC71'},
                line_kws={'color': '#E74C3C', 'lw': 4}, ax=ax6a)
    legend_el6 = [
        Line2D([0],[0], marker='o', color='w', label='Titik Hijau: Data Kejadian (Satu Acara)',
               markerfacecolor='#2ECC71', markersize=10),
        Line2D([0],[0], color='#E74C3C', lw=4, label='Garis Merah: Tren/Pola Pemborosan')
    ]
    ax6a.legend(handles=legend_el6, loc='upper left', frameon=True, fontsize=11, facecolor='white')
    ax6a.set_title('Dampak Rasio Makanan terhadap Volume Food Waste', fontsize=15, fontweight='bold', pad=20)
    ax6a.set_xlabel('Rasio Makanan per Tamu (Porsi Makanan / Jumlah Tamu)', fontsize=12, fontweight='bold')
    ax6a.set_ylabel('Jumlah Makanan Terbuang (kg)', fontsize=12, fontweight='bold')
    plt.tight_layout()
    st.pyplot(fig6a); plt.close()

    monthly_data_fw = fw.set_index('Timestamp').resample('MS').mean(numeric_only=True)

    fig6b, (ax6b1, ax6b2) = plt.subplots(2, 1, figsize=(15, 9), facecolor='white',
                                          sharex=True, gridspec_kw={'hspace': 0.15})
    ax6b1.plot(monthly_data_fw.index, monthly_data_fw['food_per_guest_ratio'],
               marker='o', color='#3498DB', linewidth=3, markersize=8)
    ax6b1.set_ylabel('Average Ratio\n(Food/Guest)', fontsize=12, fontweight='bold')
    ax6b1.set_title('Monthly Production Efficiency Trend', fontsize=13, fontweight='bold')
    ax6b1.grid(True, alpha=0.3)

    ax6b2.plot(monthly_data_fw.index, monthly_data_fw['Wastage Food Amount'],
               marker='s', color='#E67E22', linestyle='--', linewidth=3, markersize=8)
    ax6b2.set_ylabel('Average Waste\n(kg)', fontsize=12, fontweight='bold')
    ax6b2.set_title('Monthly Food Waste Trend', fontsize=13, fontweight='bold')
    ax6b2.grid(True, alpha=0.3)
    ax6b2.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
    ax6b2.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
    plt.xticks(rotation=45)

    plt.suptitle('Monthly Trend of Production Efficiency and Food Waste',
                 fontsize=15, fontweight='bold', y=1.02)
    plt.tight_layout()
    st.pyplot(fig6b); plt.close()

    st.markdown("""
    <div class="insight-box">
        <strong>Insight:</strong> Rasio makanan rendah (tamu sedikit) menghasilkan limbah per kapita tertinggi.
        Untuk acara kecil, gunakan <strong>Content-Based Filtering</strong> dengan penawaran porsi individual real-time.
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<hr class='custom-divider'>", unsafe_allow_html=True)

    # ──────────────────────────────────────────────────────
    # BQ7: Efektivitas Promo — DYNAMIC
    # ──────────────────────────────────────────────────────
    st.markdown(
        f'<div class="section-header">BQ7 · Efektivitas Promo dalam Meningkatkan Penjualan & Menekan Waste ({period_label})</div>',
        unsafe_allow_html=True
    )

    if df_promo.empty:
        st.info("Tidak cukup data untuk BQ7 pada filter yang dipilih.")
    else:
        df_promo_vis  = df_promo.rename(columns={'restaurant_type_label': 'Tipe Restoran'})
        order_rank_bq7 = (
            df_promo_vis.groupby('Tipe Restoran', observed=False)['quantity_sold']
            .mean().sort_values(ascending=False).index
        )
        my_pal = {'Ada Promo': '#1D3557', 'Tanpa Promo': '#A8DADC'}

        fig7, (ax7a, ax7b) = plt.subplots(1, 2, figsize=(18, 7), facecolor='white')
        sns.barplot(data=df_promo_vis, x='Tipe Restoran', y='quantity_sold',
                    hue='Promo', ax=ax7a, palette=my_pal, order=order_rank_bq7, errorbar=None)
        ax7a.set_title('Peningkatan PENJUALAN Berkat Promo', fontweight='bold', fontsize=14, pad=15)
        ax7a.set_ylabel('Rata-rata Unit Terjual', fontweight='bold')
        ax7a.set_xlabel('Tipe Restoran', fontweight='bold')
        for c in ax7a.containers:
            ax7a.bar_label(c, fmt='%.1f', padding=3, fontweight='bold', fontsize=9)
        sns.despine(ax=ax7a)

        sns.barplot(data=df_promo_vis, x='Tipe Restoran', y='leftover',
                    hue='Promo', ax=ax7b, palette=my_pal, order=order_rank_bq7, errorbar=None)
        ax7b.set_title('Dampak Promo terhadap SISA MAKANAN', fontweight='bold', fontsize=14, pad=15)
        ax7b.set_ylabel('Rata-rata Sisa (Kg/Unit)', fontweight='bold')
        ax7b.set_xlabel('Tipe Restoran', fontweight='bold')
        for c in ax7b.containers:
            ax7b.bar_label(c, fmt='%.1f', padding=3, fontweight='bold', fontsize=9)
        sns.despine(ax=ax7b)

        plt.suptitle(f'Analisis Strategi Promo: Penjualan vs Food Waste ({period_label})',
                     fontsize=15, fontweight='bold', y=1.02)
        plt.tight_layout(rect=[0, 0.03, 1, 0.98])
        st.pyplot(fig7); plt.close()

        # Dynamic insight BQ7
        try:
            promo_df    = df_promo_vis[df_promo_vis['Promo'] == 'Ada Promo'].groupby('Tipe Restoran', observed=False)
            no_promo_df = df_promo_vis[df_promo_vis['Promo'] == 'Tanpa Promo'].groupby('Tipe Restoran', observed=False)
            lift_series  = promo_df['quantity_sold'].mean() - no_promo_df['quantity_sold'].mean()
            best_promo   = lift_series.idxmax() if not lift_series.empty else "-"
            waste_diff   = no_promo_df['leftover'].mean() - promo_df['leftover'].mean()
            best_waste   = waste_diff.idxmax() if not waste_diff.empty else "-"
            st.markdown(f"""
            <div class="insight-box">
                <strong>Insight ({period_label}):</strong>
                Promo memberikan lift penjualan terbesar di <strong>{best_promo}</strong>.
                Penurunan sisa makanan terbesar berkat promo terjadi di <strong>{best_waste}</strong>.
                Promo terbukti efektif menekan waste sekaligus mendongkrak volume terjual di semua segmen yang aktif.
            </div>
            """, unsafe_allow_html=True)
        except Exception:
            st.markdown("""
            <div class="insight-box">
                <strong>Insight:</strong> Promo terbukti efektif di semua tipe restoran dalam periode yang dipilih.
            </div>
            """, unsafe_allow_html=True)

    # ── Analisis Tambahan Food Wastage ──
    st.markdown("<hr class='custom-divider'>", unsafe_allow_html=True)
    st.markdown(
        '<div class="section-header">Analisis Tambahan · Distribusi Wastage per Kategori & Musim</div>',
        unsafe_allow_html=True
    )

    col_add1, col_add2, col_add3 = st.columns(3)

    with col_add1:
        fig_a1, ax_a1 = plt.subplots(figsize=(5, 4), facecolor='white')
        fw_type = fw.groupby('Type of Food')['Wastage Food Amount'].mean().sort_values(ascending=True)
        ax_a1.barh(fw_type.index, fw_type.values, color=C_BLUE, edgecolor='none', height=0.55)
        ax_a1.set_xlabel('Avg Wastage (kg)', fontsize=8)
        ax_a1.set_title('Avg Wastage per Tipe Makanan', fontsize=9, fontweight='bold', pad=8)
        ax_a1.grid(axis='x', alpha=0.3)
        ax_a1.spines[['top', 'right']].set_visible(False)
        plt.tight_layout(); st.pyplot(fig_a1); plt.close()

    with col_add2:
        fig_a2, ax_a2 = plt.subplots(figsize=(5, 4), facecolor='white')
        fw_event = fw.groupby('Event Type')['Wastage Food Amount'].sum().sort_values(ascending=True)
        ax_a2.barh(fw_event.index, fw_event.values, color=C_ORANGE, edgecolor='none', height=0.55)
        ax_a2.set_xlabel('Total Wastage (kg)', fontsize=8)
        ax_a2.set_title('Total Wastage per Event Type', fontsize=9, fontweight='bold', pad=8)
        ax_a2.grid(axis='x', alpha=0.3)
        ax_a2.spines[['top', 'right']].set_visible(False)
        plt.tight_layout(); st.pyplot(fig_a2); plt.close()

    with col_add3:
        fig_a3, ax_a3 = plt.subplots(figsize=(5, 4), facecolor='white')
        fw_season = fw.groupby('Seasonality')['Wastage Food Amount'].sum().sort_values(ascending=True)
        ax_a3.barh(fw_season.index, fw_season.values, color=C_GREEN, edgecolor='none', height=0.55)
        ax_a3.set_xlabel('Total Wastage (kg)', fontsize=8)
        ax_a3.set_title('Total Wastage per Musim', fontsize=9, fontweight='bold', pad=8)
        ax_a3.grid(axis='x', alpha=0.3)
        ax_a3.spines[['top', 'right']].set_visible(False)
        plt.tight_layout(); st.pyplot(fig_a3); plt.close()


# ═══════════════════════════════════════════════════════════════════
# TAB 2 — AI PREDICTION (no duplicate filters — uses global sidebar)
# ═══════════════════════════════════════════════════════════════════
with tab2:

    st.markdown("<h3 class='tab2-main-title'>🤖 Real-time Surplus & Food Sales Predictor</h3>", unsafe_allow_html=True)
    st.markdown(
        "<span style='color: #64748b; font-size: 0.95rem; font-family: \"Inter\", sans-serif;'>"
        "Masukkan parameter operasional untuk melihat proyeksi penjualan dan rekomendasi AI (XGBoost Engine).</span>",
        unsafe_allow_html=True
    )

    # Form Input Utama
    with st.form("ai_prediction_form", border=False):

        # SECTION 1: Profil Restoran & Menu
        st.markdown('<div class="modern-header">Profil Restoran & Menu</div>', unsafe_allow_html=True)
        col1, col2, col3 = st.columns(3)

        with col1:
            input_id   = st.number_input("Restaurant ID", min_value=1, value=1, step=1,
                                         help="ID unik cabang restoran.")
            # Default-kan ke filter global (ambil nilai pertama yang terpilih, fallback ke list lengkap)
            default_rest_type = selected_rest[0] if selected_rest else "Casual Dining"
            input_type = st.selectbox(
                "Tipe Restoran",
                options=rest_types_all,
                index=rest_types_all.index(default_rest_type) if default_rest_type in rest_types_all else 0
            )

        with col2:
            input_menu = st.text_input("Nama Menu Item", value="Chicken Chop", placeholder="Cth: Nasi Goreng")
            default_meal = selected_meal[0] if selected_meal else "Dinner"
            input_meal = st.selectbox(
                "Waktu Sajian (Meal Type)",
                options=meal_types_all,
                index=meal_types_all.index(default_meal) if default_meal in meal_types_all else 0
            )

        with col3:
            default_wx = selected_weather[0] if selected_weather else "Sunny"
            input_weather_ai = st.selectbox(
                "Kondisi Cuaca (Weather)",
                options=weather_all,
                index=weather_all.index(default_wx) if default_wx in weather_all else 0
            )

        # SECTION 2: Struktur Finansial
        st.markdown('<div class="modern-header">Struktur Finansial (IDR)</div>', unsafe_allow_html=True)
        col4, col5, col6 = st.columns(3)

        with col4:
            input_cost    = st.number_input("Biaya Bahan Baku (HPP)", min_value=0, value=15000, step=1000)
        with col5:
            input_market  = st.number_input("Harga Pasar (Market Price)", min_value=0, value=35000, step=1000)
        with col6:
            input_selling = st.number_input("Harga Jual (Selling Price)", min_value=0, value=30000, step=1000)

        # SECTION 3: Kalender & Promosi
        st.markdown('<div class="modern-header">Promosi & Kalender Operasional</div>', unsafe_allow_html=True)
        col7, col8 = st.columns([1, 1])

        with col7:
            st.markdown(
                "<div style='margin-bottom:8px; font-weight:600; color:#0f172a; font-size:0.85rem; "
                "font-family:\"Inter\", sans-serif;'>Event & Campaign</div>",
                unsafe_allow_html=True
            )
            input_promo = st.toggle("Aktifkan Promosi / Diskon", value=True)
            input_event = st.toggle("Special Event / Hari Libur Besar", value=False)

        with col8:
            st.markdown(
                "<div style='margin-bottom:8px; font-weight:600; color:#0f172a; "
                "font-size:0.85rem; font-family:\"Inter\", sans-serif;'>"
                "Parameter Waktu</div>",
                unsafe_allow_html=True
            )

            # Hari Operasional
            hari_options = {
                "Senin": 0,
                "Selasa": 1,
                "Rabu": 2,
                "Kamis": 3,
                "Jumat": 4,
                "Sabtu": 5,
                "Minggu": 6
            }

            selected_hari = st.selectbox(
                "Hari Operasional",
                options=list(hari_options.keys()),
                index=4  # Default Jumat
            )

            input_dow = hari_options[selected_hari]

            # Bulan Operasional
            bulan_options = {
                "Januari": 1,
                "Februari": 2,
                "Maret": 3,
                "April": 4,
                "Mei": 5,
                "Juni": 6,
                "Juli": 7,
                "Agustus": 8,
                "September": 9,
                "Oktober": 10,
                "November": 11,
                "Desember": 12
            }

            current_month = datetime.datetime.now().month

            selected_bulan = st.selectbox(
                "Bulan Operasional",
                options=list(bulan_options.keys()),
                index=current_month - 1
            )

            input_month = bulan_options[selected_bulan]

            # Auto-generate weekend flag
            input_weekend = 1 if input_dow >= 5 else 0

        st.markdown("<br>", unsafe_allow_html=True)
        submit_button = st.form_submit_button("Generate AI Prediction", use_container_width=True)

    # ─────────────────────────────────────────────
    # PROCESS POST REQUEST TO FASTAPI & DISPLAY RESULT
    # ─────────────────────────────────────────────
    if submit_button:
        with st.spinner("AI Engine sedang memproses data..."):
            time.sleep(0.5)

            try:
                payload = {
                    "restaurant_id":          int(input_id),
                    "restaurant_type":         str(input_type),
                    "menu_item_name":          str(input_menu),
                    "meal_type":               str(input_meal),
                    "typical_ingredient_cost": float(input_cost),
                    "observed_market_price":   float(input_market),
                    "actual_selling_price":    float(input_selling),
                    "weather_condition":       str(input_weather_ai),
                    "has_promotion":           bool(input_promo),
                    "special_event":           bool(input_event),
                    "day_of_week":             int(input_dow),
                    "is_weekend":              int(input_weekend),
                    "month":                   int(input_month)
                }

                response = requests.post(AI_SERVICE_URL, json=payload, timeout=10)

                if response.status_code == 200:
                    st.toast("Prediksi berhasil di-generate!")
                    result = response.json()

                    pred_qty         = result.get("predicted_sales_qty", 0)
                    status_surplus   = result.get("status", "Unknown")
                    recommendation   = result.get("recommendation", "Tidak ada rekomendasi.")
                    r2_val           = result.get("model_r2_score", "N/A")

                    st.markdown("<hr style='border-color: #e2e8f0; margin: 2rem 0;'>", unsafe_allow_html=True)

                    res_col1, res_col2 = st.columns([1, 1.5])

                    with res_col1:
                        if status_surplus == "Surplus Kritis":
                            badge_class = "badge-danger"
                        elif status_surplus == "Waspada":
                            badge_class = "badge-warn"
                        else:
                            badge_class = "badge-safe"

                        st.markdown(f"""
                        <div class="result-card">
                            <div class="result-title">Estimasi Volume Terjual</div>
                            <div class="result-value">{pred_qty} <span style="font-size:1.2rem; color:#94a3b8; font-weight:500; font-family:'Inter', sans-serif;">Porsi</span></div>
                            <span class="{badge_class}">{status_surplus.upper()}</span>
                        </div>
                        """, unsafe_allow_html=True)

                    with res_col2:
                        st.markdown(f"""
                        <div class="result-card" style="background: #f8fafc;">
                            <div class="result-title" style="color: #0f172a;">AI Insight & Tindakan</div>
                            <p style="font-size: 1rem; color: #334155; margin-top: 12px; margin-bottom: 20px; line-height: 1.6; font-family:'Inter', sans-serif;">
                                {recommendation}
                            </p>
                            <div style="font-size: 0.75rem; color: #94a3b8; font-family: monospace;">
                                ✨ Powered by XGBoost Regressor (Akurasi R²: {r2_val})
                            </div>
                        </div>
                        """, unsafe_allow_html=True)

                else:
                    st.toast("Gagal menghubungi server!", icon="❌")
                    st.error(f"Error {response.status_code}: {response.text}")

            except requests.exceptions.ConnectionError:
                st.toast("Koneksi ke Microservice terputus!", icon="⚠️")
                st.error("❌ **Gagal Terhubung ke AI Service**")
                st.info(
                    "💡 **Solusi:** Pastikan Uvicorn server sudah running:\n"
                    "```bash\nuvicorn app:app --host 0.0.0.0 --port 8000 --reload\n```"
                )
            except Exception as e:
                st.error(f"Terjadi kesalahan internal: {str(e)}")


# ─────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────
st.markdown("<br><br>", unsafe_allow_html=True)
st.markdown("""
<div style="text-align:center; padding:20px; border-top:1px solid #D5E8F3; font-size:0.75rem; color:#AEB6BF;">
    ♻️ <strong style="color:#5D6D7E;">KeduaKali</strong> · Platform Cerdas Penyelamat Makanan Leftover & Barang Imperfect<br>
    <span style="margin-top:4px; display:block;">Capstone CC26-PSU226 · Sustainable & Responsible Consumption · 2026</span>
</div>
""", unsafe_allow_html=True)