"""
app.py — Streamlit frontend for Customer Segmentation
Uses back.py as the backend engine.
"""

import streamlit as st
import plotly.graph_objects as go
import numpy as np

# ── Import backend ──────────────────────────────────────────────────────────
from back import (
    load_data,
    get_features,
    train_model,
    get_cluster_info,
    predict_customer,
    CLUSTER_COLORS,
)

# ── Page configuration ──────────────────────────────────────────────────────
st.set_page_config(
    page_title="Customer Segmentation · KMeans",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS for premium dark look ────────────────────────────────────────
st.markdown("""
<style>
/* ── global ─────────────────────────────────────────────── */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

/* dark background override */
.stApp {
    background: #0f172a; /* Solid dark blue/slate */
}

/* ── header banner ─────────────────────────────────────── */
.hero-banner {
    background: #4f46e5; /* Solid indigo */
    border-radius: 18px;
    padding: 2.2rem 2.6rem;
    margin-bottom: 1.8rem;
    box-shadow: 0 14px 40px rgba(102,126,234,.35);
}
.hero-banner h1 {
    color: #FFFFFF;
    font-size: 2.2rem;
    font-weight: 800;
    margin: 0 0 .3rem 0;
    letter-spacing: -0.5px;
}
.hero-banner p {
    color: rgba(255,255,255,.82);
    font-size: 1.05rem;
    margin: 0;
}

/* ── cluster info cards ────────────────────────────────── */
.cluster-card {
    background: rgba(255,255,255,0.06);
    backdrop-filter: blur(14px);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 16px;
    padding: 1.3rem 1.4rem;
    margin-bottom: 1rem;
    transition: transform .2s ease, box-shadow .2s ease;
}
.cluster-card:hover {
    transform: translateY(-4px);
    box-shadow: 0 12px 32px rgba(0,0,0,.25);
}
.cluster-dot {
    display: inline-block;
    width: 14px; height: 14px;
    border-radius: 50%;
    margin-right: 8px;
    vertical-align: middle;
    box-shadow: 0 0 8px currentColor;
}
.cluster-card h4 {
    color: #fff;
    font-size: 0.9rem;
    font-weight: 700;
    margin: 0 0 .45rem 0;
    white-space: nowrap;
}
.cluster-card p {
    color: rgba(255,255,255,.72);
    font-size: .88rem;
    margin: .15rem 0;
    line-height: 1.55;
}
.cluster-card .stat {
    color: rgba(255,255,255,.55);
    font-size: .78rem;
}

/* ── sidebar styling ───────────────────────────────────── */
section[data-testid="stSidebar"] {
    background: #1e293b; /* Solid slate */
}
section[data-testid="stSidebar"] h1,
section[data-testid="stSidebar"] h2,
section[data-testid="stSidebar"] h3,
section[data-testid="stSidebar"] label {
    color: #e0e0e0 !important;
}

/* prediction result box */
.pred-box {
    background: #1e1b4b; /* Solid dark indigo */
    border: 1.5px solid rgba(102,126,234,.45);
    border-radius: 16px;
    padding: 1.5rem;
    margin-top: .8rem;
    text-align: center;
}
.pred-box h3 {
    color: #667eea; font-weight: 700; margin: 0 0 .3rem 0;
}
.pred-box p {
    color: rgba(255,255,255,.78); font-size: .92rem; margin: .2rem 0;
}
.pred-dot-legend {
    display: inline-block;
    width: 12px; height: 12px;
    border-radius: 50%;
    background: #0a1128;
    border: 2px solid #667eea;
    margin-right: 6px;
    vertical-align: middle;
}

/* section headers */
.section-head {
    color: #e0e0e0;
    font-size: 1.25rem;
    font-weight: 700;
    margin: 1.6rem 0 .8rem 0;
    letter-spacing: -.3px;
}

/* button in-and-out animation */
@keyframes pulseButton {
    0% { transform: scale(1); }
    50% { transform: scale(1.02); }
    100% { transform: scale(1); }
}

.stButton button {
    animation: pulseButton 2s infinite ease-in-out;
    transition: all 0.2s ease !important;
}

.stButton button:hover {
    transform: none !important;
    border-color: #667eea !important;
    color: #667eea !important;
}
.stButton button:active {
    transform: scale(0.95) !important;
}
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# DATA & MODEL  (cached so it only runs once)
# ══════════════════════════════════════════════════════════════════════════════

@st.cache_data
def _load_and_train():
    df = load_data()
    x = get_features(df)
    kmeans, labels = train_model(x)
    info = get_cluster_info(x, labels, kmeans)
    return df, x, kmeans, labels, info

df, x, kmeans, labels, cluster_info = _load_and_train()

# Build a quick lookup: cluster_id → info dict
info_by_id = {ci["cluster_id"]: ci for ci in cluster_info}


# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR — Predict a New Customer
# ══════════════════════════════════════════════════════════════════════════════

with st.sidebar:
    st.markdown("## 🔮 Predict New Customer")
    st.markdown(
        "<p style='color:rgba(255,255,255,.55); font-size:.88rem;'>"
        "Enter a customer's details to see which segment they belong to.</p>",
        unsafe_allow_html=True,
    )

    income_input = st.number_input(
        "Annual Income (k$)", min_value=0, max_value=200, value=50, step=1
    )
    score_input = st.number_input(
        "Spending Score (1-100)", min_value=1, max_value=100, value=50, step=1
    )

    predict_btn = st.button("🚀  Predict Cluster", use_container_width=True)

    # Store prediction state
    if predict_btn:
        pred_id = predict_customer(kmeans, income_input, score_input)
        st.session_state["pred"] = {
            "income": income_input,
            "score": score_input,
            "cluster_id": pred_id,
        }

    # Show prediction result
    if "pred" in st.session_state:
        p = st.session_state["pred"]
        ci = info_by_id[p["cluster_id"]]
        st.markdown(
            f"""
            <div class="pred-box">
                <h3>Cluster Identified</h3>
                <p style="font-size:1.15rem; font-weight:700; color:#fff;">
                    <span class="cluster-dot" style="background:{ci['color']};color:{ci['color']};"></span>
                    {ci['label']}
                </p>
                <p>{ci['description']}</p>
                <p class="stat" style="margin-top:.6rem;">
                    <span class="pred-dot-legend"></span>
                    The dark-blue dot on the chart represents this customer.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("---")
    st.markdown(
        "<p style='color:rgba(255,255,255,.35); font-size:.75rem; text-align:center;'>"
        "Built with Streamlit · KMeans Clustering</p>",
        unsafe_allow_html=True,
    )


# ══════════════════════════════════════════════════════════════════════════════
# MAIN AREA
# ══════════════════════════════════════════════════════════════════════════════

# ── Hero Banner ──────────────────────────────────────────────────────────────
st.markdown(
    """
    <div class="hero-banner">
        <h1>🛍️ Customer Segmentation</h1>
        <p>KMeans clustering on Annual Income &amp; Spending Score —
           identify your most valuable customer groups at a glance.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# ── Scatter Chart ────────────────────────────────────────────────────────────
st.markdown('<p class="section-head">📊 Customer Clusters</p>', unsafe_allow_html=True)

fig = go.Figure()

# One trace per cluster (sorted by rank so legend is stable)
for ci in sorted(cluster_info, key=lambda c: c["rank"]):
    cid = ci["cluster_id"]
    mask = labels == cid
    fig.add_trace(go.Scatter(
        x=x[mask, 0],
        y=x[mask, 1],
        mode="markers",
        name=ci["label"],
        marker=dict(
            size=9,
            color=ci["color"],
            opacity=0.85,
            line=dict(width=0.6, color="rgba(255,255,255,0.25)"),
        ),
        hovertemplate=(
            "<b>%{text}</b><br>"
            "Income: %{x} k$<br>"
            "Spending: %{y}<extra></extra>"
        ),
        text=[ci["label"]] * int(mask.sum()),
    ))

# Centroids
fig.add_trace(go.Scatter(
    x=kmeans.cluster_centers_[:, 0],
    y=kmeans.cluster_centers_[:, 1],
    mode="markers",
    name="Centroids",
    marker=dict(
        size=14,
        color="#ffffff",
        symbol="x",
        line=dict(width=2, color="#000"),
    ),
    hovertemplate="Centroid<br>Income: %{x:.1f} k$<br>Spending: %{y:.1f}<extra></extra>",
))

# Predicted customer dot (dark blue, prominent)
if "pred" in st.session_state:
    p = st.session_state["pred"]
    fig.add_trace(go.Scatter(
        x=[p["income"]],
        y=[p["score"]],
        mode="markers",
        name="🆕 New Customer",
        marker=dict(
            size=16,
            color="#0a1128",
            symbol="circle",
            line=dict(width=3, color="#667eea"),
        ),
        hovertemplate=(
            "<b>New Customer</b><br>"
            f"Income: {p['income']} k$<br>"
            f"Spending: {p['score']}<extra></extra>"
        ),
    ))

fig.update_layout(
    template="plotly_dark",
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(15,12,41,0.6)",
    xaxis=dict(
        title="Annual Income (k$)",
        gridcolor="rgba(255,255,255,0.06)",
        zeroline=False,
    ),
    yaxis=dict(
        title="Spending Score (1-100)",
        gridcolor="rgba(255,255,255,0.06)",
        zeroline=False,
    ),
    legend=dict(
        orientation="h",
        yanchor="bottom",
        y=1.02,
        xanchor="center",
        x=0.5,
        font=dict(size=12, color="#ffffff"),
    ),
    margin=dict(l=40, r=20, t=60, b=45),
    height=540,
    font=dict(family="Inter, sans-serif"),
)

st.plotly_chart(fig, use_container_width=True, theme=None)


# ── Cluster Details Cards ────────────────────────────────────────────────────
st.markdown('<p class="section-head">📋 Cluster Breakdown</p>', unsafe_allow_html=True)

cols = st.columns(len(cluster_info))

for col, ci in zip(cols, sorted(cluster_info, key=lambda c: c["rank"])):
    with col:
        st.markdown(
            f"""
            <div class="cluster-card">
                <h4>
                    <span class="cluster-dot" style="background:{ci['color']};color:{ci['color']};"></span>
                    {ci['label']}
                </h4>
                <p>{ci['description']}</p>
                <p class="stat">
                    👥 {ci['size']} customers<br>
                    💵 Avg Income: {ci['avg_income']} k$<br>
                    🛒 Avg Spending: {ci['avg_spending']}
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

# ── Footer ───────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    "<p style='text-align:center; color:rgba(255,255,255,.25); font-size:.8rem;'>"
    "Customer Segmentation Dashboard · Powered by KMeans Clustering & Streamlit"
    "</p>",
    unsafe_allow_html=True,
)