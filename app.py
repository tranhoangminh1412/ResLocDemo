# Streamlit demo: Restaurant Location Selector — Simplified for Non-technical Users
# -------------------------------------------------
# Simplified, user-friendly demo. Assumptions:
# - Map uses OpenStreetMap (folium) only.
# - Data is provided by the backend in production. For the demo we generate synthetic
#   California-centered data so the map shows CA.
# - Weight adjustments are simplified to *three* group sliders (Demographic, Competition,
#   Site-specific). No per-feature sliders, no advanced map settings.
# - UI is intentionally simple for middle → upper-middle aged restaurant owners.
# -------------------------------------------------

import os
import numpy as np
import pandas as pd
import streamlit as st
from io import StringIO

# Try to import folium + streamlit_folium. If unavailable, show friendly message.
try:
    import folium
    from streamlit_folium import st_folium
    _FOLIUM_OK = True
except Exception:
    _FOLIUM_OK = False

st.set_page_config(page_title="Restaurant Location Selector — Simple Demo", layout="wide")

# -------------------------
# 1) Demo data generator (California-centered)
# -------------------------

def make_demo_data(seed: int = 7, n: int = 40,
                   center=(36.7783, -119.4179)) -> pd.DataFrame:
    """Generate synthetic CA-centered candidates. In production, backend will supply real data.
    """
    rng = np.random.default_rng(seed)
    lat0, lon0 = center
    lat = lat0 + (rng.normal(0, 0.6, n))  # wider spread to cover CA
    lon = lon0 + (rng.normal(0, 1.4, n))

    population_density = rng.normal(3000, 1200, n).clip(300, 15000)
    median_income = rng.normal(70000, 18000, n).clip(20000, 200000)
    asian_pop_share = rng.uniform(0.01, 0.4, n)
    foot_traffic = rng.normal(1500, 700, n).clip(50, 10000)
    competitor_count = rng.poisson(5, n) + rng.integers(0, 4, n)
    rent_per_sqft = rng.normal(30, 10, n).clip(8, 150)
    crime_index = rng.normal(45, 15, n).clip(5, 100)
    rating = rng.normal(4.0, 0.4, n).clip(2.5, 5.0)
    growth_index = rng.normal(1.03, 0.07, n).clip(0.8, 1.4)
    size_sqft = rng.integers(600, 4000, n)
    parking = rng.choice([0, 1], size=n, p=[0.4, 0.6])
    cuisine = rng.choice(["Asian", "Italian", "Mexican", "American", "Coffee"], size=n)

    df = pd.DataFrame({
        "name": [f"Candidate #{i+1}" for i in range(n)],
        "lat": lat,
        "lon": lon,
        "population_density": population_density,
        "median_income": median_income,
        "asian_pop_share": asian_pop_share,
        "foot_traffic": foot_traffic,
        "competitor_count": competitor_count,
        "rent_per_sqft": rent_per_sqft,
        "crime_index": crime_index,
        "rating": rating,
        "growth_index": growth_index,
        "size_sqft": size_sqft,
        "parking": parking,
        "cuisine": cuisine,
    })
    return df

# In production this variable will be filled by the backend. For demo, generate CA data.
data = make_demo_data()

# ------------------------------------
# 2) Very simple staged inputs (wizard-style hints)
# ------------------------------------
st.sidebar.header("Step 1 — Demographic")
population_level = st.sidebar.selectbox("Location type", ["Small town", "Suburban", "City"], index=2)
market_saturation = st.sidebar.selectbox("Market saturation", ["Developing", "Mature"], index=0)
preferred_cuisines = st.sidebar.multiselect("Cuisine types (pick one or two)", sorted(data["cuisine"].unique()), default=["Asian"]) 

st.sidebar.header("Step 2 — Competition")
competition_sensitivity = st.sidebar.selectbox("How much do you care about nearby competitors?", ["Not much", "Somewhat", "Very much"], index=1)

st.sidebar.header("Step 3 — Site preferences")
rent_min, rent_max = st.sidebar.slider("Desired rent $/sqft (range)", 5, 200, (5, 60))
size_min, size_max = st.sidebar.slider("Wanted size (sqft)", 200, 5000, (800, 2000))
parking_pref = st.sidebar.checkbox("Prefer listings with more parking", value=False)

# ------------------------------------
# 3) Simple weight controls: only three group sliders
# ------------------------------------
st.sidebar.header("Adjust what matters most")
st.sidebar.markdown("Use these three sliders to say what matters most: Demographics, Competition, or Site factors.")
st.sidebar.markdown("<div class='step-badge'><div class='step-number'>1</div><div style='margin-left:6px'><div class='step-label'>Demographics</div><div style='font-size:12px;color:#3f536b'></div></div></div>", unsafe_allow_html=True)
group_demo_mult = st.sidebar.slider("How important is having a big audience closeby matter to you?", 0.0, 2.0, 1.0, 0.05)
st.sidebar.markdown("<div class='step-badge'><div class='step-number'>2</div><div style='margin-left:6px'><div class='step-label'>Competition</div><div style='font-size:12px;color:#3f536b'>Set how much nearby competitors matter</div></div></div>", unsafe_allow_html=True)
group_comp_mult = st.sidebar.slider("How much do you care about having nearby competitors?", 0.0, 2.0, 1.0, 0.05)
st.sidebar.markdown("<div class='step-badge'><div class='step-number'>3</div><div style='margin-left:6px'><div class='step-label'>Site</div><div style='font-size:12px;color:#3f536b'>Pick size, rent, and parking preference</div></div></div>", unsafe_allow_html=True)
group_site_mult = st.sidebar.slider("How important is the site's specifications matching your needs?", 0.0, 2.0, 1.0, 0.05)


# Build simple weight dictionary from group multipliers
# We map groups to the original feature set but only via these three multipliers.
BASE_WEIGHTS = {
    "population_density": (1.0, +1),
    "median_income": (1.0, +1),
    "asian_pop_share": (1.0, +1),
    "competitor_count": (1.0, -1),
    "rent_per_sqft": (1.0, -1),
    "foot_traffic": (1.0, +1),
    "crime_index": (1.0, -1),
    "growth_index": (1.0, +1),
}

WEIGHTS = {}
for k, (base, direction) in BASE_WEIGHTS.items():
    if k in ["population_density", "median_income", "asian_pop_share"]:
        w = base * group_demo_mult
    elif k in ["competitor_count", "rent_per_sqft"]:
        w = base * group_comp_mult
    else:
        w = base * group_site_mult
    WEIGHTS[k] = (w, direction)

# ------------------------------------
# 4) Score helpers
# ------------------------------------

def zscore(s: pd.Series) -> pd.Series:
    mu, sigma = s.mean(), s.std(ddof=0)
    if sigma == 0:
        return pd.Series(0.0, index=s.index)
    return (s - mu) / sigma


def compute_scores(df: pd.DataFrame, weights: dict,
                   rent_min=0, rent_max=1e9, size_min=0, size_max=1e9, parking_pref=False, cuisines=None):
    df2 = df.copy()
    # Apply simple filters from steps (these are friendly and minimal)
    df2 = df2[(df2["rent_per_sqft"] >= rent_min) & (df2["rent_per_sqft"] <= rent_max)]
    if "size_sqft" in df2.columns:
        df2 = df2[(df2["size_sqft"] >= size_min) & (df2["size_sqft"] <= size_max)]
    if parking_pref and "parking" in df2.columns:
        df2 = df2[df2["parking"] == 1]
    if cuisines:
        if "cuisine" in df2.columns:
            df2 = df2[df2["cuisine"].isin(cuisines)]

    score_parts = []
    for col, (w, direction) in weights.items():
        if col not in df2.columns:
            continue
        zs = zscore(df2[col]) * direction
        score_parts.append(w * zs)
        df2[f"contrib_{col}"] = w * zs

    df2["score"] = np.sum(score_parts, axis=0) if score_parts else 0.0
    df2 = df2.sort_values("score", ascending=False)
    return df2

scored = compute_scores(data, WEIGHTS, rent_min=rent_min, rent_max=rent_max, size_min=size_min, size_max=size_max, parking_pref=parking_pref, cuisines=preferred_cuisines)

# ------------------------------------
# 5) UI: headline and short explanation
# ------------------------------------

# Add custom background + styling to make UI cleaner and more official
st.markdown(
    """
    <style>
    /* Set full-page background */
    .main {
        background-color: #ffffff !important;
    }

    /* Sidebar background */
    section[data-testid="stSidebar"] {
        background-color: #000000 !important;
    }

    /* Make headers more professional */
    h1, h2, h3, h4, h5, h6 {
        color: #ffffff !important;
        font-family: 'Segoe UI', sans-serif !important;
    }

    /* Improve table readability */
    .stDataFrame {
        background-color: black !important;
    }

    /* Improve button look */
    .stButton>button {
        background-color: #2e7cff;
        color: white;
        border-radius: 6px;
        padding: 0.6rem 1.2rem;
        border: none;
    }
    .stButton>button:hover {
        background-color: #1b5fd6;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# App header + polished UI
# Use a compact banner, clear numbering, high-contrast text, and icons for visual cues
st.markdown(
    """
    <style>
    /* Banner */
    .app-banner {
        display: flex;
        align-items: center;
        gap: 16px;
        padding: 18px 24px;
        background: linear-gradient(90deg, #ffffff 0%, #f8fbff 100%);
        border-radius: 8px;
        box-shadow: 0 6px 20px rgba(0,0,0,0.06);
        margin-bottom: 18px;
    }
    .app-logo {
        width: 72px;
        height: 72px;
        border-radius: 10px;
        background: linear-gradient(180deg,#2e7cff,#1b5fd6);
        display:flex;align-items:center;justify-content:center;color:white;font-weight:700;font-size:28px
    }
    .app-title { font-size: 22px; color: #0b243b; margin:0; font-weight:700; }
    .app-sub { color:#41515f; margin:0; font-size:13px }

    /* Sidebar step badges */
    .step-badge { display:inline-flex; align-items:center; gap:8px; padding:8px 10px; border-radius:8px; background:#ffffff; box-shadow:0 2px 6px rgba(0,0,0,0.04); margin-bottom:8px }
    .step-number { background:#2e7cff; color:white; font-weight:700; padding:6px 10px; border-radius:6px }
    .step-label { font-weight:600; color:black }

    /* Increase contrast and size for main headings */
    .stHeader { color: #000000 !important; font-size:18px !important; }

    /* Make table text larger for readability */
    .stDataFrame table td, .stDataFrame table th { font-size: 14px !important }
    </style>

    <div class="app-banner">
        <div class="app-logo">RS</div>
        <div>
            <div class="app-title">Restaurant Location Selector</div>
            <div class="app-sub">A simple, official-looking tool for evaluating restaurant site candidates</div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# Sidebar: show numbered steps with icons and short captions (friendly language)


# Short official instruction copy for the main page
st.markdown("**How to use**: 1) Select a location type and cuisine. 2) Use the Competition slider to set importance. 3) Move the three blue sliders to say what matters most. Click a map marker to view details.", unsafe_allow_html=True)


# ------------------------------------
# 6) Map visualization (OpenStreetMap / Folium) centered on California
# ------------------------------------

if not _FOLIUM_OK:
    st.error("This demo requires folium and streamlit-folium. Please install: pip install folium streamlit-folium")
else:
    TOP_N = st.slider("How many top locations to show?", 5, 40, 12)
    center_lat, center_lon = 36.7783, -119.4179
    fmap = folium.Map(location=[center_lat, center_lon], zoom_start=6, tiles="OpenStreetMap")

    top_df = scored.head(TOP_N)
    for _, r in top_df.iterrows():
        popup_html = f"<b>{r['name']}</b><br>Score: {r['score']:.2f}<br>Rent: ${r['rent_per_sqft']:.2f}/sqft/mo<br>Traffic: {int(r['foot_traffic'])}<br>Competitors: {int(r['competitor_count'])}"
        folium.CircleMarker(
            location=[r["lat"], r["lon"]],
            radius=6 + 10 * ((r["score"] - top_df["score"].min()) / max(1e-6, top_df["score"].max() - top_df["score"].min())) if not top_df.empty else 6,
            fill=True,
            fill_opacity=0.8,
            popup=folium.Popup(popup_html, max_width=300),
        ).add_to(fmap)

    st_folium(fmap, width=900, height=600)

# ------------------------------------
# 7) Ranked table + simple explainability
# ------------------------------------

st.subheader("Top candidates")
show_cols = ["name", "score", "rent_per_sqft", "median_income", "population_density", "foot_traffic", "competitor_count", "size_sqft", "parking", "cuisine"]
st.dataframe(scored.head(25)[[c for c in show_cols if c in scored.columns]], use_container_width=True)

with st.expander("Explain a selected candidate"):
    options = list(scored["name"].head(25)) if not scored.empty else []
    pick = st.selectbox("Choose a location", options)
    if pick:
        row = scored.loc[scored["name"] == pick].iloc[0]
        st.markdown(f"**{pick}** — composite score: **{row['score']:.2f}**")
        contrib_cols = [c for c in scored.columns if c.startswith("contrib_")]
        explain_df = (row[contrib_cols].rename(lambda c: c.replace("contrib_", "")).to_frame(name="value")).sort_values("value", ascending=False)
        st.dataframe(explain_df)

# ------------------------------------
# 8) Download results
# ------------------------------------

csv = scored.to_csv(index=False).encode("utf-8")
st.download_button("Download scored candidates (CSV)", csv, file_name="scored_locations_demo.csv")

st.caption("This simple demo centers on California and uses only three sliders to keep things easy for non-technical users.")
