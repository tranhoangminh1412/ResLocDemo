import streamlit as st
import pandas as pd
import numpy as np
import folium
import DataPipeline
import dictionaries
import time
from geopy.geocoders import Nominatim
from geopy.extra.rate_limiter import RateLimiter

def escape_dollars(text):
    if isinstance(text, str):
        return text.replace("$", "\\$")
    return text


GEOCODE_CACHE_PATH = "geocode_cache.csv"


st.set_page_config(
    page_title="Restaurant Site Selector",
    layout="wide"
)

st.markdown("""
<style>
/* Collapse the ghost vertical padding block created after rerun */
div[data-testid="stVerticalBlock"] > div:nth-child(1) {
    height: 0 !important;
    min-height: 0 !important;
    padding: 0 !important;
    margin: 0 !important;
    overflow: hidden !important;
}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<style>
/* Remove Streamlit auto-padding ABOVE BOTH columns */
div[data-testid="column"] > div:first-child {
    margin-top: 0 !important;
    padding-top: 0 !important;
}

/* Remove internal vertical-blocks that push results downward */
div[data-testid="column"] div[data-testid="stVerticalBlock"] > div:first-child {
    height: 0 !important;
    padding: 0 !important;
    margin: 0 !important;
}

/* Left = sticky map */
.sticky-map {
    position: sticky;
    top: 0;
    height: calc(100vh - 110px);
    overflow: hidden;
}

/* Right = scroll panel */
.scroll-panel {
    height: calc(100vh - 110px);
    overflow-y: auto;
    padding-right: 12px;
}
</style>
""", unsafe_allow_html=True)


try:
    import folium
    from streamlit_folium import st_folium
    _FOLIUM_OK = True
except Exception:
    _FOLIUM_OK = False


# =============================
#     GEOCODING HELPERS
# =============================

def _make_geolocator(user_agent="resloc_app"):
    return Nominatim(user_agent=user_agent, timeout=10)


def geocode_addresses(
    df,
    address_col="full_address",
    cache_path=GEOCODE_CACHE_PATH,
    user_agent="resloc_app",
    show_progress=True,
):
    try:
        cache = pd.read_csv(cache_path, dtype=str).set_index("address")
    except Exception:
        cache = pd.DataFrame(columns=["address", "lat", "lon"]).set_index("address")

    geolocator = _make_geolocator(user_agent=user_agent)
    geocode_limited = RateLimiter(
        geolocator.geocode,
        min_delay_seconds=1,
        max_retries=2,
        error_wait_seconds=2.0,
    )

    addresses = df[address_col].astype(str).fillna("").unique().tolist()
    to_query = [a for a in addresses if a not in cache.index]

    total = len(to_query)
    progress_bar = st.progress(0.0) if (show_progress and total > 0) else None

    for i, addr in enumerate(to_query, start=1):
        try:
            res = geocode_limited(addr)
            if res:
                lat, lon = float(res.latitude), float(res.longitude)
            else:
                lat, lon = None, None
        except Exception:
            lat, lon = None, None

        cache.loc[addr] = {"lat": lat, "lon": lon}

        if progress_bar:
            progress_bar.progress(i / max(1, total))

    try:
        cache.reset_index().to_csv(cache_path, index=False)
    except Exception:
        pass

    merged = df.copy()
    merged["address"] = merged[address_col].astype(str)
    merged = merged.merge(
        cache.reset_index(), how="left", left_on="address", right_on="address"
    )
    merged["lat"] = pd.to_numeric(merged["lat"], errors="coerce")
    merged["lon"] = pd.to_numeric(merged["lon"], errors="coerce")

    return merged


# =============================
# 1) LOAD BASE DATA
# =============================

@st.cache_data
def get_site_level_df():
    site_level_df, _, _ = DataPipeline.data_cleaning()
    return site_level_df


site_level_df = get_site_level_df()
Avg_SqFt_range = site_level_df["Avg_SqFt"].agg(["min", "max"])


# =============================
# 2) SIDEBAR INPUTS
# =============================

st.sidebar.title("Your Restaurant Preferences")

# ---- Weights ---- #
st.sidebar.subheader("Step 1 – What matters most?")

demo_raw = st.sidebar.slider("Demographics importance", 0.0, 1.0, 0.4, 0.05)
comp_raw = st.sidebar.slider("Competition importance", 0.0, 1.0, 0.4, 0.05)
site_raw = st.sidebar.slider("Site (rent & size) importance", 0.0, 1.0, 0.2, 0.05)

total_raw = demo_raw + comp_raw + site_raw
if total_raw == 0:
    Demo_weight = 0.4
    Comp_weight = 0.25
    Site_weight = 0.35
else:
    Demo_weight = demo_raw / total_raw
    Comp_weight = comp_raw / total_raw
    Site_weight = site_raw / total_raw

weights = {
    "Demo_weight": Demo_weight,
    "Comp_weight": Comp_weight,
    "Site_weight": Site_weight,
}

# ---- Cuisine ---- #
st.sidebar.subheader("Step 2 – Restaurant type")

Restaurant_Types = list(dictionaries.asian_food_keywords.keys())
Restaurant_type_input = st.sidebar.selectbox(
    "Cuisine (type of Asian restaurant)",
    options=Restaurant_Types,
    index=Restaurant_Types.index("Vietnamese") if "Vietnamese" in Restaurant_Types else 0,
)

# ---- Site Size ---- #
st.sidebar.subheader("Step 3 – Site size preference")

min_sqft = int(Avg_SqFt_range["min"])
max_sqft = int(Avg_SqFt_range["max"])
default_sqft = int((min_sqft + max_sqft) / 2)

Avg_SqFt_input = st.sidebar.slider(
    "Preferred average square feet",
    min_value=min_sqft,
    max_value=max_sqft,
    value=default_sqft,
    step=50,
)

# ---- Demographics ---- #
st.sidebar.subheader("Step 4 – Area demographics")

Percentage_Asian_choice = st.sidebar.selectbox(
    "Market saturation (Asian population share)", ["Low", "Medium", "High"], index=2
)
Median_INCTOT_choice = st.sidebar.selectbox(
    "Typical income level", ["Low", "Medium", "High"], index=1
)
Median_Yearly_Population_choice = st.sidebar.selectbox(
    "Location type (population size)", ["Low", "Medium", "High"], index=1
)

inputs = {
    "Restaurant_type_input": Restaurant_type_input,
    "Avg_SqFt_input": Avg_SqFt_input,
    "Percentage_Asian_input": Percentage_Asian_choice.lower(),
    "Median_INCTOT_input": Median_INCTOT_choice.lower(),
    "Median_Yearly_Population_input": Median_Yearly_Population_choice.lower(),
}


# =============================
# 3) MAIN LOGIC + GEOCODING
# =============================

st.title("Restaurant Site Selector")
st.write("We help you find the best rental locations in California for your new Asian restaurant!")

run_button = st.button("Find Best Locations")

if "final_results" not in st.session_state:
    st.session_state["final_results"] = None

if run_button:
    with st.spinner("Finding best sites based on your preferences…"):
        results = DataPipeline.calculate_final_scores(weights, inputs)

        if results is not None and not results.empty:
            top50 = results.head(50).copy()
            geo_top50 = geocode_addresses(
                top50, address_col="full_address", show_progress=True
            )

            results_geo = results.merge(
                geo_top50[["full_address", "lat", "lon"]],
                on="full_address",
                how="left",
            )

            st.session_state["final_results"] = results_geo
        else:
            st.session_state["final_results"] = results

final_merged_df_sorted = st.session_state["final_results"]

if final_merged_df_sorted is None:
    st.info("Set your preferences then click 'Find Best Locations'.")
    st.stop()

if final_merged_df_sorted.empty:
    st.warning("No sites matched. Try adjusting filters.")
    st.stop()

# =============================
# 4) FIXED MAP + SCROLLABLE RESULTS LAYOUT
# =============================

# =============================
# MAP RENDER FUNCTION
# =============================
def render_map_from_results(df, center=(36.7783, -119.4179), zoom_start=6):
    fmap = folium.Map(location=center, zoom_start=zoom_start, tiles="OpenStreetMap")

    lat_col = next((c for c in ["lat", "Lat", "latitude"] if c in df.columns), None)
    lon_col = next((c for c in ["lon", "Lon", "lng", "longitude"] if c in df.columns), None)

    if lat_col is None or lon_col is None:
        return fmap

    score_min = df["fit_score"].min()
    score_max = df["fit_score"].max()
    score_range = max(score_max - score_min, 1e-6)

    for _, r in df.iterrows():
        if pd.isna(r[lat_col]) or pd.isna(r[lon_col]):
            continue

        marker_id = int(r["marker_id"])

        radius = 6 + max(0, (r["fit_score"] - score_min) / score_range * 10)

        marker = folium.CircleMarker(
            location=[float(r[lat_col]), float(r[lon_col])],
            radius=radius,
            color="#2e7cff",
            fill=True,
            fill_color="#2e7cff",
            fill_opacity=0.9
        )

        marker.add_to(fmap)

        # Expose markers to JS
        marker.get_root().html.add_child(folium.Element(f"""
            <script>
                if (window.markerMap === undefined) {{
                    window.markerMap = {{}};
                }}
                window.markerMap["marker-{marker_id}"] = {{
                    marker: document.getElementsByClassName('leaflet-interactive')[document.getElementsByClassName('leaflet-interactive').length-1]
                }};
            </script>
        """))

    try:
        fmap.fit_bounds(df[[lat_col, lon_col]].dropna().values.tolist(), padding=(30,30))
    except:
        pass

    return fmap

# ---- Custom CSS layout ----
st.markdown("""
<style>
    .gmaps-container {
        display: flex;
        height: calc(100vh - 150px);  /* full height minus top Streamlit header */
        width: 100%;
        overflow: hidden;
        margin-top: 10px;
    }

    .map-panel {
        flex: 3;
        height: 100%;
        position: sticky;
        top: 0;
        overflow: hidden;
    }

    .results-panel {
        flex: 2;
        height: 100%;
        overflow-y: scroll;
        padding: 10px 18px;
        background-color: #111;
        color: white;
    }

    /* --- Cards --- */
    .result-card {
        background: #1e1e1e;
        border-radius: 14px;
        padding: 18px;
        margin-bottom: 16px;
        border: 1px solid #333;
        box-shadow: 0 2px 8px rgba(0,0,0,0.55);
        transition: .2s;
    }
    .result-card:hover {
        transform: translateY(-3px);
        border-color: #4e8cff;
        background: #272727;
    }
    .best-tag {
        background: #2e7cff;
        color: #fff;
        padding: 3px 7px;
        font-size: 11px;
        border-radius: 6px;
        margin-left: 6px;
    }
</style>
""", unsafe_allow_html=True)

# =========================================================
# 5) FIXED MAP + SCROLLABLE RESULTS USING STREAMLIT COLUMNS
# =========================================================

# Assign unique IDs
final_merged_df_sorted = final_merged_df_sorted.copy()
final_merged_df_sorted["marker_id"] = range(1, len(final_merged_df_sorted) + 1)

# Build map
fmap = render_map_from_results(final_merged_df_sorted)

# Create two columns: LEFT map, RIGHT cards
left_col, right_col = st.columns([3, 2], gap="small")

# Force both columns to align from very top
st.markdown("""
<style>
/* Reset Streamlit padding/margins that push content downward */
section.main > div {
    padding-top: 0 !important;
    margin-top: 0 !important;
}

/* Column padding fix */
div[data-testid="column"] > div:nth-child(1) {
    padding-top: 0 !important;
    margin-top: 0 !important;
}

/* Make BOTH column wrappers sticky at top */
.sticky-col {
    position: sticky;
    top: 0;
    height: calc(100vh - 120px);
    overflow: hidden;
}

/* Scrollable right column */
.scrollable-panel {
    height: calc(100vh - 120px);
    overflow-y: auto;
    padding-right: 10px;
}
</style>
""", unsafe_allow_html=True)

# -------------------------
# LEFT: MAP (fixed height)
# -------------------------
with left_col:
    st.markdown('<div class="sticky-map">', unsafe_allow_html=True)
    st_folium(fmap, height=750, width=None)
    st.markdown('</div>', unsafe_allow_html=True)

# -------------------------
# RIGHT: SCROLLABLE RESULTS
# -------------------------
with right_col:
    st.markdown('<div class="sticky-col">', unsafe_allow_html=True)
    st.markdown('<div class="scrollable-panel">', unsafe_allow_html=True)

    st.markdown("## 📍 Best Matches")

    top_n = final_merged_df_sorted.head(10)

    for idx, (_, row) in enumerate(top_n.iterrows()):
        best_tag = "<span class='best-tag'>BEST</span>" if idx == 0 else ""
        price = "$" + str(row["Price"]).lstrip("$")

        min_sqft = row.get("Min_SqFt", row["Avg_SqFt"])
        max_sqft = row.get("Max_SqFt", row["Avg_SqFt"])
        sqft_text = (
            f"{int(min_sqft):,} SF"
            if int(min_sqft) == int(max_sqft)
            else f"{int(min_sqft):,}–{int(max_sqft):,} SF"
        )

        st.markdown(f"""
            <div class="result-card" id="card-{row['marker_id']}">
                <h4>🍜 {row['Name']} {best_tag}</h4>
                <p><strong>🏷️ {price} • {sqft_text}</strong></p>
                <p>📍 {row['full_address']}</p>
            </div>
        """, unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)

# -------------------------
# HOVER SYNC JS
# -------------------------
st.markdown("""
<script>
function highlight(id){
    const el = window.markerMap?.["marker-"+id]?.marker;
    if(!el) return;
    el.style.stroke = "yellow";
    el.style.strokeWidth = "4px";
}
function reset(id){
    const el = window.markerMap?.["marker-"+id]?.marker;
    if(!el) return;
    el.style.stroke = "";
    el.style.strokeWidth = "";
}

setTimeout(() => {
    document.querySelectorAll("[id^='card-']").forEach(card => {
        const id = card.id.replace("card-", "");
        card.onmouseenter = () => highlight(id);
        card.onmouseleave = () => reset(id);
    });
}, 1000);
</script>
""", unsafe_allow_html=True)


# =============================
# 6) TABLE + DOWNLOAD
# =============================

st.subheader("Detailed Table (Top 50)")
st.dataframe(final_merged_df_sorted.head(50), use_container_width=True)

csv = final_merged_df_sorted.to_csv(index=False).encode("utf-8")
st.download_button("Download CSV", csv, "scored_sites.csv")
