import streamlit as st
import pandas as pd
import numpy as np
import html
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

/* Left column sticky */
.sticky-map {
    position: sticky;
    top: 0;
    height: calc(100vh - 110px);
    overflow: hidden;
}

/* Right column scroll area */
.scroll-panel {
    height: calc(100vh - 110px);
    overflow-y: auto;
    padding-right: 12px;
    margin-top: 0 !important;
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
    # ---- load cache (if exists) ----
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

    # helper: decide if we should geocode an address
    def needs_geocode(addr: str) -> bool:
        if addr not in cache.index:
            return True
        row = cache.loc[addr]
        lat = row.get("lat", None)
        lon = row.get("lon", None)
        # re-query if missing / empty
        if pd.isna(lat) or pd.isna(lon) or lat == "" or lon == "":
            return True
        return False

    addresses = df[address_col].astype(str).fillna("").unique().tolist()
    to_query = [a for a in addresses if needs_geocode(a)]

    total = len(to_query)
    progress_bar = st.progress(0.0) if (show_progress and total > 0) else None

    # ---- geocode missing addresses ----
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

    # ---- save cache ----
    try:
        cache.reset_index().to_csv(cache_path, index=False)
    except Exception:
        pass

    # ---- merge back onto df ----
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
def load_data():
    site_level_df, county_level_df, zip_level_df = DataPipeline.data_cleaning()
    return site_level_df, county_level_df, zip_level_df
site_level_df, county_level_df, zip_level_df = load_data()

# =============================
# 2) SIDEBAR INPUTS (PREMIUM UI)
# =============================

st.sidebar.markdown(
    """
    <style>
        .streamlit-expanderHeader {
            font-size: 1rem !important;
            font-weight: 600 !important;
        }
        .streamlit-expanderContent {
            background: #f7f7f7 !important;
            border-radius: 10px !important;
            padding: 12px !important;
        }
        hr {
            margin-top: 8px;
            margin-bottom: 12px;
            border: none;
            border-top: 1px solid #ddd;
        }
    </style>
    """,
    unsafe_allow_html=True
)

st.sidebar.title("🍽️ Your Restaurant Preferences")

# ============================================================
# ⭐ STEP 1 — IMPORTANCE WEIGHTS
# ============================================================
with st.sidebar.expander("⭐ Step 1 — What matters most to you?", expanded=True):

    st.markdown(
        "<small>Move the sliders to show what you care about most.</small>",
        unsafe_allow_html=True
    )

    demo_pct = st.slider(
        "👨‍👩‍👧‍👦 Customer Proximity",
        0, 100, 40,
        help="How important is it for your restaurant to be close to your target customers? Higher values prioritize locations with more nearby residents, workers, or foot traffic that matches your customer profile."
    )

    comp_pct = st.slider(
        "⚔️ Competition Avoidance",
        0, 100, 40,
        help="How strongly do you want to stay away from competing restaurants? A higher value increases the priority for areas with fewer similar businesses nearby."
    )

    site_pct = st.slider(
        "🏠 Site Quality / Rent Factor",
        0, 100, 20,
        help="How much do rental prices influence your decision? A higher value favors locations with lower rent costs, even if customer traffic or visibility is lower."
    )

# Convert % to normalized weights
total_pct = demo_pct + comp_pct + site_pct
if total_pct == 0:
    Demo_weight = 0.4
    Comp_weight = 0.25
    Site_weight = 0.35
else:
    Demo_weight = demo_pct / total_pct
    Comp_weight = comp_pct / total_pct
    Site_weight = site_pct / total_pct

weights = {
    "Demo_weight": Demo_weight,
    "Comp_weight": Comp_weight,
    "Site_weight": Site_weight,
}

st.sidebar.markdown("<hr>", unsafe_allow_html=True)

# ============================================================
# 🍜 STEP 2 — YOUR RESTAURANT (Cuisine + Size + Menu Price)
# ============================================================
with st.sidebar.expander("🍜 Step 2 — Your Restaurant", expanded=False):

    st.markdown(
        "<small>Tell us what kind of restaurant you want to build.</small>",
        unsafe_allow_html=True
    )

    # ------------------ Cuisine ------------------ #
    Restaurant_Types = list(dictionaries.asian_food_keywords.keys())

    Restaurant_type_input = st.selectbox(
        "🍱 Cuisine Type",
        options=Restaurant_Types,
        index=Restaurant_Types.index("Vietnamese") if "Vietnamese" in Restaurant_Types else 0,
        help="Choose your restaurant's main cuisine"
    )

    # ------------------ Size ------------------ #
    min_sqft = int(site_level_df["Avg_SqFt"].quantile(0.1))
    max_sqft = int(site_level_df["Avg_SqFt"].quantile(0.9))
    default_sqft = int(site_level_df["Avg_SqFt"].mean())

    def sqft_label(val):
        if val < 1600:
            return "🧋 Small shop (like boba stores)"
        elif val > 3200:
            return "🍽️ Large dining restaurant"
        else:
            return "🍜 Medium-sized casual restaurant"

    Avg_SqFt_input = st.slider(
        "📐 Preferred space (SqFt)",
        min_sqft, max_sqft, default_sqft,
        help="Select your ideal restaurant size"
    )
    st.caption(f"**{sqft_label(Avg_SqFt_input)}**")

    # ------------------ Menu Price ------------------ #
    Median_INCTOT_choice_raw = st.selectbox(
        "💵 Menu Price Level",
        [
            "Affordable (below $15)",
            "Average ($15 – $30)",
            "Costly (above $30)"
        ],
        index=1,
        help="Based on how much your customers typically spend per meal"
    )

    # Convert to low/medium/high
    if "Affordable" in Median_INCTOT_choice_raw:
        Median_INCTOT_choice = "low"
    elif "Average" in Median_INCTOT_choice_raw:
        Median_INCTOT_choice = "medium"
    else:
        Median_INCTOT_choice = "high"

st.sidebar.markdown("<hr>", unsafe_allow_html=True)

# ============================================================
# 🌎 STEP 3 — MARKET DEMOGRAPHICS (Asian Share + Population)
# ============================================================
with st.sidebar.expander("🌎 Step 3 — Your Market", expanded=False):

    st.markdown(
        "<small>Pick the type of area you want to open in.</small>",
        unsafe_allow_html=True
    )

    # ---- Asian Population Share (IQR-based labels) ---- #

    # Compute IQR thresholds
    asian_25 = county_level_df["Percentage_Asian"].quantile(0.25)
    asian_75 = county_level_df["Percentage_Asian"].quantile(0.75)

    asian_25_fmt = f"{asian_25:.1f}%"
    asian_75_fmt = f"{asian_75:.1f}%"

    Percentage_Asian_choice_raw = st.selectbox(
        "🧑‍🍳 Asian Population Share",
        [
            f"Low (below {asian_25_fmt})",
            f"Medium ({asian_25_fmt} – {asian_75_fmt})",
            f"High (above {asian_75_fmt})"
        ],
        index=1,
        help="Higher share means more customers familiar with Asian cuisine, which may help your restaurant."
    )

    # Convert readable label → low/medium/high
    if "Low" in Percentage_Asian_choice_raw:
        Percentage_Asian_choice = "low"
    elif "Medium" in Percentage_Asian_choice_raw:
        Percentage_Asian_choice = "medium"
    else:
        Percentage_Asian_choice = "high"

    # ------------------ Population ------------------ #
    pop_30 = county_level_df["Median_Yearly_Population"].quantile(0.3)
    pop_70 = county_level_df["Median_Yearly_Population"].quantile(0.7)

    pop_formatted_30 = f"{int(pop_30):,}"
    pop_formatted_70 = f"{int(pop_70):,}"

    Median_Yearly_Population_choice_raw = st.selectbox(
        "🏙️ Population Size",
        [
            f"Rural (below {pop_formatted_30} people)",
            f"Suburb ({pop_formatted_30}–{pop_formatted_70} people)",
            f"Metropolitan (above {pop_formatted_70} people)"
        ],
        index=1,
        help="Bigger populations bring more foot traffic"
    )

    if "Rural" in Median_Yearly_Population_choice_raw:
        Median_Yearly_Population_choice = "low"
    elif "Suburb" in Median_Yearly_Population_choice_raw:
        Median_Yearly_Population_choice = "medium"
    else:
        Median_Yearly_Population_choice = "high"

# ============================================================
# 🎯 FINAL INPUT PACKAGE
# ============================================================
inputs = {
    "Restaurant_type_input": Restaurant_type_input,
    "Avg_SqFt_input": Avg_SqFt_input,
    "Percentage_Asian_input": Percentage_Asian_choice,
    "Median_INCTOT_input": Median_INCTOT_choice,
    "Median_Yearly_Population_input": Median_Yearly_Population_choice,
}

# =============================
# 3) MAIN LOGIC + GEOCODING
# =============================

st.title("Restaurant Site Selector")
st.write("We help you find the best rental locations in California for your new Asian restaurant!")

# Freeze user inputs so the results don't change before clicking the button
if "frozen_inputs" not in st.session_state:
    st.session_state["frozen_inputs"] = None

run_button = st.button("Find Best Locations")

if "final_results" not in st.session_state:
    st.session_state["final_results"] = None

if run_button:
    # Store current user settings so they don't change until next click
    st.session_state["frozen_inputs"] = inputs.copy()
    with st.spinner("Finding best sites based on your preferences…"):
        # only keep top 10 in the pipeline result
        results = DataPipeline.calculate_final_scores(weights, inputs, top_n=50)

        if results is not None and not results.empty:
            top10 = results.head(10).copy()
            geo_top10 = geocode_addresses(
                top10, address_col="full_address", show_progress=True
            )

            results_geo = results.merge(
                geo_top10[["full_address", "lat", "lon"]],
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
        rank = int(r.get("rank", 9999))

        # base radius from score
        radius = 6 + max(0, (r["fit_score"] - score_min) / score_range * 10)

        # --- styling based on rank ---
        if rank == 1:
            color = "#D3AF37"   # gold
            radius = radius + 1.5
        elif rank == 2:
            color = "#C0C0C0"   # orange
            radius = radius + 1
        elif rank == 3:
            color = "#CD7F32"
            radius = radius + 0.5
        else:
            color = "#2e7cff"   # default blue

        marker = folium.CircleMarker(
            location=[float(r[lat_col]), float(r[lon_col])],
            radius=radius,
            color=color,
            fill=True,
            fill_color=color,
            fill_opacity=0.9,
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
        fmap.fit_bounds(df[[lat_col, lon_col]].dropna().values.tolist(), padding=(30, 30))
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
# -------------------------
# RIGHT: SCROLLABLE RESULTS
# -------------------------
with right_col:
    st.markdown('<div class="scroll-panel">', unsafe_allow_html=True)

    st.markdown("## ⭐ Best Matches")

    top_n = final_merged_df_sorted.head(5)
    
    # --- Pretty maps from old code ---
    pretty_income = {"low": "affordable", "medium": "standard priced", "high": "high-cost"}
    pretty_population = {"low": "Small", "medium": "Medium", "high": "Big"}
    pretty_asian = {"low": "Low", "medium": "Medium", "high": "High"}

    f_inputs = st.session_state["frozen_inputs"]

    income_pref = f_inputs["Median_INCTOT_input"]
    pop_pref = f_inputs["Median_Yearly_Population_input"]
    asian_pref = f_inputs["Percentage_Asian_input"]

    for idx, (_, row) in enumerate(top_n.iterrows()):

        best_tag = "<span class='best-tag'>BEST</span>" if idx == 0 else ""

        price = "$" + str(row["Price"]).lstrip("$")

        # --- SqFt logic ---
        min_sqft = row.get("Min_SqFt", row["Avg_SqFt"])
        max_sqft = row.get("Max_SqFt", row["Avg_SqFt"])
        sqft_text = (
            f"{int(min_sqft):,} SF"
            if int(min_sqft) == int(max_sqft)
            else f"{int(min_sqft):,}–{int(max_sqft):,} SF"
        )

        # --- DEMO LINES (restored from old code) ---
        demo_lines = []

        # Line 1 – Menu type
        if row.get("Dummy_Median_INCTOT", 0) == 100:
            demo_lines.append(
                f"Good for {pretty_income[income_pref]} menus"
            )

        # Line 2 – Market + Asian concentration
        if row.get("Dummy_Median_Yearly_Population", 0) == 100:
            pop_line = f"{pretty_population[pop_pref]} market size"

            if row.get("Dummy_Percentage_Asian", 0) == 100:
                pop_line += (
                    f" with {pretty_asian[asian_pref]} Asian population concentration"
                )

            demo_lines.append(pop_line)

        # Convert to HTML
        demo_html = "<br>".join(
            [f"<span style='opacity:0.6;'>{line}</span>" for line in demo_lines]
        )

        # --- RENDER CARD ---
        st.markdown(f"""
            <div class="result-card" id="card-{row['marker_id']}">
                <h4>{row['Name']} {best_tag}</h4>
                <p><strong>🏷️ {price} • {sqft_text}</strong></p>
                <p>📍 {row['full_address']}</p>
                {demo_html}
            </div>
        """, unsafe_allow_html=True)

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

st.subheader("Best 50 Locations Summary")

# Build trimmed-down table
export_df = final_merged_df_sorted.head(50).copy()

# Keep only required columns
export_df = export_df[["Name", "full_address", "Price", "details"]]

# Rename columns for export
export_df = export_df.rename(columns={
    "full_address": "Address",
    "Price": "Rental Cost",
    "details": "Description"
})

# Show table in UI
st.dataframe(export_df, use_container_width=True)

# Download version
csv = export_df.to_csv(index=False).encode("utf-8")
st.download_button("Download CSV", csv, "scored_sites.csv")

