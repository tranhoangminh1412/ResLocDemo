import streamlit as st
import pandas as pd
import numpy as np
import folium
import DataPipeline
import dictionaries

st.set_page_config(
    page_title="Restaurant Site Selector",
    layout="wide"
)

try:
    import folium
    from streamlit_folium import st_folium
    _FOLIUM_OK = True
except Exception:
    _FOLIUM_OK = False

# -------------------------
# 1) Load data needed for input ranges
# -------------------------

@st.cache_data
def get_site_level_df():
    site_level_df, _, _ = DataPipeline.data_cleaning()
    return site_level_df

site_level_df = get_site_level_df()
Avg_SqFt_range = site_level_df["Avg_SqFt"].agg(["min", "max"])

# -------------------------
# 2) Sidebar – user inputs
# -------------------------

st.sidebar.title("Your Restaurant Preferences")

# ---- Top-level weights (Demo / Comp / Site) ----
st.sidebar.subheader("Step 1 – What matters most?")

demo_raw = st.sidebar.slider("Demographics importance", 0.0, 1.0, 0.4, 0.05)
comp_raw = st.sidebar.slider("Competition importance", 0.0, 1.0, 0.4, 0.05)
site_raw = st.sidebar.slider("Site (rent & size) importance", 0.0, 1.0, 0.2, 0.05)

total_raw = demo_raw + comp_raw + site_raw
if total_raw == 0:
    Demo_weight = 0.4
    Comp_weight = 0.4
    Site_weight = 0.2
else:
    Demo_weight = demo_raw / total_raw
    Comp_weight = comp_raw / total_raw
    Site_weight = site_raw / total_raw

st.sidebar.markdown(
    f"**Normalized weights**  \n"
    f"- Demographic: `{Demo_weight:.2f}`  \n"
    f"- Competition: `{Comp_weight:.2f}`  \n"
    f"- Site: `{Site_weight:.2f}`"
)

weights = {
    "Demo_weight": Demo_weight,
    "Comp_weight": Comp_weight,
    "Site_weight": Site_weight,
}

# ---- Restaurant Type (Cuisine) ----
st.sidebar.subheader("Step 2 – Restaurant type")

Restaurant_Types = list(dictionaries.asian_food_keywords.keys())
Restaurant_type_input = st.sidebar.selectbox(
    "Cuisine (type of Asian restaurant)",
    options=Restaurant_Types,
    index=Restaurant_Types.index("Vietnamese") if "Vietnamese" in Restaurant_Types else 0,
)

# ---- Site preference: Size ----
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

# ---- Demographic preferences (Low / Medium / High) ----
st.sidebar.subheader("Step 4 – Area demographics")

Percentage_Asian_Options = ["Low", "Medium", "High"]
Median_INCTOT_Options = ["Low", "Medium", "High"]
Median_Yearly_Population_Options = ["Low", "Medium", "High"]

Percentage_Asian_choice = st.sidebar.selectbox(
    "Market saturation (Asian population share)",
    options=Percentage_Asian_Options,
    index=2,  # default: High
)

Median_INCTOT_choice = st.sidebar.selectbox(
    "Typical income level (customer spending power)",
    options=Median_INCTOT_Options,
    index=1,  # default: Medium
)

Median_Yearly_Population_choice = st.sidebar.selectbox(
    "Location type (population size)",
    options=Median_Yearly_Population_Options,
    index=1,  # default: Medium
)

# Map display labels to the lowercase strings expected by DataPipeline.scaling()
inputs = {
    "Restaurant_type_input": Restaurant_type_input,
    "Avg_SqFt_input": Avg_SqFt_input,
    "Percentage_Asian_input": Percentage_Asian_choice.lower(),
    "Median_INCTOT_input": Median_INCTOT_choice.lower(),
    "Median_Yearly_Population_input": Median_Yearly_Population_choice.lower(),
}

# -------------------------
# 3) Main page – run scoring
# -------------------------

st.title("Restaurant Site Selector")
st.write(
    "This tool helps you compare available restaurant sites in California based on "
    "demographics, competition, and site characteristics tailored to your Asian restaurant."
)

run_button = st.button("Find Best Locations")

# Initialize session_state to hold results
if "final_results" not in st.session_state:
    st.session_state["final_results"] = None

# When button is clicked: compute and store results
if run_button:
    with st.spinner("Calculating the best sites for your restaurant..."):
        st.session_state["final_results"] = DataPipeline.calculate_final_scores(weights, inputs)

# Always read from session_state
final_merged_df_sorted = st.session_state["final_results"]

if final_merged_df_sorted is None:
    st.info("Set your preferences in the sidebar, then click **Find Best Locations** to see suggested sites.")
else:
    if final_merged_df_sorted.empty:
        st.warning("No sites matched the current filters. Try adjusting your preferences.")
    else:
        st.subheader("Top 10 Recommended Locations")

        map_col, results_col = st.columns([3, 2])

        # ------------------ MAP ------------------ #
        def render_map_from_results(df, center=(36.7783, -119.4179), zoom_start=6):
            fmap = folium.Map(location=center, zoom_start=zoom_start, tiles="OpenStreetMap")
            if df is None or df.empty:
                return fmap

            possible_lat_cols = ["lat", "latitude", "Lat", "Latitude"]
            possible_lon_cols = ["lon", "lng", "longitude", "Longitude", "Lon"]

            lat_col = next((c for c in possible_lat_cols if c in df.columns), None)
            lon_col = next((c for c in possible_lon_cols if c in df.columns), None)

            if lat_col is None or lon_col is None:
                return fmap

            if "fit_score" in df.columns:
                score_min = df["fit_score"].min()
                score_max = df["fit_score"].max()
                score_range = max(score_max - score_min, 1e-6)
            else:
                score_min = score_max = None
                score_range = 1.0

            for _, r in df.iterrows():
                lat = r.get(lat_col)
                lon = r.get(lon_col)
                if pd.isna(lat) or pd.isna(lon):
                    continue

                name = r.get("Name", "Candidate")
                address = r.get("full_address", "")

                sqft = r.get("Avg_SqFt", None)
                total_rent = r.get("Total_Rent", None)
                rent_per_sqft = None
                if sqft not in [None, 0, np.nan] and total_rent not in [None, np.nan]:
                    rent_per_sqft = total_rent / sqft

                score = r.get("fit_score", None)
                if score_min is not None and score is not None and not pd.isna(score):
                    radius = 6 + max(0, (score - score_min) / score_range * 10)
                else:
                    radius = 6

                rent_text = f"${rent_per_sqft:.2f} / sqft / mo" if rent_per_sqft is not None else "N/A"
                sqft_text = f"{int(sqft)}" if sqft is not None and not pd.isna(sqft) else "N/A"
                score_text = f"{score:.2f}" if score is not None and not pd.isna(score) else "N/A"

                popup_html = (
                    f"<b>{name}</b><br>{address}"
                    f"<br>Fit score: {score_text}"
                    f"<br>Est. rent per sqft: {rent_text}"
                    f"<br>Size: {sqft_text} sqft"
                )

                folium.CircleMarker(
                    location=[float(lat), float(lon)],
                    radius=radius,
                    color=None,
                    fill=True,
                    fill_color="#2e7cff",
                    fill_opacity=0.9,
                    popup=folium.Popup(popup_html, max_width=300),
                ).add_to(fmap)

            try:
                bounds = df[[lat_col, lon_col]].dropna().values.tolist()
                if bounds:
                    fmap.fit_bounds(bounds, padding=(30, 30))
            except Exception:
                pass

            return fmap

        with map_col:
            if _FOLIUM_OK:
                missing_latlon = not any(c in final_merged_df_sorted.columns for c in ["lat", "latitude", "Lat", "Latitude"]) \
                                 or not any(c in final_merged_df_sorted.columns for c in ["lon", "lng", "longitude", "Longitude", "Lon"])
                if missing_latlon:
                    st.info(
                        "Map is shown without markers because the data does not contain latitude/longitude "
                        "columns yet. Once you add coordinates (e.g., 'lat' and 'lon'), markers will appear here."
                    )
                    fmap = folium.Map(location=(36.7783, -119.4179), zoom_start=6, tiles="OpenStreetMap")
                else:
                    fmap = render_map_from_results(final_merged_df_sorted)

                st_folium(fmap, width=900, height=650)
            else:
                st.error("To view the map, please install `folium` and `streamlit-folium`: `pip install folium streamlit-folium`")

        # ------------------ TEXT RESULTS ------------------ #
        with results_col:
            top_n = final_merged_df_sorted.head(10).copy()
            restaurant_col = inputs["Restaurant_type_input"]

            for i, row in top_n.iterrows():
                st.markdown(f"### {i+1}. {row['Name']}")
                st.markdown(f"- **Address:** {row['full_address']}")
                st.markdown(f"- **Details:** {row.get('details', '')}")
                st.markdown(f"- **Overall Fit Score:** `{row['fit_score']:,.2f}`")
                st.markdown(f"- **Average Total Monthly Rent:** `${row['Price']}`")
                st.markdown(f"- **Average Total Square Feet:** `{row['Avg_SqFt']}`")
                st.markdown(f"- **Percentage Asian in County:** `{row['Percentage_Asian']:,.2f}%`")
                st.markdown(f"- **Median Personal Total Income in County:** `${row['Median_INCTOT']:,.0f}`")
                st.markdown(f"- **Median Population in County:** `{row['Median_Yearly_Population']:,.0f}`")
                st.markdown(f"- **Median Price of other restaurants in ZIP:** `${row['Median_price_mid']:,.2f}`")
                st.markdown(f"- **Total other restaurants in ZIP:** `{row['Count_total_restaurant']}`")

                if restaurant_col in row.index:
                    st.markdown(f"- **Total other {restaurant_col} restaurants in ZIP:** `{row[restaurant_col]}`")

                st.markdown("---")

        st.subheader("Detailed Table (Top 50)")
        show_cols = [
            "Name",
            "full_address",
            "fit_score",
            "Price",
            "Avg_SqFt",
            "Percentage_Asian",
            "Median_INCTOT",
            "Median_Yearly_Population",
            "Median_price_mid",
            "Count_total_restaurant",
        ]
        if restaurant_col not in show_cols and restaurant_col in final_merged_df_sorted.columns:
            show_cols.append(restaurant_col)

        st.dataframe(
            final_merged_df_sorted.head(50)[[c for c in show_cols if c in final_merged_df_sorted.columns]],
            use_container_width=True,
        )

        csv = final_merged_df_sorted.to_csv(index=False).encode("utf-8")
        st.download_button(
            "Download all scored locations (CSV)",
            csv,
            file_name="scored_restaurant_sites.csv",
        )