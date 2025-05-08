# app.py
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go # Keep for potential future use, though not strictly needed now
from io import BytesIO

# --- 0. App Configuration ---
st.set_page_config(
    page_title="Bangladesh Climate Impact Simulator",
    page_icon="🇧🇩",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- 1. Data Mocking ---
REGIONS = ["Dhaka", "Chittagong", "Rajshahi", "Khulna", "Sylhet", "Barisal", "Rangpur", "Mymensingh"]
COASTAL_DISTRICTS = ["Khulna", "Satkhira", "Bagerhat", "Patuakhali", "Barguna", "Barisal", "Jhalokati", "Pirojpur", "Bhola", "Lakshmipur", "Noakhali", "Feni", "Chattogram", "Cox's Bazar"]
CROPS = ["Rice", "Wheat", "Jute"]
SCENARIOS = {
    "SSP1-2.6": {"temp_factor": 1.5, "rain_factor": 1.05, "slr_factor": 0.3, "crop_impact_factor": 0.8},
    "SSP2-4.5": {"temp_factor": 2.7, "rain_factor": 1.10, "slr_factor": 0.5, "crop_impact_factor": 1.0},
    "SSP5-8.5": {"temp_factor": 4.4, "rain_factor": 1.15, "slr_factor": 0.8, "crop_impact_factor": 1.2}
}
YEARS = list(range(2025, 2101))

BASE_TEMP = {region: 25 + np.random.uniform(-1, 1) for region in REGIONS}
BASE_RAINFALL = {region: 2000 + np.random.uniform(-200, 200) for region in REGIONS}
BASE_CROP_YIELD = {crop: 100 for crop in CROPS}

VULNERABLE_DISTRICTS_COORDS = {
    "Khulna": {"lat": 22.8167, "lon": 89.5667, "base_slr_vulnerability": 1.2},
    "Satkhira": {"lat": 22.35, "lon": 89.075, "base_slr_vulnerability": 1.3},
    "Barisal": {"lat": 22.7010, "lon": 90.3535, "base_slr_vulnerability": 1.1},
    "Cox's Bazar": {"lat": 21.4272, "lon": 92.0058, "base_slr_vulnerability": 0.9},
    "Patuakhali": {"lat": 22.3583, "lon": 90.3297, "base_slr_vulnerability": 1.4},
    "Bhola": {"lat": 22.6833, "lon": 90.65, "base_slr_vulnerability": 1.5}
}

@st.cache_data(ttl=3600)
def generate_mock_data():
    data = []
    for year in YEARS:
        for region in REGIONS:
            for scenario_name, params in SCENARIOS.items():
                year_fraction = (year - 2025) / (2100 - 2025)
                temp_increase = year_fraction * params["temp_factor"] * (1 + np.random.uniform(-0.1, 0.1))
                temperature = BASE_TEMP[region] + temp_increase
                rainfall_change = year_fraction * (params["rain_factor"] -1) * 100
                rainfall = BASE_RAINFALL[region] * (1 + rainfall_change/100) * (1 + np.random.uniform(-0.05, 0.05))
                slr = year_fraction * params["slr_factor"] * 100
                rice_yield_change = -year_fraction * params["crop_impact_factor"] * np.random.uniform(5, 15) + (params["rain_factor"]-1)*10
                wheat_yield_change = -year_fraction * params["crop_impact_factor"] * np.random.uniform(10, 25) - (params["temp_factor"]/2)
                jute_yield_change = -year_fraction * params["crop_impact_factor"] * np.random.uniform(2, 10) + (params["rain_factor"]-1)*5
                data.append({
                    "Year": year, "Region": region, "Scenario": scenario_name,
                    "Temperature (degC)": temperature, "Rainfall (mm)": rainfall,
                    "Projected SLR (cm)": slr,
                    "Rice Yield Change (%)": np.clip(rice_yield_change, -100, 50),
                    "Wheat Yield Change (%)": np.clip(wheat_yield_change, -100, 20),
                    "Jute Yield Change (%)": np.clip(jute_yield_change, -100, 30)
                })
    df = pd.DataFrame(data)
    slr_vulnerable_data = []
    for year in YEARS:
        for scenario_name, params in SCENARIOS.items():
            year_fraction = (year - 2025) / (2100 - 2025)
            for district, props in VULNERABLE_DISTRICTS_COORDS.items():
                district_slr = year_fraction * params["slr_factor"] * 100 * props["base_slr_vulnerability"]
                slr_vulnerable_data.append({
                    "Year": year, "Scenario": scenario_name, "District": district,
                    "Projected SLR (cm)": district_slr, "lat": props["lat"], "lon": props["lon"]
                })
    slr_df = pd.DataFrame(slr_vulnerable_data)
    return df, slr_df

df_projections, df_slr_vulnerable = generate_mock_data()

# --- 2. Helper Functions for UI ---
def get_filtered_data(df, region, scenario, year=None):
    filtered = df[(df['Region'] == region) & (df['Scenario'] == scenario)]
    if year:
        filtered = filtered[filtered['Year'] == year]
    return filtered

def get_slr_filtered_data(df, scenario, year):
    return df[(df['Scenario'] == scenario) & (df['Year'] == year)]

# REMOVED: fig_to_bytes function as we will rely on Plotly's built-in PNG export

def df_to_csv_bytes(data_frame):
    return data_frame.to_csv(index=False).encode('utf-8')

# --- 3. Sidebar Controls ---
st.sidebar.title("🇧🇩 Climate Simulator Controls")
st.sidebar.markdown("Adjust parameters to explore climate impacts.")

tooltips = {
    "region": "Select a division of Bangladesh.",
    "scenario": (
        "IPCC Emission Scenarios (SSPs):\n"
        "- SSP1-2.6: Low emissions, sustainable.\n"
        "- SSP2-4.5: Medium emissions, current path.\n"
        "- SSP5-8.5: High emissions, fossil-fuel intensive."
    ),
    "year": "Select projection year (2025-2100).",
    "temp_chart": "Annual avg. temperature projection. Hover over chart for options (incl. PNG download).",
    "rain_chart": "Annual avg. rainfall projection. Hover over chart for options (incl. PNG download).",
    "slr_map": "Projected sea-level rise for coastal districts. Circle size indicates SLR. Data is illustrative.",
    "slr_chart": "Projected avg. coastal sea-level rise trend. Hover for options (incl. PNG download).",
    "crop_yield_chart": "Projected % change in crop yields. Hover for options (incl. PNG download).",
    "crop_yield_trend": "Projected trend of % crop yield change. Hover for options (incl. PNG download)."
}

selected_region = st.sidebar.selectbox("Select Region/Division", REGIONS, index=REGIONS.index("Dhaka"), help=tooltips["region"])
selected_scenario = st.sidebar.selectbox("Select Emission Scenario (SSP)", list(SCENARIOS.keys()), index=1, help=tooltips["scenario"])
selected_year = st.sidebar.slider("Select Year", min_value=min(YEARS), max_value=max(YEARS), value=2050, step=5, help=tooltips["year"])

current_year_data = get_filtered_data(df_projections, selected_region, selected_scenario, selected_year)
time_series_data_region_scenario = get_filtered_data(df_projections, selected_region, selected_scenario)

# --- 4. Main Application ---
st.title(f"Climate Change Impact Simulation for Bangladesh")
st.markdown(f"Displaying projections for **{selected_region}** under **{selected_scenario}** for **{selected_year}**.")
st.markdown("---")

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🌡️ Climate Projections", "🌊 Sea Level Rise", "🌾 Crop Yields",
    "🔄 Scenario Comparison", "ℹ️ About & Data"
])

with tab1:
    st.header(f"Climate Projections for {selected_region} ({selected_scenario})")
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Temperature Trend", help=tooltips["temp_chart"])
        if not time_series_data_region_scenario.empty:
            fig_temp = px.line(time_series_data_region_scenario, x="Year", y="Temperature (degC)", title=f"Avg. Temperature Trend")
            fig_temp.add_hline(y=BASE_TEMP[selected_region], line_dash="dot", annotation_text="Baseline Temp", annotation_position="bottom right")
            st.plotly_chart(fig_temp, use_container_width=True)
            # REMOVED: Custom PNG download button
        else:
            st.warning("No temperature data.")
    with col2:
        st.subheader("Rainfall Trend", help=tooltips["rain_chart"])
        if not time_series_data_region_scenario.empty:
            fig_rain = px.line(time_series_data_region_scenario, x="Year", y="Rainfall (mm)", title=f"Avg. Rainfall Trend")
            fig_rain.add_hline(y=BASE_RAINFALL[selected_region], line_dash="dot", annotation_text="Baseline Rainfall", annotation_position="bottom right")
            st.plotly_chart(fig_rain, use_container_width=True)
            # REMOVED: Custom PNG download button
        else:
            st.warning("No rainfall data.")
    if not time_series_data_region_scenario.empty:
        st.download_button(
            label=f"Download Climate Data ({selected_region}, {selected_scenario}) (CSV)",
            data=df_to_csv_bytes(time_series_data_region_scenario[['Year', 'Temperature (degC)', 'Rainfall (mm)']]),
            file_name=f"climate_data_{selected_region}_{selected_scenario}.csv", mime="text/csv"
        )

with tab2:
    st.header(f"Sea Level Rise (SLR) Projections ({selected_scenario}, Year {selected_year})")
    slr_data_current_year_scenario = get_slr_filtered_data(df_slr_vulnerable, selected_scenario, selected_year)
    col1, col2 = st.columns([0.6, 0.4])
    with col1:
        st.subheader("Vulnerable Districts SLR Map", help=tooltips["slr_map"])
        if not slr_data_current_year_scenario.empty:
            map_data = slr_data_current_year_scenario.copy()
            map_data['size'] = map_data['Projected SLR (cm)'].apply(lambda x: max(x, 1))
            st.map(map_data[['lat', 'lon', 'size']].rename(columns={'Projected SLR (cm)': 'SLR (cm)'}), zoom=6, use_container_width=True)
            st.caption("Circle size proportional to SLR. Illustrative data.")
            
            fig_slr_districts = px.bar(
                slr_data_current_year_scenario.sort_values("Projected SLR (cm)", ascending=False),
                x="District", y="Projected SLR (cm)",
                title=f"SLR in Vulnerable Districts",
                color="Projected SLR (cm)", color_continuous_scale=px.colors.sequential.Reds
            )
            st.plotly_chart(fig_slr_districts, use_container_width=True)
            # REMOVED: Custom PNG download button
        else:
            st.warning("No SLR data for map/bar chart.")
    with col2:
        st.subheader("Coastal Average SLR Trend", help=tooltips["slr_chart"])
        slr_trend_data = df_projections[(df_projections['Scenario'] == selected_scenario) & (df_projections['Region'] == REGIONS[0])]
        if not slr_trend_data.empty:
            fig_slr_trend = px.line(slr_trend_data, x="Year", y="Projected SLR (cm)", title=f"Coastal Avg. SLR Trend")
            st.plotly_chart(fig_slr_trend, use_container_width=True)
            # REMOVED: Custom PNG download button
        else:
            st.warning("No coastal avg. SLR trend data.")
    if not slr_data_current_year_scenario.empty:
        st.download_button(
            label=f"Download District SLR Data ({selected_scenario}, {selected_year}) (CSV)",
            data=df_to_csv_bytes(slr_data_current_year_scenario[['District', 'Projected SLR (cm)', 'lat', 'lon']]),
            file_name=f"slr_districts_{selected_scenario}_{selected_year}.csv", mime="text/csv"
        )

with tab3:
    st.header(f"Crop Yield Projections for {selected_region} ({selected_scenario})")
    st.subheader(f"Yield Change in {selected_year}", help=tooltips["crop_yield_chart"])
    if not current_year_data.empty:
        crop_yield_data_current = current_year_data[["Rice Yield Change (%)", "Wheat Yield Change (%)", "Jute Yield Change (%)"]].melt(var_name="Crop", value_name="Yield Change (%)")
        if not crop_yield_data_current.empty: # Added safety check
            fig_crop_yield = px.bar(crop_yield_data_current, x="Crop", y="Yield Change (%)", color="Crop", title=f"Crop Yield Change")
            fig_crop_yield.add_hline(y=0, line_dash="dot")
            st.plotly_chart(fig_crop_yield, use_container_width=True)
            # REMOVED: Custom PNG download button
        else:
            st.warning("No current year crop data to plot.")
    else:
        st.warning("No crop yield data for current year.")

    st.subheader("Crop Yield Change Trends", help=tooltips["crop_yield_trend"])
    if not time_series_data_region_scenario.empty:
        crop_cols = ["Rice Yield Change (%)", "Wheat Yield Change (%)", "Jute Yield Change (%)"]
        crop_yield_trends = time_series_data_region_scenario[["Year"] + crop_cols].melt(id_vars="Year", value_vars=crop_cols, var_name="Crop", value_name="Yield Change (%)")
        if not crop_yield_trends.empty: # Added safety check
            fig_crop_trends = px.line(crop_yield_trends, x="Year", y="Yield Change (%)", color="Crop", title=f"Crop Yield Change Trends")
            fig_crop_trends.add_hline(y=0, line_dash="dot")
            st.plotly_chart(fig_crop_trends, use_container_width=True)
            # REMOVED: Custom PNG download button
        else:
            st.warning("No crop trend data to plot.")
    else:
        st.warning("No crop yield trend data.")
    if not time_series_data_region_scenario.empty:
        st.download_button(
            label=f"Download Crop Yield Data ({selected_region}, {selected_scenario}) (CSV)",
            data=df_to_csv_bytes(time_series_data_region_scenario[['Year'] + ["Rice Yield Change (%)", "Wheat Yield Change (%)", "Jute Yield Change (%)"]]),
            file_name=f"crop_yield_data_{selected_region}_{selected_scenario}.csv", mime="text/csv"
        )

with tab4:
    st.header(f"Scenario Comparison for {selected_region}")
    st.markdown("Comparing projections across SSP scenarios for selected region. Hover over charts for options (incl. PNG download).")
    comparison_data_region = df_projections[df_projections['Region'] == selected_region]

    st.subheader("Temperature Projections by Scenario")
    if not comparison_data_region.empty:
        fig_temp_comp = px.line(comparison_data_region, x="Year", y="Temperature (degC)", color="Scenario", title=f"Temperature Comparison")
        st.plotly_chart(fig_temp_comp, use_container_width=True)
        # REMOVED: Custom PNG download button
    else:
        st.warning("No temp comparison data.")

    st.subheader("Rainfall Projections by Scenario")
    if not comparison_data_region.empty:
        fig_rain_comp = px.line(comparison_data_region, x="Year", y="Rainfall (mm)", color="Scenario", title=f"Rainfall Comparison")
        st.plotly_chart(fig_rain_comp, use_container_width=True)
        # REMOVED: Custom PNG download button
    else:
        st.warning("No rain comparison data.")
    
    st.subheader("Coastal Average SLR by Scenario")
    slr_comp_data = df_projections[df_projections['Region'] == REGIONS[0]][['Year', 'Scenario', 'Projected SLR (cm)']]
    if not slr_comp_data.empty:
        fig_slr_comp = px.line(slr_comp_data, x="Year", y="Projected SLR (cm)", color="Scenario", title=f"Coastal Avg. SLR Comparison")
        st.plotly_chart(fig_slr_comp, use_container_width=True)
        # REMOVED: Custom PNG download button
    else:
        st.warning("No SLR comparison data.")

    st.subheader(f"Crop Yield Change for {CROPS[0]} (Rice) by Scenario")
    if not comparison_data_region.empty:
        # Ensure the column exists before trying to plot
        crop_col_name = f"{CROPS[0]} Yield Change (%)"
        if crop_col_name in comparison_data_region.columns:
            fig_crop_comp = px.line(comparison_data_region, x="Year", y=crop_col_name, color="Scenario", title=f"{CROPS[0]} Yield Comparison")
            st.plotly_chart(fig_crop_comp, use_container_width=True)
            # REMOVED: Custom PNG download button
        else:
            st.warning(f"Column '{crop_col_name}' not found for comparison.")
    else:
        st.warning(f"No data for {CROPS[0]} yield comparison.")

    if not comparison_data_region.empty:
        st.download_button(
            label=f"Download All Scenario Comparison Data ({selected_region}) (CSV)",
            data=df_to_csv_bytes(comparison_data_region),
            file_name=f"scenario_comp_data_{selected_region}.csv", mime="text/csv"
        )

with tab5:
    st.header("About this Application & Data Source")
    st.markdown("""
    This web application simulates potential climate change impacts on Bangladesh.
    **Data Source:** MOCK DATA, for illustrative purposes, loosely based on IPCC trends.
    **Disclaimer:** For educational use. Not for real-world decision-making.
    **Tech Stack:** Python, Streamlit, Pandas, NumPy, Plotly.
    **PNG Export:** Hover over charts and use the camera icon to download as PNG.
    """)
    st.subheader("Full Mock Dataset Preview (First 100 rows)")
    if not df_projections.empty:
        st.dataframe(df_projections.head(100))
        st.download_button(
            label="Download Full Mock Projection Dataset (CSV)",
            data=df_to_csv_bytes(df_projections),
            file_name="full_mock_climate_projections_bangladesh.csv", mime="text/csv"
        )
    if not df_slr_vulnerable.empty:
        st.dataframe(df_slr_vulnerable.head(100))
        st.download_button(
            label="Download Full Mock SLR Vulnerable District Dataset (CSV)",
            data=df_to_csv_bytes(df_slr_vulnerable),
            file_name="full_mock_slr_vulnerable_districts_bangladesh.csv", mime="text/csv"
        )

st.sidebar.markdown("---")
st.sidebar.info("Built with Streamlit.")
st.sidebar.markdown("This is a mock-up. Consult peer-reviewed science for real decisions.")
