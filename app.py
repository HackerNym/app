# app.py
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from io import BytesIO

# --- 0. App Configuration ---
st.set_page_config(
    page_title="Bangladesh Climate Impact Simulator",
    page_icon="🇧🇩",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- 1. Data Mocking ---
# This is where you would integrate real datasets (e.g., CMIP6 processed data).
# For now, we'll generate mock data.

REGIONS = ["Dhaka", "Chittagong", "Rajshahi", "Khulna", "Sylhet", "Barisal", "Rangpur", "Mymensingh"]
COASTAL_DISTRICTS = ["Khulna", "Satkhira", "Bagerhat", "Patuakhali", "Barguna", "Barisal", "Jhalokati", "Pirojpur", "Bhola", "Lakshmipur", "Noakhali", "Feni", "Chattogram", "Cox's Bazar"] # Subset of actual coastal districts for simplicity
CROPS = ["Rice", "Wheat", "Jute"]
SCENARIOS = {
    "SSP1-2.6": {"temp_factor": 1.5, "rain_factor": 1.05, "slr_factor": 0.3, "crop_impact_factor": 0.8}, # Optimistic
    "SSP2-4.5": {"temp_factor": 2.7, "rain_factor": 1.10, "slr_factor": 0.5, "crop_impact_factor": 1.0}, # Middle of the road
    "SSP5-8.5": {"temp_factor": 4.4, "rain_factor": 1.15, "slr_factor": 0.8, "crop_impact_factor": 1.2}  # Pessimistic
}
YEARS = list(range(2025, 2101))

# Baseline values (these would ideally be historical averages)
BASE_TEMP = {region: 25 + np.random.uniform(-1, 1) for region in REGIONS} # Avg annual temp C
BASE_RAINFALL = {region: 2000 + np.random.uniform(-200, 200) for region in REGIONS} # Avg annual rainfall mm
BASE_CROP_YIELD = {crop: 100 for crop in CROPS} # Baseline yield index

# Vulnerable district lat/lon for map (approximate)
VULNERABLE_DISTRICTS_COORDS = {
    "Khulna": {"lat": 22.8167, "lon": 89.5667, "base_slr_vulnerability": 1.2},
    "Satkhira": {"lat": 22.35, "lon": 89.075, "base_slr_vulnerability": 1.3},
    "Barisal": {"lat": 22.7010, "lon": 90.3535, "base_slr_vulnerability": 1.1},
    "Cox's Bazar": {"lat": 21.4272, "lon": 92.0058, "base_slr_vulnerability": 0.9},
    "Patuakhali": {"lat": 22.3583, "lon": 90.3297, "base_slr_vulnerability": 1.4},
    "Bhola": {"lat": 22.6833, "lon": 90.65, "base_slr_vulnerability": 1.5}
}


@st.cache_data(ttl=3600) # Cache data for 1 hour
def generate_mock_data():
    """
    Generates mock climate projection data.
    In a real application, this function would load and process actual climate model outputs.
    """
    data = []
    for year in YEARS:
        for region in REGIONS:
            for scenario_name, params in SCENARIOS.items():
                # Temperature: linear increase + noise, stronger for higher emission scenarios
                # Increase from 2025 (year 0) to 2100 (year 75)
                year_fraction = (year - 2025) / (2100 - 2025)
                temp_increase = year_fraction * params["temp_factor"] * (1 + np.random.uniform(-0.1, 0.1)) # Add some yearly noise
                temperature = BASE_TEMP[region] + temp_increase

                # Rainfall: percentage change + noise
                rainfall_change = year_fraction * (params["rain_factor"] -1) * 100 # % change
                rainfall = BASE_RAINFALL[region] * (1 + rainfall_change/100) * (1 + np.random.uniform(-0.05, 0.05))

                # Sea Level Rise (SLR) - simplified as a general coastal value, not region specific for this data point
                # More detailed SLR impact will be shown per vulnerable district
                slr = year_fraction * params["slr_factor"] * 100 # SLR in cm, max based on scenario by 2100

                # Crop Yields (% change from baseline)
                # Simplified: higher temps generally negative, rainfall changes complex
                # Using a generic impact factor for simplicity
                rice_yield_change = -year_fraction * params["crop_impact_factor"] * np.random.uniform(5, 15) \
                                    + (params["rain_factor"]-1)*10 # Some positive effect from moderate rain increase
                wheat_yield_change = -year_fraction * params["crop_impact_factor"] * np.random.uniform(10, 25) \
                                     - (params["temp_factor"]/2) # Wheat more sensitive to heat
                jute_yield_change = -year_fraction * params["crop_impact_factor"] * np.random.uniform(2, 10) \
                                    + (params["rain_factor"]-1)*5 # Jute somewhat resilient

                data.append({
                    "Year": year,
                    "Region": region,
                    "Scenario": scenario_name,
                    "Temperature (degC)": temperature,
                    "Rainfall (mm)": rainfall,
                    "Projected SLR (cm)": slr, # General coastal SLR
                    "Rice Yield Change (%)": np.clip(rice_yield_change, -100, 50),
                    "Wheat Yield Change (%)": np.clip(wheat_yield_change, -100, 20),
                    "Jute Yield Change (%)": np.clip(jute_yield_change, -100, 30)
                })
    df = pd.DataFrame(data)
    
    # Add specific SLR impact for vulnerable districts
    slr_vulnerable_data = []
    for year in YEARS:
        for scenario_name, params in SCENARIOS.items():
            year_fraction = (year - 2025) / (2100 - 2025)
            for district, props in VULNERABLE_DISTRICTS_COORDS.items():
                # District-specific SLR: general SLR * district vulnerability factor
                district_slr = year_fraction * params["slr_factor"] * 100 * props["base_slr_vulnerability"]
                slr_vulnerable_data.append({
                    "Year": year,
                    "Scenario": scenario_name,
                    "District": district,
                    "Projected SLR (cm)": district_slr,
                    "lat": props["lat"],
                    "lon": props["lon"]
                })
    slr_df = pd.DataFrame(slr_vulnerable_data)
    
    return df, slr_df

# Load data
df_projections, df_slr_vulnerable = generate_mock_data()

# --- 2. Helper Functions for UI ---
def get_filtered_data(df, region, scenario, year=None):
    """Filters data based on selections."""
    filtered = df[(df['Region'] == region) & (df['Scenario'] == scenario)]
    if year:
        filtered = filtered[filtered['Year'] == year]
    return filtered

def get_slr_filtered_data(df, scenario, year):
    return df[(df['Scenario'] == scenario) & (df['Year'] == year)]

def fig_to_bytes(fig):
    """Converts a Plotly figure to bytes for download."""
    buf = BytesIO()
    fig.write_image(buf, format="png", scale=2) # Increase scale for better resolution
    buf.seek(0)
    return buf.getvalue()

def df_to_csv_bytes(data_frame):
    """Converts a Pandas DataFrame to CSV bytes for download."""
    return data_frame.to_csv(index=False).encode('utf-8')

# --- 3. Sidebar Controls ---
st.sidebar.title("🇧🇩 Climate Simulator Controls")
st.sidebar.markdown("Adjust the parameters below to explore different climate change impacts on Bangladesh.")

# Tooltip dictionary
tooltips = {
    "region": "Select a division of Bangladesh to see localized projections.",
    "scenario": (
        "IPCC Emission Scenarios (Shared Socioeconomic Pathways - SSPs):\n"
        "- **SSP1-2.6**: Low emissions, sustainable development (Approx. +2.6 W/m^2 radiative forcing by 2100).\n"
        "- **SSP2-4.5**: Medium emissions, business-as-usual (Approx. +4.5 W/m^2).\n"
        "- **SSP5-8.5**: High emissions, fossil-fuel intensive (Approx. +8.5 W/m^2)."
    ),
    "year": "Select the projection year (2025-2100).",
    "temp_chart": "Annual average temperature projection for the selected region and scenario.",
    "rain_chart": "Annual average rainfall projection for the selected region and scenario.",
    "slr_map": "Projected sea-level rise for selected coastal districts under the chosen scenario and year. Size of circle indicates SLR magnitude.",
    "slr_chart": "Projected average sea-level rise trend along the coast for the selected scenario.",
    "crop_yield_chart": "Projected percentage change in major crop yields for the selected region, scenario, and year.",
    "crop_yield_trend": "Projected trend of percentage change in major crop yields over time."
}

selected_region = st.sidebar.selectbox(
    "Select Region/Division",
    REGIONS,
    index=REGIONS.index("Dhaka"), # Default to Dhaka
    help=tooltips["region"]
)

selected_scenario = st.sidebar.selectbox(
    "Select Emission Scenario (SSP)",
    list(SCENARIOS.keys()),
    index=1, # Default to SSP2-4.5
    help=tooltips["scenario"]
)

selected_year = st.sidebar.slider(
    "Select Year",
    min_value=min(YEARS),
    max_value=max(YEARS),
    value=2050,
    step=5,
    help=tooltips["year"]
)

# Filter data based on sidebar selections for single year views
current_year_data = get_filtered_data(df_projections, selected_region, selected_scenario, selected_year)
# Filter data for time series views (no year filter initially)
time_series_data_region_scenario = get_filtered_data(df_projections, selected_region, selected_scenario)


# --- 4. Main Application ---
st.title(f"Climate Change Impact Simulation for Bangladesh")
st.markdown(f"Displaying projections for **{selected_region}** under **{selected_scenario}** for the year **{selected_year}**.")
st.markdown("---")

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🌡️ Climate Projections",
    "🌊 Sea Level Rise",
    "🌾 Crop Yields",
    "🔄 Scenario Comparison",
    "ℹ️ About & Data"
])

with tab1:
    st.header(f"Climate Projections for {selected_region} ({selected_scenario})")
    
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Temperature Trend", help=tooltips["temp_chart"])
        if not time_series_data_region_scenario.empty:
            fig_temp = px.line(
                time_series_data_region_scenario,
                x="Year",
                y="Temperature (degC)",
                title=f"Avg. Temperature Trend ({selected_region}, {selected_scenario})"
            )
            fig_temp.add_hline(y=BASE_TEMP[selected_region], line_dash="dot", annotation_text="Baseline Temp", annotation_position="bottom right")
            st.plotly_chart(fig_temp, use_container_width=True)
            st.download_button(
                label="Download Temperature Chart (PNG)",
                data=fig_to_bytes(fig_temp),
                file_name=f"temp_chart_{selected_region}_{selected_scenario}.png",
                mime="image/png"
            )
        else:
            st.warning("No temperature data to display for current selections.")

    with col2:
        st.subheader("Rainfall Trend", help=tooltips["rain_chart"])
        if not time_series_data_region_scenario.empty:
            fig_rain = px.line(
                time_series_data_region_scenario,
                x="Year",
                y="Rainfall (mm)",
                title=f"Avg. Rainfall Trend ({selected_region}, {selected_scenario})"
            )
            fig_rain.add_hline(y=BASE_RAINFALL[selected_region], line_dash="dot", annotation_text="Baseline Rainfall", annotation_position="bottom right")
            st.plotly_chart(fig_rain, use_container_width=True)
            st.download_button(
                label="Download Rainfall Chart (PNG)",
                data=fig_to_bytes(fig_rain),
                file_name=f"rain_chart_{selected_region}_{selected_scenario}.png",
                mime="image/png"
            )
        else:
            st.warning("No rainfall data to display for current selections.")
    
    st.download_button(
        label=f"Download Climate Projection Data for {selected_region} ({selected_scenario}) (CSV)",
        data=df_to_csv_bytes(time_series_data_region_scenario[['Year', 'Temperature (degC)', 'Rainfall (mm)']]),
        file_name=f"climate_projections_{selected_region}_{selected_scenario}.csv",
        mime="text/csv"
    )


with tab2:
    st.header(f"Sea Level Rise (SLR) Projections ({selected_scenario}, Year {selected_year})")
    
    # Filter SLR data for the selected scenario and year
    slr_data_current_year_scenario = get_slr_filtered_data(df_slr_vulnerable, selected_scenario, selected_year)

    col1, col2 = st.columns([0.6, 0.4]) # Map takes more space
    with col1:
        st.subheader("Vulnerable Districts SLR Map", help=tooltips["slr_map"])
        if not slr_data_current_year_scenario.empty:
            # Create a 'size' column for the map, ensuring non-zero sizes
            map_data = slr_data_current_year_scenario.copy()
            map_data['size'] = map_data['Projected SLR (cm)'].apply(lambda x: max(x, 1)) # Min size 1 for visibility
            
            # Use st.map - simpler but less customizable. For more complex maps, consider pydeck or folium.
            # st.map needs columns named 'lat', 'lon', 'size' (optional), 'color' (optional)
            st.map(map_data[['lat', 'lon', 'size']].rename(columns={'Projected SLR (cm)': 'SLR (cm)'}), zoom=6, use_container_width=True)
            st.caption("Circle size on map is proportional to projected SLR. Data is illustrative.")
            
            # Alternative: Bar chart for SLR by district (more explicit)
            fig_slr_districts = px.bar(
                slr_data_current_year_scenario.sort_values("Projected SLR (cm)", ascending=False),
                x="District",
                y="Projected SLR (cm)",
                title=f"SLR in Vulnerable Districts ({selected_scenario}, {selected_year})",
                color="Projected SLR (cm)",
                color_continuous_scale=px.colors.sequential.Reds
            )
            st.plotly_chart(fig_slr_districts, use_container_width=True)
            st.download_button(
                label="Download SLR Districts Chart (PNG)",
                data=fig_to_bytes(fig_slr_districts),
                file_name=f"slr_districts_chart_{selected_scenario}_{selected_year}.png",
                mime="image/png"
            )

        else:
            st.warning("No SLR data to display for current selections.")

    with col2:
        st.subheader("Coastal Average SLR Trend", help=tooltips["slr_chart"])
        # Use the general 'Projected SLR (cm)' from df_projections for the coastal average trend
        # We need data for the selected scenario across all years, averaged (or just one region's value if consistent)
        slr_trend_data = df_projections[(df_projections['Scenario'] == selected_scenario) & (df_projections['Region'] == REGIONS[0])] # Take one region as proxy for coastal avg
        
        if not slr_trend_data.empty:
            fig_slr_trend = px.line(
                slr_trend_data,
                x="Year",
                y="Projected SLR (cm)",
                title=f"Coastal Average SLR Trend ({selected_scenario})"
            )
            fig_slr_trend.update_yaxes(title_text="Projected SLR (cm)")
            st.plotly_chart(fig_slr_trend, use_container_width=True)
            st.download_button(
                label="Download Coastal SLR Trend Chart (PNG)",
                data=fig_to_bytes(fig_slr_trend),
                file_name=f"slr_trend_chart_{selected_scenario}.png",
                mime="image/png"
            )
        else:
            st.warning("No coastal average SLR trend data.")

    st.download_button(
        label=f"Download Vulnerable District SLR Data ({selected_scenario}, {selected_year}) (CSV)",
        data=df_to_csv_bytes(slr_data_current_year_scenario[['District', 'Projected SLR (cm)', 'lat', 'lon']]),
        file_name=f"slr_vulnerable_districts_{selected_scenario}_{selected_year}.csv",
        mime="text/csv"
    )

with tab3:
    st.header(f"Crop Yield Projections for {selected_region} ({selected_scenario})")

    st.subheader(f"Yield Change in {selected_year}", help=tooltips["crop_yield_chart"])
    if not current_year_data.empty:
        crop_yield_data_current = current_year_data[[
            "Rice Yield Change (%)", "Wheat Yield Change (%)", "Jute Yield Change (%)"
        ]].melt(var_name="Crop", value_name="Yield Change (%)")

        fig_crop_yield = px.bar(
            crop_yield_data_current,
            x="Crop",
            y="Yield Change (%)",
            color="Crop",
            title=f"Crop Yield Change ({selected_region}, {selected_scenario}, {selected_year})"
        )
        fig_crop_yield.add_hline(y=0, line_dash="dot")
        st.plotly_chart(fig_crop_yield, use_container_width=True)
        st.download_button(
            label="Download Crop Yield Chart (PNG)",
            data=fig_to_bytes(fig_crop_yield),
            file_name=f"crop_yield_chart_{selected_region}_{selected_scenario}_{selected_year}.png",
            mime="image/png"
        )
    else:
        st.warning("No crop yield data for current selections.")

    st.subheader("Crop Yield Change Trends", help=tooltips["crop_yield_trend"])
    if not time_series_data_region_scenario.empty:
        crop_cols = ["Rice Yield Change (%)", "Wheat Yield Change (%)", "Jute Yield Change (%)"]
        crop_yield_trends = time_series_data_region_scenario[["Year"] + crop_cols].melt(
            id_vars="Year", value_vars=crop_cols, var_name="Crop", value_name="Yield Change (%)"
        )
        fig_crop_trends = px.line(
            crop_yield_trends,
            x="Year",
            y="Yield Change (%)",
            color="Crop",
            title=f"Crop Yield Change Trends ({selected_region}, {selected_scenario})"
        )
        fig_crop_trends.add_hline(y=0, line_dash="dot")
        st.plotly_chart(fig_crop_trends, use_container_width=True)
        st.download_button(
            label="Download Crop Trends Chart (PNG)",
            data=fig_to_bytes(fig_crop_trends),
            file_name=f"crop_trends_chart_{selected_region}_{selected_scenario}.png",
            mime="image/png"
        )
    else:
        st.warning("No crop yield trend data.")
    
    st.download_button(
        label=f"Download Crop Yield Data for {selected_region} ({selected_scenario}) (CSV)",
        data=df_to_csv_bytes(time_series_data_region_scenario[['Year'] + crop_cols]),
        file_name=f"crop_yield_data_{selected_region}_{selected_scenario}.csv",
        mime="text/csv"
    )


with tab4:
    st.header(f"Scenario Comparison for {selected_region}")
    st.markdown("This section allows comparing projections across different SSP emission scenarios for the selected region.")

    # Filter data for the selected region across all scenarios
    comparison_data_region = df_projections[df_projections['Region'] == selected_region]

    st.subheader("Temperature Projections by Scenario")
    if not comparison_data_region.empty:
        fig_temp_comp = px.line(
            comparison_data_region,
            x="Year",
            y="Temperature (degC)",
            color="Scenario",
            title=f"Temperature Comparison ({selected_region})"
        )
        st.plotly_chart(fig_temp_comp, use_container_width=True)
        st.download_button(
            label="Download Temp. Comparison Chart (PNG)",
            data=fig_to_bytes(fig_temp_comp),
            file_name=f"temp_comp_chart_{selected_region}.png",
            mime="image/png"
        )
    else:
        st.warning("No data for temperature comparison.")

    st.subheader("Rainfall Projections by Scenario")
    if not comparison_data_region.empty:
        fig_rain_comp = px.line(
            comparison_data_region,
            x="Year",
            y="Rainfall (mm)",
            color="Scenario",
            title=f"Rainfall Comparison ({selected_region})"
        )
        st.plotly_chart(fig_rain_comp, use_container_width=True)
        st.download_button(
            label="Download Rainfall Comparison Chart (PNG)",
            data=fig_to_bytes(fig_rain_comp),
            file_name=f"rain_comp_chart_{selected_region}.png",
            mime="image/png"
        )
    else:
        st.warning("No data for rainfall comparison.")
    
    # SLR Coastal Average comparison
    st.subheader("Coastal Average SLR by Scenario")
    # We need data across all scenarios, all years, averaged (or one region's value if consistent)
    slr_comp_data = df_projections[df_projections['Region'] == REGIONS[0]][['Year', 'Scenario', 'Projected SLR (cm)']] # Take one region as proxy for coastal avg
    if not slr_comp_data.empty:
        fig_slr_comp = px.line(
            slr_comp_data,
            x="Year",
            y="Projected SLR (cm)",
            color="Scenario",
            title=f"Coastal Average SLR Comparison"
        )
        st.plotly_chart(fig_slr_comp, use_container_width=True)
        st.download_button(
            label="Download SLR Comparison Chart (PNG)",
            data=fig_to_bytes(fig_slr_comp),
            file_name=f"slr_comp_chart.png",
            mime="image/png"
        )
    else:
        st.warning("No data for SLR comparison.")


    st.subheader(f"Crop Yield Change for {CROPS[0]} (Rice) by Scenario") # Example for Rice
    if not comparison_data_region.empty:
        fig_crop_comp = px.line(
            comparison_data_region,
            x="Year",
            y=f"{CROPS[0]} Yield Change (%)", # Example for Rice
            color="Scenario",
            title=f"{CROPS[0]} Yield Change Comparison ({selected_region})"
        )
        st.plotly_chart(fig_crop_comp, use_container_width=True)
        st.download_button(
            label=f"Download {CROPS[0]} Yield Comparison Chart (PNG)",
            data=fig_to_bytes(fig_crop_comp),
            file_name=f"crop_{CROPS[0]}_comp_chart_{selected_region}.png",
            mime="image/png"
        )
    else:
        st.warning(f"No data for {CROPS[0]} yield comparison.")

    st.download_button(
        label=f"Download All Scenario Comparison Data for {selected_region} (CSV)",
        data=df_to_csv_bytes(comparison_data_region),
        file_name=f"scenario_comparison_data_{selected_region}.csv",
        mime="text/csv"
    )

with tab5:
    st.header("About this Application & Data Source")
    st.markdown("""
    This web application simulates potential climate change impacts on Bangladesh, focusing on temperature, rainfall,
    sea level rise, and major crop yields. It uses mock data designed to reflect trends from IPCC emission scenarios.

    **Key Features:**
    - **Region Selection:** View projections for different divisions of Bangladesh.
    - **Interactive Projections:** Explore trends for temperature, rainfall, sea level rise, and crop yields.
    - **Scenario Analysis:** Compare impacts under different IPCC SSP scenarios (SSP1-2.6, SSP2-4.5, SSP5-8.5).
    - **Dynamic Controls:** Use sliders and dropdowns to customize your view.
    - **Export Options:** Download charts as PNG and data as CSV.

    **Data Source & Methodology:**
    - The data used in this simulation is **MOCK DATA**, generated for illustrative purposes.
    - The trends are loosely based on projections from sources like IPCC AR6, CMIP6 model ensembles, and regional climate studies for Bangladesh.
    - **Temperature:** Generally increasing, with higher increases under higher emission scenarios.
    - **Rainfall:** Patterns are more complex; this simulation shows a slight general increase, with more variability. Real projections vary significantly by region and season.
    - **Sea Level Rise (SLR):** Cumulative increase over time, more pronounced in higher emission scenarios. Vulnerability mapping is simplified.
    - **Crop Yields:** Estimated based on simplified agro-climatic responses to temperature and rainfall changes. Rice, Wheat, and Jute are considered. Actual crop responses are highly complex and depend on many factors (CO2 fertilization, adaptation, water management, etc.).

    **Disclaimer:**
    This tool is for educational and illustrative purposes only. The projections are based on simplified models and mock data.
    For accurate and detailed climate impact assessments, please refer to scientific literature, official government reports, and data from recognized climate research institutions (e.g., IPCC, World Bank Climate Change Knowledge Portal, Bangladesh Meteorological Department, Department of Environment).

    **Code Structure for Real Data Integration:**
    The data generation logic is primarily within the `generate_mock_data()` function. To use real data:
    1.  Prepare your datasets (e.g., from CMIP6 downscaled netCDF files, or processed CSVs).
    2.  Modify `generate_mock_data()` to load and process these files into Pandas DataFrames matching the structure used here (`df_projections`, `df_slr_vulnerable`).
    3.  Ensure column names like 'Year', 'Region', 'Scenario', 'Temperature (degC)', etc., are consistent.
    4.  Update `BASE_TEMP`, `BASE_RAINFALL` if you have historical baseline data.

    **Tech Stack:**
    - Python
    - Streamlit (for the web application framework)
    - Pandas (for data manipulation)
    - NumPy (for numerical operations)
    - Plotly (for interactive charts)
    """)

    st.subheader("Full Mock Dataset Preview (First 100 rows)")
    st.dataframe(df_projections.head(100))
    st.download_button(
        label="Download Full Mock Projection Dataset (CSV)",
        data=df_to_csv_bytes(df_projections),
        file_name="full_mock_climate_projections_bangladesh.csv",
        mime="text/csv"
    )
    st.dataframe(df_slr_vulnerable.head(100))
    st.download_button(
        label="Download Full Mock SLR Vulnerable District Dataset (CSV)",
        data=df_to_csv_bytes(df_slr_vulnerable),
        file_name="full_mock_slr_vulnerable_districts_bangladesh.csv",
        mime="text/csv"
    )

st.sidebar.markdown("---")
st.sidebar.info("Built with Streamlit by an AI assistant.")
st.sidebar.markdown("This is a mock-up. For real-world decisions, consult peer-reviewed climate science.")