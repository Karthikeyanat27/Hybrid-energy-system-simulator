# app.py
"""
Hybrid Energy System Simulator (Streamlit UI) - with day/hour/minute timestep options

Run:
streamlit run app.py

Install:
pip install streamlit requests pandas numpy matplotlib openpyxl
"""

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

from model import (
    Params,
    fetch_open_meteo_hourly,
    load_weather_from_csv,
    run_simulation,
    build_excel_bytes,
)

# -----------------------------
# Page setup
# -----------------------------
st.set_page_config(page_title="Hybrid Energy System Simulator", layout="wide")
st.title("Hybrid Energy System Simulator")
st.caption("Hybrid Solar + Wind → Storage → E-bike Charging (supports day/hour/minute time steps)")

# -----------------------------
# Sidebar: Inputs
# -----------------------------
st.sidebar.header("Input Panel")

scenario_name = st.sidebar.text_input("Scenario name", value="Baseline_Eindhoven")

st.sidebar.subheader("Weather input")
weather_mode = st.sidebar.radio("Weather source", options=["Online (Open-Meteo)", "Upload CSV"], index=0)
timezone = st.sidebar.text_input("Timezone", value="Europe/Amsterdam")

if weather_mode == "Online (Open-Meteo)":
    st.sidebar.markdown("**Location**")
    latitude = st.sidebar.number_input("Latitude", value=51.448, format="%.6f")
    longitude = st.sidebar.number_input("Longitude", value=5.490, format="%.6f")

    st.sidebar.markdown("**Date range**")
    start_date = st.sidebar.text_input("Start date (YYYY-MM-DD)", value="2024-01-01")
    end_date = st.sidebar.text_input("End date (YYYY-MM-DD)", value="2024-12-31")
    uploaded_csv = None
else:
    st.sidebar.markdown("Upload CSV with columns: **time, G, v**")
    uploaded_csv = st.sidebar.file_uploader("Upload CSV", type=["csv"])
    latitude = st.sidebar.number_input("Latitude (optional)", value=51.448, format="%.6f")
    longitude = st.sidebar.number_input("Longitude (optional)", value=5.490, format="%.6f")
    start_date = st.sidebar.text_input("Start date label (optional)", value="CSV_START")
    end_date = st.sidebar.text_input("End date label (optional)", value="CSV_END")

st.sidebar.subheader("Simulation time step")

timestep_choice = st.sidebar.selectbox(
    "Choose time step",
    options=["1 day", "1 hour", "30 minutes", "15 minutes", "10 minutes"],
    index=1
)

timestep_to_hours = {
    "1 day": 24.0,
    "1 hour": 1.0,
    "30 minutes": 0.5,
    "15 minutes": 0.25,
    "10 minutes": 10.0 / 60.0,
}
dt_hours = timestep_to_hours[timestep_choice]

v_bus = st.sidebar.number_input("DC bus voltage (V)", value=36.0, format="%.2f")

st.sidebar.subheader("PV system")
n_pv = st.sidebar.number_input("Number of PV panels", min_value=0, value=4, step=1)
p_pv_panel_rated_w = st.sidebar.number_input("Rated power per PV panel (W)", value=527.0, format="%.2f")
pv_derate = st.sidebar.number_input("PV derating factor", value=0.85, format="%.3f")
eta_pv_dcdc = st.sidebar.number_input("PV DC-DC (MPPT) efficiency", value=0.95, format="%.3f")

st.sidebar.subheader("Wind system (generic curve)")
p_w_rated_w = st.sidebar.number_input("Wind rated power (W)", value=1080.0, format="%.2f")
v_ci = st.sidebar.number_input("Cut-in wind speed v_ci (m/s)", value=3.0, format="%.2f")
v_r = st.sidebar.number_input("Rated wind speed v_r (m/s)", value=12.0, format="%.2f")
v_co = st.sidebar.number_input("Cut-out wind speed v_co (m/s)", value=25.0, format="%.2f")
eta_rect = st.sidebar.number_input("Rectifier efficiency", value=0.95, format="%.3f")
eta_mech = st.sidebar.number_input("Mechanical-electrical efficiency", value=0.90, format="%.3f")

st.sidebar.subheader("Storage (lead-acid)")
e_bat_max_wh = st.sidebar.number_input("Storage capacity (Wh)", value=12600.0, format="%.1f")
soc_init = st.sidebar.number_input("Initial SOC (0-1)", value=0.20, min_value=0.0, max_value=1.0, format="%.2f")
p_ch_max_w = st.sidebar.number_input("Max battery charging power P_ch,max (W)", value=1260.0, format="%.1f")
eta_ch = st.sidebar.number_input("Charging efficiency η_ch", value=0.87, format="%.3f")
eta_dis = st.sidebar.number_input("Discharging efficiency η_dis", value=0.87, format="%.3f")
soc_service_min = st.sidebar.number_input("Minimum SOC to serve bikes", value=0.10, min_value=0.0, max_value=1.0, format="%.2f")

st.sidebar.subheader("E-bike charging")
e_bike_wh = st.sidebar.number_input("E-bike energy per full charge (Wh)", value=504.0, format="%.1f")
p_bike_w = st.sidebar.number_input("Charger power per bike (W)", value=150.0, format="%.1f")
eta_bike_charger = st.sidebar.number_input("Bike charger efficiency", value=0.95, format="%.3f")
n_ports = st.sidebar.number_input("Number of charging ports", min_value=1, value=5, step=1)

st.sidebar.subheader("Demand model")
max_bikes_per_day = st.sidebar.number_input("Max bikes per day", min_value=0, value=25, step=1)
random_seed = st.sidebar.number_input("Random seed", min_value=0, value=42, step=1)

st.sidebar.subheader("Export")
include_timeseries_excel = st.sidebar.checkbox("Include time series in Excel (bigger file)", value=True)

# -----------------------------
# Cache weather
# -----------------------------
@st.cache_data(show_spinner=False)
def cached_fetch_weather(lat, lon, start_date, end_date, tz):
    return fetch_open_meteo_hourly(lat, lon, start_date, end_date, tz)

# -----------------------------
# Run Simulation
# -----------------------------
run_btn = st.button("Run Simulation", type="primary")

if "results" not in st.session_state:
    st.session_state["results"] = None

if run_btn:
    with st.spinner("Loading weather data and running simulation..."):
        params = Params(
            scenario_name=scenario_name,
            timezone=timezone,
            latitude=float(latitude),
            longitude=float(longitude),
            start_date=start_date,
            end_date=end_date,

            timestep_label=timestep_choice,
            dt_hours=float(dt_hours),

            v_bus=float(v_bus),

            n_pv=int(n_pv),
            p_pv_panel_rated_w=float(p_pv_panel_rated_w),
            pv_derate=float(pv_derate),
            eta_pv_dcdc=float(eta_pv_dcdc),

            p_w_rated_w=float(p_w_rated_w),
            v_ci=float(v_ci),
            v_r=float(v_r),
            v_co=float(v_co),
            eta_rect=float(eta_rect),
            eta_mech=float(eta_mech),

            e_bat_max_wh=float(e_bat_max_wh),
            soc_init=float(soc_init),
            p_ch_max_w=float(p_ch_max_w),
            eta_ch=float(eta_ch),
            eta_dis=float(eta_dis),
            soc_service_min=float(soc_service_min),

            e_bike_wh=float(e_bike_wh),
            p_bike_w=float(p_bike_w),
            eta_bike_charger=float(eta_bike_charger),
            n_ports=int(n_ports),

            max_bikes_per_day=int(max_bikes_per_day),
            random_seed=int(random_seed),
        )

        if weather_mode == "Online (Open-Meteo)":
            weather_raw = cached_fetch_weather(latitude, longitude, start_date, end_date, timezone)
        else:
            if uploaded_csv is None:
                st.error("Please upload a CSV file for CSV mode.")
                st.stop()
            csv_df = pd.read_csv(uploaded_csv)
            weather_raw = load_weather_from_csv(csv_df)

        results = run_simulation(params, weather_raw)
        st.session_state["results"] = results

# -----------------------------
# Outputs
# -----------------------------
results = st.session_state["results"]
if results is None:
    st.info("Fill inputs in the sidebar and click **Run Simulation**.")
    st.stop()

summary = results["summary"]
ts = results["timeseries"]
daily = results["daily"]
monthly = results["monthly"]

st.subheader("Key Outputs")

c1, c2, c3, c4, c5, c6 = st.columns(6)
c1.metric("PV energy (kWh)", f"{summary['solar_generated_kwh']:.1f}")
c2.metric("Wind energy (kWh)", f"{summary['wind_generated_kwh']:.1f}")
c3.metric("Total generated (kWh)", f"{summary['total_generated_kwh']:.1f}")
c4.metric("Spill (kWh)", f"{summary['spilled_kwh']:.1f}", f"{summary['spill_percent']:.1f}%")
c5.metric("E-bike delivered (kWh)", f"{summary['ebike_delivered_kwh']:.1f}")
c6.metric("Service rate (%)", f"{summary['service_rate_percent']:.1f}")

c7, c8, c9, c10 = st.columns(4)
c7.metric("Charges served (equiv.)", f"{summary['ebike_charges_served']:.1f}")
c8.metric("Demand bikes/year", f"{summary['bikes_demand_total']}")
c9.metric("Unserved (kWh)", f"{summary['unserved_kwh']:.2f}")
ttf = summary["time_to_full_storage_hours"]
c10.metric("Time to full storage", "Not reached" if ttf is None else f"{ttf:.0f} h")

st.subheader("Export")
excel_bytes = build_excel_bytes(results, include_timeseries=include_timeseries_excel)
filename = f"{summary['scenario_name']}_{summary['timestep_label'].replace(' ', '')}.xlsx"
st.download_button(
    label="Download Excel Results",
    data=excel_bytes,
    file_name=filename,
    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
)

# -----------------------------
# Plot helpers
# -----------------------------
def plot_series(x, y, title, xlabel, ylabel, ylim=None):
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(x, y)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if ylim is not None:
        ax.set_ylim(*ylim)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


# -----------------------------
# Tabs
# -----------------------------
tabs = st.tabs(["Weather", "Generation", "Storage", "E-bike Charging", "Monthly Summary", "Tables"])

with tabs[0]:
    st.subheader("Weather Inputs")
    st.write(f"Time step: **{summary['timestep_label']}** (Δt = {summary['dt_hours']:.4f} hours)")
    st.pyplot(plot_series(ts["time"], ts["G"], "Irradiance G(t)", "Time", "W/m²"))
    st.pyplot(plot_series(ts["time"], ts["v"], "Wind speed v(t)", "Time", "m/s"))

with tabs[1]:
    st.subheader("Generation Outputs")
    st.pyplot(plot_series(ts["time"], ts["P_solar"], "PV Power P_solar(t)", "Time", "W"))
    st.pyplot(plot_series(ts["time"], ts["P_wind"], "Wind Power P_wind(t)", "Time", "W"))
    st.pyplot(plot_series(ts["time"], ts["P_in"], "Total Input Power P_in(t) = PV + Wind", "Time", "W"))

with tabs[2]:
    st.subheader("Storage Behavior")
    st.pyplot(plot_series(ts["time"], ts["SOC"], "Storage SOC(t)", "Time", "SOC", ylim=(0, 1.05)))
    st.pyplot(plot_series(ts["time"], ts["P_charge"], "Battery Accepted Charge Power P_charge(t)", "Time", "W"))
    st.pyplot(plot_series(ts["time"], ts["P_spill"], "Spilled Power P_spill(t)", "Time", "W"))

with tabs[3]:
    st.subheader("E-bike Charging")
    st.pyplot(plot_series(ts["time"], ts["P_load"], "E-bike Load Power P_load(t)", "Time", "W"))

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(pd.to_datetime(daily["date"]), daily["bikes_demand"], label="Demand (bikes/day)")
    ax.plot(pd.to_datetime(daily["date"]), daily["served_bikes_equiv"], label="Served (bikes/day equiv.)")
    ax.plot(pd.to_datetime(daily["date"]), daily["unmet_bikes_equiv"], label="Unmet (bikes/day equiv.)")
    ax.set_title("Daily demand vs served vs unmet")
    ax.set_xlabel("Date")
    ax.set_ylabel("Bikes/day")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    st.pyplot(fig)

with tabs[4]:
    st.subheader("Monthly Summary (kWh)")

    fig1, ax1 = plt.subplots(figsize=(12, 5))
    ax1.bar(monthly["month"], monthly["solar_kwh"], label="PV kWh")
    ax1.bar(monthly["month"], monthly["wind_kwh"], bottom=monthly["solar_kwh"], label="Wind kWh")
    ax1.set_title("Monthly Renewable Generation (PV + Wind)")
    ax1.set_xlabel("Month")
    ax1.set_ylabel("kWh")
    ax1.tick_params(axis="x", rotation=45)
    ax1.grid(True, axis="y", alpha=0.3)
    ax1.legend()
    fig1.tight_layout()
    st.pyplot(fig1)

    fig2, ax2 = plt.subplots(figsize=(12, 5))
    ax2.plot(monthly["month"], monthly["bikes_kwh"], marker="o", label="Delivered to e-bikes (kWh)")
    ax2.plot(monthly["month"], monthly["spill_kwh"], marker="o", label="Spilled energy (kWh)")
    ax2.set_title("Monthly E-bike Delivery vs Spilled Energy")
    ax2.set_xlabel("Month")
    ax2.set_ylabel("kWh")
    ax2.tick_params(axis="x", rotation=45)
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    fig2.tight_layout()
    st.pyplot(fig2)

with tabs[5]:
    st.subheader("Tables")
    st.write("Time series (first 200 rows):")
    st.dataframe(ts.head(200))
    st.write("Daily:")
    st.dataframe(daily.head(60))
    st.write("Monthly:")
    st.dataframe(monthly)
