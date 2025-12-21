# model.py
"""
Hybrid Energy System Simulator (Model Engine) - with dynamic timestep support

Supports timestep options:
- 1 day (24h): aggregates hourly PV/wind power into daily energy, then simulates daily steps
- 1 hour: uses hourly weather directly
- 30/15/10 minutes: interpolates hourly weather to finer timestep

Weather input options:
- Online (Open-Meteo hourly): time series at 1-hour resolution
- Uploaded CSV (any resolution): time series is used directly and can be resampled

Outputs:
- "timeseries": DataFrame at the simulation timestep (day/hour/min)
- "daily": daily demand/served/unmet
- "monthly": monthly energy summary
- "summary": key KPIs

Install:
pip install requests pandas numpy openpyxl
"""

from __future__ import annotations
from dataclasses import dataclass, asdict
from typing import Dict, Optional
import numpy as np
import pandas as pd
import requests


# -----------------------------
# A) Parameter container
# -----------------------------
@dataclass
class Params:
    # Identity
    scenario_name: str

    # Weather/meta
    timezone: str
    latitude: float
    longitude: float
    start_date: str
    end_date: str

    # Time step control
    timestep_label: str       # e.g., "1 day", "1 hour", "15 minutes"
    dt_hours: float           # numerical dt in hours

    # DC bus
    v_bus: float

    # PV
    n_pv: int
    p_pv_panel_rated_w: float
    pv_derate: float
    eta_pv_dcdc: float

    # Wind (generic curve)
    p_w_rated_w: float
    v_ci: float
    v_r: float
    v_co: float
    eta_rect: float
    eta_mech: float

    # Storage (lead-acid)
    e_bat_max_wh: float
    soc_init: float
    p_ch_max_w: float
    eta_ch: float
    eta_dis: float
    soc_service_min: float

    # E-bike + charging station
    e_bike_wh: float
    p_bike_w: float
    eta_bike_charger: float
    n_ports: int

    # Demand model
    max_bikes_per_day: int
    random_seed: int


# -----------------------------
# B) Weather fetchers
# -----------------------------
def fetch_open_meteo_hourly(
    latitude: float,
    longitude: float,
    start_date: str,
    end_date: str,
    timezone: str,
) -> pd.DataFrame:
    """
    Fetch hourly weather from Open-Meteo archive API (no key):
    - shortwave_radiation (W/m^2) -> G
    - windspeed_10m (m/s)         -> v

    Returns: DataFrame columns: time, G, v (hourly)
    """
    url = "https://archive-api.open-meteo.com/v1/archive"
    params = {
        "latitude": latitude,
        "longitude": longitude,
        "start_date": start_date,
        "end_date": end_date,
        "hourly": "shortwave_radiation,windspeed_10m",
        "timezone": timezone,
    }
    resp = requests.get(url, params=params, timeout=60)
    resp.raise_for_status()
    data = resp.json()

    time = pd.to_datetime(data["hourly"]["time"])
    G = np.array(data["hourly"]["shortwave_radiation"], dtype=float)
    v = np.array(data["hourly"]["windspeed_10m"], dtype=float)

    df = pd.DataFrame({"time": time, "G": G, "v": v}).sort_values("time").reset_index(drop=True)
    return df


def load_weather_from_csv(csv_df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize uploaded CSV into columns: time, G, v
    Required columns: time, G, v
    """
    required = {"time", "G", "v"}
    missing = required - set(csv_df.columns)
    if missing:
        raise ValueError(f"CSV is missing required columns: {sorted(missing)}")

    df = csv_df.copy()
    df["time"] = pd.to_datetime(df["time"])
    df["G"] = pd.to_numeric(df["G"], errors="coerce")
    df["v"] = pd.to_numeric(df["v"], errors="coerce")
    df = df.dropna(subset=["time", "G", "v"]).sort_values("time").reset_index(drop=True)
    return df


# -----------------------------
# C) Power models
# -----------------------------
def pv_power_w(pv_rated_total_w: float, irradiance_wm2: np.ndarray, derate: float, eta_dcdc: float) -> np.ndarray:
    """
    PV model:
    P_raw = P_rated_total * (G/1000) * derate
    clamp <= P_rated_total
    P_out = eta_dcdc * P_raw
    """
    g_pu = irradiance_wm2 / 1000.0
    p_raw = pv_rated_total_w * g_pu * derate
    p_raw = np.minimum(p_raw, pv_rated_total_w)
    p_out = eta_dcdc * p_raw
    return np.maximum(p_out, 0.0)


def wind_power_w(
    p_rated_w: float,
    wind_ms: np.ndarray,
    v_ci: float,
    v_r: float,
    v_co: float,
    eta_rect: float,
    eta_mech: float,
) -> np.ndarray:
    """
    Generic wind power curve:
    - v < v_ci: 0
    - v_ci <= v < v_r: cubic ramp
    - v_r <= v <= v_co: rated
    - v > v_co: 0
    Then apply efficiencies.
    """
    v = wind_ms.astype(float)
    p_raw = np.zeros_like(v)

    mask2 = (v >= v_ci) & (v < v_r)
    denom = (v_r**3 - v_ci**3) if (v_r**3 - v_ci**3) != 0 else 1.0
    p_raw[mask2] = p_rated_w * ((v[mask2] ** 3 - v_ci**3) / denom)

    mask3 = (v >= v_r) & (v <= v_co)
    p_raw[mask3] = p_rated_w

    p_out = eta_rect * eta_mech * p_raw
    return np.maximum(p_out, 0.0)


# -----------------------------
# D) Resampling helpers
# -----------------------------
def _ensure_time_index(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["time"] = pd.to_datetime(out["time"])
    out = out.sort_values("time").reset_index(drop=True)
    return out


def resample_weather_to_minutes(weather_hourly: pd.DataFrame, minutes: int) -> pd.DataFrame:
    """
    Convert hourly weather (time, G, v) to finer resolution using time interpolation.
    - Creates a new time grid at 'minutes' resolution
    - Interpolates G and v linearly in time

    Returns DataFrame: time, G, v at new resolution.
    """
    df = _ensure_time_index(weather_hourly)
    if minutes <= 0:
        raise ValueError("minutes must be positive")

    # Set time index for resampling
    df_i = df.set_index("time")[["G", "v"]]

    # Resample to new frequency and interpolate
    freq = f"{minutes}min"
    df_res = df_i.resample(freq).interpolate(method="time")

    # Drop any NaN at ends (if present)
    df_res = df_res.dropna()

    out = df_res.reset_index()
    return out


def aggregate_to_daily_from_hourly_for_sim(
    weather_hourly: pd.DataFrame,
    params: Params,
) -> pd.DataFrame:
    """
    For daily timestep simulation:
    - Compute PV and wind power hourly first (using hourly G, v)
    - Convert to daily energy (Wh)
    - Convert back to daily-average power (W) for a 24h step
    - Also provide daily avg G and v for plotting (mean)

    Returns DataFrame columns:
    time (daily timestamp), G (mean), v (mean),
    P_solar (daily avg W), P_wind (daily avg W), P_in (daily avg W)
    """
    dfh = _ensure_time_index(weather_hourly)
    dfh["date"] = dfh["time"].dt.date

    p_pv_total_rated = params.n_pv * params.p_pv_panel_rated_w

    # Hourly power (W)
    p_solar_h = pv_power_w(
        pv_rated_total_w=p_pv_total_rated,
        irradiance_wm2=dfh["G"].to_numpy(),
        derate=params.pv_derate,
        eta_dcdc=params.eta_pv_dcdc,
    )
    p_wind_h = wind_power_w(
        p_rated_w=params.p_w_rated_w,
        wind_ms=dfh["v"].to_numpy(),
        v_ci=params.v_ci,
        v_r=params.v_r,
        v_co=params.v_co,
        eta_rect=params.eta_rect,
        eta_mech=params.eta_mech,
    )

    dfh["P_solar_h"] = p_solar_h
    dfh["P_wind_h"] = p_wind_h

    # Daily energy (Wh) since hourly -> multiply by 1h and sum
    daily = dfh.groupby("date").agg(
        G_mean=("G", "mean"),
        v_mean=("v", "mean"),
        E_solar_Wh=("P_solar_h", lambda x: float((x * 1.0).sum())),
        E_wind_Wh=("P_wind_h", lambda x: float((x * 1.0).sum())),
    ).reset_index()

    # Daily average power over 24h (W)
    daily["P_solar"] = daily["E_solar_Wh"] / 24.0
    daily["P_wind"] = daily["E_wind_Wh"] / 24.0
    daily["P_in"] = daily["P_solar"] + daily["P_wind"]

    # Daily timestamp (midnight)
    daily_time = pd.to_datetime(daily["date"].astype(str))
    out = pd.DataFrame({
        "time": daily_time,
        "G": daily["G_mean"].astype(float),
        "v": daily["v_mean"].astype(float),
        "P_solar": daily["P_solar"].astype(float),
        "P_wind": daily["P_wind"].astype(float),
        "P_in": daily["P_in"].astype(float),
    })
    return out


# -----------------------------
# E) Core simulation on a given timeseries grid
# -----------------------------
def simulate_on_grid(params: Params, grid: pd.DataFrame, precomputed_generation: bool = False) -> Dict[str, object]:
    """
    Run simulation using 'grid' DataFrame that matches dt_hours.

    grid must contain columns:
    - time (datetime)
    - G, v (for plotting)
    - If precomputed_generation=False: must contain G and v; we compute P_solar, P_wind, P_in
    - If precomputed_generation=True: must contain P_solar, P_wind, P_in already
    """
    df = _ensure_time_index(grid)
    df["date"] = df["time"].dt.date
    df["month"] = df["time"].dt.to_period("M").astype(str)

    p_pv_total_rated = params.n_pv * params.p_pv_panel_rated_w
    if not precomputed_generation:
        df["P_solar"] = pv_power_w(
            pv_rated_total_w=p_pv_total_rated,
            irradiance_wm2=df["G"].to_numpy(),
            derate=params.pv_derate,
            eta_dcdc=params.eta_pv_dcdc,
        )
        df["P_wind"] = wind_power_w(
            p_rated_w=params.p_w_rated_w,
            wind_ms=df["v"].to_numpy(),
            v_ci=params.v_ci,
            v_r=params.v_r,
            v_co=params.v_co,
            eta_rect=params.eta_rect,
            eta_mech=params.eta_mech,
        )
        df["P_in"] = df["P_solar"] + df["P_wind"]

    # Daily demand generation (seeded)
    rng = np.random.default_rng(params.random_seed)
    unique_days = pd.Index(sorted(df["date"].unique()))
    daily_bikes_demand = pd.Series(
        rng.integers(low=0, high=params.max_bikes_per_day + 1, size=len(unique_days)),
        index=unique_days,
        name="bikes_demand",
    )
    daily_demand_wh = daily_bikes_demand.astype(float) * float(params.e_bike_wh)

    # Station limits
    p_load_max = params.n_ports * params.p_bike_w  # W

    # Allocate arrays
    n = len(df)
    E = np.zeros(n, dtype=float)
    SOC = np.zeros(n, dtype=float)
    P_charge = np.zeros(n, dtype=float)
    P_load = np.zeros(n, dtype=float)
    P_spill = np.zeros(n, dtype=float)

    daily_served_wh = pd.Series(0.0, index=unique_days, name="served_wh")
    daily_unmet_wh = pd.Series(0.0, index=unique_days, name="unmet_wh")

    # Initial state
    E[0] = params.soc_init * params.e_bat_max_wh
    SOC[0] = E[0] / params.e_bat_max_wh

    current_day = df.loc[0, "date"]
    remaining_day_demand_wh = float(daily_demand_wh.loc[current_day])

    for t in range(n):
        day = df.loc[t, "date"]
        if day != current_day:
            if remaining_day_demand_wh > 0:
                daily_unmet_wh.loc[current_day] += remaining_day_demand_wh
            current_day = day
            remaining_day_demand_wh = float(daily_demand_wh.loc[current_day])

        if t > 0:
            E_t = E[t - 1]
            SOC_t = SOC[t - 1]
        else:
            E_t = E[0]
            SOC_t = SOC[0]

        # A) E-bike load request
        if SOC_t <= params.soc_service_min:
            p_load_req = 0.0
        else:
            if remaining_day_demand_wh > 0:
                e_bike_step_cap_wh = params.eta_bike_charger * p_load_max * params.dt_hours
                e_to_bikes_wh = min(remaining_day_demand_wh, e_bike_step_cap_wh)
                p_load_req = e_to_bikes_wh / (params.eta_bike_charger * params.dt_hours)
            else:
                p_load_req = 0.0

        # B) Battery charge acceptance with cutoff
        if E_t >= params.e_bat_max_wh:
            p_charge_req = 0.0
        else:
            p_charge_req = min(float(df.loc[t, "P_in"]), params.p_ch_max_w)

        P_spill[t] = max(0.0, float(df.loc[t, "P_in"]) - p_charge_req)

        # C) Update battery energy
        e_charge_added_wh = params.eta_ch * p_charge_req * params.dt_hours
        e_dis_removed_wh = (p_load_req * params.dt_hours) / params.eta_dis if p_load_req > 0 else 0.0

        e_next = E_t + e_charge_added_wh - e_dis_removed_wh
        e_next = min(max(e_next, 0.0), params.e_bat_max_wh)

        P_charge[t] = p_charge_req
        P_load[t] = p_load_req
        E[t] = e_next
        SOC[t] = E[t] / params.e_bat_max_wh

        # D) Served energy update
        e_delivered_wh = params.eta_bike_charger * P_load[t] * params.dt_hours
        if e_delivered_wh > 0:
            served_now = min(remaining_day_demand_wh, e_delivered_wh)
            daily_served_wh.loc[current_day] += served_now
            remaining_day_demand_wh = max(0.0, remaining_day_demand_wh - served_now)

    if remaining_day_demand_wh > 0:
        daily_unmet_wh.loc[current_day] += remaining_day_demand_wh

    # Attach results
    df["P_charge"] = P_charge
    df["P_load"] = P_load
    df["P_spill"] = P_spill
    df["E_storage_Wh"] = E
    df["SOC"] = SOC

    # Daily table
    daily = pd.DataFrame({
        "date": unique_days,
        "bikes_demand": daily_bikes_demand.values,
        "demand_wh": daily_demand_wh.values,
        "served_wh": daily_served_wh.values,
        "unmet_wh": daily_unmet_wh.values,
    })
    daily["served_bikes_equiv"] = daily["served_wh"] / params.e_bike_wh
    daily["unmet_bikes_equiv"] = daily["unmet_wh"] / params.e_bike_wh

    # Monthly aggregation (kWh)
    monthly = df.groupby("month").agg(
        solar_kwh=("P_solar", lambda x: float((x * params.dt_hours).sum() / 1000.0)),
        wind_kwh=("P_wind", lambda x: float((x * params.dt_hours).sum() / 1000.0)),
        in_kwh=("P_in", lambda x: float((x * params.dt_hours).sum() / 1000.0)),
        spill_kwh=("P_spill", lambda x: float((x * params.dt_hours).sum() / 1000.0)),
        bikes_kwh=("P_load", lambda x: float((params.eta_bike_charger * x * params.dt_hours).sum() / 1000.0)),
        soc_mean=("SOC", "mean"),
    ).reset_index()

    # Summary
    solar_kwh = float((df["P_solar"] * params.dt_hours).sum() / 1000.0)
    wind_kwh = float((df["P_wind"] * params.dt_hours).sum() / 1000.0)
    total_kwh = float((df["P_in"] * params.dt_hours).sum() / 1000.0)
    spill_kwh = float((df["P_spill"] * params.dt_hours).sum() / 1000.0)
    bikes_kwh = float((params.eta_bike_charger * df["P_load"] * params.dt_hours).sum() / 1000.0)

    total_bikes_demand = int(daily["bikes_demand"].sum())
    demand_kwh = float(daily["demand_wh"].sum() / 1000.0)
    unserved_kwh = float(daily["unmet_wh"].sum() / 1000.0)

    charges_served = float(daily["served_wh"].sum() / params.e_bike_wh)
    charges_unserved = float(daily["unmet_wh"].sum() / params.e_bike_wh)

    full_idx = np.where(df["SOC"].to_numpy() >= 0.999)[0]
    time_to_full_hours = None
    if len(full_idx) > 0:
        time_to_full_hours = float(full_idx[0] * params.dt_hours)

    spill_pct = (spill_kwh / total_kwh * 100.0) if total_kwh > 0 else 0.0
    service_rate_pct = (bikes_kwh / demand_kwh * 100.0) if demand_kwh > 0 else 0.0

    summary = {
        "scenario_name": params.scenario_name,
        "timestep_label": params.timestep_label,
        "dt_hours": params.dt_hours,
        "pv_installed_w": params.n_pv * params.p_pv_panel_rated_w,
        "wind_rated_w": params.p_w_rated_w,
        "solar_generated_kwh": solar_kwh,
        "wind_generated_kwh": wind_kwh,
        "total_generated_kwh": total_kwh,
        "spilled_kwh": spill_kwh,
        "spill_percent": spill_pct,
        "ebike_delivered_kwh": bikes_kwh,
        "ebike_charges_served": charges_served,
        "bikes_demand_total": total_bikes_demand,
        "demand_kwh": demand_kwh,
        "unserved_kwh": unserved_kwh,
        "unserved_charges_equiv": charges_unserved,
        "service_rate_percent": service_rate_pct,
        "time_to_full_storage_hours": time_to_full_hours,
        "final_soc": float(df["SOC"].iloc[-1]),
    }

    return {
        "inputs": asdict(params),
        "timeseries": df,
        "daily": daily,
        "monthly": monthly,
        "summary": summary,
    }


# -----------------------------
# F) Main entry: prepare grid then simulate
# -----------------------------
def run_simulation(params: Params, weather_raw: pd.DataFrame, weather_is_hourly: bool = True) -> Dict[str, object]:
    """
    weather_raw must contain: time, G, v

    If params.dt_hours == 24: daily timestep approach:
    - requires hourly-like raw data (best if Open-Meteo hourly)
    - aggregates hourly PV/wind power to daily average power
    - simulates on daily grid

    If params.dt_hours < 1: resamples to minutes via interpolation

    If params.dt_hours == 1: uses as-is (or resamples if CSV isn't hourly but you still want 1h)
    """
    w = _ensure_time_index(weather_raw)

    if params.dt_hours >= 24.0 - 1e-9:
        # Daily simulation: compute hourly PV/wind then aggregate to daily-average power
        daily_grid = aggregate_to_daily_from_hourly_for_sim(w, params)
        return simulate_on_grid(params, daily_grid, precomputed_generation=True)

    # Minute-level (or 30min etc.) simulation
    if params.dt_hours < 1.0 - 1e-9:
        minutes = int(round(params.dt_hours * 60))
        grid = resample_weather_to_minutes(w, minutes=minutes)
        return simulate_on_grid(params, grid, precomputed_generation=False)

    # Hourly simulation
    if abs(params.dt_hours - 1.0) < 1e-9:
        # If CSV isn't hourly, resample to hourly (mean then interpolate)
        # If already hourly, this keeps it unchanged.
        df_i = w.set_index("time")[["G", "v"]].resample("1H").mean().interpolate(method="time").dropna().reset_index()
        return simulate_on_grid(params, df_i, precomputed_generation=False)

    # Fallback: if dt is some other value (e.g., 2 hours), we can resample to that
    minutes = int(round(params.dt_hours * 60))
    grid = resample_weather_to_minutes(w, minutes=minutes)
    return simulate_on_grid(params, grid, precomputed_generation=False)


# -----------------------------
# G) Excel export
# -----------------------------
def build_excel_bytes(results: Dict[str, object], include_timeseries: bool = True) -> bytes:
    """
    Export results to Excel in memory.
    Sheets:
    - Inputs
    - Summary
    - Monthly
    - Daily
    - TimeSeries (optional)
    """
    from io import BytesIO

    inputs_df = pd.DataFrame([results["inputs"]])
    summary_df = pd.DataFrame([results["summary"]])
    monthly_df = results["monthly"].copy()
    daily_df = results["daily"].copy()

    output = BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        inputs_df.to_excel(writer, index=False, sheet_name="Inputs")
        summary_df.to_excel(writer, index=False, sheet_name="Summary")
        monthly_df.to_excel(writer, index=False, sheet_name="Monthly")
        daily_df.to_excel(writer, index=False, sheet_name="Daily")

        if include_timeseries:
            ts = results["timeseries"].copy()
            ts.to_excel(writer, index=False, sheet_name="TimeSeries")

    return output.getvalue()
