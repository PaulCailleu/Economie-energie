import streamlit as st
import pandas as pd
import numpy as np
import altair as alt

# ===================== #
#     PARAMÈTRES FIXES  #
# ===================== #

YEARS = 20
DISCOUNT_RATE = 0.05
EXISTING_DIESEL_CAP = 16.0  # MW déjà installés


DEFAULT_PARAMS = {
    # CAPEX (investissements initiaux)
    "capex_solar": 935_000,      # €/MW
    "capex_wind": 1_850_000,     # €/MW
    "capex_diesel": 800_000,     # €/MW
    "capex_bat_p": 180_000,      # €/MW
    "capex_bat_e": 400_000,      # €/MWh

    # O&M fixes annuels
    "omfix_solar": 23_000,       # €/MW/an
    "omfix_wind": 45_000,        # €/MW/an
    "omfix_diesel": 50_000,      # €/MW/an
    "omfix_bat_p": 5_000,        # €/MW/an
    "omfix_bat_e": 5_000,        # €/MWh/an

    # Coûts variables
    "fuel_diesel": 47,           # €/MWh
    "omvar_diesel": 0,           # €/MWh

    # CO2 – indicateur
    "co2_factor_diesel": 0.24,   # tCO2/MWh
    "co2_price": 0,              # €/tCO₂ (non utilisé dans les coûts ici)

    # Coût de l'énergie non servie
    "cost_END": 10_000,          # €/MWh
}

PARAM_LABELS = {
    "capex_solar": "CAPEX solaire (€/MW installé)",
    "capex_wind": "CAPEX éolien (€/MW installé)",
    "capex_diesel": "CAPEX groupes diesel (€/MW installé)",
    "capex_bat_p": "CAPEX batterie – puissance (€/MW)",
    "capex_bat_e": "CAPEX batterie – énergie (€/MWh)",
    "omfix_solar": "O&M fixe solaire (€/MW/an)",
    "omfix_wind": "O&M fixe éolien (€/MW/an)",
    "omfix_diesel": "O&M fixe diesel (€/MW/an)",
    "omfix_bat_p": "O&M fixe batterie – puissance (€/MW/an)",
    "omfix_bat_e": "O&M fixe batterie – énergie (€/MWh/an)",
    "fuel_diesel": "Coût carburant diesel (€/MWh)",
    "omvar_diesel": "O&M variable diesel (€/MWh)",
    "co2_factor_diesel": "Facteur d'émission diesel (tCO₂/MWh élec)",
    "cost_END": "Coût de l'énergie non servie (€/MWh)",
}


# ===================== #
#   PROFIL DÉMO         #
# ===================== #

def generate_demo_profiles(n_days=7):
    hours = np.arange(n_days * 24)

    load = 12 + 5 * np.sin(2 * np.pi * hours / 24 - np.pi / 2)
    load = np.clip(load, 7, 24)

    cf_solar = np.maximum(0, np.sin(2 * np.pi * (hours % 24) / 24 - np.pi / 2))
    cf_solar = cf_solar * 0.9

    rng = np.random.default_rng(42)
    cf_wind = 0.4 + 0.2 * rng.standard_normal(len(hours))
    cf_wind = np.clip(cf_wind, 0, 1)

    return pd.DataFrame(
        {"hour": hours, "load": load, "cf_solar": cf_solar, "cf_wind": cf_wind}
    )


# ===================== #
#  SIMULATION "EXCEL"   #
# ===================== #

def simulate_rule_based(profiles, K_solar, K_wind, K_diesel, K_bat_p, K_bat_e):
    load = profiles["load"].values
    cf_solar = profiles["cf_solar"].values
    cf_wind = profiles["cf_wind"].values
    T = len(load)

    P_solar = np.zeros(T)
    P_wind = np.zeros(T)
    P_diesel = np.zeros(T)
    P_ch = np.zeros(T)
    P_dis = np.zeros(T)
    E_bat = np.zeros(T)
    END = np.zeros(T)
    Spill = np.zeros(T)

    e = K_bat_e  # batterie démarre pleine

    for t in range(T):
        P_solar_max = cf_solar[t] * K_solar
        P_wind_max = cf_wind[t] * K_wind
        ren_dispo = P_solar_max + P_wind_max
        d = load[t]

        if ren_dispo >= d:
            P_solar[t] = min(P_solar_max, d)
            reste = d - P_solar[t]
            P_wind[t] = min(P_wind_max, reste)

            surplus = ren_dispo - d
            charge_possible = min(surplus, K_bat_p, max(0.0, K_bat_e - e))
            P_ch[t] = charge_possible
            e += charge_possible

            Spill[t] = surplus - charge_possible

        else:
            P_solar[t] = P_solar_max
            P_wind[t] = P_wind_max
            deficit = d - ren_dispo

            dis_possible = min(deficit, K_bat_p, e)
            P_dis[t] = dis_possible
            e -= dis_possible
            deficit -= dis_possible

            diesel_possible = min(deficit, K_diesel)
            P_diesel[t] = diesel_possible
            deficit -= diesel_possible

            END[t] = max(0.0, deficit)

        e = min(max(e, 0.0), K_bat_e)
        E_bat[t] = e

    return pd.DataFrame(
        {
            "load": load,
            "P_solar": P_solar,
            "P_wind": P_wind,
            "P_diesel": P_diesel,
            "P_ch": P_ch,
            "P_dis": P_dis,
            "E_bat": E_bat,
            "END": END,
            "Spill": Spill,
        }
    )


# ===================== #
#   CALCUL DES INDICS   #
# ===================== #

def compute_metrics(ts, caps, params):
    p = params
    K_solar = caps["solar"]
    K_wind = caps["wind"]
    K_diesel = caps["diesel"]
    K_bat_p = caps["bat_p"]
    K_bat_e = caps["bat_e"]

    total_load = float(ts["load"].sum())
    E_diesel = float(ts["P_diesel"].sum())
    E_END = float(ts["END"].sum())

    K_diesel_new = max(0.0, K_diesel - EXISTING_DIESEL_CAP)

    C_inv = (
        p["capex_solar"] * K_solar
        + p["capex_wind"] * K_wind
        + p["capex_diesel"] * K_diesel_new
        + p["capex_bat_p"] * K_bat_p
        + p["capex_bat_e"] * K_bat_e
    )

    C_om_fix = (
        p["omfix_solar"] * K_solar
        + p["omfix_wind"] * K_wind
        + p["omfix_diesel"] * K_diesel
        + p["omfix_bat_p"] * K_bat_p
        + p["omfix_bat_e"] * K_bat_e
    )

    C_fuel = E_diesel * p["fuel_diesel"]
    C_om_var = E_diesel * p["omvar_diesel"]
    C_END = E_END * p["cost_END"]

    CO2 = E_diesel * p["co2_factor_diesel"]
    C_CO2 = CO2 * p["co2_price"]  # 0 par défaut

    C_annual = C_om_fix + C_fuel + C_om_var + C_END

    PV_factor = (1 - (1 + DISCOUNT_RATE) ** -YEARS) / DISCOUNT_RATE
    C_total = C_inv + C_annual * PV_factor
    PV_CO2 = CO2 * PV_factor

    unavailability_hours = int((ts["END"] > 0).sum())

    energy_served = total_load - E_END
    LCOE = (C_total / (energy_served * PV_factor)) if energy_served > 0 else None

    return {
        "C_inv": C_inv,
        "C_om_fix": C_om_fix,
        "C_fuel": C_fuel,
        "C_om_var": C_om_var,
        "C_END": C_END,
        "C_CO2": C_CO2,
        "C_annual": C_annual,
        "C_total": C_total,
        "CO2": CO2,
        "PV_CO2": PV_CO2,
        "E_diesel": E_diesel,
        "E_END": E_END,
        "END_ratio": E_END / total_load if total_load > 0 else 0.0,
        "total_load": total_load,
        "unavailability_hours": unavailability_hours,
        "LCOE": LCOE,
        "PV_factor": PV_factor,
    }


# ===================== #
#   GRID SEARCH COMPLET #
# ===================== #

def grid_search_optimal_mix(
    profiles,
    params,
    objective_mode,
    max_end_ratio,
    solar_range,
    wind_range,
    diesel_range,
    bat_p_range,
    bat_e_range,
):
    best_caps = None
    best_metrics = None
    best_ts = None
    best_value = None

    rows = []

    total_combos = (
        len(solar_range) * len(wind_range) * len(diesel_range) * len(bat_p_range) * len(bat_e_range)
    )
    progress = st.progress(0)
    done = 0
    denom = max(1, total_combos)

    for Ks in solar_range:
        for Kw in wind_range:
            for Kd in diesel_range:
                for Kbp in bat_p_range:
                    for Kbe in bat_e_range:
                        caps = {
                            "solar": float(Ks),
                            "wind": float(Kw),
                            "diesel": float(Kd),
                            "bat_p": float(Kbp),
                            "bat_e": float(Kbe),
                        }

                        ts = simulate_rule_based(
                            profiles, caps["solar"], caps["wind"], caps["diesel"], caps["bat_p"], caps["bat_e"]
                        )
                        metrics = compute_metrics(ts, caps, params)
                        feasible = metrics["END_ratio"] <= max_end_ratio

                        rows.append(
                            {
                                "solar": caps["solar"],
                                "wind": caps["wind"],
                                "diesel": caps["diesel"],
                                "bat_p": caps["bat_p"],
                                "bat_e": caps["bat_e"],
                                "C_total": metrics["C_total"],
                                "LCOE": metrics["LCOE"] if metrics["LCOE"] is not None else np.nan,
                                "CO2": metrics["CO2"],
                                "END_ratio": metrics["END_ratio"],
                                "feasible": feasible,
                                "phase": "full",
                            }
                        )

                        if feasible:
                            value = metrics["C_total"] if objective_mode == "cost" else metrics["CO2"]
                            if (best_value is None) or (value < best_value):
                                best_value = value
                                best_caps = caps
                                best_metrics = metrics
                                best_ts = ts

                        done += 1
                        progress.progress(done / denom)

    df_scenarios = pd.DataFrame(rows)
    return best_caps, best_metrics, best_ts, df_scenarios


# ===================== #
#   OPTI RAPIDE (COARSE→FINE)
# ===================== #

def grid_search_optimal_mix_fast(
    profiles,
    params,
    objective_mode,
    max_end_ratio,
    solar_range,
    wind_range,
    diesel_range,
    bat_p_range,
    bat_e_range,
    coarse_stride=2,
    refine_radius_steps=1,
    keep_rows=True,
):
    def eval_caps(caps):
        ts = simulate_rule_based(
            profiles, caps["solar"], caps["wind"], caps["diesel"], caps["bat_p"], caps["bat_e"]
        )
        metrics = compute_metrics(ts, caps, params)
        feasible = metrics["END_ratio"] <= max_end_ratio
        value = metrics["C_total"] if objective_mode == "cost" else metrics["CO2"]
        return ts, metrics, feasible, value

    def as_float_array(arr):
        return np.array(list(arr), dtype=float)

    Ks = as_float_array(solar_range)
    Kw = as_float_array(wind_range)
    Kd = as_float_array(diesel_range)
    Kbp = as_float_array(bat_p_range)
    Kbe = as_float_array(bat_e_range)

    Ks_c = Ks[::coarse_stride] if len(Ks) else Ks
    Kw_c = Kw[::coarse_stride] if len(Kw) else Kw
    Kd_c = Kd[::coarse_stride] if len(Kd) else Kd
    Kbp_c = Kbp[::coarse_stride] if len(Kbp) else Kbp
    Kbe_c = Kbe[::coarse_stride] if len(Kbe) else Kbe

    best_caps = None
    best_metrics = None
    best_ts = None
    best_value = None
    rows = []

    total_coarse = len(Ks_c) * len(Kw_c) * len(Kd_c) * len(Kbp_c) * len(Kbe_c)
    progress = st.progress(0)
    done = 0
    denom = max(1, total_coarse)

    for s in Ks_c:
        for w in Kw_c:
            for d in Kd_c:
                for bp in Kbp_c:
                    for be in Kbe_c:
                        caps = {"solar": float(s), "wind": float(w), "diesel": float(d),
                                "bat_p": float(bp), "bat_e": float(be)}
                        ts, metrics, feasible, value = eval_caps(caps)

                        if keep_rows:
                            rows.append({
                                "solar": caps["solar"], "wind": caps["wind"], "diesel": caps["diesel"],
                                "bat_p": caps["bat_p"], "bat_e": caps["bat_e"],
                                "C_total": metrics["C_total"],
                                "LCOE": metrics["LCOE"] if metrics["LCOE"] is not None else np.nan,
                                "CO2": metrics["CO2"],
                                "END_ratio": metrics["END_ratio"],
                                "feasible": feasible,
                                "phase": "coarse",
                            })

                        if feasible and ((best_value is None) or (value < best_value)):
                            best_value = value
                            best_caps, best_metrics, best_ts = caps, metrics, ts

                        done += 1
                        progress.progress(done / denom)

    if best_caps is None:
        st.warning("Aucun scénario faisable trouvé en phase coarse → fallback grid-search complet.")
        return grid_search_optimal_mix(
            profiles, params, objective_mode, max_end_ratio,
            solar_range, wind_range, diesel_range, bat_p_range, bat_e_range
        )

    def idx_nearest(arr, v):
        return int(np.argmin(np.abs(arr - v)))

    i_s = idx_nearest(Ks, best_caps["solar"])
    i_w = idx_nearest(Kw, best_caps["wind"])
    i_d = idx_nearest(Kd, best_caps["diesel"])
    i_bp = idx_nearest(Kbp, best_caps["bat_p"])
    i_be = idx_nearest(Kbe, best_caps["bat_e"])

    def neighborhood(arr, i0):
        lo = max(0, i0 - refine_radius_steps)
        hi = min(len(arr) - 1, i0 + refine_radius_steps)
        return arr[lo:hi + 1]

    Ks_f = neighborhood(Ks, i_s)
    Kw_f = neighborhood(Kw, i_w)
    Kd_f = neighborhood(Kd, i_d)
    Kbp_f = neighborhood(Kbp, i_bp)
    Kbe_f = neighborhood(Kbe, i_be)

    total_fine = len(Ks_f) * len(Kw_f) * len(Kd_f) * len(Kbp_f) * len(Kbe_f)
    done = 0
    denom = max(1, total_fine)
    st.caption(f"Raffinement local : {total_fine} combinaisons")

    for s in Ks_f:
        for w in Kw_f:
            for d in Kd_f:
                for bp in Kbp_f:
                    for be in Kbe_f:
                        caps = {"solar": float(s), "wind": float(w), "diesel": float(d),
                                "bat_p": float(bp), "bat_e": float(be)}
                        ts, metrics, feasible, value = eval_caps(caps)

                        if keep_rows:
                            rows.append({
                                "solar": caps["solar"], "wind": caps["wind"], "diesel": caps["diesel"],
                                "bat_p": caps["bat_p"], "bat_e": caps["bat_e"],
                                "C_total": metrics["C_total"],
                                "LCOE": metrics["LCOE"] if metrics["LCOE"] is not None else np.nan,
                                "CO2": metrics["CO2"],
                                "END_ratio": metrics["END_ratio"],
                                "feasible": feasible,
                                "phase": "refine",
                            })

                        if feasible and (value < best_value):
                            best_value = value
                            best_caps, best_metrics, best_ts = caps, metrics, ts

                        done += 1
                        progress.progress(done / denom)

    df_scenarios = pd.DataFrame(rows) if keep_rows else pd.DataFrame()
    return best_caps, best_metrics, best_ts, df_scenarios


# ===================== #
#   HEATMAPS (GALERIE)  #
# ===================== #

def best_per_cell(df, x, y, metric="C_total"):
    d = df.copy()
    d = d.sort_values(metric, ascending=True)
    d = d.drop_duplicates(subset=[x, y], keep="first")
    return d


def heatmap_rect(df, x, y, z="C_total", title=None):
    return (
        alt.Chart(df)
        .mark_rect()
        .encode(
            x=alt.X(f"{x}:Q", title=x),
            y=alt.Y(f"{y}:Q", title=y),
            color=alt.Color(f"{z}:Q", title=z),
            tooltip=[
                alt.Tooltip(f"{x}:Q", title=x),
                alt.Tooltip(f"{y}:Q", title=y),
                alt.Tooltip("solar:Q", title="solar (MW)"),
                alt.Tooltip("wind:Q", title="wind (MW)"),
                alt.Tooltip("diesel:Q", title="diesel (MW)"),
                alt.Tooltip("bat_p:Q", title="bat_p (MW)"),
                alt.Tooltip("bat_e:Q", title="bat_e (MWh)"),
                alt.Tooltip("C_total:Q", format=",.0f"),
                alt.Tooltip("LCOE:Q", format=",.2f"),
                alt.Tooltip("CO2:Q", format=",.0f"),
                alt.Tooltip("END_ratio:Q", format=".4%"),
                alt.Tooltip("feasible:N", title="feasible"),
                alt.Tooltip("phase:N", title="phase"),
            ],
        )
        .properties(height=260, title=title)
        .interactive()
    )


# ===================== #
#       STREAMLIT       #
# ===================== #

st.set_page_config(page_title="Mix de l'île - simulation & coûts", layout="wide")

st.title("⚡ Système électrique isolé de l'île – simulation type Excel & analyse techno-économique")

st.markdown(
    """
Cet outils, permet de simuler les perfomances d'un mix énergétique à partir d'un fichier de données météo et de consommation.
"""
)

# -------------------------- #
#  0️⃣ DONNÉES & HYPOTHÈSES  #
# -------------------------- #

st.markdown("## Hypothèses de coût et données météo et consommation")

col0a, col0b = st.columns([3, 3])

with col0a:
    st.markdown("### Profil de charge et de production")
    uploaded = st.file_uploader(
        "Charger un CSV (colonnes : hour, load, cf_wind, cf_solar)",
        type=["csv"],
    )

    if uploaded is not None:
        profiles = pd.read_csv(uploaded, decimal=",")
        for col in ["load", "cf_wind", "cf_solar"]:
            profiles[col] = pd.to_numeric(profiles[col], errors="coerce")
        st.success("Profil chargé ✅")
    else:
        st.info("Aucun fichier : profil **démo** (7 jours).")
        profiles = generate_demo_profiles(n_days=7)

    st.caption("Aperçu des données")
    st.dataframe(profiles.head())

with col0b:
    st.markdown("### Hypothèses techno-économiques")

    params = {}

    with st.expander("CAPEX (€/MW ou €/MWh)", expanded=False):
        c1, c2 = st.columns(2)
        with c1:
            for key in ["capex_solar", "capex_wind", "capex_diesel"]:
                params[key] = st.number_input(PARAM_LABELS[key], value=float(DEFAULT_PARAMS[key]), step=10.0)
        with c2:
            for key in ["capex_bat_p", "capex_bat_e"]:
                params[key] = st.number_input(PARAM_LABELS[key], value=float(DEFAULT_PARAMS[key]), step=10.0)

    with st.expander("O&M fixes (€/MW/an ou €/MWh/an)", expanded=False):
        c1, c2 = st.columns(2)
        with c1:
            for key in ["omfix_solar", "omfix_wind", "omfix_diesel"]:
                params[key] = st.number_input(PARAM_LABELS[key], value=float(DEFAULT_PARAMS[key]), step=10.0)
        with c2:
            for key in ["omfix_bat_p", "omfix_bat_e"]:
                params[key] = st.number_input(PARAM_LABELS[key], value=float(DEFAULT_PARAMS[key]), step=10.0)

    with st.expander("Variables, CO₂ (indicateur), END", expanded=False):
        c1, c2 = st.columns(2)
        with c1:
            for key in ["fuel_diesel", "omvar_diesel"]:
                params[key] = st.number_input(PARAM_LABELS[key], value=float(DEFAULT_PARAMS[key]), step=1.0)
        with c2:
            for key in ["co2_factor_diesel", "cost_END"]:
                params[key] = st.number_input(PARAM_LABELS[key], value=float(DEFAULT_PARAMS[key]), step=1.0)

        params["co2_price"] = 0.0  # fixé à 0 pour coller à ton cadre

st.markdown("### Objectif du mode optimisation")

c1, c2 = st.columns(2)
with c1:
    max_end_ratio = st.slider(
        "Indisponibilité max (END / demande annuelle)",
        0.0, 1.0, 0.0, 0.1
    )
with c2:
    objective_mode_label = st.radio(
        "Objectif optimisation",
        ["Minimiser le coût (C_total)", "Minimiser les émissions (CO2)"],
    )
    objective_mode = "cost" if objective_mode_label.startswith("Minimiser le coût") else "co2"


# ------------------------------- #
#  1️⃣ MODE D'ÉTUDE               #
# ------------------------------- #

st.markdown("## Choix du type d'étude")

mode = st.radio(
    "Mode",
    [" Tester un scénario ", " Optimiser pour la méteo et la"],
)

current_caps = None
current_metrics = None
current_ts = None
df_scenarios = None

# ===================== #
#     MODE SCÉNARIO     #
# ===================== #

if mode.startswith("🧪"):
    st.markdown("## 2️⃣ Scénario – capacités installées")

    c1, c2, c3 = st.columns(3)
    with c1:
        solar_cap = st.slider("Solaire (MW)", 0.0, 150.0, 10.0, 1.0)
        wind_cap = st.slider("Éolien (MW)", 0.0, 150.0, 10.0, 1.0)
    with c2:
        diesel_cap = st.slider("Diesel (MW)", 0.0, 150.0, 24.0, 1.0)
    with c3:
        bat_p_cap = st.slider("Batterie puissance (MW)", 0.0, 50.0, 2.0, 0.5)
        bat_e_cap = st.slider("Batterie énergie (MWh)", 0.0, 200.0, 4.0, 1.0)

    if st.button("▶️ Lancer la simulation"):
        current_caps = {
            "solar": float(solar_cap),
            "wind": float(wind_cap),
            "diesel": float(diesel_cap),
            "bat_p": float(bat_p_cap),
            "bat_e": float(bat_e_cap),
        }
        with st.spinner("Simulation horaire…"):
            current_ts = simulate_rule_based(
                profiles,
                current_caps["solar"],
                current_caps["wind"],
                current_caps["diesel"],
                current_caps["bat_p"],
                current_caps["bat_e"],
            )
            current_metrics = compute_metrics(current_ts, current_caps, params)
        st.success("Simulation terminée ✅")

# ===================== #
#   MODE OPTIMISATION   #
# ===================== #

else:
    st.markdown("## 2️⃣ Optimisation – définition de la grille + accélération")

    st.info("Tu définis des plages via sliders + un pas. Ensuite optimisation rapide (coarse → refine).")

    c1, c2 = st.columns(2)
    with c1:
        solar_rng = st.slider("Solaire (MW) – plage", 0.0, 250.0, (0.0, 20.0), 1.0)
        solar_step = st.slider("Solaire – pas (MW)", 1.0, 50.0, 5.0, 1.0)

        wind_rng = st.slider("Éolien (MW) – plage", 0.0, 200.0, (0.0, 30.0), 1.0)
        wind_step = st.slider("Éolien – pas (MW)", 1.0, 50.0, 5.0, 1.0)

        diesel_rng = st.slider("Diesel (MW) – plage", 0.0, 200.0, (8.0, 24.0), 1.0)
        diesel_step = st.slider("Diesel – pas (MW)", 1.0, 50.0, 4.0, 1.0)

    with c2:
        bat_p_rng = st.slider("Batterie P (MW) – plage", 0.0, 80.0, (0.0, 4.0), 0.5)
        bat_p_step = st.slider("Batterie P – pas (MW)", 0.5, 20.0, 2.0, 0.5)

        bat_e_rng = st.slider("Batterie E (MWh) – plage", 0.0, 300.0, (0.0, 8.0), 1.0)
        bat_e_step = st.slider("Batterie E – pas (MWh)", 1.0, 80.0, 4.0, 1.0)

    st.markdown("### Accélération coarse → refine")
    c3, c4 = st.columns(2)
    with c3:
        coarse_stride = st.slider(
            "Coarse stride (1 = complet, 2 = 1 point sur 2, 3 = 1 sur 3)",
            1, 5, 2, 1
        )
    with c4:
        refine_radius_steps = st.slider(
            "Refine radius (en nb de pas autour du meilleur coarse)",
            0, 4, 1, 1
        )

    def make_range(vmin, vmax, step):
        if vmax < vmin:
            vmin, vmax = vmax, vmin
        return np.arange(vmin, vmax + 1e-9, step)

    solar_range = make_range(*solar_rng, solar_step)
    wind_range = make_range(*wind_rng, wind_step)
    diesel_range = make_range(*diesel_rng, diesel_step)
    bat_p_range = make_range(*bat_p_rng, bat_p_step)
    bat_e_range = make_range(*bat_e_rng, bat_e_step)

    total_full = len(solar_range) * len(wind_range) * len(diesel_range) * len(bat_p_range) * len(bat_e_range)
    st.caption(f"Grille complète : **{total_full}** combinaisons (si coarse_stride=1).")

    if st.button("🔍 Lancer l'optimisation rapide"):
        with st.spinner("Optimisation coarse → refine…"):
            best_caps, best_metrics, best_ts, df_scenarios = grid_search_optimal_mix_fast(
                profiles,
                params,
                objective_mode,
                max_end_ratio,
                solar_range,
                wind_range,
                diesel_range,
                bat_p_range,
                bat_e_range,
                coarse_stride=coarse_stride,
                refine_radius_steps=refine_radius_steps,
                keep_rows=True,
            )

        if best_caps is None:
            st.error("Aucun scénario faisable trouvé (même après fallback).")
        else:
            current_caps = best_caps
            current_metrics = best_metrics
            current_ts = best_ts
            st.success("Optimisation terminée ✅")

# ------------------------------- #
#   3️⃣ SYNTHÈSE + GALERIE         #
# ------------------------------- #

if current_metrics is not None and current_ts is not None and current_caps is not None:
    st.markdown("## 3️⃣ Synthèse économique & fiabilité")

    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric("total C_I", f"{current_metrics['C_inv']:.0f} €")
    with c2:
        st.metric("total C_O&M", f"{current_metrics['C_om_fix']:.0f} €/an")
    with c3:
        st.metric("Total C_Fuel", f"{current_metrics['C_fuel']:.0f} €/an")

    c4, c5, c6 = st.columns(3)
    with c4:
        st.metric("Total C_LOLE", f"{current_metrics['C_END']:.0f} €/an")
    with c5:
        st.metric("total present costs", f"{current_metrics['C_total']:.0f} €_2026")
    with c6:
        st.metric("System LCOE", f"{current_metrics['LCOE']:.1f} €_2026/MWh" if current_metrics["LCOE"] is not None else "n/a")

    st.metric("CO2_Emissions", f"{current_metrics['CO2']:.0f} tCO2/an")

    f1, f2 = st.columns(2)
    with f1:
        st.metric("Heures d'indisponibilité", f"{current_metrics['unavailability_hours']}")
    with f2:
        st.metric("END / demande", f"{current_metrics['END_ratio']*100:.3f} %")

    st.markdown("### Capacités installées (solution)")
    st.write(current_caps)

    # ------------------------------- #
    #   🧊 GALERIE HEATMAPS           #
    # ------------------------------- #
    if df_scenarios is not None and len(df_scenarios) > 0:
        st.markdown("## 🧊 Galerie de heatmaps – coûts (toutes les vues utiles)")

        show_only_feasible = st.checkbox(
            "Afficher seulement les scénarios faisables (END_ratio ≤ contrainte)",
            value=True,
        )

        df_hm = df_scenarios.copy()
        if show_only_feasible:
            df_hm = df_hm[df_hm["feasible"]].copy()

        if len(df_hm) == 0:
            st.warning("Aucun scénario à afficher avec ce filtre.")
        else:
            pairs = [
                ("solar", "wind"),
                ("solar", "diesel"),
                ("wind", "diesel"),
                ("solar", "bat_p"),
                ("solar", "bat_e"),
                ("wind", "bat_p"),
                ("wind", "bat_e"),
                ("diesel", "bat_p"),
                ("diesel", "bat_e"),
                ("bat_p", "bat_e"),
            ]

            st.subheader("Heatmaps – C_total (meilleur coût par case)")
            for i in range(0, len(pairs), 2):
                cols = st.columns(2)
                for j in range(2):
                    if i + j >= len(pairs):
                        break
                    x, y = pairs[i + j]
                    df_cell = best_per_cell(df_hm, x, y, metric="C_total")
                    with cols[j]:
                        st.altair_chart(
                            heatmap_rect(df_cell, x, y, z="C_total", title=f"C_total — {x} vs {y}"),
                            use_container_width=True,
                        )

            with st.expander("Voir aussi les heatmaps de LCOE", expanded=False):
                for i in range(0, len(pairs), 2):
                    cols = st.columns(2)
                    for j in range(2):
                        if i + j >= len(pairs):
                            break
                        x, y = pairs[i + j]
                        df_cell = best_per_cell(df_hm, x, y, metric="LCOE")
                        with cols[j]:
                            st.altair_chart(
                                heatmap_rect(df_cell, x, y, z="LCOE", title=f"LCOE — {x} vs {y}"),
                                use_container_width=True,
                            )

            with st.expander("Voir aussi les heatmaps de CO2", expanded=False):
                for i in range(0, len(pairs), 2):
                    cols = st.columns(2)
                    for j in range(2):
                        if i + j >= len(pairs):
                            break
                        x, y = pairs[i + j]
                        df_cell = best_per_cell(df_hm, x, y, metric="CO2")
                        with cols[j]:
                            st.altair_chart(
                                heatmap_rect(df_cell, x, y, z="CO2", title=f"CO2 — {x} vs {y}"),
                                use_container_width=True,
                            )

            with st.expander("Table des scénarios (top 300 par coût)", expanded=False):
                st.dataframe(df_scenarios.sort_values("C_total").head(300))

    # ------------------------------- #
    #   4️⃣ VISUS SYSTÈME             #
    # ------------------------------- #

    st.markdown("## 4️⃣ Compte rendu visuel du fonctionnement")

    st.subheader("Courbe de charge (48 premières heures)")
    n_show = min(48, len(current_ts))
    df_plot = current_ts.head(n_show).copy()
    df_plot["hour_idx"] = np.arange(len(df_plot))
    st.line_chart(df_plot.set_index("hour_idx")[["load"]])

    st.subheader("Production par filière (empilée, 48 premières heures)")
    df_prod = df_plot.rename(
        columns={
            "P_solar": "Solaire",
            "P_wind": "Éolien",
            "P_dis": "Batterie (décharge)",
            "P_diesel": "Diesel",
        }
    )
    prod = df_prod[["hour_idx", "Solaire", "Éolien", "Batterie (décharge)", "Diesel"]].melt(
        id_vars="hour_idx", var_name="Filière", value_name="Puissance (MW)"
    )
    chart = (
        alt.Chart(prod)
        .mark_area()
        .encode(
            x=alt.X("hour_idx:Q", title="Heure"),
            y=alt.Y("Puissance (MW):Q", stack="zero"),
            color=alt.Color("Filière:N", title="Filière"),
            tooltip=["hour_idx", "Filière", "Puissance (MW)"],
        )
    )
    st.altair_chart(chart, use_container_width=True)

    st.subheader("Énergie non servie et surplus renouvelable (48 premières heures)")
    df_end = df_plot.rename(columns={"END": "Énergie non servie", "Spill": "Surplus EnR"})
    st.line_chart(df_end.set_index("hour_idx")[["Énergie non servie", "Surplus EnR"]])

    st.markdown("## 🔋 Batterie (5 jours)")
    n_show_bat = min(5 * 24, len(current_ts))
    bat_plot = current_ts.head(n_show_bat).copy()
    bat_plot["hour_idx"] = np.arange(len(bat_plot))
    bat_power = bat_plot[["hour_idx", "P_ch", "P_dis"]].melt(
        id_vars="hour_idx", var_name="Mode", value_name="Puissance (MW)"
    )
    bat_power_chart = (
        alt.Chart(bat_power)
        .mark_line()
        .encode(
            x="hour_idx",
            y="Puissance (MW)",
            color=alt.Color("Mode:N", title="Mode"),
            tooltip=["hour_idx", "Mode", "Puissance (MW)"],
        )
    )
    st.altair_chart(bat_power_chart, use_container_width=True)

else:
    st.info("➡️ Lance une simulation (scénario) ou une optimisation pour afficher les résultats.")
