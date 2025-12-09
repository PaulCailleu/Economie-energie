import streamlit as st
import pandas as pd
import numpy as np
import altair as alt

# ===================== #
#     PARAMÈTRES FIXES  #
# ===================== #

YEARS = 20
DISCOUNT_RATE = 0.05


def annuity_factor(r=DISCOUNT_RATE, n=YEARS):
    return r / (1 - (1 + r) ** -n)


A = annuity_factor()

# Paramètres techno-éco par défaut (à adapter si besoin)
DEFAULT_PARAMS = {
    "capex_solar": 600_000,      # €/MW
    "capex_wind": 1_300_000,     # €/MW
    "capex_diesel": 600_000,     # €/MW
    "capex_bat_p": 150_000,      # €/MW
    "capex_bat_e": 250_000,      # €/MWh

    "omfix_solar": 15_000,       # €/MW/an
    "omfix_wind": 40_000,        # €/MW/an
    "omfix_diesel": 20_000,      # €/MW/an
    "omfix_bat_p": 5_000,        # €/MW/an
    "omfix_bat_e": 5_000,        # €/MWh/an

    "fuel_diesel": 350,          # €/MWh
    "omvar_diesel": 5,           # €/MWh

    "co2_factor_diesel": 0.7,    # tCO2/MWh
    "co2_price": 80,             # €/tCO2

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
    "co2_price": "Prix du CO₂ (€/tCO₂)",
    "cost_END": "Coût de l'énergie non servie (€/MWh)",
}

# ===================== #
#   PROFIL DÉMO         #
# ===================== #

def generate_demo_profiles(n_days=7):
    """Profil fictif si l'utilisateur n’upload pas de données."""
    hours = np.arange(n_days * 24)
    # Demande base 12 MW + variation journalière
    load = 12 + 5 * np.sin(2 * np.pi * hours / 24 - np.pi / 2)
    load = np.clip(load, 7, 24)

    # Solaire : seulement le jour
    cf_solar = np.maximum(0, np.sin(2 * np.pi * (hours % 24) / 24 - np.pi / 2))
    cf_solar = cf_solar * 0.9

    # Éolien : bruit autour de 0.4
    rng = np.random.default_rng(42)
    cf_wind = 0.4 + 0.2 * rng.standard_normal(len(hours))
    cf_wind = np.clip(cf_wind, 0, 1)

    df = pd.DataFrame({
        "hour": hours,
        "load": load,
        "cf_solar": cf_solar,
        "cf_wind": cf_wind,
    })
    return df


# ===================== #
#  SIMULATION "EXCEL"   #
# ===================== #

def simulate_rule_based(profiles, K_solar, K_wind, K_diesel, K_bat_p, K_bat_e):
    """
    Simulation horaire avec les règles type Excel :
    - priorité aux EnR (solaire + éolien),
    - si surplus : charge batterie puis curtailment,
    - si déficit : décharge batterie, puis diesel, puis END.
    """
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

    # Batterie démarre pleine (comme dans l'énoncé)
    e = K_bat_e

    for t in range(T):
        # Potentiel renouvelable
        P_solar_max = cf_solar[t] * K_solar
        P_wind_max = cf_wind[t] * K_wind
        ren_dispo = P_solar_max + P_wind_max
        d = load[t]

        if ren_dispo >= d:
            # Toute la demande couverte par les EnR
            P_solar[t] = min(P_solar_max, d)
            reste = d - P_solar[t]
            P_wind[t] = min(P_wind_max, reste)

            surplus = ren_dispo - d

            # Charge batterie avec le surplus
            charge_possible = min(surplus, K_bat_p, max(0.0, K_bat_e - e))
            P_ch[t] = charge_possible
            e += charge_possible

            Spill[t] = surplus - charge_possible

            P_diesel[t] = 0.0
            P_dis[t] = 0.0
            END[t] = 0.0

        else:
            # Renouvelables insuffisants : on utilise tout
            P_solar[t] = P_solar_max
            P_wind[t] = P_wind_max
            deficit = d - ren_dispo

            # Décharge batterie
            dis_possible = min(deficit, K_bat_p, e)
            P_dis[t] = dis_possible
            e -= dis_possible
            deficit -= dis_possible

            # Diesel
            diesel_possible = min(deficit, K_diesel)
            P_diesel[t] = diesel_possible
            deficit -= diesel_possible

            # Ce qui reste, c'est de l'END
            END[t] = max(0.0, deficit)

            P_ch[t] = 0.0
            Spill[t] = 0.0

        # On borne le niveau de batterie
        e = min(max(e, 0.0), K_bat_e)
        E_bat[t] = e

    ts = pd.DataFrame({
        "load": load,
        "P_solar": P_solar,
        "P_wind": P_wind,
        "P_diesel": P_diesel,
        "P_ch": P_ch,
        "P_dis": P_dis,
        "E_bat": E_bat,
        "END": END,
        "Spill": Spill,
    })
    return ts


# ===================== #
#   CALCUL DES INDICS   #
# ===================== #

def compute_metrics(ts, caps, params):
    """
    Calcule les coûts, CO2, END, heures d'indisponibilité, LCOE, etc.
    """
    p = params
    K_solar = caps["solar"]
    K_wind = caps["wind"]
    K_diesel = caps["diesel"]
    K_bat_p = caps["bat_p"]
    K_bat_e = caps["bat_e"]

    total_load = float(ts["load"].sum())
    E_diesel = float(ts["P_diesel"].sum())   # MWh/an
    E_END = float(ts["END"].sum())           # MWh/an

    # Investissements (payés année 0)
    C_inv = (
        p["capex_solar"] * K_solar
        + p["capex_wind"] * K_wind
        + p["capex_diesel"] * K_diesel
        + p["capex_bat_p"] * K_bat_p
        + p["capex_bat_e"] * K_bat_e
    )

    # O&M fixes annuels
    C_om_fix = (
        p["omfix_solar"] * K_solar
        + p["omfix_wind"] * K_wind
        + p["omfix_diesel"] * K_diesel
        + p["omfix_bat_p"] * K_bat_p
        + p["omfix_bat_e"] * K_bat_e
    )

    # Coûts variables diesel
    C_fuel = E_diesel * p["fuel_diesel"]
    C_om_var = E_diesel * p["omvar_diesel"]

    # Coût END
    C_END = E_END * p["cost_END"]

    # CO2
    CO2 = E_diesel * p["co2_factor_diesel"]
    C_CO2 = CO2 * p["co2_price"]

    # Coût annuel total (sans actualisation)
    C_annual = C_om_fix + C_fuel + C_om_var + C_END + C_CO2

    # Facteur de PV des coûts annuels
    PV_factor = (1 - (1 + DISCOUNT_RATE) ** -YEARS) / DISCOUNT_RATE

    C_total = C_inv + C_annual * PV_factor

    # Heures d'indisponibilité (au moins un peu d'END)
    unavailability_hours = int((ts["END"] > 0).sum())

    # LCOE (€/MWh servi) : coût total / énergie servie actualisée
    energy_served_per_year = total_load - E_END
    if energy_served_per_year > 0:
        PV_energy_served = energy_served_per_year * PV_factor
        LCOE = C_total / PV_energy_served
    else:
        LCOE = None

    # OPEX annuel "technique" (sans END ni CO2)
    OPEX_annual_technical = C_om_fix + C_fuel + C_om_var

    metrics = {
        "C_inv": C_inv,
        "C_om_fix": C_om_fix,
        "C_fuel": C_fuel,
        "C_om_var": C_om_var,
        "C_END": C_END,
        "C_CO2": C_CO2,
        "C_annual": C_annual,
        "C_total": C_total,
        "CO2": CO2,
        "E_diesel": E_diesel,
        "E_END": E_END,
        "END_ratio": E_END / total_load if total_load > 0 else 0.0,
        "total_load": total_load,
        "unavailability_hours": unavailability_hours,
        "LCOE": LCOE,
        "OPEX_annual_technical": OPEX_annual_technical,
    }
    return metrics


# ===================== #
#   GRID SEARCH OPTI    #
# ===================== #

def grid_search_optimal_mix(profiles, params, objective_mode, max_end_ratio,
                            solar_range, wind_range, diesel_range,
                            bat_p_range, bat_e_range):
    """
    Recherche du mix optimal sur une grille de capacités, en utilisant
    la simulation rule-based. ATTENTION : le nombre de combinaisons
    peut vite exploser si les pas sont trop fins.
    """
    best_caps = None
    best_metrics = None
    best_ts = None
    best_value = None

    total_combos = (
        len(solar_range)
        * len(wind_range)
        * len(diesel_range)
        * len(bat_p_range)
        * len(bat_e_range)
    )
    progress = st.progress(0)
    done = 0

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
                            profiles,
                            caps["solar"],
                            caps["wind"],
                            caps["diesel"],
                            caps["bat_p"],
                            caps["bat_e"],
                        )
                        metrics = compute_metrics(ts, caps, params)

                        # Contrôle de l'indisponibilité
                        if metrics["END_ratio"] > max_end_ratio:
                            done += 1
                            progress.progress(done / total_combos)
                            continue

                        if objective_mode == "cost":
                            value = metrics["C_total"]
                        else:  # "co2"
                            value = metrics["CO2"]

                        if (best_value is None) or (value < best_value):
                            best_value = value
                            best_caps = caps
                            best_metrics = metrics
                            best_ts = ts

                        done += 1
                        progress.progress(done / total_combos)

    return best_caps, best_metrics, best_ts


# ===================== #
#       STREAMLIT       #
# ===================== #

st.set_page_config(page_title="Mix de l'île - simulation & coûts", layout="wide")

st.title("⚡ Système électrique isolé de l'île – simulation type Excel & analyse techno-économique")

st.markdown(
    """
Cette application permet de :

1. **Analyser les coûts** d’un mix (CAPEX, OPEX, CO₂, LCOE, heures d'indisponibilité).  
2. **Choisir un mix de capacités** (MW solaire, éolien, diesel, batterie) ou
   **chercher un mix quasi-optimal** sur une grille (coût ou émissions).  
3. Visualiser un **compte rendu graphique** : charge, production empilée, énergie non servie et curtailment.
"""
)

# -------------------------- #
#  0️⃣ DONNÉES & HYPOTHÈSES  #
# -------------------------- #

st.markdown("## 0️⃣ Données d'entrée & hypothèses de coût")

col0a, col0b = st.columns([2, 3])

with col0a:
    st.markdown("### Profil de charge et de production")
    uploaded = st.file_uploader(
        "📁 Charger un fichier CSV (colonnes : hour, load, cf_wind, cf_solar)",
        type=["csv"],
    )

    if uploaded is not None:
        # 1) lecture en tenant compte de la virgule comme séparateur décimal
        profiles = pd.read_csv(uploaded, decimal=',')

        # 2) s'assurer que les colonnes numériques sont bien en float
        for col in ["load", "cf_wind", "cf_solar"]:
            profiles[col] = pd.to_numeric(profiles[col], errors="coerce")

        st.success("Profil de charge/production chargé ✅")
    else:
        st.info("Aucun fichier fourni : utilisation d'un profil **démonstration** (7 jours).")
        profiles = generate_demo_profiles(n_days=7)

    st.caption("Aperçu des données horaire (quelques premières lignes)")
    st.dataframe(profiles.head())

with col0b:
    st.markdown("### Hypothèses techno-économiques")

    params = {}
    with st.expander("CAPEX (investissement) – en €/MW ou €/MWh", expanded=False):
        col_capex1, col_capex2 = st.columns(2)
        with col_capex1:
            for key in ["capex_solar", "capex_wind", "capex_diesel"]:
                label = PARAM_LABELS[key]
                params[key] = st.number_input(label, value=float(DEFAULT_PARAMS[key]), step=10.0)
        with col_capex2:
            for key in ["capex_bat_p", "capex_bat_e"]:
                label = PARAM_LABELS[key]
                params[key] = st.number_input(label, value=float(DEFAULT_PARAMS[key]), step=10.0)

    with st.expander("O&M fixes annuels – en €/MW/an ou €/MWh/an", expanded=False):
        col_om1, col_om2 = st.columns(2)
        with col_om1:
            for key in ["omfix_solar", "omfix_wind", "omfix_diesel"]:
                label = PARAM_LABELS[key]
                params[key] = st.number_input(label, value=float(DEFAULT_PARAMS[key]), step=10.0)
        with col_om2:
            for key in ["omfix_bat_p", "omfix_bat_e"]:
                label = PARAM_LABELS[key]
                params[key] = st.number_input(label, value=float(DEFAULT_PARAMS[key]), step=10.0)

    with st.expander("Coûts variables, CO₂ et coût de l'énergie non servie", expanded=False):
        col_var1, col_var2 = st.columns(2)
        with col_var1:
            for key in ["fuel_diesel", "omvar_diesel"]:
                label = PARAM_LABELS[key]
                params[key] = st.number_input(label, value=float(DEFAULT_PARAMS[key]), step=10.0)
        with col_var2:
            for key in ["co2_factor_diesel", "co2_price", "cost_END"]:
                label = PARAM_LABELS[key]
                params[key] = st.number_input(label, value=float(DEFAULT_PARAMS[key]), step=10.0)

st.markdown("### Hypothèses de fiabilité et critère d'optimisation")

col_hyp1, col_hyp2 = st.columns(2)
with col_hyp1:
    max_end_ratio = st.slider(
        "Indisponibilité maximale autorisée (END / demande annuelle)",
        min_value=0.0,
        max_value=0.05,
        value=0.001,
        step=0.001,
    )
with col_hyp2:
    objective_mode_label = st.radio(
        "Critère utilisé en mode 'mix quasi-optimal'",
        options=[
            "Minimiser le coût actualisé (donc le LCOE)",
            "Minimiser les émissions de CO₂",
        ],
    )
    objective_mode = "cost" if objective_mode_label.startswith("Minimiser le coût") else "co2"

st.markdown("---")

# ------------------------------- #
#  1️⃣ CHOIX DU TYPE D'ÉTUDE      #
# ------------------------------- #

st.markdown("## 1️⃣ Choix du type d'étude")

mode = st.radio(
    "Comment souhaitez-vous explorer le mix de production ?",
    [
        "🧪 Tester un scénario avec des capacités installées choisies",
        "🔍 Chercher un mix quasi-optimal (grid search)",
    ],
)

current_caps = None
current_metrics = None
current_ts = None

# ===================== #
#     MODE SCÉNARIO     #
# ===================== #

if mode.startswith("🧪"):
    st.markdown("## 2️⃣ Définir un scénario de capacités installées")

    col1, col2, col3 = st.columns(3)
    with col1:
        solar_cap = st.number_input(
            "Capacité solaire installée (MW)", min_value=0.0, value=10.0, step=1.0
        )
        wind_cap = st.number_input(
            "Capacité éolienne installée (MW)", min_value=0.0, value=10.0, step=1.0
        )
    with col2:
        diesel_cap = st.number_input(
            "Capacité installée de groupes diesel (MW)", min_value=0.0, value=24.0, step=1.0
        )
    with col3:
        bat_p_cap = st.number_input(
            "Puissance batterie installée (MW)", min_value=0.0, value=2.0, step=0.5
        )
        bat_e_cap = st.number_input(
            "Capacité énergétique batterie (MWh)", min_value=0.0, value=4.0, step=1.0
        )

    if st.button("▶️ Lancer la simulation du scénario"):
        current_caps = {
            "solar": solar_cap,
            "wind": wind_cap,
            "diesel": diesel_cap,
            "bat_p": bat_p_cap,
            "bat_e": bat_e_cap,
        }

        with st.spinner("Simulation horaire (règles Excel-like)…"):
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
    st.markdown("## 2️⃣ Chercher un mix de production quasi-optimal (grid search)")

    st.markdown(
        """
La recherche se fait sur une **grille de capacités**.
Plus les pas sont fins, plus c'est précis, mais plus c'est long.
"""
    )

    colr1, colr2 = st.columns(2)
    with colr1:
        solar_min = st.number_input("Solaire min (MW)", 0.0, 100.0, 0.0, 1.0)
        solar_max = st.number_input("Solaire max (MW)", 0.0, 100.0, 20.0, 1.0)
        solar_step = st.number_input("Pas solaire (MW)", 0.1, 50.0, 5.0, 0.1)

        wind_min = st.number_input("Éolien min (MW)", 0.0, 100.0, 0.0, 1.0)
        wind_max = st.number_input("Éolien max (MW)", 0.0, 100.0, 30.0, 1.0)
        wind_step = st.number_input("Pas éolien (MW)", 0.1, 50.0, 5.0, 0.1)

    with colr2:
        diesel_min = st.number_input("Diesel min (MW)", 0.0, 100.0, 8.0, 1.0)
        diesel_max = st.number_input("Diesel max (MW)", 0.0, 100.0, 24.0, 1.0)
        diesel_step = st.number_input("Pas diesel (MW)", 0.1, 50.0, 4.0, 0.1)

        bat_p_min = st.number_input("Puissance batterie min (MW)", 0.0, 100.0, 0.0, 0.5)
        bat_p_max = st.number_input("Puissance batterie max (MW)", 0.0, 100.0, 4.0, 0.5)
        bat_p_step = st.number_input("Pas puissance batterie (MW)", 0.1, 50.0, 2.0, 0.1)

        bat_e_min = st.number_input("Énergie batterie min (MWh)", 0.0, 200.0, 0.0, 1.0)
        bat_e_max = st.number_input("Énergie batterie max (MWh)", 0.0, 200.0, 8.0, 1.0)
        bat_e_step = st.number_input("Pas énergie batterie (MWh)", 0.1, 200.0, 4.0, 0.1)

    if st.button("🔍 Lancer la recherche du mix quasi-optimal"):
        solar_range = np.arange(solar_min, solar_max + 1e-9, solar_step)
        wind_range = np.arange(wind_min, wind_max + 1e-9, wind_step)
        diesel_range = np.arange(diesel_min, diesel_max + 1e-9, diesel_step)
        bat_p_range = np.arange(bat_p_min, bat_p_max + 1e-9, bat_p_step)
        bat_e_range = np.arange(bat_e_min, bat_e_max + 1e-9, bat_e_step)

        total_combos = (
            len(solar_range)
            * len(wind_range)
            * len(diesel_range)
            * len(bat_p_range)
            * len(bat_e_range)
        )

        if total_combos > 2000:
            st.warning(
                f"Attention : {total_combos} combinaisons à tester, "
                "cela peut être long. Réduis les plages ou augmente les pas si besoin."
            )

        with st.spinner("Recherche du mix quasi-optimal (grid search, règles Excel-like)…"):
            best_caps, best_metrics, best_ts = grid_search_optimal_mix(
                profiles,
                params,
                objective_mode,
                max_end_ratio,
                solar_range,
                wind_range,
                diesel_range,
                bat_p_range,
                bat_e_range,
            )

        if best_caps is None:
            st.error(
                "Aucun mix ne respecte la contrainte d'indisponibilité avec la grille choisie. "
                "Essaie d'élargir les plages (surtout diesel / batterie) "
                "ou d'autoriser une indisponibilité un peu plus élevée."
            )
        else:
            current_caps = best_caps
            current_metrics = best_metrics
            current_ts = best_ts
            st.success("Recherche terminée ✅")

# ------------------------------- #
#   3️⃣ SYNTHÈSE COÛTS & FIABILITÉ #
# ------------------------------- #

if current_metrics is not None and current_ts is not None and current_caps is not None:
    st.markdown("## 3️⃣ Synthèse économique et de fiabilité")

    colc1, colc2, colc3, colc4 = st.columns(4)
    with colc1:
        st.metric(
            "LCOE (coût moyen actualisé)",
            f"{current_metrics['LCOE']:.1f} €/MWh" if current_metrics["LCOE"] is not None else "n/a",
        )
    with colc2:
        st.metric("CAPEX total (année 0)", f"{current_metrics['C_inv']:.0f} €")
    with colc3:
        st.metric(
            "OPEX annuel technique (sans END ni CO₂)",
            f"{current_metrics['OPEX_annual_technical']:.0f} €/an",
        )
    with colc4:
        st.metric("Émissions annuelles de CO₂", f"{current_metrics['CO2']:.0f} tCO₂/an")

    colf1, colf2 = st.columns(2)
    with colf1:
        st.metric(
            "Heures d'indisponibilité par an",
            f"{current_metrics['unavailability_hours']}",
        )
    with colf2:
        st.metric(
            "END / demande annuelle",
            f"{current_metrics['END_ratio']*100:.3f} %",
        )

    st.markdown("### Capacités installées du scénario étudié")
    st.write(current_caps)

    # ------------------------------- #
    #   4️⃣ COMPTE RENDU VISUEL       #
    # ------------------------------- #

    st.markdown("## 4️⃣ Compte rendu visuel du fonctionnement du système")

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
        id_vars="hour_idx",
        var_name="Filière de production",
        value_name="Puissance (MW)",
    )

    chart = (
        alt.Chart(prod)
        .mark_area()
        .encode(
            x=alt.X("hour_idx:Q", title="Heure"),
            y=alt.Y("Puissance (MW):Q", stack="zero"),
            color=alt.Color("Filière de production:N", title="Filière"),
            tooltip=["hour_idx", "Filière de production", "Puissance (MW)"],
        )
    )

    st.altair_chart(chart, use_container_width=True)

    st.subheader("Énergie non servie et surplus renouvelable (48 premières heures)")
    df_end = df_plot.rename(
        columns={"END": "Énergie non servie", "Spill": "Surplus EnR (curtailment)"}
    )
    st.line_chart(df_end.set_index("hour_idx")[["Énergie non servie", "Surplus EnR (curtailment)"]])

else:
    st.info("➡️ Définis un scénario ou lance une optimisation pour afficher les indicateurs et les graphiques.")
