"""Dashboard Streamlit : profils bruts (charge et facteurs de charge) avec zoom/pan."""
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st


EXCEL_DEFAULT = Path("5EA14_project_isolated-system.xlsx")
REF_COL = 8  # colonne "Ref case" dans la feuille summary

st.set_page_config(layout="wide", page_title="Profils bruts énergie")


@st.cache_data(show_spinner=False)
def load_inputs(excel_path: Path) -> tuple[pd.DataFrame, dict]:
    """Charge les séries brutes (load, facteurs de charge) et les capacités de référence pour info."""
    summary = pd.read_excel(excel_path, sheet_name="summary", header=None)
    capacities = {
        "solar": float(summary.loc[2, REF_COL]),
        "wind": float(summary.loc[3, REF_COL]),
        "diesel": float(summary.loc[4, REF_COL]),
    }
    om_costs = {
        "solar": float(summary.loc[18, 2]) if not pd.isna(summary.loc[18, 2]) else 25000.0,
        "wind": float(summary.loc[19, 2]) if not pd.isna(summary.loc[19, 2]) else 25000.0,
        "diesel": float(summary.loc[20, 2]) if not pd.isna(summary.loc[20, 2]) else 50000.0,
    }
    inv_costs = {
        "solar": float(summary.loc[11, 2]) if not pd.isna(summary.loc[11, 2]) else 1_000_000.0,
        "wind": float(summary.loc[12, 2]) if not pd.isna(summary.loc[12, 2]) else 1_000_000.0,
        "diesel": float(summary.loc[13, 2]) if not pd.isna(summary.loc[13, 2]) else 1_000_000.0,
        "storage_power": float(summary.loc[14, 2]) if not pd.isna(summary.loc[14, 2]) else 200_000.0,
        "storage_energy": float(summary.loc[15, 2]) if not pd.isna(summary.loc[15, 2]) else 300_000.0,
    }
    costs = {
        "diesel": float(summary.loc[7, 2]),
        "unserved": float(summary.loc[6, 2]),
    }
    storage_defaults = {
        "power": float(summary.loc[7, 8]) if not pd.isna(summary.loc[7, 8]) else 0.0,
        "energy": float(summary.loc[8, 8]) if not pd.isna(summary.loc[8, 8]) else 0.0,
    }

    raw = pd.read_excel(excel_path, sheet_name="Ref case", header=None)
    data = (
        raw.iloc[4:, [4, 5, 19, 20]]
        .rename(columns={4: "time", 5: "load", 19: "solar_cf", 20: "wind_cf"})
        .dropna(subset=["load"])
    )
    data["time"] = pd.to_datetime(data["time"])
    for col_name in ("load", "solar_cf", "wind_cf"):
        data[col_name] = pd.to_numeric(data[col_name], errors="coerce")
    data = data.dropna(subset=["time", "load"]).reset_index(drop=True)
    return data, {
        "capacities": capacities,
        "costs": costs,
        "inv_costs": inv_costs,
        "om_costs": om_costs,
        "storage_defaults": storage_defaults,
    }


def dispatch_min_cost(
    load: pd.Series,
    solar_cf: pd.Series,
    wind_cf: pd.Series,
    capacities: dict,
    costs: dict,
    storage: dict,
) -> pd.DataFrame:
    """
    Dispatch greedy pour minimiser le coût variable :
    - Solaire et éolien couvrent la demande
    - Surplus ENR charge la batterie
    - Batterie décharge pour réduire le résiduel
    - Diesel couvre le reste, sinon énergie non servie
    """
    n = len(load)
    load_np = load.to_numpy().astype(float)
    solar_av = capacities["solar"] * solar_cf.to_numpy()
    wind_av = capacities["wind"] * wind_cf.to_numpy()
    diesel_cap = capacities["diesel"]
    p_batt = capacities["storage_power"]
    e_batt = capacities["storage_energy"]

    alloc = {k: [] for k in ["solar", "wind", "battery", "diesel", "unserved", "battery_charge", "soc"]}
    soc = min(max(storage.get("soc_init", 0.0), 0.0), storage["energy"])

    for i in range(len(load_np)):
        remaining = load_np[i]
        # Solaire
        solar_take = min(remaining, solar_av[i])
        remaining -= solar_take
        solar_surplus = max(0.0, solar_av[i] - solar_take)
        alloc["solar"].append(solar_take)

        # Éolien
        wind_take = min(remaining, wind_av[i])
        remaining -= wind_take
        wind_surplus = max(0.0, wind_av[i] - wind_take)
        alloc["wind"].append(wind_take)

        # Surplus renouvelable éventuel pour charger la batterie
        surplus = solar_surplus + wind_surplus
        charge = min(p_batt, surplus, e_batt - soc)
        soc += charge
        alloc["battery_charge"].append(charge)

        # Décharge pour couvrir le résiduel
        discharge = min(remaining, p_batt, soc)
        soc -= discharge
        remaining -= discharge
        alloc["battery"].append(discharge)

        # Diesel
        diesel_take = min(remaining, diesel_cap)
        remaining -= diesel_take
        alloc["diesel"].append(diesel_take)

        # Non servi
        alloc["unserved"].append(max(remaining, 0))
        alloc["soc"].append(soc)

    dispatch = pd.DataFrame(alloc, index=load.index)
    dispatch["variable_cost"] = (
        dispatch["solar"] * costs["solar"]
        + dispatch["wind"] * costs["wind"]
        + dispatch["battery"] * costs["battery"]
        + dispatch["diesel"] * costs["diesel"]
        + dispatch["unserved"] * costs["unserved"]
    )
    return dispatch


def main() -> None:
    st.title("Profils bruts : charge et facteurs de charge")
    st.caption("Lecture directe du fichier Excel, sans modification ni choix de scénario.")

    excel_path = st.text_input("Classeur Excel", value=str(EXCEL_DEFAULT))
    excel_path = Path(excel_path)
    if not excel_path.exists():
        st.error(f"Fichier introuvable : {excel_path}")
        return

    data, defaults = load_inputs(excel_path)
    capacities_ref = defaults["capacities"]
    costs_ref = defaults["costs"]
    inv_costs_ref = defaults["inv_costs"]
    om_costs_ref = defaults["om_costs"]
    storage_defaults = defaults["storage_defaults"]

    st.subheader("Capacités de référence (info)")
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Solaire installé (réf.)", f"{capacities_ref['solar']} MW")
    with col2:
        st.metric("Éolien installé (réf.)", f"{capacities_ref['wind']} MW")

    st.subheader("Graphique interactif (zoom/pan)")
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(
        go.Scatter(
            x=data["time"],
            y=data["load"],
            name="Demande (MW)",
            line=dict(color="black"),
            hovertemplate="%{x|%Y-%m-%d %H:%M}<br>Demande: %{y:.2f} MW",
        ),
        secondary_y=False,
    )
    fig.add_trace(
        go.Scatter(
            x=data["time"],
            y=data["solar_cf"],
            name="Solar CF",
            line=dict(color="#f6c344"),
            hovertemplate="%{x|%Y-%m-%d %H:%M}<br>Solar CF: %{y:.2f}",
        ),
        secondary_y=True,
    )
    fig.add_trace(
        go.Scatter(
            x=data["time"],
            y=data["wind_cf"],
            name="Wind CF",
            line=dict(color="#4e79a7"),
            hovertemplate="%{x|%Y-%m-%d %H:%M}<br>Wind CF: %{y:.2f}",
        ),
        secondary_y=True,
    )
    fig.update_yaxes(title_text="Charge (MW)", secondary_y=False)
    fig.update_yaxes(title_text="Facteur de charge", range=[0, 1], secondary_y=True)
    fig.update_layout(
        hovermode="x unified",
        legend=dict(orientation="h", y=-0.2),
        margin=dict(l=0, r=0, t=40, b=0),
        height=450,
    )
    st.plotly_chart(fig, use_container_width=True)

    with st.expander("Statistiques rapides"):
        stats_df = pd.DataFrame(
            {
                "Demande (MW)": data["load"],
                "Solar CF": data["solar_cf"],
                "Wind CF": data["wind_cf"],
            }
        ).describe().T
        st.dataframe(stats_df)

    st.caption("Charge totale (axe gauche) et facteurs de charge solaire/éolien (axe droit) sur les données brutes. Aucune modification ni sélection de scénario.")

    st.subheader("Allocation à coût variable minimal")
    col_cap1, col_cap2, col_cap3, col_cap4 = st.columns(4)
    with col_cap1:
        cap_solar = st.number_input("Capacité solaire (MW)", value=float(capacities_ref["solar"]), min_value=0.0, step=1.0)
    with col_cap2:
        cap_wind = st.number_input("Capacité éolienne (MW)", value=float(capacities_ref["wind"]), min_value=0.0, step=1.0)
    with col_cap3:
        cap_diesel = st.number_input("Capacité diesel (MW)", value=float(capacities_ref["diesel"]), min_value=0.0, step=1.0)
    with col_cap4:
        cap_storage_power = st.number_input("Puissance batterie (MW)", value=float(storage_defaults["power"]), min_value=0.0, step=0.5)
    col_cap5, col_cap6 = st.columns(2)
    with col_cap5:
        cap_storage_energy = st.number_input("Énergie batterie (MWh)", value=float(storage_defaults["energy"]), min_value=0.0, step=1.0)
    with col_cap6:
        soc_init = st.number_input("SOC initial batterie (MWh)", value=0.0, min_value=0.0, max_value=float(max(0.0, cap_storage_energy)))

    col_cost1, col_cost2, col_cost3, col_cost4 = st.columns(4)
    with col_cost1:
        cost_solar = st.number_input("Coût variable solaire (€/MWh)", value=0.0, min_value=0.0, step=1.0)
    with col_cost2:
        cost_wind = st.number_input("Coût variable éolien (€/MWh)", value=0.0, min_value=0.0, step=1.0)
    with col_cost3:
        cost_diesel = st.number_input("Coût variable diesel (€/MWh)", value=float(costs_ref["diesel"]), min_value=0.0, step=1.0)
    with col_cost4:
        cost_unserved = st.number_input("Coût énergie non servie (€/MWh)", value=float(costs_ref["unserved"]), min_value=0.0, step=100.0)
    col_cost5 = st.columns(1)[0]
    with col_cost5:
        cost_battery = st.number_input("Coût variable batterie (€/MWh déchargé)", value=0.0, min_value=0.0, step=1.0)

    st.subheader("Coûts de maintenance annuels (€/MW.an)")
    col_om1, col_om2, col_om3, col_om4 = st.columns(4)
    with col_om1:
        om_solar = st.number_input("O&M solaire (€/MW.an)", value=float(om_costs_ref["solar"]), min_value=0.0, step=500.0)
    with col_om2:
        om_wind = st.number_input("O&M éolien (€/MW.an)", value=float(om_costs_ref["wind"]), min_value=0.0, step=500.0)
    with col_om3:
        om_diesel = st.number_input("O&M diesel (€/MW.an)", value=float(om_costs_ref["diesel"]), min_value=0.0, step=500.0)
    with col_om4:
        om_storage = st.number_input("O&M batterie (€/MW.an)", value=0.0, min_value=0.0, step=500.0)

    st.subheader("Hypothèses d'investissement")
    col_inv1, col_inv2, col_inv3, col_inv4, col_inv5 = st.columns(5)
    with col_inv1:
        inv_solar = st.number_input("CAPEX solaire (€/MW)", value=float(inv_costs_ref["solar"]), min_value=0.0, step=1000.0)
    with col_inv2:
        inv_wind = st.number_input("CAPEX éolien (€/MW)", value=float(inv_costs_ref["wind"]), min_value=0.0, step=1000.0)
    with col_inv3:
        inv_diesel = st.number_input("CAPEX diesel (€/MW)", value=float(inv_costs_ref["diesel"]), min_value=0.0, step=1000.0)
    with col_inv4:
        rate = st.number_input("Taux d'intérêt (%)", value=5.0, min_value=0.0, step=0.1) / 100.0
    with col_inv5:
        inv_storage_power = st.number_input("CAPEX batterie (€/MW puissance)", value=float(inv_costs_ref["storage_power"]), min_value=0.0, step=1000.0)
    years = st.number_input("Horizon (années)", value=20, min_value=1, step=1)
    inv_storage_energy = st.number_input("CAPEX batterie (€/MWh énergie)", value=float(inv_costs_ref["storage_energy"]), min_value=0.0, step=1000.0)

    capacities_run = {
        "solar": cap_solar,
        "wind": cap_wind,
        "diesel": cap_diesel,
        "storage_power": cap_storage_power,
        "storage_energy": cap_storage_energy,
    }
    costs_run = {
        "solar": cost_solar,
        "wind": cost_wind,
        "diesel": cost_diesel,
        "unserved": cost_unserved,
        "battery": cost_battery,
    }

    # Annuité pour annualiser l'investissement
    if rate == 0:
        annuity = 1 / years
    else:
        annuity = rate * (1 + rate) ** years / ((1 + rate) ** years - 1)
    annual_capex = (
        cap_solar * inv_solar * annuity
        + cap_wind * inv_wind * annuity
        + cap_diesel * inv_diesel * annuity
        + cap_storage_power * inv_storage_power * annuity
        + cap_storage_energy * inv_storage_energy * annuity
    )
    annual_om = (
        cap_solar * om_solar
        + cap_wind * om_wind
        + cap_diesel * om_diesel
        + cap_storage_power * om_storage
    )

    dispatch = dispatch_min_cost(
        load=data["load"],
        solar_cf=data["solar_cf"],
        wind_cf=data["wind_cf"],
        capacities=capacities_run,
        costs=costs_run,
        storage={
            "power": cap_storage_power,
            "energy": cap_storage_energy,
            "soc_init": soc_init,
        },
    )

    st.caption(
        f"Coût variable total : {dispatch['variable_cost'].sum():,.0f} € "
        f"(sur {len(dispatch)} pas horaires)"
    )

    st.subheader("Dispatch (stack sous la demande)")
    dispatch_plot = make_subplots(specs=[[{"secondary_y": False}]])
    dispatch_plot.add_trace(
        go.Scatter(
            x=data["time"],
            y=data["load"],
            name="Demande",
            line=dict(color="black", width=1.2),
            hovertemplate="%{x|%Y-%m-%d %H:%M}<br>Demande: %{y:.2f} MW",
        )
    )
    for src, color in [
        ("solar", "#f6c344"),
        ("wind", "#4e79a7"),
        ("battery", "#7bc6cc"),
        ("diesel", "#d47251"),
        ("unserved", "#b30000"),
    ]:
        dispatch_plot.add_trace(
            go.Scatter(
                x=data["time"],
                y=dispatch[src],
                name=src.capitalize(),
                stackgroup="one",
                line=dict(width=0.5, color=color),
                hovertemplate="%{x|%Y-%m-%d %H:%M}<br>" + src + ": %{y:.2f} MW",
            )
        )
    dispatch_plot.update_layout(
        hovermode="x unified",
        legend=dict(orientation="h", y=-0.2),
        margin=dict(l=0, r=0, t=40, b=0),
        height=500,
    )
    dispatch_plot.update_yaxes(title_text="Puissance (MW)")
    st.plotly_chart(dispatch_plot, use_container_width=True)

    st.subheader("Coûts cumulés (stack)")
    steps = pd.Series(range(1, len(dispatch) + 1), index=dispatch.index)
    cum_cost = dispatch["variable_cost"].cumsum()
    cum_capex = annual_capex * (steps / len(dispatch))
    cum_om = annual_om * (steps / len(dispatch))
    cum_total = cum_capex + cum_om + cum_cost
    cost_fig = go.Figure()
    cost_fig.add_trace(
        go.Scatter(
            x=data["time"],
            y=cum_capex,
            name="CAPEX annualisé cumulé",
            stackgroup="cost",
            line=dict(width=0, color="#c4c8ff"),
            fillcolor="#c4c8ff",
            hovertemplate="%{x|%Y-%m-%d %H:%M}<br>CAPEX cumulé: %{y:,.0f} €",
        )
    )
    cost_fig.add_trace(
        go.Scatter(
            x=data["time"],
            y=cum_om,
            name="O&M cumulé",
            stackgroup="cost",
            line=dict(width=0, color="#9ac5a0"),
            fillcolor="#9ac5a0",
            hovertemplate="%{x|%Y-%m-%d %H:%M}<br>O&M cumulé: %{y:,.0f} €",
        )
    )
    cost_fig.add_trace(
        go.Scatter(
            x=data["time"],
            y=cum_cost,
            name="Variable cumulé",
            stackgroup="cost",
            line=dict(width=0, color="#8882c2"),
            fillcolor="#8882c2",
            hovertemplate="%{x|%Y-%m-%d %H:%M}<br>Variable cumulé: %{y:,.0f} €",
        )
    )
    cost_fig.add_trace(
        go.Scatter(
            x=data["time"],
            y=cum_total,
            name="Total cumulé",
            line=dict(color="#444", width=1.2),
            hovertemplate="%{x|%Y-%m-%d %H:%M}<br>Total: %{y:,.0f} €",
        )
    )
    cost_fig.update_layout(
        hovermode="x unified",
        margin=dict(l=0, r=0, t=40, b=0),
        height=350,
        yaxis_title="€ cumulés",
    )
    st.plotly_chart(cost_fig, use_container_width=True)


if __name__ == "__main__":
    main()
