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
    costs = {
        "diesel": float(summary.loc[7, 2]),
        "unserved": float(summary.loc[6, 2]),
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
    return data, {"capacities": capacities, "costs": costs}


def dispatch_min_cost(load: pd.Series, solar_cf: pd.Series, wind_cf: pd.Series, capacities: dict, costs: dict) -> pd.DataFrame:
    """
    Dispatch greedy pour minimiser le coût variable :
    on trie les sources par coût croissant, puis on affecte jusqu'à couvrir la demande.
    """
    n = len(load)
    avail = {
        "solar": capacities["solar"] * solar_cf.to_numpy(),
        "wind": capacities["wind"] * wind_cf.to_numpy(),
        "diesel": pd.Series([capacities["diesel"]] * n, index=load.index).to_numpy(),
    }
    order = sorted(["solar", "wind", "diesel"], key=lambda k: costs[k])
    load_np = load.to_numpy().astype(float)
    alloc = {k: [] for k in ["solar", "wind", "diesel", "unserved"]}

    for i in range(len(load_np)):
        remaining = load_np[i]
        for src in order:
            take = min(remaining, avail[src][i])
            alloc[src].append(take)
            remaining -= take
        alloc["unserved"].append(max(remaining, 0))

    dispatch = pd.DataFrame(alloc, index=load.index)
    dispatch["variable_cost"] = (
        dispatch["solar"] * costs["solar"]
        + dispatch["wind"] * costs["wind"]
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
    col_cap1, col_cap2, col_cap3 = st.columns(3)
    with col_cap1:
        cap_solar = st.number_input("Capacité solaire (MW)", value=float(capacities_ref["solar"]), min_value=0.0, step=1.0)
    with col_cap2:
        cap_wind = st.number_input("Capacité éolienne (MW)", value=float(capacities_ref["wind"]), min_value=0.0, step=1.0)
    with col_cap3:
        cap_diesel = st.number_input("Capacité diesel (MW)", value=float(capacities_ref["diesel"]), min_value=0.0, step=1.0)

    col_cost1, col_cost2, col_cost3, col_cost4 = st.columns(4)
    with col_cost1:
        cost_solar = st.number_input("Coût variable solaire (€/MWh)", value=0.0, min_value=0.0, step=1.0)
    with col_cost2:
        cost_wind = st.number_input("Coût variable éolien (€/MWh)", value=0.0, min_value=0.0, step=1.0)
    with col_cost3:
        cost_diesel = st.number_input("Coût variable diesel (€/MWh)", value=float(costs_ref["diesel"]), min_value=0.0, step=1.0)
    with col_cost4:
        cost_unserved = st.number_input("Coût énergie non servie (€/MWh)", value=float(costs_ref["unserved"]), min_value=0.0, step=100.0)

    capacities_run = {"solar": cap_solar, "wind": cap_wind, "diesel": cap_diesel}
    costs_run = {
        "solar": cost_solar,
        "wind": cost_wind,
        "diesel": cost_diesel,
        "unserved": cost_unserved,
    }

    dispatch = dispatch_min_cost(
        load=data["load"],
        solar_cf=data["solar_cf"],
        wind_cf=data["wind_cf"],
        capacities=capacities_run,
        costs=costs_run,
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

    st.subheader("Coûts variables cumulés")
    cum_cost = dispatch["variable_cost"].cumsum()
    cost_fig = go.Figure(
        go.Scatter(
            x=data["time"],
            y=cum_cost,
            name="Coût cumulé",
            line=dict(color="#444", width=1.5),
            hovertemplate="%{x|%Y-%m-%d %H:%M}<br>Coût cumulé: %{y:,.0f} €",
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
