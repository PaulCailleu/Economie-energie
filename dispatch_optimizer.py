"""
Compute hourly dispatch that minimizes variable cost to meet load using
installed generation capacities from the isolated system workbook.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

default_colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

plt.rcParams.update({
    "text.usetex": True,         # Utiliser LaTeX pour le rendu des textes
    "font.family": "serif",      # Utiliser une police de caractères avec empattement (serif)
    "font.size" : "18",
    "legend.fontsize": 10,
    #"text.latex.preamble": r'\usepackage{bm}'
})


SCENARIO_COLS = {"Ref case": 8, "Case 1": 9, "Case 2": 10}


def read_inputs(excel_path: Path, scenario: str) -> tuple[pd.DataFrame, dict]:
    """Load load profile, capacity factors, capacities, and variable costs."""
    if scenario not in SCENARIO_COLS:
        raise ValueError(f"Scenario must be one of {list(SCENARIO_COLS)}")

    summary = pd.read_excel(excel_path, sheet_name="summary", header=None)
    col = SCENARIO_COLS[scenario]
    capacities = {
        "solar": float(summary.loc[2, col]),
        "wind": float(summary.loc[3, col]),
        "diesel": float(summary.loc[4, col]),
    }
    costs = {"diesel": float(summary.loc[7, 2]), "unserved": float(summary.loc[6, 2])}

    raw = pd.read_excel(excel_path, sheet_name="Ref case", header=None)
    data = (
        raw.iloc[4:, [4, 5, 19, 20]]
        .rename(
            columns={4: "time", 5: "load", 19: "solar_cf", 20: "wind_cf"},
        )
        .dropna(subset=["load"])
    )
    data["time"] = pd.to_datetime(data["time"])
    for col_name in ("load", "solar_cf", "wind_cf"):
        data[col_name] = pd.to_numeric(data[col_name], errors="coerce")
    data = data.dropna(subset=["time", "load"]).reset_index(drop=True)
    return data, {"capacities": capacities, "costs": costs}


def dispatch_min_cost(
    data: pd.DataFrame, capacities: dict, costs: dict
) -> pd.DataFrame:
    """Greedy optimal dispatch (zero-marginal solar/wind, costly diesel, penalty unserved)."""
    load = data["load"].to_numpy()
    solar_avail = capacities["solar"] * data["solar_cf"].to_numpy()
    wind_avail = capacities["wind"] * data["wind_cf"].to_numpy()

    solar = np.minimum(load, solar_avail)
    residual = load - solar

    wind = np.minimum(residual, wind_avail)
    residual -= wind

    diesel = np.minimum(residual, capacities["diesel"])
    residual -= diesel

    unserved = np.maximum(residual, 0)
    total_cost = diesel * costs["diesel"] + unserved * costs["unserved"]

    dispatch = data.copy()
    dispatch[["solar", "wind", "diesel", "unserved"]] = np.column_stack(
        [solar, wind, diesel, unserved]
    )
    dispatch["variable_cost"] = total_cost
    return dispatch


def _ensure_interactive_backend() -> None:
    """If current backend is non-interactive (e.g., Agg), try to switch to an interactive one."""
    backend = matplotlib.get_backend().lower()
    if "agg" in backend:
        for candidate in ("TkAgg", "MacOSX", "Qt5Agg"):
            try:
                plt.switch_backend(candidate)
                return
            except Exception:
                continue


def plot_profile(
    data: pd.DataFrame, capacities: dict, scenario: str, output: Path, show: bool
) -> None:
    """Plot raw load and available solar/wind profiles to visualize resource shapes."""
    if show:
        _ensure_interactive_backend()
    solar_avail = capacities["solar"] * data["solar_cf"]
    wind_avail = capacities["wind"] * data["wind_cf"]

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(data["time"], data["load"], color="black", linewidth=1.0, label="Demande brute")
    ax.plot(data["time"], solar_avail, color="#f6c344", linewidth=0.8, label="Solar dispo")
    ax.plot(data["time"], wind_avail, color="#4e79a7", linewidth=0.8, label="Wind dispo")
    ax.set_title(f"Profils bruts (load et ressources) - {scenario}")
    ax.set_ylabel("Puissance (MW)")
    ax.margins(x=0)
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(output, dpi=200)
    if show:
        plt.show()
    plt.close(fig)


def plot_dispatch(
    dispatch: pd.DataFrame, scenario: str, output: Path, show: bool
) -> None:
    """Create stacked area plot showing how supply meets load."""
    if show:
        _ensure_interactive_backend()
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.stackplot(
        dispatch["time"],
        dispatch["solar"],
        dispatch["wind"],
        dispatch["diesel"],
        dispatch["unserved"],
        labels=["Solar", "Wind", "Diesel", "Non-servi"],
        colors=["#f6c344", "#4e79a7", "#d47251", "#b30000"],
        linewidth=0,
    )
    ax.plot(dispatch["time"], dispatch["load"], color="black", linewidth=1.0, label="Demande")
    ax.set_title(f"Dispatch minimisant le coût variable ({scenario})")
    ax.set_ylabel("Puissance (MW)")
    ax.legend(loc="upper right")
    ax.margins(x=0)
    fig.tight_layout()
    fig.savefig(output, dpi=200)
    if show:
        plt.show()
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Calcule un dispatch à coût variable minimal pour l'année."
    )
    parser.add_argument(
        "--excel",
        type=Path,
        default=Path("5EA14_project_isolated-system.xlsx"),
        help="Chemin du classeur d'entrée.",
    )
    parser.add_argument(
        "--scenario",
        type=str,
        default="Case 1",
        choices=list(SCENARIO_COLS),
        help="Colonne de capacités à utiliser (Ref case, Case 1, Case 2).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("dispatch.png"),
        help="Nom du fichier PNG de sortie.",
    )
    parser.add_argument(
        "--plot-profile",
        action="store_true",
        help="Génère un plot des profils bruts (load, solaire dispo, vent dispo) avant le dispatch.",
    )
    parser.add_argument(
        "--profile-output",
        type=Path,
        default=Path("profile.png"),
        help="Nom du PNG pour le plot des profils bruts.",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Affiche la figure à l'écran en plus de sauvegarder le PNG.",
    )
    args = parser.parse_args()

    data, params = read_inputs(args.excel, args.scenario)

    if args.plot_profile:
        plot_profile(data, params["capacities"], args.scenario, args.profile_output, args.show)
        print(f"Profil brut sauvegardé dans {args.profile_output}")

    dispatch = dispatch_min_cost(data, params["capacities"], params["costs"])

    total_var_cost = dispatch["variable_cost"].sum()
    print(f"Coût variable total {args.scenario}: {total_var_cost:,.0f} €")
    print(dispatch[["solar", "wind", "diesel", "unserved"]].describe())

    plot_dispatch(dispatch, args.scenario, args.output, args.show)
    print(f"Graphique sauvegardé dans {args.output}")


if __name__ == "__main__":
    main()
