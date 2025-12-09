"""
Compute hourly dispatch that minimizes variable cost to meet load using
installed generation capacities from the isolated system workbook.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


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


def plot_dispatch(
    dispatch: pd.DataFrame, scenario: str, output: Path, show: bool
) -> None:
    """Create stacked area plot showing how supply meets load."""
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
        "--show",
        action="store_true",
        help="Affiche la figure à l'écran en plus de sauvegarder le PNG.",
    )
    args = parser.parse_args()

    data, params = read_inputs(args.excel, args.scenario)
    dispatch = dispatch_min_cost(data, params["capacities"], params["costs"])

    total_var_cost = dispatch["variable_cost"].sum()
    print(f"Coût variable total {args.scenario}: {total_var_cost:,.0f} €")
    print(dispatch[["solar", "wind", "diesel", "unserved"]].describe())

    plot_dispatch(dispatch, args.scenario, args.output, args.show)
    print(f"Graphique sauvegardé dans {args.output}")


if __name__ == "__main__":
    main()
