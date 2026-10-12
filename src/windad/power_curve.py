"""Courbe de puissance empirique par la méthode des intervalles (bins).

Pour chaque turbine, on découpe la vitesse de vent en intervalles de 0,5 m/s et
on retient la médiane de la puissance dans chaque intervalle. La médiane est
robuste : quelques jours d'anomalies ne déforment pas la courbe de référence.
La dispersion normale autour de la courbe est mesurée par l'écart absolu médian.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

BIN_WIDTH = 0.5


@dataclass
class PowerCurve:
    centers: np.ndarray   # centre des intervalles de vent (m/s)
    median: np.ndarray    # puissance médiane (kW)
    spread: np.ndarray    # dispersion robuste (kW), 1,4826 × MAD

    def expected(self, ws: np.ndarray) -> np.ndarray:
        return np.interp(ws, self.centers, self.median, left=0.0, right=0.0)

    def scale(self, ws: np.ndarray) -> np.ndarray:
        return np.interp(ws, self.centers, self.spread)


def fit_power_curve(ws: np.ndarray, power: np.ndarray, min_count: int = 20) -> PowerCurve:
    # On écarte les arrêts manifestes (vent suffisant, puissance nulle), qui
    # tireraient la médiane vers le bas dans les intervalles peu remplis.
    keep = ~((ws > 4) & (power <= 1))
    ws, power = ws[keep], power[keep]
    edges = np.arange(0, np.nanmax(ws) + BIN_WIDTH, BIN_WIDTH)
    idx = np.digitize(ws, edges) - 1
    centers, med, spread = [], [], []
    for b in range(len(edges) - 1):
        p = power[idx == b]
        if p.size < min_count:
            continue
        m = np.median(p)
        centers.append(edges[b] + BIN_WIDTH / 2)
        med.append(m)
        spread.append(max(1.4826 * np.median(np.abs(p - m)), 10.0))
    return PowerCurve(np.array(centers), np.array(med), np.array(spread))


def fit_all(scada: pd.DataFrame) -> dict[str, PowerCurve]:
    return {wt: fit_power_curve(g["wind_speed"].to_numpy(), g["power_kw"].to_numpy())
            for wt, g in scada.groupby("turbine")}


def add_expected(scada: pd.DataFrame, curves: dict[str, PowerCurve]) -> pd.DataFrame:
    """Ajoute la puissance attendue, le résidu et le résidu normalisé."""
    out = []
    for wt, g in scada.groupby("turbine", sort=False):
        g = g.copy()
        c = curves[wt]
        ws = g["wind_speed"].to_numpy()
        g["power_expected_kw"] = c.expected(ws)
        g["residual_kw"] = g["power_kw"] - g["power_expected_kw"]
        g["z"] = g["residual_kw"] / c.scale(ws)
        out.append(g)
    return pd.concat(out, ignore_index=True)
