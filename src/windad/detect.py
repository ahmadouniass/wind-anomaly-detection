"""Détection d'anomalies de performance à partir des résidus à la courbe de puissance.

Quatre détecteurs, du plus net au plus subtil :

* ``arret``            : vent suffisant mais production quasi nulle ;
* ``plafonnement``     : production bloquée sur un palier sous la puissance attendue ;
* ``sous_performance`` : résidu normalisé très négatif sur au moins une heure ;
* ``derive``           : baisse lente du rendement, repérée en comparant chaque
                         turbine à ses voisines jour après jour.

Chaque pas de temps reçoit au plus un label (par ordre de priorité), puis les pas
consécutifs de même label sont regroupés en événements.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

PRIORITY = ["arret", "plafonnement", "sous_performance", "derive"]
STEP_HOURS = 1 / 6  # pas de 10 minutes


@dataclass
class DetectionConfig:
    rated_kw: float = 2050.0
    min_expected_frac: float = 0.10   # on ne juge que si la production attendue est significative
    stop_frac: float = 0.02           # « quasi nulle » : moins de 2 % de la puissance nominale
    stop_min_expected_frac: float = 0.05
    plateau_std_frac: float = 0.01    # palier : écart-type sur 1 h < 1 % de la nominale
    plateau_gap_frac: float = 0.10    # et au moins 10 % sous l'attendu
    z_threshold: float = -3.0         # seuil sur le résidu normalisé
    min_steps: dict | None = None     # durée minimale d'un événement, par détecteur
    gap_steps: int = 3                # trous tolérés à l'intérieur d'un événement (30 min)
    under_gap_steps: int = 12         # 2 h
    drift_onset: float = 0.99         # début de la dérive : premier jour sous 99 %
    plateau_gap_steps: int = 72       # un bridage peut être masqué quelques heures par un vent faible
    drift_gap_steps: int = 288        # pour une dérive, jusqu'à 2 jours de pas jugés normaux
    drift_window_days: int = 7
    drift_threshold: float = 0.97     # rendement relatif aux voisines sous 97 %
    #   (sur données simulées, le bruit du rendement relatif ne descend pas sous 0,976 ;
    #    sur données réelles, ce seuil doit être recalé sur l'historique)

    def __post_init__(self) -> None:
        if self.min_steps is None:
            self.min_steps = {"arret": 3, "plafonnement": 6, "sous_performance": 6, "derive": 144}


def _point_labels(g: pd.DataFrame, cfg: DetectionConfig) -> np.ndarray:
    p = g["power_kw"].to_numpy()
    pe = g["power_expected_kw"].to_numpy()
    z = g["z"].to_numpy()
    judged = pe > cfg.min_expected_frac * cfg.rated_kw

    # Un arrêt est sans ambiguïté : on le juge dès 5 % de production attendue
    stop = (pe > cfg.stop_min_expected_frac * cfg.rated_kw) & (p < cfg.stop_frac * cfg.rated_kw)

    rolling_std = pd.Series(p).rolling(6, center=True, min_periods=4).std().to_numpy()
    plateau = (judged & (rolling_std < cfg.plateau_std_frac * cfg.rated_kw)
               & (p > 0.05 * cfg.rated_kw) & (pe - p > cfg.plateau_gap_frac * cfg.rated_kw))

    under = judged & (z < cfg.z_threshold)

    labels = np.full(len(g), "", dtype=object)
    labels[under] = "sous_performance"
    labels[plateau] = "plafonnement"
    labels[stop] = "arret"
    return labels


def daily_performance_index(scada: pd.DataFrame, cfg: DetectionConfig) -> pd.DataFrame:
    """Rendement journalier (production / attendu) de chaque turbine, rapporté à
    la médiane des autres turbines du parc le même jour.

    Diviser par les voisines retire les effets communs (densité de l'air,
    saisonnalité, biais de la courbe de référence) et isole ce qui est propre à
    une turbine.
    """
    ok = (scada["label"] == "") & (scada["power_expected_kw"] > 0.2 * cfg.rated_kw)
    d = scada[ok].assign(day=scada.loc[ok, "time"].dt.floor("D"))
    daily = d.groupby(["day", "turbine"])[["power_kw", "power_expected_kw"]].sum()
    pi = (daily["power_kw"] / daily["power_expected_kw"]).unstack("turbine")
    rel = pd.DataFrame(index=pi.index, columns=pi.columns, dtype=float)
    for wt in pi.columns:
        others = pi.drop(columns=wt).median(axis=1)
        rel[wt] = pi[wt] / others
    smooth = rel.rolling(cfg.drift_window_days, min_periods=4).median()
    return smooth


def _drift_days(r: pd.Series, threshold: float, onset: float) -> set:
    """Jours en dérive : sous le seuil, puis prolongés vers l'arrière tant que le
    rendement reste sous ``onset``. La médiane glissante réagit avec retard ;
    remonter jusqu'au début de la baisse évite de sous-estimer les pertes."""
    below = (r < threshold).to_numpy()
    soft = (r < onset).to_numpy()
    flag = below.copy()
    for i in np.flatnonzero(below):
        j = i - 1
        while j >= 0 and soft[j] and not flag[j]:
            flag[j] = True
            j -= 1
    return set(r.index[flag])


def _runs(labels: np.ndarray, judged: np.ndarray, gaps: dict[str, int]) -> list[tuple[str, int, int]]:
    """Regroupe les pas de même label en événements.

    Les pas où l'on ne peut pas juger (vent trop faible) sont neutres : ils ne
    coupent pas un événement. Seuls les pas jugés normaux comptent comme trou.
    """
    events, cur, start, last, holes = [], None, 0, 0, 0
    for i, lab in enumerate(labels):
        if lab == "":
            if judged[i] and cur is not None:
                holes += 1
                if holes > gaps[cur]:
                    events.append((cur, start, last))
                    cur, holes = None, 0
            continue
        if lab == cur:
            last, holes = i, 0
            continue
        if cur is not None:
            events.append((cur, start, last))
        cur, start, last, holes = lab, i, i, 0
    if cur is not None:
        events.append((cur, start, last))
    return events


def detect(scada: pd.DataFrame, cfg: DetectionConfig | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Renvoie (scada annoté avec la colonne ``label``, table des événements)."""
    cfg = cfg or DetectionConfig()
    scada = scada.sort_values(["turbine", "time"], ignore_index=True)
    scada["label"] = ""
    for wt, g in scada.groupby("turbine", sort=False):
        scada.loc[g.index, "label"] = _point_labels(g, cfg)

    # Dérive lente : jours où le rendement relatif glissant passe sous le seuil
    rel = daily_performance_index(scada, cfg)
    day = scada["time"].dt.floor("D")
    for wt in rel.columns:
        bad_days = _drift_days(rel[wt].dropna(), cfg.drift_threshold, cfg.drift_onset)
        m = (scada["turbine"] == wt) & day.isin(bad_days) & (scada["label"] == "") \
            & (scada["power_expected_kw"] > cfg.min_expected_frac * cfg.rated_kw)
        scada.loc[m, "label"] = "derive"

    rows = []
    for wt, g in scada.groupby("turbine", sort=False):
        gaps = {k: cfg.gap_steps for k in PRIORITY} | {
            "plafonnement": cfg.plateau_gap_steps, "sous_performance": cfg.under_gap_steps,
            "derive": cfg.drift_gap_steps}
        judged = (g["power_expected_kw"] > cfg.min_expected_frac * cfg.rated_kw).to_numpy()
        for kind, i0, i1 in _runs(g["label"].to_numpy(), judged, gaps):
            seg = g.iloc[i0 : i1 + 1]
            flagged = seg[seg["label"] == kind]
            if len(flagged) < cfg.min_steps[kind]:
                continue
            rows.append(dict(
                turbine=wt, detector=kind, start=seg["time"].iloc[0], end=seg["time"].iloc[-1],
                n_steps=len(flagged),
                energy_expected_kwh=flagged["power_expected_kw"].sum() * STEP_HOURS,
                energy_actual_kwh=flagged["power_kw"].sum() * STEP_HOURS,
            ))
    events = pd.DataFrame(rows)
    if not events.empty:
        events["loss_kwh"] = (events["energy_expected_kwh"] - events["energy_actual_kwh"]).clip(lower=0)
        events = events.sort_values(["turbine", "start"], ignore_index=True)
        events.insert(0, "event_id", [f"E{i:03d}" for i in range(1, len(events) + 1)])
    return scada, events
