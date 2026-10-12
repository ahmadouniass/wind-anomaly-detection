"""Évaluation sur données simulées, où l'on connaît la vérité terrain.

On ne retient comme « anomalies à détecter » que celles qui ont réellement
coûté de l'énergie : un bridage à 60 % par vent faible ne change rien à la
production et ne peut pas être vu.
"""

from __future__ import annotations

import pandas as pd

STEP_HOURS = 1 / 6
MIN_TRUE_LOSS_KWH = 500.0


def true_losses(truth: pd.DataFrame, scada: pd.DataFrame) -> pd.DataFrame:
    truth = truth.copy()
    losses = []
    for t in truth.itertuples():
        m = (scada["turbine"] == t.turbine) & scada["time"].between(t.start, t.end)
        g = scada.loc[m]
        losses.append(((g["power_clean_kw"] - g["power_kw"]).clip(lower=0)).sum() * STEP_HOURS)
    truth["true_loss_kwh"] = losses
    return truth


def evaluate(truth: pd.DataFrame, events: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Renvoie (vérité annotée, rappel par type, résumé global)."""
    tol = pd.Timedelta("1D")
    truth = truth.copy()
    truth["detecte"] = False
    events = events.copy()
    events["vrai_positif"] = False
    truth["bien_diagnostique"] = False
    for i, t in truth.iterrows():
        m = ((events["turbine"] == t["turbine"]) & (events["start"] <= t["end"] + tol)
             & (events["end"] >= t["start"] - tol))
        if m.any():
            truth.loc[i, "detecte"] = True
            events.loc[m, "vrai_positif"] = True
            if "categorie" in events:
                truth.loc[i, "bien_diagnostique"] = (events.loc[m, "categorie"] == t["kind"]).any()

    impact = truth[truth["true_loss_kwh"] >= MIN_TRUE_LOSS_KWH]
    recall = impact.groupby("kind").agg(
        a_detecter=("detecte", "size"), detectees=("detecte", "sum"),
        bien_diagnostiquees=("bien_diagnostique", "sum"))
    summary = {
        "rappel_global": impact["detecte"].mean(),
        "diagnostic_correct": impact.loc[impact["detecte"], "bien_diagnostique"].mean(),
        "precision": events["vrai_positif"].mean(),
        "perte_vraie_mwh": impact["true_loss_kwh"].sum() / 1000,
        "perte_estimee_mwh": events.loc[events["vrai_positif"], "loss_kwh"].sum() / 1000,
        "faux_positifs": int((~events["vrai_positif"]).sum()),
    }
    return truth, recall, summary
