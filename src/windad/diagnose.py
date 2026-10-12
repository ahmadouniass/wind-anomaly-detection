"""Diagnostic des événements détectés.

Pour chaque événement, on calcule quelques indicateurs explicatifs, puis on
applique des règles métier simples et lisibles. Ces mêmes indicateurs servent
d'outils à l'agent LLM (module ``agent``), qui peut les combiner plus librement.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .power_curve import BIN_WIDTH


def fit_pitch_reference(scada: pd.DataFrame) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Pitch médian par intervalle de vent, par turbine (période normale uniquement)."""
    ref = {}
    normal = scada[(scada["label"] == "") & (scada["power_kw"] > 0)]
    for wt, g in normal.groupby("turbine"):
        b = (g["wind_speed"] // BIN_WIDTH) * BIN_WIDTH + BIN_WIDTH / 2
        med = g.groupby(b)["pitch_deg"].median()
        ref[wt] = (med.index.to_numpy(), med.to_numpy())
    return ref


def event_features(event: pd.Series, scada: pd.DataFrame, alarms: pd.DataFrame,
                   pitch_ref: dict) -> dict:
    """Indicateurs explicatifs d'un événement."""
    wt, t0, t1 = event["turbine"], event["start"], event["end"]
    win = scada[(scada["time"] >= t0) & (scada["time"] <= t1)]
    own = win[(win["turbine"] == wt) & (win["label"] == event["detector"])]
    feats = {
        "duree_h": round((t1 - t0).total_seconds() / 3600 + 1 / 6, 1),
        "vent_moyen_ms": round(own["wind_speed"].mean(), 1),
        "rendement": round(own["power_kw"].sum() / max(own["power_expected_kw"].sum(), 1e-9), 3),
    }
    if "temp_c" in own:
        feats["temperature_moyenne_c"] = round(own["temp_c"].mean(), 1)
    if "pitch_deg" in own and wt in pitch_ref and event["detector"] != "arret":
        x, y = pitch_ref[wt]
        expected_pitch = np.interp(own["wind_speed"], x, y)
        feats["ecart_pitch_deg"] = round(float(np.median(own["pitch_deg"] - expected_pitch)), 2)

    # Les voisines souffrent-elles en même temps ? (effet parc ou effet turbine)
    nb = win[(win["turbine"] != wt) & win["time"].isin(own["time"])]
    if len(nb):
        feats["rendement_voisines"] = round(
            nb["power_kw"].sum() / max(nb["power_expected_kw"].sum(), 1e-9), 3)

    # Alarmes de la turbine pendant l'événement ou dans les 24 h qui précèdent :
    # un arrêt peut commencer par vent faible et n'être détecté que plus tard.
    a = alarms[(alarms["turbine"] == wt) & (alarms["time"] >= t0 - pd.Timedelta("24h"))
               & (alarms["time"] <= t1)]
    a = a[a["code"] != 900]  # avertissements sans arrêt
    feats["alarmes"] = [f"{r.code} – {r.description}" for r in a.itertuples()]
    return feats


def rule_based_diagnosis(detector: str, f: dict) -> tuple[str, str, str]:
    """Renvoie (catégorie, cause probable, action recommandée)."""
    nb = f.get("rendement_voisines", 1.0)
    if detector == "arret":
        if f["alarmes"]:
            return ("arret", f"Arrêt sur défaut ({f['alarmes'][0]})",
                    "Intervention de maintenance corrective")
        return ("arret", "Arrêt sans alarme associée",
                "Vérifier le planning de maintenance et la disponibilité")
    if detector == "plafonnement":
        if nb < 0.85:
            return ("bridage", "Bridage à l'échelle du parc",
                    "Vérifier les consignes du gestionnaire de réseau")
        return ("bridage", "Bridage propre à la turbine",
                "Vérifier les consignes de derating et les protections thermiques")
    if f.get("temperature_moyenne_c", 10) < 2:
        return ("givrage", "Givrage probable des pales",
                "Vérifier les capteurs de givre et la procédure de redémarrage")
    if detector == "derive":
        if abs(f.get("ecart_pitch_deg", 0)) > 1:
            return ("degradation", "Désalignement de pitch probable",
                    "Contrôler l'étalonnage des pales")
        return ("degradation", "Dégradation de performance",
                "Inspection des pales (encrassement, érosion)")
    if nb < 0.85:
        return ("autre", "Sous-performance commune au parc",
                "Comparer aux données météo (densité de l'air, turbulence)")
    return "autre", "Sous-performance non expliquée", "Analyse détaillée par un expert"


def diagnose_events(events: pd.DataFrame, scada: pd.DataFrame, alarms: pd.DataFrame) -> pd.DataFrame:
    pitch_ref = fit_pitch_reference(scada)
    rows = []
    for _, ev in events.iterrows():
        f = event_features(ev, scada, alarms, pitch_ref)
        cat, cause, action = rule_based_diagnosis(ev["detector"], f)
        rows.append({**f, "categorie": cat, "cause_probable": cause, "action": action,
                     "alarmes": "; ".join(f["alarmes"])})
    return pd.concat([events.reset_index(drop=True), pd.DataFrame(rows)], axis=1)
