"""Chargement de données SCADA réelles au format interne du projet.

Le format interne est une table longue : turbine, time, wind_speed, power_kw,
pitch_deg, rotor_rpm, temp_c. Pour un autre jeu de données, il suffit d'adapter
le dictionnaire de correspondance des colonnes.
"""

from __future__ import annotations

import pandas as pd

# Correspondance pour les données ouvertes du parc de La Haute Borne.
# Noms à vérifier sur l'en-tête du fichier téléchargé.
LA_HAUTE_BORNE_COLUMNS = {
    "Wind_turbine_name": "turbine",
    "Date_time": "time",
    "Ws_avg": "wind_speed",
    "P_avg": "power_kw",
    "Ba_avg": "pitch_deg",
    "Rs_avg": "rotor_rpm",
    "Ot_avg": "temp_c",
}

REQUIRED = ["turbine", "time", "wind_speed", "power_kw"]


def load_scada(path: str, columns: dict[str, str] = LA_HAUTE_BORNE_COLUMNS,
               sep: str = ";") -> pd.DataFrame:
    """Lit un fichier CSV et le convertit au format interne."""
    df = pd.read_csv(path, sep=sep, usecols=lambda c: c in columns)
    df = df.rename(columns=columns)
    missing = [c for c in REQUIRED if c not in df.columns]
    if missing:
        raise ValueError(f"Colonnes manquantes après renommage : {missing}")
    df["time"] = pd.to_datetime(df["time"], utc=True).dt.tz_localize(None)
    df = df.dropna(subset=["wind_speed", "power_kw"])
    return df.sort_values(["turbine", "time"], ignore_index=True)
