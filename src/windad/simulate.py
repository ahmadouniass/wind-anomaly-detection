"""Génération de données SCADA simulées pour un petit parc éolien.

Le format imite celui des données ouvertes de parcs éoliens : un pas de temps de
10 minutes, une ligne par turbine et par pas de temps. Des anomalies de types
connus sont injectées, ce qui permet d'évaluer la détection (vérité terrain).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

TURBINES = ["WT01", "WT02", "WT03", "WT04"]
RATED_KW = 2050.0
CUT_IN, RATED_WS, CUT_OUT = 3.0, 12.5, 25.0
STEP = pd.Timedelta("10min")


@dataclass
class SimulatedFarm:
    scada: pd.DataFrame    # séries temporelles 10 min
    alarms: pd.DataFrame   # journal des alarmes
    static: pd.DataFrame   # caractéristiques des turbines
    truth: pd.DataFrame    # anomalies injectées (vérité terrain)


def nominal_power(ws: np.ndarray) -> np.ndarray:
    """Courbe de puissance théorique (kW) d'une turbine de 2,05 MW."""
    frac = (ws**3 - CUT_IN**3) / (RATED_WS**3 - CUT_IN**3)
    p = RATED_KW * np.clip(frac, 0, 1)
    p[(ws < CUT_IN) | (ws > CUT_OUT)] = 0.0
    return p


def nominal_pitch(ws: np.ndarray) -> np.ndarray:
    """Angle de pitch (°) : nul sous la vitesse nominale, croissant au-delà."""
    return np.where(ws > RATED_WS, (ws - RATED_WS) * 2.2, 0.0)


def _wind_series(n: int, rng: np.random.Generator, months: np.ndarray) -> np.ndarray:
    """Vitesse de vent commune au parc : AR(1) gaussien transformé en Weibull."""
    from scipy import stats

    z = np.empty(n)
    z[0] = rng.normal()
    phi = 0.995
    eps = rng.normal(scale=np.sqrt(1 - phi**2), size=n)
    for t in range(1, n):
        z[t] = phi * z[t - 1] + eps[t]
    u = stats.norm.cdf(z)
    # Vent plus fort en hiver : paramètre d'échelle saisonnier
    scale = 7.5 + 1.5 * np.cos(2 * np.pi * (months - 1) / 12)
    return stats.weibull_min.ppf(u, c=2.1, scale=scale)


def simulate_farm(start: str = "2024-01-01", days: int = 365, seed: int = 42) -> SimulatedFarm:
    rng = np.random.default_rng(seed)
    times = pd.date_range(start, periods=days * 144, freq=STEP)
    n = len(times)
    months = times.month.to_numpy()
    hours = times.hour.to_numpy() + times.minute.to_numpy() / 60

    ws_farm = _wind_series(n, rng, months)
    doy = times.dayofyear.to_numpy()
    temp = 11 - 8 * np.cos(2 * np.pi * (doy - 15) / 365) + 4 * np.sin(2 * np.pi * (hours - 9) / 24)
    temp = temp + rng.normal(0, 1.5, n)

    frames, truth_rows, alarm_rows = [], [], []
    for k, wt in enumerate(TURBINES):
        ws = np.clip(ws_farm * (1 + 0.03 * (k - 1.5) / 1.5) + rng.normal(0, 0.35, n), 0, None)
        p_nom = nominal_power(ws)
        power = p_nom * (1 + rng.normal(0, 0.03, n)) + rng.normal(0, 15, n)
        power = np.clip(power, 0, RATED_KW * 1.02)
        power[p_nom == 0] = 0.0
        pitch = nominal_pitch(ws) + rng.normal(0, 0.3, n)
        pitch[ws < CUT_IN] = 85.0
        rotor = np.where(ws >= CUT_IN, np.clip(6 + 0.8 * ws, 0, 16.8), 0.5) + rng.normal(0, 0.2, n)
        power_clean = power.copy()  # production sans anomalie, pour mesurer la vraie perte
        label = np.array([""] * n, dtype=object)

        def window(min_len: int, max_len: int, months_ok: set[int] | None = None) -> slice:
            """Tire une fenêtre libre (sans anomalie déjà injectée)."""
            for _ in range(500):
                length = int(rng.integers(min_len, max_len))
                i0 = int(rng.integers(0, n - length))
                if months_ok and months[i0] not in months_ok:
                    continue
                if (label[i0 : i0 + length] == "").all():
                    return slice(i0, i0 + length)
            raise RuntimeError("Impossible de placer l'anomalie")

        def record(kind: str, sl: slice, detail: str) -> None:
            label[sl] = kind
            truth_rows.append(dict(turbine=wt, kind=kind, start=times[sl.start],
                                   end=times[sl.stop - 1], detail=detail))

        # 1) Arrêts sur défaut (6 h à 3 jours), avec alarme
        for _ in range(3):
            sl = window(36, 432)
            code = int(rng.choice([1001, 2040, 3120]))
            power[sl], rotor[sl], pitch[sl] = 0.0, 0.3, 88.0
            record("arret", sl, f"alarme {code}")
            alarm_rows.append(dict(turbine=wt, time=times[sl.start], code=code,
                                   description={1001: "Défaut convertisseur",
                                                2040: "Surchauffe génératrice",
                                                3120: "Défaut système de pitch"}[code]))

        # 2) Bridage : puissance plafonnée à 60 % (1 à 4 jours)
        for _ in range(2):
            sl = window(144, 576)
            cap = 0.6 * RATED_KW
            over = power[sl] > cap
            power[sl] = np.where(over, cap + rng.normal(0, 5, over.size), power[sl])
            pitch[sl] = np.where(over, pitch[sl] + 4, pitch[sl])
            record("bridage", sl, "plafond 60 %")

        # 3) Givrage en hiver : perte de 30 à 50 % par grand froid (12 à 48 h)
        sl = window(72, 288, months_ok={1, 2, 12})
        temp[sl] = rng.uniform(-4, 0.5, sl.stop - sl.start)
        power[sl] *= rng.uniform(0.5, 0.7)
        record("givrage", sl, "température < 1 °C")

        # 4) Dégradation progressive (une seule turbine) : -8 % sur 3 semaines,
        #    avec un décalage de pitch qui l'explique
        if wt == "WT03":
            sl = window(21 * 144, 21 * 144 + 1)
            ramp = np.linspace(0.97, 0.90, sl.stop - sl.start)
            power[sl] *= ramp
            pitch[sl] += 2.0
            record("degradation", sl, "désalignement de pitch (+2°)")

        frames.append(pd.DataFrame({
            "turbine": wt, "time": times, "wind_speed": ws, "power_kw": power,
            "pitch_deg": pitch, "rotor_rpm": rotor, "temp_c": temp,
            "power_clean_kw": power_clean, "true_anomaly": label,
        }))

    # Quelques alarmes « bruit » sans perte d'énergie, comme dans la réalité
    for _ in range(25):
        alarm_rows.append(dict(turbine=str(rng.choice(TURBINES)), time=times[int(rng.integers(0, n))],
                               code=900, description="Avertissement capteur (sans arrêt)"))

    static = pd.DataFrame({"turbine": TURBINES, "rated_kw": RATED_KW, "hub_height_m": 80,
                           "rotor_diameter_m": 82, "commissioning_year": [2009, 2009, 2009, 2010]})
    return SimulatedFarm(
        scada=pd.concat(frames, ignore_index=True),
        alarms=pd.DataFrame(alarm_rows).sort_values("time", ignore_index=True),
        static=static,
        truth=pd.DataFrame(truth_rows).sort_values(["turbine", "start"], ignore_index=True),
    )
