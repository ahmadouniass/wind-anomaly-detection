"""Figures du rapport."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

COLORS = {"": "#B8BEC6", "arret": "#C0392B", "plafonnement": "#E67E22",
          "sous_performance": "#8E44AD", "derive": "#1F4E79"}
LABELS = {"": "Normal", "arret": "Arrêt", "plafonnement": "Plafonnement",
          "sous_performance": "Sous-performance", "derive": "Dérive"}
CAT_COLORS = {"arret": "#C0392B", "bridage": "#E67E22", "givrage": "#2E86C1",
              "degradation": "#1F4E79", "autre": "#7F8C8D"}
CAT_LABELS = {"arret": "Arrêt", "bridage": "Bridage", "givrage": "Givrage",
              "degradation": "Dégradation", "autre": "Autre"}


def _style(ax):
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(alpha=0.25)


def power_curve(scada: pd.DataFrame, curves: dict, turbine: str, path: str) -> None:
    g = scada[scada["turbine"] == turbine]
    fig, ax = plt.subplots(figsize=(8, 5))
    for lab in ["", "derive", "sous_performance", "plafonnement", "arret"]:
        d = g[g["label"] == lab]
        if len(d):
            ax.scatter(d["wind_speed"], d["power_kw"], s=2 if lab == "" else 4,
                       c=COLORS[lab], alpha=0.35 if lab == "" else 0.8, label=LABELS[lab], rasterized=True)
    c = curves[turbine]
    ax.plot(c.centers, c.median, color="black", lw=2, label="Courbe de référence")
    ax.set(xlabel="Vitesse du vent (m/s)", ylabel="Puissance (kW)", xlim=(0, 25),
           title=f"{turbine} : courbe de puissance et points anormaux")
    ax.legend(markerscale=4, frameon=False, loc="upper left")
    _style(ax)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def timeline(events: pd.DataFrame, path: str) -> None:
    turbines = sorted(events["turbine"].unique())
    fig, ax = plt.subplots(figsize=(11, 0.6 * len(turbines) + 1.6))
    for i, wt in enumerate(turbines):
        for ev in events[events["turbine"] == wt].itertuples():
            width = max((ev.end - ev.start) / pd.Timedelta("1D"), 0.8)
            ax.barh(i, width, left=ev.start, height=0.55, color=CAT_COLORS[ev.categorie])
    ax.set_yticks(range(len(turbines)), turbines)
    ax.invert_yaxis()
    handles = [plt.Rectangle((0, 0), 1, 1, color=CAT_COLORS[k]) for k in CAT_COLORS
               if k in set(events["categorie"])]
    ax.legend(handles, [CAT_LABELS[k] for k in CAT_COLORS if k in set(events["categorie"])],
              ncol=5, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.12))
    ax.set_title("Événements détectés, par turbine et par cause probable")
    _style(ax)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def losses_by_cause(events: pd.DataFrame, path: str) -> None:
    t = events.groupby(["turbine", "categorie"])["loss_kwh"].sum().unstack(fill_value=0) / 1000
    t = t.loc[:, [c for c in CAT_COLORS if c in t.columns]]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    bottom = np.zeros(len(t))
    for cat in t.columns:
        ax.bar(t.index, t[cat], bottom=bottom, color=CAT_COLORS[cat], label=CAT_LABELS[cat], width=0.6)
        bottom += t[cat].to_numpy()
    for x, total in enumerate(bottom):
        ax.text(x, total + bottom.max() * 0.01, f"{total:.0f}", ha="center", va="bottom")
    ax.set(ylabel="Énergie perdue estimée (MWh)", title="Pertes d'énergie estimées par turbine et par cause")
    ax.legend(frameon=False)
    _style(ax)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def drift_index(rel: pd.DataFrame, threshold: float, path: str) -> None:
    fig, ax = plt.subplots(figsize=(10, 4))
    for wt in rel.columns:
        ax.plot(rel.index, rel[wt], lw=1.4, label=wt)
    ax.set_ylim(min(0.9, float(rel.min().min()) - 0.01), 1.03)
    ax.axhline(threshold, color="#C0392B", ls="--", lw=1, label=f"Seuil de dérive ({threshold:.2f})")
    ax.set(ylabel="Rendement relatif aux voisines", title="Indice de performance journalier (médiane glissante 7 jours)")
    ax.legend(frameon=False, ncol=5)
    _style(ax)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
