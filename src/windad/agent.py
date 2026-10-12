"""Agent de diagnostic : un LLM qui enquête sur un événement avec des outils.

Le diagnostic par règles (module ``diagnose``) est rapide et lisible, mais
rigide. L'agent reçoit un événement et dispose d'outils pour consulter les
données : indicateurs de l'événement, alarmes, comparaison aux voisines,
historique de la turbine. Il décide lui-même quels outils appeler, puis rend un
diagnostic structuré (cause, niveau de confiance, justification, action).

Nécessite le paquet ``anthropic`` et la variable d'environnement
``ANTHROPIC_API_KEY``. Le modèle se règle avec ``WINDAD_MODEL``.
"""

from __future__ import annotations

import json
import os
from typing import Any

import pandas as pd

from .diagnose import event_features, fit_pitch_reference

DEFAULT_MODEL = os.environ.get("WINDAD_MODEL", "claude-sonnet-5-5")

SYSTEM_PROMPT = """Tu es un ingénieur performance d'un exploitant de parcs éoliens.
On te confie un événement de sous-production détecté automatiquement sur une turbine.
Utilise les outils pour rassembler des éléments, puis établis le diagnostic le plus probable.

Causes possibles : arrêt sur défaut, arrêt planifié, bridage (propre à la turbine ou au parc),
givrage, dégradation progressive (encrassement, érosion, désalignement de pitch),
problème de capteur, effet météo commun au parc. Si les éléments ne suffisent pas, dis-le.

Réponds en français, à la fin, par un unique objet JSON :
{"cause": "...", "confiance": "faible|moyenne|forte", "justification": "...", "action": "..."}"""

TOOLS = [
    {
        "name": "get_event_features",
        "description": "Indicateurs de l'événement : durée, vent moyen, rendement (production / attendu), "
                       "température moyenne, écart de pitch au comportement normal, rendement des voisines, alarmes.",
        "input_schema": {"type": "object", "properties": {"event_id": {"type": "string"}},
                         "required": ["event_id"]},
    },
    {
        "name": "get_alarms",
        "description": "Alarmes d'une turbine entre deux dates (format ISO), y compris les simples avertissements.",
        "input_schema": {"type": "object", "properties": {
            "turbine": {"type": "string"}, "start": {"type": "string"}, "end": {"type": "string"}},
            "required": ["turbine", "start", "end"]},
    },
    {
        "name": "compare_with_neighbors",
        "description": "Rendement de chaque turbine du parc pendant la période de l'événement.",
        "input_schema": {"type": "object", "properties": {"event_id": {"type": "string"}},
                         "required": ["event_id"]},
    },
    {
        "name": "get_turbine_history",
        "description": "Autres événements détectés sur la même turbine, avec leur cause probable par règles.",
        "input_schema": {"type": "object", "properties": {"turbine": {"type": "string"}},
                         "required": ["turbine"]},
    },
]


class DiagnosticTools:
    """Fonctions exposées à l'agent. Elles renvoient du JSON sérialisable."""

    def __init__(self, scada: pd.DataFrame, events: pd.DataFrame, alarms: pd.DataFrame):
        self.scada, self.events, self.alarms = scada, events.set_index("event_id"), alarms
        self.pitch_ref = fit_pitch_reference(scada)

    def get_event_features(self, event_id: str) -> dict:
        ev = self.events.loc[event_id]
        return {"turbine": ev["turbine"], "detecteur": ev["detector"],
                "debut": str(ev["start"]), "fin": str(ev["end"]),
                "perte_kwh": round(float(ev["loss_kwh"])),
                **event_features(ev, self.scada, self.alarms, self.pitch_ref)}

    def get_alarms(self, turbine: str, start: str, end: str) -> list[dict]:
        a = self.alarms[(self.alarms["turbine"] == turbine)
                        & self.alarms["time"].between(pd.Timestamp(start), pd.Timestamp(end))]
        return [{"date": str(r.time), "code": int(r.code), "description": r.description}
                for r in a.itertuples()]

    def compare_with_neighbors(self, event_id: str) -> dict:
        ev = self.events.loc[event_id]
        w = self.scada[self.scada["time"].between(ev["start"], ev["end"])
                       & (self.scada["power_expected_kw"] > 0)]
        g = w.groupby("turbine")[["power_kw", "power_expected_kw"]].sum()
        return (g["power_kw"] / g["power_expected_kw"]).round(3).to_dict()

    def get_turbine_history(self, turbine: str) -> list[dict]:
        ev = self.events[self.events["turbine"] == turbine]
        cols = [c for c in ["detector", "start", "end", "loss_kwh", "cause_probable"] if c in ev]
        return json.loads(ev[cols].reset_index().to_json(orient="records", date_format="iso"))

    def call(self, name: str, args: dict) -> Any:
        return getattr(self, name)(**args)


def run_agent(event_id: str, tools: DiagnosticTools, client=None, model: str = DEFAULT_MODEL,
              max_turns: int = 8, verbose: bool = False) -> dict:
    """Fait enquêter l'agent sur un événement et renvoie son diagnostic (dict)."""
    if client is None:
        import anthropic
        client = anthropic.Anthropic()

    messages = [{"role": "user", "content": f"Diagnostique l'événement {event_id}."}]
    for _ in range(max_turns):
        resp = client.messages.create(model=model, max_tokens=1500, system=SYSTEM_PROMPT,
                                      tools=TOOLS, messages=messages)
        messages.append({"role": "assistant", "content": resp.content})
        if resp.stop_reason != "tool_use":
            text = "".join(b.text for b in resp.content if b.type == "text")
            return _parse_json(text)
        results = []
        for block in resp.content:
            if block.type != "tool_use":
                continue
            try:
                out = tools.call(block.name, block.input)
            except Exception as exc:  # l'agent voit l'erreur et peut corriger son appel
                out = {"erreur": str(exc)}
            if verbose:
                print(f"  -> {block.name}({block.input})")
            results.append({"type": "tool_result", "tool_use_id": block.id,
                            "content": json.dumps(out, ensure_ascii=False, default=str)})
        messages.append({"role": "user", "content": results})
    return {"cause": "indéterminée", "confiance": "faible",
            "justification": "Nombre maximal d'étapes atteint", "action": "Analyse manuelle"}


def _parse_json(text: str) -> dict:
    start, end = text.rfind("{"), text.rfind("}")
    try:
        return json.loads(text[start : end + 1])
    except (ValueError, json.JSONDecodeError):
        return {"cause": "indéterminée", "confiance": "faible", "justification": text.strip(),
                "action": "Analyse manuelle"}
