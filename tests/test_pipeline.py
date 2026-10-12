"""Tests automatiques : ils vérifient que le pipeline continue de fonctionner
après chaque modification (``pytest``)."""

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

from windad.agent import DiagnosticTools, run_agent  # noqa: E402
from windad.detect import _runs, detect  # noqa: E402
from windad.diagnose import diagnose_events, rule_based_diagnosis  # noqa: E402
from windad.evaluate import evaluate, true_losses  # noqa: E402
from windad.power_curve import add_expected, fit_all, fit_power_curve  # noqa: E402
from windad.simulate import RATED_KW, nominal_power, simulate_farm  # noqa: E402


@pytest.fixture(scope="module")
def pipeline():
    farm = simulate_farm(days=120, seed=3)
    scada = add_expected(farm.scada, fit_all(farm.scada))
    scada, events = detect(scada)
    events = diagnose_events(events, scada, farm.alarms)
    return farm, scada, events


def test_power_curve_recovers_nominal_curve():
    rng = np.random.default_rng(0)
    ws = rng.uniform(0, 20, 20000)
    p = nominal_power(ws) + rng.normal(0, 20, ws.size)
    c = fit_power_curve(ws, p)
    grid = np.array([5.0, 8.0, 11.0, 15.0])
    assert np.allclose(c.expected(grid), nominal_power(grid), atol=0.03 * RATED_KW)


def test_runs_merges_short_gaps_and_ignores_low_wind():
    labels = np.array(["arret", "", "arret", "", "", "", "", "arret"], dtype=object)
    judged = np.array([True, True, True, False, False, False, False, True])
    # le trou de 1 pas est toléré ; les 4 pas non jugés (vent faible) ne coupent pas
    assert _runs(labels, judged, {"arret": 1}) == [("arret", 0, 7)]
    judged[3:7] = True
    assert len(_runs(labels, judged, {"arret": 1})) == 2


def test_no_event_on_clean_data():
    farm = simulate_farm(days=120, seed=5)
    scada = farm.scada.assign(power_kw=farm.scada["power_clean_kw"])
    scada = add_expected(scada, fit_all(scada))
    _, events = detect(scada)
    assert len(events) <= 1  # au plus une fausse alerte sur 120 jours et 4 turbines


def test_detection_and_losses(pipeline):
    farm, scada, events = pipeline
    truth, _, summary = evaluate(true_losses(farm.truth, scada), events)
    assert summary["rappel_global"] >= 0.8
    assert summary["precision"] >= 0.9
    # l'énergie perdue estimée est du bon ordre de grandeur
    assert 0.7 <= summary["perte_estimee_mwh"] / summary["perte_vraie_mwh"] <= 1.1


def test_events_have_required_columns(pipeline):
    _, _, events = pipeline
    for col in ["event_id", "turbine", "detector", "start", "end", "loss_kwh",
                "categorie", "cause_probable", "action"]:
        assert col in events.columns
    assert (events["loss_kwh"] >= 0).all()
    assert (events["end"] >= events["start"]).all()


def test_rules():
    assert rule_based_diagnosis("arret", {"alarmes": ["1001 – Défaut"]})[0] == "arret"
    assert rule_based_diagnosis("sous_performance", {"alarmes": [], "temperature_moyenne_c": -2})[0] == "givrage"
    assert rule_based_diagnosis("derive", {"alarmes": [], "ecart_pitch_deg": 2.0})[1].startswith("Désalignement")


class FakeClient:
    """Imite l'API : un appel d'outil, puis une réponse finale en JSON."""

    def __init__(self, event_id):
        self.calls = 0
        self.event_id = event_id
        self.messages = self

    def create(self, **kwargs):
        self.calls += 1
        if self.calls == 1:
            block = SimpleNamespace(type="tool_use", id="t1", name="get_event_features",
                                    input={"event_id": self.event_id})
            return SimpleNamespace(stop_reason="tool_use", content=[block])
        # la 2e requête doit contenir le résultat de l'outil
        last = kwargs["messages"][-1]["content"][0]
        assert last["type"] == "tool_result" and "perte_kwh" in last["content"]
        text = 'Diagnostic : {"cause": "Arrêt sur défaut", "confiance": "forte", ' \
               '"justification": "puissance nulle et alarme", "action": "maintenance"}'
        return SimpleNamespace(stop_reason="end_turn", content=[SimpleNamespace(type="text", text=text)])


def test_agent_loop_with_fake_client(pipeline):
    farm, scada, events = pipeline
    tools = DiagnosticTools(scada, events, farm.alarms)
    eid = events["event_id"].iloc[0]
    out = run_agent(eid, tools, client=FakeClient(eid))
    assert out["cause"] == "Arrêt sur défaut" and out["confiance"] == "forte"


def test_agent_tools_return_serialisable(pipeline):
    import json
    farm, scada, events = pipeline
    tools = DiagnosticTools(scada, events, farm.alarms)
    eid = events["event_id"].iloc[0]
    wt = events["turbine"].iloc[0]
    for name, args in [("get_event_features", {"event_id": eid}),
                       ("compare_with_neighbors", {"event_id": eid}),
                       ("get_turbine_history", {"turbine": wt}),
                       ("get_alarms", {"turbine": wt, "start": "2024-01-01", "end": "2024-12-31"})]:
        json.dumps(tools.call(name, args), default=str)
