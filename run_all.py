"""Pipeline complet : données -> courbe de puissance -> détection -> diagnostic -> pertes.

Exemples :
    python run_all.py                          # données simulées, avec évaluation
    python run_all.py --data scada.csv         # données réelles (format La Haute Borne)
    python run_all.py --agent 3                # + diagnostic par l'agent LLM des 3 plus gros événements
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

import pandas as pd  # noqa: E402

from windad import plots  # noqa: E402
from windad.data import load_scada  # noqa: E402
from windad.detect import DetectionConfig, daily_performance_index, detect  # noqa: E402
from windad.diagnose import diagnose_events  # noqa: E402
from windad.power_curve import add_expected, fit_all  # noqa: E402
from windad.simulate import simulate_farm  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", help="CSV de données SCADA réelles (sinon données simulées)")
    ap.add_argument("--alarms", help="CSV des alarmes (colonnes turbine, time, code, description)")
    ap.add_argument("--agent", type=int, default=0, help="nombre d'événements à confier à l'agent LLM")
    ap.add_argument("--out", default="outputs")
    args = ap.parse_args()

    out = Path(args.out)
    (out / "figures").mkdir(parents=True, exist_ok=True)
    truth = None

    if args.data:
        scada = load_scada(args.data)
        alarms = (pd.read_csv(args.alarms, parse_dates=["time"]) if args.alarms
                  else pd.DataFrame(columns=["turbine", "time", "code", "description"]))
    else:
        farm = simulate_farm()
        scada, alarms, truth = farm.scada, farm.alarms, farm.truth
        print(f"Données simulées : {scada['turbine'].nunique()} turbines, "
              f"{len(scada):,} pas de 10 min, {len(truth)} anomalies injectées")

    cfg = DetectionConfig()
    curves = fit_all(scada)
    scada = add_expected(scada, curves)
    scada, events = detect(scada, cfg)
    events = diagnose_events(events, scada, alarms)
    events.to_csv(out / "evenements.csv", index=False)

    total = events["loss_kwh"].sum() / 1000
    print(f"\n{len(events)} événements détectés, perte estimée totale : {total:.0f} MWh")
    print(events.groupby("cause_probable")["loss_kwh"].agg(["count", "sum"])
          .assign(sum=lambda d: (d["sum"] / 1000).round(1))
          .rename(columns={"count": "événements", "sum": "perte_MWh"})
          .sort_values("perte_MWh", ascending=False).to_string())

    if truth is not None:
        from windad.evaluate import evaluate, true_losses
        truth, recall, summary = evaluate(true_losses(truth, scada), events)
        print("\nÉvaluation sur la vérité terrain (anomalies ayant coûté au moins 500 kWh) :")
        print(recall.to_string())
        print(json.dumps({k: round(float(v), 3) for k, v in summary.items()}, indent=2, ensure_ascii=False))
        truth.to_csv(out / "verite_terrain.csv", index=False)
        with open(out / "evaluation.json", "w") as fh:
            json.dump({k: round(float(v), 3) for k, v in summary.items()}, fh, indent=2, ensure_ascii=False)

    worst = events.groupby("turbine")["loss_kwh"].sum().idxmax()
    plots.power_curve(scada, curves, worst, out / "figures" / "courbe_puissance.png")
    plots.timeline(events, out / "figures" / "chronologie.png")
    plots.losses_by_cause(events, out / "figures" / "pertes.png")
    # L'indice est recalculé sans les labels de dérive, pour voir la baisse elle-même
    before_drift = scada.assign(label=scada["label"].replace("derive", ""))
    plots.drift_index(daily_performance_index(before_drift, cfg), cfg.drift_threshold,
                      out / "figures" / "indice_derive.png")

    if args.agent:
        from windad.agent import DiagnosticTools, run_agent
        tools = DiagnosticTools(scada, events, alarms)
        results = []
        for eid in events.nlargest(args.agent, "loss_kwh")["event_id"]:
            print(f"\nAgent — {eid}")
            diag = run_agent(eid, tools, verbose=True)
            print(json.dumps(diag, indent=2, ensure_ascii=False))
            results.append({"event_id": eid, **diag})
        pd.DataFrame(results).to_csv(out / "diagnostics_agent.csv", index=False)

    print(f"\nRésultats écrits dans {out}/")


if __name__ == "__main__":
    main()
