"""Orchestrator: one command runs the whole lifecycle.

    python run_pipeline.py all                  ingest -> transform -> train -> monitor
    python run_pipeline.py all --auto-retrain   ... and retrain automatically if the monitor says so
    python run_pipeline.py monitor              check the current model only
"""
import argparse

import ingest
import monitor
import train
import transform


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("step", choices=["ingest", "transform", "train", "monitor", "all"])
    ap.add_argument("--start", default="2021-01-01")
    ap.add_argument("--end", default=ingest.default_end())
    ap.add_argument("--auto-retrain", action="store_true")
    a = ap.parse_args()

    if a.step in ("ingest", "all"):
        ingest.ingest(a.start, a.end)
    if a.step in ("transform", "all"):
        transform.run()
    if a.step in ("train", "all"):
        train.train()
    if a.step in ("monitor", "all"):
        report = monitor.monitor()
        print(monitor.summary(report))
        if report["decision"] == "RETRAIN" and a.auto_retrain:
            print("[pipeline] drift detected -> retraining on all available data")
            train.train()
            print(monitor.summary(monitor.monitor()))


if __name__ == "__main__":
    main()
