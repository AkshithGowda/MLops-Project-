"""Stage 5 - monitoring and feedback loop: data drift (PSI + KS test) and live accuracy, with a retrain decision."""
import argparse
import json

import joblib
import matplotlib
import numpy as np
import pandas as pd
from scipy.stats import ks_2samp

from config import DRIFT_FEATURES, FEATURES, GOLD, MODELS, REPORTS, TARGET

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

PSI_DRIFT = 0.20          # PSI >= 0.2 is the usual "significant shift" threshold
DRIFT_SHARE = 0.30        # retrain if at least 30 % of the monitored features drifted ...
MAE_RATIO = 1.25          # retrain outright if the live error is 25 % worse than at training time
MAE_SOFT = 1.10           # ... or 10 % worse while the inputs have drifted


def psi(ref, cur, bins: int = 10) -> float:
    """Population Stability Index of `cur` against `ref`, using the reference's quantile bins."""
    ref, cur = np.asarray(ref, float), np.asarray(cur, float)
    edges = np.unique(np.quantile(ref, np.linspace(0, 1, bins + 1)))
    if len(edges) < 3:
        return 0.0
    edges[0], edges[-1] = -np.inf, np.inf
    r = np.clip(np.histogram(ref, edges)[0] / len(ref), 1e-4, None)
    c = np.clip(np.histogram(cur, edges)[0] / len(cur), 1e-4, None)
    return float(np.sum((c - r) * np.log(c / r)))


def drift_table(ref: pd.DataFrame, cur: pd.DataFrame) -> dict:
    out = {}
    for f in DRIFT_FEATURES:
        p = psi(ref[f], cur[f])
        out[f] = {"psi": round(p, 3), "ks_p_value": float(f"{ks_2samp(ref[f], cur[f]).pvalue:.3g}"),
                  "ref_mean": round(float(ref[f].mean()), 2), "cur_mean": round(float(cur[f].mean()), 2),
                  "drifted": bool(p >= PSI_DRIFT)}
    return out


def monitor(start=None, end=None, window_days=60, model_dir=MODELS, tag="monitoring") -> dict:
    model = joblib.load(model_dir / "model.joblib")
    meta = json.loads((model_dir / "metadata.json").read_text(encoding="utf-8"))
    ref = pd.read_parquet(model_dir / "reference.parquet")
    gold = pd.read_parquet(GOLD / "features.parquet").dropna(subset=[TARGET])
    if end is None:
        end = gold["date"].max()
    end = pd.Timestamp(end)
    start = pd.Timestamp(start) if start else end - pd.Timedelta(days=window_days - 1)
    cur = gold[(gold["date"] >= start) & (gold["date"] <= end)].copy()
    if len(cur) < 30:
        raise SystemExit("Monitoring window has too few rows.")

    # Weather is seasonal, so a monsoon week must be compared with past monsoons, not with the yearly average.
    # If the model never saw these months, fall back to all training rows - that itself is a warning sign.
    same_season = ref[ref["date"].dt.month.isin(set(cur["date"].dt.month))]
    matched = len(same_season) >= 100
    drift = drift_table(same_season if matched else ref, cur)
    share = sum(v["drifted"] for v in drift.values()) / len(drift)

    cur["pred"] = model.predict(cur[FEATURES])
    cur["abs_err"] = (cur["pred"] - cur[TARGET]).abs()
    live_mae = float(cur["abs_err"].mean())
    base_mae = float((cur["tmax"] - cur[TARGET]).abs().mean())
    train_mae = meta["test_metrics"]["mae"]
    ratio = live_mae / train_mae

    # Input drift alone is a warning (WATCH). Retrain only when accuracy is actually suffering,
    # or when drift is combined with a model that has never seen this season.
    drifted = share >= DRIFT_SHARE
    reasons = []
    if drifted:
        reasons.append(f"{share:.0%} of monitored features drifted (PSI >= {PSI_DRIFT})")
    if ratio >= MAE_RATIO:
        reasons.append(f"live MAE is {ratio:.2f}x the MAE measured at training time")
    elif drifted and ratio >= MAE_SOFT:
        reasons.append(f"live MAE is already {ratio:.2f}x the training MAE")
    elif drifted and not matched:
        reasons.append("no training data for this season")
    if ratio >= MAE_RATIO or (drifted and (ratio >= MAE_SOFT or not matched)):
        decision = "RETRAIN"
    elif drifted:
        decision = "WATCH"
    else:
        decision = "OK"
    report = {
        "window": [str(start.date()), str(end.date())], "rows": int(len(cur)),
        "model": meta["model_name"], "model_trained_on": meta["window"],
        "reference": (f"season-matched training rows ({len(same_season)})" if matched
                      else "all training rows (the model has no training data for these months)"),
        "drift": drift, "share_features_drifted": round(share, 2),
        "live_mae": round(live_mae, 3), "persistence_baseline_mae": round(base_mae, 3),
        "training_test_mae": round(train_mae, 3), "mae_ratio": round(ratio, 2),
        "decision": decision, "reasons": reasons,
    }
    REPORTS.mkdir(parents=True, exist_ok=True)
    (REPORTS / f"{tag}.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    plot(report, cur, REPORTS / f"{tag}.png")
    return report


def plot(report, cur, path):
    fig, ax = plt.subplots(1, 2, figsize=(11, 3.8))
    names = list(report["drift"])
    vals = [report["drift"][n]["psi"] for n in names]
    ax[0].barh(names, vals, color=["#c0392b" if v >= PSI_DRIFT else "#2e8b57" for v in vals])
    ax[0].axvline(PSI_DRIFT, color="k", ls="--", lw=1)
    ax[0].set_xlabel("PSI (dashed line = 0.2 drift threshold)")
    ax[0].set_title("Feature drift vs training data")
    ax[0].invert_yaxis()
    daily = cur.groupby("date")["abs_err"].mean()
    ax[1].plot(daily.index, daily.values, color="#444", lw=1, label="daily abs. error")
    ax[1].axhline(report["training_test_mae"], color="#2e8b57", ls="--", label="MAE at training")
    ax[1].axhline(report["live_mae"], color="#c0392b", ls="-", label="live MAE")
    ax[1].set_ylabel("deg C")
    ax[1].set_title(f"Live accuracy ({report['window'][0]} to {report['window'][1]})")
    ax[1].legend(fontsize=8)
    fig.autofmt_xdate()
    fig.suptitle(f"Decision: {report['decision']}", fontweight="bold")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def summary(r: dict) -> str:
    lines = [f"[monitor] window {r['window'][0]} .. {r['window'][1]} ({r['rows']} rows), model {r['model']}",
             f"[monitor] reference: {r['reference']}"]
    for f, v in r["drift"].items():
        lines.append(f"[monitor]   {f:<13} PSI={v['psi']:<6} KS p={v['ks_p_value']:<9} mean {v['ref_mean']} -> {v['cur_mean']}"
                     f"{'   DRIFT' if v['drifted'] else ''}")
    lines.append(f"[monitor] live MAE {r['live_mae']} (training {r['training_test_mae']}, persistence {r['persistence_baseline_mae']})")
    lines.append(f"[monitor] DECISION: {r['decision']}" + (" - " + "; ".join(r["reasons"]) if r["reasons"] else ""))
    return "\n".join(lines)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Check the live model for data drift and accuracy loss")
    ap.add_argument("--start")
    ap.add_argument("--end")
    ap.add_argument("--window-days", type=int, default=60)
    a = ap.parse_args()
    print(summary(monitor(a.start, a.end, a.window_days)))
