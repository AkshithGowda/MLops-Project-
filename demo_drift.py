"""Demonstration of the feedback loop (simulates 'trained in the dry season, deployed into the monsoon').

1. Train model v1 only on dry-season months (Dec-May).
2. Monitor it on the 2026 monsoon window -> drift is detected, retraining is recommended.
3. Retrain on all data available *before* the window (so the window stays unseen) and monitor again.
4. Finish with a normal full-data production model.
"""
import monitor
import train

WINDOW = ("2026-06-15", "2026-09-15")
DRY = [12, 1, 2, 3, 4, 5]

print("=" * 70, "\nSTEP 1: train v1 on dry-season data only (months 12,1,2,3,4,5)\n" + "=" * 70)
train.train(months=DRY)

print("=" * 70, "\nSTEP 2: monitor v1 on the monsoon window\n" + "=" * 70)
r1 = monitor.monitor(*WINDOW, tag="demo_v1_monsoon")
print(monitor.summary(r1))

print("=" * 70, "\nSTEP 3: retrain on all data before the window, monitor again\n" + "=" * 70)
train.train(end="2026-06-14")
r2 = monitor.monitor(*WINDOW, tag="demo_v2_after_retrain")
print(monitor.summary(r2))

print("=" * 70, "\nSTEP 4: final production model on all data\n" + "=" * 70)
train.train()
print(f"\nSummary: v1 live MAE {r1['live_mae']} ({r1['decision']})  ->  retrained live MAE {r2['live_mae']} ({r2['decision']})")
