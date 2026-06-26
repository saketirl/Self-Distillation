import wandb
import pandas as pd
from pathlib import Path

api = wandb.Api()

ENTITY = "saketirl"
PROJECT = "qwen3-continual-learning"

GROUPS = {
    "tooluse_stiefel":  ["pwc7enqf", "emnjwmbe", "r18uum80"],
    "spider_stiefel":   ["8u3mny6x", "kqnscm81", "114tsguh"],
    "cot_math_stiefel": ["u35afgtn", "buiimgt8", "6v6m5jyv"],
    "tooluse_adam":     ["ambhqs6x", "4aeefy9e", "qlan0m41"],
    "spider_adam":      ["t9o7sc1o", "qbtwoui8", "q311u5a7"],
    "cot_math_adam":    ["xx722xio", "08h7v0ad", "sn798vp5"],
}

out_dir = Path("outputs/wandb_csvs")
out_dir.mkdir(parents=True, exist_ok=True)

for tag, run_ids in GROUPS.items():
    all_history = []
    all_summary = []

    for run_id in run_ids:
        try:
            run = api.run(f"{ENTITY}/{PROJECT}/{run_id}")
            print(f"  [{tag}] {run_id}: {run.name} — fetching all rows...", flush=True)

            rows = list(run.scan_history(keys=["train/loss", "_step"]))
            hist = pd.DataFrame(rows)
            hist = hist.dropna(subset=["train/loss"])
            hist["run_id"] = run_id
            hist["run_name"] = run.name
            print(f"    -> {len(hist)} optimizer steps", flush=True)
            all_history.append(hist)

            summary = dict(run.summary)
            summary["run_id"] = run_id
            summary["run_name"] = run.name
            all_summary.append(summary)

        except Exception as e:
            print(f"  [{tag}] {run_id}: ERROR - {e}", flush=True)

    if all_history:
        pd.concat(all_history, ignore_index=True).to_csv(out_dir / f"{tag}_history.csv", index=False)
    if all_summary:
        pd.DataFrame(all_summary).to_csv(out_dir / f"{tag}_summary.csv", index=False)
    print(f"  Saved: {tag}_history.csv + {tag}_summary.csv\n", flush=True)

print("Done.")
