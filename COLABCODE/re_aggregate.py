import sys
import json
import numpy as np
from pathlib import Path

# Add COLABCODE to path
sys.path.insert(0, str(Path(r"C:\Users\Tran Duc Le\Documents\RESEARCHAGENTFINAL\projects\XGATE\RELATED_DATA\XGATE_Public\COLABCODE")))
from run_all_experiments import NpEncoder, bootstrap_ci

root = Path(r"C:\Users\Tran Duc Le\Documents\RESEARCHAGENTFINAL\projects\XGATE\ablation-results-colab2")
runs = []
for p in root.glob("run_*_seed*/ablation_metrics.json"):
    print(f"Loaded {p}")
    runs.append(json.loads(p.read_text(encoding="utf-8")))

if not runs:
    print("No runs found!")
    sys.exit(1)

ABLATION_CONFIGS = ["KD_only", "KD_ECD", "KD_EGAT", "Full_XGATE"]

aggregated = {}
for config_name in ABLATION_CONFIGS:
    aggregated[config_name] = {}
    for metric in ["f1_macro", "precision_macro", "recall_macro",
                   "roc_auc_macro", "fpr_macro", "accuracy",
                   "spearman_rho", "delta_L",
                   "adv_f1_macro", "adv_fpr_macro", "latency_ms"]:
        vals = [r[config_name][metric] for r in runs
                if config_name in r and metric in r[config_name]]
        if not vals:
            continue
        arr = np.array(vals, dtype=float)
        lo, hi = bootstrap_ci(arr) if len(arr) > 1 else (arr[0], arr[0])
        aggregated[config_name][metric] = {
            "values": arr.tolist(),
            "mean":   float(arr.mean()),
            "std":    float(arr.std(ddof=1) if len(arr) > 1 else 0.0),
            "ci95":   [lo, hi],
        }

stats_path = root / "ABLATION_STATS.json"
with open(stats_path, "w", encoding="utf-8") as f:
    json.dump(aggregated, f, indent=2, cls=NpEncoder)

sep = "-" * 80
header = f"{'Config':<16}{'F1-Macro':^16}{'Delta_L (Drift)':^16}{'Adv FPR':^16}{'Latency':^14}"
lines = [sep, header, sep]
for config_name, mdict in aggregated.items():
    f1 = mdict.get("f1_macro", {})
    dl = mdict.get("delta_L", {})
    afpr = mdict.get("adv_fpr_macro", {})
    lat = mdict.get("latency_ms", {})
    row = f"{config_name:<16}"
    row += f"{f1.get('mean', 0)*100:.2f}±{f1.get('std', 0)*100:.2f}%".center(16)
    row += f"{dl.get('mean', 0):.4f}±{dl.get('std', 0):.4f}".center(16)
    row += f"{afpr.get('mean', 0)*100:.2f}%".center(16)
    row += f"{lat.get('mean', 0):.2f}ms".center(14)
    lines.append(row)
lines.append(sep)
table = "\n".join(lines)

table_path = root / "ABLATION_TABLE.txt"
table_path.write_text(table, encoding="utf-8")

print(f"Aggregated {len(runs)} seeds successfully!")
print(table)
