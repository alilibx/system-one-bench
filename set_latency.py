"""Replace the latency in results.json with one measured in a separate re-run, keeping every other number.

Latency depends on the network between the caller and the API, so comparing models means timing them all from one
machine. Re-run each model with run_decisions.py into its own folder, then point this at the prediction files:
the name after "preds_" picks the results.json entry, and only its latency_ms changes.

Usage: python set_latency.py lat/preds_or_jev.jsonl lat/preds_luna.jsonl ...
"""
import json
import os
import sys

import numpy as np

results = json.load(open("results.json"))
for path in sys.argv[1:]:
    name = os.path.basename(path).removeprefix("preds_").removesuffix(".jsonl")
    if name not in results:
        sys.exit(f"{name} is not in results.json")
    lat = [r["ms"] for r in map(json.loads, open(path)) if r.get("ms") is not None and "error" not in r]
    results[name]["latency_ms"] = {"p50": round(float(np.percentile(lat, 50)), 1), "p95": round(float(np.percentile(lat, 95)), 1)}
    print(f"{name:14} p50 {results[name]['latency_ms']['p50']:7.1f} ms  p95 {results[name]['latency_ms']['p95']:8.1f} ms  ({len(lat)} calls)")
json.dump(results, open("results.json", "w"), indent=1)
