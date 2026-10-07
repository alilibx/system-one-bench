"""Score prediction files against cases.jsonl.

Every decision becomes (predicted distribution over options, gold label index, optional gold distribution).
Metrics: accuracy, macro-F1, NLL, Brier, top-label ECE (10 bins), confidence AUROC (does the top
probability separate right from wrong), selective accuracy at 50% coverage, score MAE, KL from soft gold.
A failed call or missing answer counts as wrong, with a uniform distribution.

API cost (usage.cost, USD) and token counts are summed per model when the predictions carry them.

Usage: python score.py preds_laya.jsonl preds_jev.jsonl ...  -> results.json + printed tables
"""
import json
import sys
from collections import defaultdict

import numpy as np
from sklearn.metrics import f1_score, roc_auc_score

cases = {c["id"]: c for c in (json.loads(l) for l in open("cases.jsonl"))}


def options(q):
    if q["type"] == "noul":
        return ["false", "true"]
    if q["type"] == "score":
        return [str(i) for i in range(len(q["criteria"]))]
    c = q["criteria"]
    return list(c.keys()) if isinstance(c, dict) else list(c)


def pred_dist(q, ans):
    k = options(q)
    if ans is None:
        return np.full(len(k), 1 / len(k)), False
    if q["type"] == "noul":
        p = float(ans["noul"])
        return np.array([1 - p, p]), True
    probs = ans.get("probabilities") or {}
    d = np.array([float(probs.get(o, 0.0)) for o in k])
    if d.sum() <= 0:  # fall back to the argmax when no distribution came back
        d = np.array([1.0 if o == str(ans.get("choice")) else 0.0 for o in k])
    return d / d.sum(), True


def gold_idx(q, g):
    k = options(q)
    if q["type"] == "choice":
        return k.index(g["label"])
    return int(g["label"])


def ece(conf, correct, bins=10):
    conf, correct = np.asarray(conf), np.asarray(correct, float)
    e = 0.0
    edges = np.linspace(0, 1, bins + 1)
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            e += m.mean() * abs(conf[m].mean() - correct[m].mean())
    return e


def metrics(rows):
    y = np.array([r["y"] for r in rows])
    yhat = np.array([r["yhat"] for r in rows])
    conf = np.array([r["conf"] for r in rows])
    correct = (y == yhat).astype(float)
    out = {"n": len(rows), "failed": int(sum(not r["ok"] for r in rows)),
           "acc": correct.mean(),
           "macro_f1": f1_score(y, yhat, average="macro"),
           "nll": float(np.mean([-np.log(max(r["p"][r["y"]], 1e-6)) for r in rows])),
           "brier": float(np.mean([((r["p"] - np.eye(len(r["p"]))[r["y"]]) ** 2).sum() for r in rows])),
           "ece": ece(conf, correct),
           "p_true_zero": float(np.mean([r["p"][r["y"]] < 1e-4 for r in rows]))}
    out["conf_auroc"] = roc_auc_score(correct, conf) if 0 < correct.mean() < 1 else float("nan")
    order = np.argsort(-conf)
    out["acc_at_50cov"] = correct[order[: max(1, len(order) // 2)]].mean()
    sc = [r for r in rows if r["type"] == "score"]
    if sc:
        out["score_mae"] = float(np.mean([abs((np.arange(len(r["p"])) * r["p"]).sum() - r["y"]) for r in sc]))
    soft = [r for r in rows if r.get("gd") is not None]
    if soft:
        out["kl_gold"] = float(np.mean([(r["gd"] * (np.log(np.clip(r["gd"], 1e-9, 1)) - np.log(np.clip(r["p"], 1e-6, 1)))).sum()
                                        for r in soft]))
        out["soft_acc"] = float(np.mean([r["gd"][r["yhat"]] for r in soft]))
    return {k: (round(float(v), 4) if isinstance(v, (float, np.floating)) else v) for k, v in out.items()}


def load(path):
    preds = {}
    for l in open(path):
        r = json.loads(l)
        if r["id"] not in preds or "error" in preds[r["id"]]:
            preds[r["id"]] = r
    rows = []
    for cid, c in cases.items():
        if cid not in preds:  # not run by this model (e.g. an Arabic-only checkpoint); failures are logged as errors
            continue
        pr = preds[cid]
        for qid, q in c["questions"].items():
            ans = (pr.get("answers") or {}).get(qid)
            p, ok = pred_dist(q, ans)
            g = c["gold"][qid]
            rows.append({"suite": c["suite"], "workflow": c.get("workflow"), "type": q["type"], "qid": qid, "ok": ok, "p": p,
                         "y": gold_idx(q, g), "yhat": int(p.argmax()), "conf": float(p.max()),
                         "gd": np.array(g["dist"]) / sum(g["dist"]) if g.get("dist") and c["suite"] == "typed-decisions" else None})
    lat = [r["ms"] for r in preds.values() if r.get("ms") is not None and "error" not in r]
    usage = [r["usage"] for r in preds.values() if r.get("usage")]
    cost = {"usd": round(sum(u.get("cost") or 0 for u in usage), 4), "input_tokens": sum(u.get("input_tokens") or 0 for u in usage),
            "output_tokens": sum(u.get("output_tokens") or 0 for u in usage), "calls": len(usage)} if usage else None
    return rows, lat, cost


if __name__ == "__main__":
    results = {}
    for path in sys.argv[1:]:
        name = path.removeprefix("preds_").removesuffix(".jsonl")
        rows, lat, cost = load(path)
        by = defaultdict(list)
        for r in rows:
            by[("suite", r["suite"])].append(r)
            by[("type", r["type"])].append(r)
            if r["workflow"]:
                by[("workflow", r["workflow"])].append(r)
        results[name] = {f"{k}:{v}": metrics(rs) for (k, v), rs in sorted(by.items())}
        results[name]["all"] = metrics(rows)
        results[name]["cost"] = cost
        results[name]["latency_ms"] = {"p50": round(float(np.percentile(lat, 50)), 1), "p95": round(float(np.percentile(lat, 95)), 1)} if lat else None
    json.dump(results, open("results.json", "w"), indent=1)
    keys = sorted({k for v in results.values() for k in v if k.startswith("suite:")})
    cols = ["acc", "macro_f1", "nll", "brier", "ece", "conf_auroc", "acc_at_50cov", "failed"]
    for k in keys + ["all"]:
        print(f"\n## {k}")
        print("model".ljust(28) + "".join(c.rjust(13) for c in cols))
        for name, res in results.items():
            m = res.get(k)
            if m:
                print(name.ljust(28) + "".join(str(m.get(c, "")).rjust(13) for c in cols))
    print("\n## cost and latency")
    for name, res in results.items():
        print(name.ljust(28), res["cost"], res["latency_ms"])
