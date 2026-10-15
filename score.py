"""Score prediction files against cases.jsonl.

Every decision becomes (predicted distribution over options, gold label index, optional gold distribution).
Metrics: accuracy, macro-F1, NLL, Brier, top-label ECE (10 bins), confidence AUROC (does the top
probability separate right from wrong), selective accuracy at 50% coverage, score MAE, KL from soft gold.
A failed call or missing answer counts as wrong, with a uniform distribution.

Box questions (image cases only) are scored by intersection over union with the gold box: accuracy is the share with
IoU >= 0.5, plus mean IoU, and calibration compares the answer's confidence with that hit. They stay out of "all".

API cost (usage.cost, USD) and token counts are summed per model when the predictions carry them.

The cases file must match data.lock.json (--unverified skips the check), and each run records the frozen hash it was
scored on as data_sha256.

Usage: python score.py preds_laya.jsonl preds_jev.jsonl ...  -> results.json + printed tables
       python score.py --merge preds_or_jev_114.jsonl            -> adds or replaces just these runs in results.json
       python score.py --cases image_cases.jsonl --out results_vision.json vision/preds_*.jsonl
"""
import argparse
import json
import os
from collections import defaultdict

import numpy as np
from sklearn.metrics import f1_score, roc_auc_score

import data_lock

cases = {}
BOX_HIT = 0.5


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
        top = ans.get("choice", ans.get("score"))
        d = np.array([1.0 if o == str(top) else 0.0 for o in k])
    if d.sum() <= 0:  # an answer we can't read counts as failed, like a missing one
        return np.full(len(k), 1 / len(k)), False
    return d / d.sum(), True


def gold_idx(q, g):
    k = options(q)
    if q["type"] == "choice":
        return k.index(g["label"])
    return int(g["label"])


def iou(a, b):
    w = max(0.0, min(a["right"], b["right"]) - max(a["left"], b["left"]))
    h = max(0.0, min(a["bottom"], b["bottom"]) - max(a["top"], b["top"]))
    union = (a["right"] - a["left"]) * (a["bottom"] - a["top"]) + (b["right"] - b["left"]) * (b["bottom"] - b["top"]) - w * h
    return w * h / union if union > 0 else 0.0


def box_row(c, q, ans, g):
    b = (ans or {}).get("box")
    ok = isinstance(b, dict) and all(isinstance(b.get(k), (int, float)) for k in ("top", "bottom", "left", "right"))
    v = iou(b, g["box"]) if ok else 0.0
    return {"suite": c["suite"], "type": "box", "ok": ok, "iou": v, "hit": v >= BOX_HIT,
            "conf": float(ans.get("confidence") or 0.0) if ok else 0.0}


def box_metrics(rows):
    hit = np.array([r["hit"] for r in rows], float)
    conf = np.array([r["conf"] for r in rows])
    out = {"n": len(rows), "failed": int(sum(not r["ok"] for r in rows)), "acc": hit.mean(),
           "mean_iou": float(np.mean([r["iou"] for r in rows])), "ece": ece(conf, hit),
           "mean_conf": float(conf.mean())}
    out["conf_auroc"] = roc_auc_score(hit, conf) if 0 < hit.mean() < 1 else float("nan")
    return {k: (round(float(v), 4) if isinstance(v, (float, np.floating)) else v) for k, v in out.items()}


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
            g = c["gold"][qid]
            if q["type"] == "box":
                rows.append(box_row(c, q, ans, g))
                continue
            p, ok = pred_dist(q, ans)
            rows.append({"suite": c["suite"], "workflow": c.get("workflow"), "type": q["type"], "qid": qid, "ok": ok, "p": p,
                         "y": gold_idx(q, g), "yhat": int(p.argmax()), "conf": float(p.max()),
                         "gd": np.array(g["dist"]) / sum(g["dist"]) if g.get("dist") and c["suite"] == "typed-decisions" else None})
    lat = [r["ms"] for r in preds.values() if r.get("ms") is not None and "error" not in r]
    usage = [r["usage"] for r in preds.values() if r.get("usage")]
    priced = [u["cost"] for u in usage if u.get("cost") is not None]  # the TypeSafe API reports tokens but no cost
    cost = {"usd": round(sum(priced), 4) if priced else None, "input_tokens": sum(u.get("input_tokens") or 0 for u in usage),
            "output_tokens": sum(u.get("output_tokens") or 0 for u in usage), "calls": len(usage)} if usage else None
    return rows, lat, cost


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("preds", nargs="+")
    ap.add_argument("--cases", default="cases.jsonl")
    ap.add_argument("--out", default="results.json")
    ap.add_argument("--merge", action="store_true", help="keep the runs already in --out and add or replace these")
    ap.add_argument("--unverified", action="store_true", help="score even if the cases don't match data.lock.json")
    a = ap.parse_args()
    data_sha = None if a.unverified else data_lock.verified_sha(a.cases)
    cases.update({c["id"]: c for c in (json.loads(l) for l in open(a.cases))})
    results = json.load(open(a.out)) if a.merge and os.path.exists(a.out) else {}
    for path in a.preds:
        name = os.path.basename(path).removeprefix("preds_").removesuffix(".jsonl")
        rows, lat, cost = load(path)
        boxes = [r for r in rows if r["type"] == "box"]
        rows = [r for r in rows if r["type"] != "box"]
        by = defaultdict(list)
        for r in rows:
            by[("suite", r["suite"])].append(r)
            by[("type", r["type"])].append(r)
            if r["workflow"]:
                by[("workflow", r["workflow"])].append(r)
        res = {f"{k}:{v}": metrics(rs) for (k, v), rs in sorted(by.items())}
        bys = defaultdict(list)
        for r in boxes:
            bys[r["suite"]].append(r)
        res.update({f"suite:{k}": box_metrics(rs) for k, rs in sorted(bys.items())})
        if boxes:
            res["type:box"] = box_metrics(boxes)
        res["all"] = metrics(rows) if rows else None
        res["cost"] = cost
        res["data_sha256"] = data_sha
        res["latency_ms"] = {"p50": round(float(np.percentile(lat, 50)), 1), "p95": round(float(np.percentile(lat, 95)), 1)} if lat else None
        results[name] = res
    json.dump(results, open(a.out, "w"), indent=1)
    keys = sorted({k for v in results.values() for k in v if k.startswith("suite:")})
    cols = ["acc", "macro_f1", "nll", "brier", "ece", "conf_auroc", "acc_at_50cov", "mean_iou", "failed"]
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
