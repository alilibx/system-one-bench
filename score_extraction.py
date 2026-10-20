"""Score extraction runs against extraction_cases.jsonl, into results_extraction.json.

Every field gets a score from 0 to 1, by type:
  text      token F1 with the best gold answer, after SQuAD's normalisation (lowercase, no punctuation or articles).
            When the passage doesn't hold the answer (SQuAD 2.0), only null scores, and null scores 0 when it does.
  number    right when it equals the gold value rounded to the gold's decimals (TAT-QA rounds to 2).
  date      right when it's the same day.
  list      F1 between the predicted and gold items, each pair matched once by token F1 (as DROP scores its spans).
  estimate  right when the median is within the gold's tolerance: 25% of a wine's price, 2 of its points.
  forecast  right when the forecast's mean absolute error beats repeating the last year (the seasonal-naive forecast).
A failed call or an answer that can't be read scores 0.

A suite's score is the mean over its fields, and the Extraction Index is the mean over suites, equal weight per suite,
0 to 100, with one sub-index for each of extract, compute, estimate and forecast. Alongside:
  calibration   ECE and AUROC of each answer's confidence against an exact hit (text, number, date and list)
  null          on SQuAD 2.0, how often an unanswerable question gets null, and the AUROC of p_null
  intervals     how often the truth falls inside the 80% interval (0.8 is calibrated), and for forecasts MASE, sMAPE and
                weighted quantile loss
API cost (usage.cost, USD) and token counts are summed per run when the predictions carry them.

Usage: python score_extraction.py extraction/preds_clairic.jsonl extraction/preds_llm_*.jsonl
       python score_extraction.py --merge extraction/preds_clairic.jsonl
"""
import argparse
import json
import math
import os
import re
import unicodedata
from collections import Counter, defaultdict

import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import roc_auc_score

import data_lock

GROUPS = {"extract": ["squad2", "piaf_fr", "arcd_ar", "sroie", "tatqa_list"], "compute": ["drop_number", "tatqa_arith"],
          "estimate": ["wine"], "forecast": ["tourism"]}
QUANTILES = {"p10": 0.1, "median": 0.5, "p90": 0.9}


def r4(x):
    return None if x is None or (isinstance(x, float) and math.isnan(x)) else round(float(x), 4)


# ---------- reading answers ----------
def num(x):
    if isinstance(x, bool):
        return None
    if isinstance(x, (int, float)):
        return float(x) if math.isfinite(x) else None
    if isinstance(x, str):
        m = re.search(r"-?\d[\d,]*\.?\d*|-?\.\d+", x)
        return float(m.group().replace(",", "")) if m else None
    return None


def value(ans):
    return ans.get("value") if isinstance(ans, dict) else ans


def item(x):
    return str(x.get("value", x.get("text", ""))) if isinstance(x, dict) else str(x)


def quantiles(ans):
    """{p10, median, p90} from an estimate or forecast answer: numbers for an estimate, lists for a forecast."""
    if not isinstance(ans, dict):
        return {"median": ans}
    iv = ans.get("interval") or {}
    q = {"median": ans.get("median", ans.get("value")), "p10": ans.get("p10"), "p90": ans.get("p90")}
    if isinstance(iv, dict):
        q["p10"], q["p90"] = q["p10"] if q["p10"] is not None else iv.get("p10"), q["p90"] if q["p90"] is not None else iv.get("p90")
    return q


# ---------- text ----------
def normalize(s):
    s = unicodedata.normalize("NFKC", str(s)).lower()
    s = "".join(" " if unicodedata.category(c)[0] in "PS" else c for c in s if unicodedata.category(c) != "Mn")
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    return s.split()


def token_f1(pred, gold):
    p, g = normalize(pred), normalize(gold)
    common = sum((Counter(p) & Counter(g)).values())
    if not p or not g:
        return float(p == g)
    if not common:
        return 0.0
    prec, rec = common / len(p), common / len(g)
    return 2 * prec * rec / (prec + rec)


def list_f1(pred, gold):
    if not pred or not gold:
        return float(not pred and not gold)
    m = np.array([[token_f1(p, g) for g in gold] for p in pred])
    rows, cols = linear_sum_assignment(-m)
    s = m[rows, cols].sum()
    prec, rec = s / len(pred), s / len(gold)
    return 0.0 if s == 0 else 2 * prec * rec / (prec + rec)


# ---------- one field ----------
def field_row(c, name, ans, g):
    t = g["type"]
    row = {"suite": c["suite"], "type": t, "field": name, "ok": ans is not None, "score": 0.0}
    if isinstance(ans, dict):
        row["conf"] = num(ans.get("confidence"))
        row["p_null"] = num(ans.get("p_null"))
    if t in ("estimate", "forecast"):
        return quant_row(row, t, ans, g)
    v = value(ans)
    if t == "text":
        row["gold_null"] = not g["answers"]
        row["pred_null"] = v is None or (isinstance(v, str) and not v.strip())
        if not row["ok"]:  # a failed call isn't a null answer
            row["exact"] = 0.0
        elif row["gold_null"] or row["pred_null"]:
            row["score"] = row["exact"] = float(row["gold_null"] and row["pred_null"])
        else:
            row["score"] = max(token_f1(v, a) for a in g["answers"])
            row["exact"] = float(any(normalize(v) == normalize(a) for a in g["answers"]))
    elif t == "number":
        p = num(v)
        row["ok"] &= p is not None
        d = g["decimals"]
        row["score"] = row["exact"] = float(p is not None and math.isclose(round(p, d), round(g["value"], d), abs_tol=1e-9))
    elif t == "date":
        m = re.match(r"\d{4}-\d{2}-\d{2}", v) if isinstance(v, str) else None
        row["ok"] &= m is not None
        row["score"] = row["exact"] = float(m is not None and m.group() == g["value"])
    elif t == "list":
        items = [item(x) for x in v] if isinstance(v, list) else None
        row["ok"] &= items is not None
        row["score"] = list_f1(items or [], g["value"])
        row["exact"] = float(row["score"] == 1.0)
    return row


def pinball(y, q, tau):
    return max(tau * (y - q), (tau - 1) * (y - q))


def quant_row(row, t, ans, g):
    q = quantiles(ans)
    y = g["value"]
    if t == "estimate":
        q = {k: num(v) for k, v in q.items()}
        m = q["median"]
        row["ok"] &= m is not None
        if m is None:
            return row
        w = g["within"]
        if "ratio" in w:
            row["score"] = float(m > 0 and max(m / y, y / m) <= w["ratio"])
            row["err"] = abs(math.log(max(m, 1e-9) / y))  # log error, for a field on a log scale
        else:
            row["score"] = float(abs(m - y) <= w["abs"])
            row["err"] = abs(m - y)
        if q["p10"] is not None and q["p90"] is not None:
            row["covered"] = float(q["p10"] <= y <= q["p90"])
        return row
    q = {k: [num(x) for x in v] if isinstance(v, list) else None for k, v in q.items()}
    m = q["median"]
    row["ok"] &= m is not None and len(m) == len(y) and None not in m
    if not row["ok"]:
        return row
    y, naive = np.array(y), np.array(g["naive"])
    mae = np.abs(np.array(m) - y).mean()
    row["score"] = float(mae < np.abs(naive - y).mean())
    row["mase"] = mae / g["scale"] if g["scale"] > 0 else None
    row["smape"] = float(np.mean([2 * abs(a - b) / (abs(a) + abs(b)) if abs(a) + abs(b) else 0.0 for a, b in zip(m, y)]))
    full = {k: v for k, v in q.items() if v is not None and len(v) == len(y) and None not in v}
    if {"p10", "p90"} <= set(full):
        row["covered"] = float(np.mean([lo <= v <= hi for lo, v, hi in zip(full["p10"], y, full["p90"])]))
        row["wql"] = float(np.mean([2 * sum(pinball(v, p, QUANTILES[k]) for v, p in zip(y, full[k])) / np.abs(y).sum()
                                    for k in QUANTILES]))
    return row


# ---------- metrics ----------
def ece(conf, hit, bins=10):
    conf, hit = np.asarray(conf), np.asarray(hit, float)
    e = 0.0
    for lo, hi in zip(np.linspace(0, 1, bins + 1)[:-1], np.linspace(0, 1, bins + 1)[1:]):
        m = (conf > lo) & (conf <= hi) if lo else (conf >= lo) & (conf <= hi)
        if m.any():
            e += m.mean() * abs(conf[m].mean() - hit[m].mean())
    return e


def auroc(y, s):
    return roc_auc_score(y, s) if 0 < np.mean(y) < 1 else None


def mean(rows, k):
    v = [r[k] for r in rows if r.get(k) is not None]
    return float(np.mean(v)) if v else None


def metrics(rows):
    out = {"n": len(rows), "failed": sum(not r["ok"] for r in rows), "score": mean(rows, "score"),
           "exact": mean(rows, "exact"), "coverage80": mean(rows, "covered"), "mase": mean(rows, "mase"),
           "smape": mean(rows, "smape"), "wql": mean(rows, "wql")}
    by_field = defaultdict(list)
    for r in rows:
        if r["type"] == "estimate":
            by_field[r["field"]].append(r)
    if by_field:
        out["fields"] = {f: {"score": r4(mean(rs, "score")), "coverage80": r4(mean(rs, "covered")),
                             "median_error": r4(float(np.median([r["err"] for r in rs if "err" in r])) if any("err" in r for r in rs) else None)}
                         for f, rs in by_field.items()}
    return {k: (r4(v) if isinstance(v, float) else v) for k, v in out.items() if v is not None}


def calibration(rows):
    rs = [r for r in rows if r.get("conf") is not None and "exact" in r and r["ok"]]
    if not rs:
        return None
    conf, hit = np.clip([r["conf"] for r in rs], 0, 1), [r["exact"] for r in rs]
    return {"n": len(rs), "mean_conf": r4(np.mean(conf)), "hit": r4(np.mean(hit)), "ece": r4(ece(conf, hit)),
            "conf_auroc": r4(auroc(hit, conf))}


def nulls(rows):
    rs = [r for r in rows if "gold_null" in r and r["suite"] == "squad2"]
    if not rs:
        return None
    un, ans = [r for r in rs if r["gold_null"]], [r for r in rs if not r["gold_null"]]
    scored = [r for r in rs if r.get("p_null") is not None]
    return {"unanswerable": len(un), "null_when_unanswerable": r4(mean(un, "exact")),
            "null_when_answerable": r4(float(np.mean([r["pred_null"] and r["ok"] for r in ans]))) if ans else None,
            "p_null_auroc": r4(auroc([r["gold_null"] for r in scored], [r["p_null"] for r in scored])) if scored else None}


def load(path, cases):
    preds = {}
    for line in open(path):
        r = json.loads(line)
        if r["id"] not in preds or "error" in preds[r["id"]]:
            preds[r["id"]] = r
    rows = []
    for cid, c in cases.items():
        if cid not in preds:  # not run (e.g. --suites); failures are logged as errors and score 0
            continue
        answers = preds[cid].get("answers") if "error" not in preds[cid] else None
        for name, g in c["gold"].items():
            rows.append(field_row(c, name, (answers or {}).get(name) if isinstance(answers, dict) else None, g))
    ok = [r for r in preds.values() if "error" not in r]
    lat = [r["ms"] for r in ok if r.get("ms") is not None]
    usage = [r["usage"] for r in ok if r.get("usage")]
    costs = [u["cost"] for u in usage if u.get("cost") is not None]
    cost = {"usd": round(sum(costs), 4) if costs else None,
            "input_tokens": sum(u.get("input_tokens", u.get("prompt_tokens")) or 0 for u in usage),
            "output_tokens": sum(u.get("output_tokens", u.get("completion_tokens")) or 0 for u in usage),
            "calls": len(usage)} if usage else None
    prompt = {r.get("prompt_sha256") for r in ok} - {None}
    return rows, lat, cost, (prompt.pop() if len(prompt) == 1 else None)


def score(rows, lat, cost, prompt_sha, data_sha):
    by = defaultdict(list)
    for r in rows:
        by[("suite", r["suite"])].append(r)
        by[("type", r["type"])].append(r)
    res = {f"{k}:{v}": metrics(rs) for (k, v), rs in sorted(by.items())}
    suite_score = {s: res[f"suite:{s}"]["score"] for s in sum(GROUPS.values(), []) if f"suite:{s}" in res}
    full = len(suite_score) == sum(map(len, GROUPS.values()))
    res["index"] = r4(100 * np.mean(list(suite_score.values()))) if full else None
    res["groups"] = {g: r4(100 * np.mean([suite_score[s] for s in ss])) for g, ss in GROUPS.items() if all(s in suite_score for s in ss)}
    res["calibration"] = calibration(rows)
    res["null"] = nulls(rows)
    res["cost"] = cost
    res["latency_ms"] = {"p50": round(float(np.percentile(lat, 50)), 1), "p95": round(float(np.percentile(lat, 95)), 1)} if lat else None
    res["data_sha256"] = data_sha
    if prompt_sha:
        res["prompt_sha256"] = prompt_sha
    return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("preds", nargs="+")
    ap.add_argument("--cases", default="extraction_cases.jsonl")
    ap.add_argument("--out", default="results_extraction.json")
    ap.add_argument("--merge", action="store_true", help="keep the runs already in --out and add or replace these")
    ap.add_argument("--unverified", action="store_true", help="score even if the cases don't match data.lock.json")
    a = ap.parse_args()
    data_sha = None if a.unverified else data_lock.verified_sha(a.cases)
    cases = {c["id"]: c for c in (json.loads(l) for l in open(a.cases))}
    results = json.load(open(a.out)) if a.merge and os.path.exists(a.out) else {}
    for path in a.preds:
        name = os.path.basename(path).removeprefix("preds_").removesuffix(".jsonl")
        results[name] = score(*load(path, cases), data_sha)
    with open(a.out, "w") as f:
        json.dump(results, f, indent=1, ensure_ascii=False)
        f.write("\n")
    suites = sum(GROUPS.values(), [])
    print("model".ljust(24) + "index".rjust(7) + "".join(g[:8].rjust(9) for g in GROUPS) + "".join(s[:11].rjust(12) for s in suites))
    for name, res in results.items():
        cell = lambda v: "" if v is None else f"{v:.1f}"
        print(name[:23].ljust(24) + cell(res["index"]).rjust(7) + "".join(cell(res["groups"].get(g)).rjust(9) for g in GROUPS)
              + "".join(cell(100 * res[f"suite:{s}"]["score"] if f"suite:{s}" in res else None).rjust(12) for s in suites))
    for name, res in results.items():
        print(f"\n{name}: calibration {res['calibration']}, null {res['null']}, cost {res['cost']}, latency {res['latency_ms']}")
