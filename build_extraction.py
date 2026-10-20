"""Build the extraction track. Every case is one extraction request body plus gold values.

Extraction models (Cloudflare's Clairic, https://blog.cloudflare.com/clairic-extraction-model) don't answer questions
from a list of options like decision models do. They read a `state` and return one typed value per named field: text
spans, numbers, dates and lists pulled from the input, estimates with an 80% interval, and forecasts. So they get their
own track, with their own cases, scored against plain LLMs prompted for the same fields.

Output: extraction_cases.jsonl, one line per case:
  {suite, id, lang, state: "<text>", fields: {name: {"type", "instructions", ...}}, gold: {name: {"type", ...}}}
Identical {state, fields} bodies are sent to every extraction model; LLMs get the same two rendered into one fixed
prompt (run_extraction.py).

Suites, 200 cases each, grouped by what Clairic says it does:
  extract   squad2 (English, half the answers aren't in the passage), piaf_fr, arcd_ar, sroie (receipts), tatqa_list
  compute   drop_number, tatqa_arith
  estimate  wine (price and points from a review)
  forecast  tourism (the next 24 months of a monthly series)
"""
import json
import random
import re
from datetime import date

from datasets import load_dataset

N = 200
SEED = 0
out = []

# Every source is pinned to the commit it was frozen from; data.lock.json holds the hashes of the result.
REVISIONS = {
    "rajpurkar/squad_v2": "3ffb306f725f7d2ce8394bc1873b24868140c412",
    "AgentPublic/piaf": "bda8c063bc7297180796cd835d1974c0bc71c521",
    "hsseinmz/arcd": "cc6906b6eda547e4ffc63b8d88ccca7e0515187a",
    "jsdnrs/ICDAR2019-SROIE": "bffe40c26759f3376ec2b3ae9031dbba54cd587c",
    "next-tat/TAT-QA": "c96247f5077eac447f63527fd3dcfdc58bb56d6a",
    "ucinlp/drop": "95cda593fae71b60b5b19f82de3fcf3298c1239c",
    "james-burton/wine_reviews": "35c2acf5531872b556e960597f92eba7beccce2a",
    "autogluon/chronos_datasets": "eeecad0b82a8c237e212ce6f8d1abecb513e2cec",
}


def load(name, *args, **kw):
    return load_dataset(name, *args, revision=REVISIONS[name], **kw)


def sample(rows, n=N):
    rows = list(rows)
    idx = list(range(len(rows)))
    random.Random(SEED).shuffle(idx)
    return [rows[i] for i in idx[:n]]


def add(suite, i, lang, state, fields, gold):
    out.append({"suite": suite, "id": f"{suite}-{i}", "lang": lang, "state": state, "fields": fields, "gold": gold})


def unique(xs):
    return list(dict.fromkeys(x.strip() for x in xs if x.strip()))


def decimals(x):
    s = repr(float(x))
    return 0 if s.endswith(".0") else len(s.split(".")[1])


# 1-3. Reading comprehension as a text field. SQuAD 2.0 keeps its unanswerable questions: the right answer is null.
QA = [("squad2", "en", "rajpurkar/squad_v2", "validation"), ("piaf_fr", "fr", "AgentPublic/piaf", "train"),
      ("arcd_ar", "ar", "hsseinmz/arcd", "validation")]
for suite, lang, name, split in QA:
    for i, r in enumerate(sample(load(name, split=split))):
        add(suite, i, lang, r["context"], {"answer": {"type": "text", "instructions": r["question"].strip()}},
            {"answer": {"type": "text", "answers": unique(r["answers"]["text"])}})

# 4. SROIE receipts: the OCR text of a scanned receipt, one line per text box, and three fields to fill in.
MONTHS = {m: i + 1 for i, m in enumerate(["JAN", "FEB", "MAR", "APR", "MAY", "JUN", "JUL", "AUG", "SEP", "OCT", "NOV", "DEC"])}


def receipt_date(s):
    """SROIE dates are day first (Malaysian receipts): 15/01/2019, 15-01-19, 22 MAR 18, 27/MAR/2018, 2018-03-07."""
    p = re.findall(r"\d+|[A-Za-z]+", s)
    if len(p[0]) == 4:
        y, m, d = p
    else:
        d, m, y = p
    m = MONTHS[m.upper()[:3]] if m.isalpha() else int(m)
    y = int(y) + 2000 if int(y) < 100 else int(y)
    d, m = (m, int(d)) if m > 12 else (int(d), m)  # one receipt writes the month first
    return date(y, m, d).isoformat()


SROIE = {"company": {"type": "text", "instructions": "The name of the company that issued the receipt."},
         "date": {"type": "date", "instructions": "The date of the receipt."},
         "total": {"type": "number", "instructions": "The total amount paid, including tax."}}
# Totals are written 193.00, RM 10.60 or $8.20; the two labelled negative are left out.
receipts = [r for r in load("jsdnrs/ICDAR2019-SROIE", split="test").remove_columns("image")
            if not r["entities"]["total"].startswith("-")]
for i, r in enumerate(sample(receipts)):
    e = r["entities"]
    add("sroie", i, "en", "\n".join(r["words"]), SROIE,
        {"company": {"type": "text", "answers": [e["company"]]},
         "date": {"type": "date", "value": receipt_date(e["date"])},
         "total": {"type": "number", "value": float(re.search(r"[\d.]+", e["total"].replace(",", "")).group()), "decimals": 2}})

# 5-6. TAT-QA: a financial report's table and paragraphs. Arithmetic answers are a number field, multi-span ones a list.
SCALE = {"": "", "percent": " Give it as a percentage, e.g. 12.5 for 12.5%.", "thousand": " Give it in thousands.",
         "million": " Give it in millions.", "billion": " Give it in billions."}


def report(d):
    rows = ["| " + " | ".join(c.strip() for c in row) + " |" for row in d["table"]["table"]]
    paras = [p["text"] for p in sorted(d["paragraphs"], key=lambda p: p["order"])]
    return "\n".join(rows) + "\n\n" + "\n\n".join(paras)


tat = load("next-tat/TAT-QA", split="validation")
arith = [(d, q) for d in tat for q in d["questions"] if q["answer_type"] == "arithmetic"]
for i, (d, q) in enumerate(sample(arith)):
    add("tatqa_arith", i, "en", report(d),
        {"answer": {"type": "number", "instructions": q["question"].strip() + SCALE[q["scale"]]}},
        {"answer": {"type": "number", "value": float(q["answer"]), "decimals": decimals(q["answer"])}})
multi = [(d, q) for d in tat for q in d["questions"] if q["answer_type"] == "multi-span"]
for i, (d, q) in enumerate(sample(multi)):
    add("tatqa_list", i, "en", report(d), {"answer": {"type": "list", "instructions": q["question"].strip()}},
        {"answer": {"type": "list", "value": [str(a) for a in q["answer"]]}})

# 7. DROP: counting and arithmetic over a passage. Only questions every annotator answered with the same number.
drop = []
for r in load("ucinlp/drop", split="validation"):
    a = r["answers_spans"]
    if set(a["types"]) == {"number"} and len(set(a["spans"])) == 1 and re.fullmatch(r"-?[\d,]*\.?\d+", a["spans"][0]):
        drop.append(r)
for i, r in enumerate(sample(drop)):
    v = r["answers_spans"]["spans"][0].replace(",", "")
    add("drop_number", i, "en", r["passage"].strip(), {"answer": {"type": "number", "instructions": r["question"].strip()}},
        {"answer": {"type": "number", "value": float(v), "decimals": decimals(v)}})

# 8. Wine reviews: estimate the price and the critic's points from the tasting note. Neither is stated in the input.
wine = load("james-burton/wine_reviews", split="test")
variety = wine.features["variety"].names
WINE = {"price": {"type": "estimate", "instructions": "The bottle's price in US dollars.", "min": 4, "max": 2000,
                  "scale": "log", "unit": "USD"},
        "points": {"type": "estimate", "instructions": "The critic's score for this wine, from 80 to 100.", "min": 80,
                   "max": 100, "step": 1}}
for i, r in enumerate(sample(r for r in wine if r["price"] is not None)):
    state = f"Wine: {variety[r['variety']]}\nCountry: {r['country']}\nRegion: {r['province']}\nTasting note: {r['description']}"
    add("wine", i, "en", state, WINE,
        {"price": {"type": "estimate", "value": r["price"], "within": {"ratio": 1.25}},
         "points": {"type": "estimate", "value": r["points"], "within": {"abs": 2}}})

# 9. Monthly tourism series (Monash): forecast the 24 months after the last 8 years shown. Scored against repeating
# the last year (the seasonal-naive forecast), so the gold carries that forecast and the in-sample error that scales it.
H, HIST, SEASON = 24, 96, 12
tour = load("autogluon/chronos_datasets", "monash_tourism_monthly", split="train")
for i, r in enumerate(sample(tour)):
    ts, y = r["timestamp"][-H - HIST:], [round(v, 2) for v in r["target"][-H - HIST:]]
    hist, fut = y[:-H], y[-H:]
    lines = "\n".join(f"{t:%Y-%m}: {v:g}" for t, v in zip(ts[:-H], hist))
    naive = [hist[len(hist) - SEASON + k % SEASON] for k in range(H)]
    scale = sum(abs(hist[k] - hist[k - SEASON]) for k in range(SEASON, len(hist))) / (len(hist) - SEASON)
    add("tourism", i, "", f"Monthly tourism numbers, one line per month:\n{lines}",
        {"forecast": {"type": "forecast", "instructions": f"The next {H} monthly values.", "steps": H}},
        {"forecast": {"type": "forecast", "value": fut, "naive": naive, "scale": round(scale, 6)}})

with open("extraction_cases.jsonl", "w") as f:
    for c in out:
        f.write(json.dumps(c, ensure_ascii=False) + "\n")
from collections import Counter
print(len(out), "cases", Counter(c["suite"] for c in out))
