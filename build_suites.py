"""Build benchmark cases. Every case is one POST /v1/systemone body plus gold labels.

Output: cases.jsonl, one line per case:
  {suite, id, state, questions, gold: {qid: {"type", "label", "dist"?}}}
Identical bodies are sent to every model.
"""
import json
import random

from datasets import load_dataset

N = 200
SEED = 0
out = []


def sample(ds, n=N):
    idx = list(range(len(ds)))
    random.Random(SEED).shuffle(idx)
    return [ds[i] for i in idx[:n]]


def add(suite, i, state, qid, q, label, dist=None):
    g = {"type": q["type"], "label": label}
    if dist is not None:
        g["dist"] = dist
    out.append({"suite": suite, "id": f"{suite}-{i}", "state": state, "questions": {qid: q}, "gold": {qid: g}})


# 1. typed-decisions (multi-question, soft gold). Full test split, 400 cases x 5 questions.
td = load_dataset("LocalLLaMA/typed-decisions", "all", split="test")
for r in td:
    qs = json.loads(r["questions"])
    gold = json.loads(r["gold"])
    g = {}
    for qid, q in qs.items():
        ga = gold[qid]
        if q["type"] == "noul":
            p = float(ga["noul"])
            g[qid] = {"type": "noul", "label": int(p >= 0.5), "dist": [1 - p, p]}
        elif q["type"] == "choice":
            keys = list(q["criteria"].keys()) if isinstance(q["criteria"], dict) else list(q["criteria"])
            probs = ga["probabilities"]
            d = [float(probs.get(k, 0.0)) for k in keys]
            g[qid] = {"type": "choice", "label": keys[max(range(len(d)), key=d.__getitem__)], "dist": d, "keys": keys}
        else:
            probs = ga["probabilities"]
            d = [float(probs.get(str(i), 0.0)) for i in range(len(q["criteria"]))]
            g[qid] = {"type": "score", "label": max(range(len(d)), key=d.__getitem__), "dist": d}
    out.append({"suite": "typed-decisions", "workflow": r["workflow"], "id": r["id"],
                "state": json.loads(r["state"]), "questions": qs, "gold": g})

# 2. AG News, 4-way topic choice
ag = load_dataset("fancyzhx/ag_news", split="test")
AG = {"world": "world news, politics, international affairs",
      "sports": "sports, athletes, games, competitions",
      "business": "business, economy, companies, markets, finance",
      "sci_tech": "science and technology, computing, internet, research"}
agk = list(AG)
for i, r in enumerate(sample(ag)):
    add("ag_news", i, {"article": r["text"]}, "topic",
        {"type": "choice", "instructions": "What is the topic of `article`?", "criteria": AG}, agk[r["label"]])

# 3. DAIR emotion, 6-way choice
em = load_dataset("dair-ai/emotion", "split", split="test")
EM = {"sadness": "the writer feels sad, down, hurt, or lonely",
      "joy": "the writer feels happy, pleased, content, or excited",
      "love": "the writer feels love, affection, tenderness, or longing for someone",
      "anger": "the writer feels angry, irritated, resentful, or hostile",
      "fear": "the writer feels afraid, anxious, nervous, or threatened",
      "surprise": "the writer feels surprised, amazed, or astonished"}
emk = list(EM)
for i, r in enumerate(sample(em)):
    add("emotion", i, {"text": r["text"]}, "emotion",
        {"type": "choice", "instructions": "Which emotion does the writer of `text` express most strongly?", "criteria": EM},
        emk[r["label"]])

# 4. Banking77, 77-way intent choice (high cardinality)
bk = load_dataset("mteb/banking77", split="test")
bnames = sorted({r["label_text"] for r in bk})
BK = {n: n.replace("_", " ") for n in bnames}
for i, r in enumerate(sample(bk)):
    add("banking77", i, {"customer_message": r["text"]}, "intent",
        {"type": "choice", "instructions": "Which banking intent best matches `customer_message`?", "criteria": BK},
        r["label_text"])

# 5. SST-5, 5-level sentiment score
s5 = load_dataset("SetFit/sst5", split="test")
S5 = ["very negative review", "negative review", "neutral or mixed review", "positive review", "very positive review"]
for i, r in enumerate(sample(s5)):
    add("sst5", i, {"review": r["text"]}, "sentiment",
        {"type": "score", "instructions": "How positive is the sentiment of `review`?", "criteria": S5}, int(r["label"]))

# 6. SST-2, noul
s2 = load_dataset("stanfordnlp/sst2", split="validation")
for i, r in enumerate(sample(s2)):
    add("sst2", i, {"review": r["sentence"]}, "positive",
        {"type": "noul", "instructions": "Does `review` express a positive opinion of the movie?"}, int(r["label"]))

# 7. BoolQ, reading-comprehension noul
bq = load_dataset("google/boolq", split="validation")
for i, r in enumerate(sample(bq)):
    add("boolq", i, {"passage": r["passage"], "question": r["question"] + "?"}, "answer_yes",
        {"type": "noul", "instructions": "According to `passage`, is the answer to `question` yes?"}, int(r["answer"]))

# 8. XNLI in English, Arabic, French: 3-way NLI choice
NLI = {"entailment": "the hypothesis is definitely true given the premise",
       "neutral": "the hypothesis might be true or false; the premise does not settle it",
       "contradiction": "the hypothesis is definitely false given the premise"}
nk = list(NLI)
for lang in ["en", "ar", "fr"]:
    xn = load_dataset("facebook/xnli", lang, split="test")
    for i, r in enumerate(sample(xn, 150)):
        add(f"xnli_{lang}", i, {"premise": r["premise"], "hypothesis": r["hypothesis"]}, "relation",
            {"type": "choice", "instructions": "How does `hypothesis` relate to `premise`?", "criteria": NLI}, nk[r["label"]])

with open("cases.jsonl", "w") as f:
    for c in out:
        f.write(json.dumps(c, ensure_ascii=False) + "\n")
from collections import Counter
print(len(out), "cases", Counter(c["suite"] for c in out))
