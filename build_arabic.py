"""Add Arabic suites to cases.jsonl (idempotent: replaces any ar_* suites already there).

Suites, and whether each is in laya-ara / laya-ara-rag training (test splits are held out, but the
task family is seen):
  ar_massive_scenario  MASSIVE ar-SA test, 18-way scenario choice          laya-ara: in-mix
  ar_massive_intent    MASSIVE ar-SA test, 20 options (gold + 19 random)   laya-ara: in-mix (their "20-option" protocol)
  ar_offensive         OSACT4 subtask A validation, noul                   laya-ara: in-mix
  ar_sentiment         Arabic tweet sentiment test, 3-way choice           zero-shot for every model
  ar_miracl_rerank     MIRACL-ar dev, pick the relevant passage of 8       laya-ara-rag: in-family
  ar_miracl_pair       MIRACL-ar dev, is this passage relevant? noul       laya-ara-rag: in-family
xnli_ar (already built by build_suites.py) is also reported in the Arabic table.
"""
import json
import random
from collections import defaultdict

from datasets import load_dataset

N = 200
SEED = 0
K_RERANK = 8
PASSAGE_CHARS = 320
out = []

# Every source is pinned to the commit it was frozen from, so an upstream edit can't change the cases.
# data.lock.json holds the hashes of the result; check with data_lock.py.
REVISIONS = {
    "SetFit/amazon_massive_scenario_ar-SA": "d82b73bba2083361552d155f31d25d3c8308adfd",
    "SetFit/amazon_massive_intent_ar-SA": "88c4768ff44705c6aae8470250a67ae91de09af7",
    "arbml/OSACT4_hatespeech": "2fe9f6399ba7f3b5228660d8a3d0eec3ef4197a1",
    "mteb/tweet_sentiment_multilingual": "bceb37eb62681b42e4147d879384917969bb1674",
    "mteb/MIRACLReranking": "ab6f54eff185a84bc1f6ab96b56bc7df87433228",
}


def load(name, *args, **kw):
    return load_dataset(name, *args, revision=REVISIONS[name], **kw)


def sample(ds, n=N, seed=SEED):
    idx = list(range(len(ds)))
    random.Random(seed).shuffle(idx)
    return [ds[i] for i in idx[:n]]


def add(suite, i, state, qid, q, label):
    out.append({"suite": suite, "id": f"{suite}-{i}", "state": state, "questions": {qid: q},
                "gold": {qid: {"type": q["type"], "label": label}}})


def pretty(name):
    return name.replace("_", " ").replace("iot", "smart home (IoT)").replace("qa", "question answering")


# MASSIVE scenario, 18-way
sc = load("SetFit/amazon_massive_scenario_ar-SA", split="test")
scen = sorted({r["label_text"] for r in sc})
SC = {s: pretty(s) for s in scen}
for i, r in enumerate(sample(sc)):
    add("ar_massive_scenario", i, {"utterance": r["text"]}, "scenario",
        {"type": "choice", "instructions": "Which scenario is the user's `utterance` (Arabic) about?", "criteria": SC},
        r["label_text"])

# MASSIVE intent, 20 options: the gold intent plus 19 distractors drawn per case
it = load("SetFit/amazon_massive_intent_ar-SA", split="test")
intents = sorted({r["label_text"] for r in it})
rng = random.Random(SEED + 1)
for i, r in enumerate(sample(it)):
    opts = rng.sample([x for x in intents if x != r["label_text"]], 19) + [r["label_text"]]
    rng.shuffle(opts)
    add("ar_massive_intent", i, {"utterance": r["text"]}, "intent",
        {"type": "choice", "instructions": "Which intent best matches the user's `utterance` (Arabic)?",
         "criteria": {o: pretty(o) for o in opts}}, r["label_text"])

# OSACT4 subtask A, offensive language
os4 = load("arbml/OSACT4_hatespeech", split="validation")
for i, r in enumerate(sample(os4)):
    add("ar_offensive", i, {"tweet": r["tweet"]}, "offensive",
        {"type": "noul", "instructions": "Is `tweet` (Arabic) offensive: does it contain insults, profanity, or attacks?"},
        int(r["offensive"] in ("OFF", 1, "1", True)))

# Arabic tweet sentiment, 3-way
ts = load("mteb/tweet_sentiment_multilingual", "arabic", split="test")
names = ["negative", "neutral", "positive"]  # cardiffnlp label ids, stored as strings "0"/"1"/"2"
SE = {"negative": "the writer expresses a negative opinion or feeling",
      "neutral": "no clear positive or negative opinion",
      "positive": "the writer expresses a positive opinion or feeling"}
for i, r in enumerate(sample(ts)):
    add("ar_sentiment", i, {"tweet": r["text"]}, "sentiment",
        {"type": "choice", "instructions": "What is the sentiment of `tweet` (Arabic)?", "criteria": SE},
        names[int(r["label"])])

# MIRACL-ar dev: listwise rerank and pairwise relevance
queries = {r["_id"]: r["text"] for r in load("mteb/MIRACLReranking", "ar-queries", split="dev")}
qrels = defaultdict(dict)
for r in load("mteb/MIRACLReranking", "ar-qrels", split="dev"):
    qrels[r["query-id"]][r["corpus-id"]] = r["score"]
corpus = load("mteb/MIRACLReranking", "ar-corpus", split="dev")
text = {r["_id"]: (r["title"] + ": " if r["title"] else "") + r["text"] for r in corpus}


def clip(s):
    return s if len(s) <= PASSAGE_CHARS else s[:PASSAGE_CHARS].rsplit(" ", 1)[0] + " …"


usable = [q for q in queries if any(v > 0 for v in qrels[q].values()) and sum(v == 0 for v in qrels[q].values()) >= K_RERANK - 1]
rng = random.Random(SEED + 2)
rng.shuffle(usable)
for i, qid in enumerate(usable[:N]):
    pos = rng.choice([d for d, v in qrels[qid].items() if v > 0])
    negs = rng.sample([d for d, v in qrels[qid].items() if v == 0], K_RERANK - 1)
    docs = negs + [pos]
    rng.shuffle(docs)
    keys = [chr(ord("a") + j) for j in range(len(docs))]
    add("ar_miracl_rerank", i, {"query": queries[qid]}, "passage",
        {"type": "choice", "instructions": "Which passage best answers `query` (Arabic)?",
         "criteria": {k: clip(text[d]) for k, d in zip(keys, docs)}}, keys[docs.index(pos)])

for i, qid in enumerate(usable[N:2 * N]):  # different queries from the rerank suite; half relevant, half not
    rel = i % 2 == 0
    doc = rng.choice([d for d, v in qrels[qid].items() if (v > 0) == rel])
    add("ar_miracl_pair", i, {"query": queries[qid], "passage": clip(text[doc])}, "relevant",
        {"type": "noul", "instructions": "Does `passage` contain the answer to `query`?"}, int(rel))

existing = [json.loads(l) for l in open("cases.jsonl")]
kept = [c for c in existing if not c["suite"].startswith("ar_")]
with open("cases.jsonl", "w") as f:
    for c in kept + out:
        f.write(json.dumps(c, ensure_ascii=False) + "\n")
from collections import Counter
print(len(kept), "existing +", len(out), "Arabic cases", Counter(c["suite"] for c in out))
