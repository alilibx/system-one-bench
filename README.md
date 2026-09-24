# Laya vs TypeSafe Jev

Head-to-head benchmark of [convaiinnovations/laya](https://huggingface.co/convaiinnovations/laya) (open weights, run locally)
against TypeSafe Jev (`jev-1.13.0`, API). Both models get byte-identical `state` + `questions` bodies.

## Live race

```bash
OMP_NUM_THREADS=1 HF_HUB_OFFLINE=1 TYPESAFE_API_KEY=<your key> .venv/bin/python server.py
```

Then open http://localhost:8061, pick a dataset and a challenger (Laya Router, laya-ara, or laya-ara-rag), and press
**Start race**. `OMP_NUM_THREADS=1` avoids a libomp deadlock when several checkpoints load in one process. Laya runs on the Apple GPU (MPS).
Jev is called from the server, so the key never reaches the browser. Get a key at [typesafe.ai](https://typesafe.ai); keep it in your shell or a secrets manager, never in this repo. "Jev parallel 1" is the fair setting:
one request in flight per model.

## Full benchmark

```bash
.venv/bin/python build_suites.py                  # -> cases.jsonl (2,050 cases, 3,650 decisions)
.venv/bin/python build_arabic.py                  # appends 1,200 Arabic cases
.venv/bin/python run_laya.py --variant ara        # Wouze/laya-ara, Arabic suites only
.venv/bin/python run_laya.py --variant ara-rag    # Wouze/laya-ara-rag, Arabic suites only
.venv/bin/python run_laya.py --variant router     # -> preds_laya_router.jsonl
TYPESAFE_API_KEY=<your key> .venv/bin/python run_jev.py
.venv/bin/python score.py preds_laya_router.jsonl preds_jev.jsonl   # -> results.json
```

## Results (2026-09-23, Apple M5, 24 GB)

Laya uses `Router()` as its model card recommends, with its shipped temperatures. Nothing was refit on test data.
Accuracy is scored against each dataset's label (typed-decisions: argmax of the soft gold).

| Suite | n | Laya acc | Jev acc | Laya ECE | Jev ECE |
|---|---|---|---|---|---|
| AG News (4-way topic) | 200 | **0.925** | 0.875 | **0.028** | 0.092 |
| XNLI English | 150 | 0.873 | **0.880** | **0.043** | 0.072 |
| XNLI French | 150 | 0.760 | **0.767** | 0.167 | **0.122** |
| XNLI Arabic | 150 | 0.680 | **0.693** | 0.234 | **0.143** |
| DAIR Emotion (6-way) | 200 | 0.535 | **0.565** | 0.351 | **0.300** |
| SST-2 (noul) | 200 | 0.885 | **0.945** | **0.049** | 0.065 |
| BoolQ (noul) | 200 | 0.815 | **0.895** | 0.093 | **0.037** |
| SST-5 (5-level score) | 200 | 0.340 | **0.635** | 0.381 | **0.181** |
| Banking77 (77-way) | 200 | 0.440 | **0.795** | 0.490 | **0.098** |
| typed-decisions (5 q/case) | 400 | 0.364 | **0.731** | 0.173 | **0.043** |
| **All decisions** | 3,650 | 0.510 | **0.755** | 0.181 | **0.058** |

Latency (one request in flight):

| | Laya on M5 GPU | Jev API |
|---|---|---|
| 1 question, p50 | **~89 ms** | ~400 ms |
| 5 questions, long state (typed-decisions), p50 | 1,140 ms | **~430 ms** |

Jev cost for the whole run: 1.30 M input tokens, about $0.055.

## Arabic (2026-09-25)

Adds [Wouze/laya-ara](https://huggingface.co/Wouze/laya-ara) (Arabic NLU fine-tune of laya-multilingual, non-commercial
licence) and [Wouze/laya-ara-rag](https://huggingface.co/Wouze/laya-ara-rag) (Arabic reranking fine-tune). Every model
runs as shipped. The Router sends Arabic to `laya-multilingual`, so its column is the stock base both fine-tunes start from.

| Suite (n=200, XNLI 150) | Router (stock) | laya-ara | laya-ara-rag | Jev | In fine-tune training? |
|---|---|---|---|---|---|
| XNLI-ar (NLI, 3-way) | 0.680 | **0.753** | 0.700 | 0.693 | laya-ara: capped XNLI-ar sample |
| MASSIVE scenario (18-way) | 0.390 | **0.695** | 0.605 | 0.635 | laya-ara: train split |
| MASSIVE intent (20 options) | 0.325 | 0.775 | 0.550 | **0.830** | laya-ara: train split |
| OSACT4 offensive (noul) | 0.825 | 0.885 | 0.845 | **0.895** | laya-ara: train split |
| Tweet sentiment (3-way) | 0.370 | 0.490 | 0.440 | **0.760** | no model (zero-shot) |
| MIRACL rerank (1 of 8 passages) | 0.315 | 0.320 | 0.715 | **0.875** | laya-ara-rag: MIRACL train |
| MIRACL pairwise relevance (noul) | 0.820 | 0.745 | 0.865 | **0.905** | laya-ara-rag: MIRACL train |
| **Mean of 7 suites** | 0.532 | 0.666 | 0.674 | **0.799** | |
| Latency p50 / p95 | **47 / 83 ms** | 54 / 125 ms | 58 / 127 ms | 396 / 550 ms | |

- Each fine-tune lifts the stock base a lot on its own task family: laya-ara on MASSIVE (+30–45 points), laya-ara-rag
  on MIRACL rerank (+40 points). laya-ara-rag's MASSIVE gains come from the small MASSIVE-ar sample in its training
  mix. laya-ara doesn't improve on reranking, and neither fine-tune closes the gap on sentiment, which no model trained on.
- laya-ara beats Jev on XNLI-ar and MASSIVE scenario, but it trained on XNLI-ar and MASSIVE-ar data. Jev leads
  everywhere else, including sentiment, the one fully zero-shot suite.
- Laya variants are 7–8× faster per call on short Arabic inputs. Jev latency was measured at 8 requests in flight;
  sequential p50 was about 400 ms, so the ratio holds.
- Our scores differ from the model cards (for example laya-ara scenario 0.695 here vs 0.865 on its card). The cards
  use their own frozen prompt templates, while every model here gets the same English instructions and criteria.

## Takeaways

- Laya is 4–5× faster on short single-question calls and edges Jev on AG News. It is roughly even on XNLI
  (including Arabic and French).
- It falls well behind on ordinal scores (SST-5), high-cardinality choices (Banking77) and multi-question
  structured states (typed-decisions, where it sits below the 0.47 majority-class baseline). This matches the
  model card's own "honest limits" section.
- The card's 0.766 typed-decisions number comes from `laya-typed-decisions`, a checkpoint fine-tuned on that
  benchmark's train split (a specialist). It is not comparable to a zero-shot generalist.
- With several questions per call, Laya on MPS is slower than Jev: each question re-encodes the whole state as its
  own sequence (batched, fp32 on MPS), so five questions cost about five times the compute. The card's ~33 ms figures are for a CUDA GPU.
- Caveat: Laya's training families include topic, emotion, NLI and intent tasks, so some public test sets may
  overlap its training data.
