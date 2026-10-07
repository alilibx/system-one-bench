# System One Bench

A benchmark for System One models: models that read a `state` and answer named questions with typed, probabilistic
answers (`noul` yes/no, `choice`, and ordered `score`) in one fast call. Every model gets byte-identical `state` +
`questions` bodies, and is scored on accuracy, calibration, latency, and cost across 16 labelled datasets (3,250 cases,
4,850 decisions) in English, French, and Arabic. A live race runs two models side by side on the same cases.

Models covered so far:

| Model | Maker | How it runs | Round |
|---|---|---|---|
| Jev 1.13 (`typesafe/jev-1.13`) | TypeSafe | OpenRouter Decisions API, or the TypeSafe API | all |
| GPT-6 Luna Decisions (`openai/gpt-6-luna-decisions`) | OpenAI | OpenRouter Decisions API | 2026-10-07 |
| Decider V1 27B (`perplexity/pplx-decider-v1-27b`) | Perplexity | OpenRouter Decisions API | 2026-10-07 |
| d1 (`liquid/d1`) | Liquid AI | OpenRouter Decisions API | 2026-10-07 |
| Clef and Clef Flash (`cloudflare/clef`, `cloudflare/clef-flash`) | Cloudflare | OpenRouter Decisions API | 2026-10-07 |
| Solar Decide (`upstage/solar-decide`) | Upstage | OpenRouter Decisions API | 2026-10-07 |
| Mercury Decide (`inception/mercury-decide:free`) | Inception | OpenRouter Decisions API, free tier | 2026-10-07 |
| Kev 4B (`jaredpalmer/kev-4b`) | Jared Palmer | OpenRouter Decisions API | 2026-10-07 |
| Tev1 4B Experimental (`togethercomputer/tev1-4b-experimental`) | Together AI | OpenRouter Decisions API | 2026-10-07 |
| [Laya](https://huggingface.co/convaiinnovations/laya) and its Arabic fine-tunes | Convai Innovations, Wouze | open weights, locally on Apple MPS | 2026-09-23, 2026-09-25 |

Every API model is called at `POST https://openrouter.ai/api/alpha/decisions`, so they all share one gateway.

**Latest results: [systemonebench.dev](https://systemonebench.dev/)**

System One Bench is a living benchmark: we keep adding models as they're released, and every new model runs the same
cases through the same scorer. The [roadmap](#roadmap) lists what's next.

## Results site

`site/` is a static page that reads `results.json` and `site/meta.json` (display names, makers, run dates, and dataset
descriptions). Its layout follows [Artificial Analysis](https://artificialanalysis.ai/): a headline **System One Index** (mean
accuracy across datasets, equal weight per dataset, 0 to 100, with English and Arabic sub-indexes), index-against-latency
and index-against-cost scatters with a Pareto line, calibration, latency and cost bar charts, and one bar chart per dataset. `.github/workflows/pages.yml` publishes it to GitHub Pages on every push to `main` that touches
`results.json` or `site/`, so re-running `score.py` and merging updates it. Each run needs an entry in `site/meta.json` with its run
date; the newest run date becomes the page's "Updated" date, and runs without an entry are listed as not shown.
The site is served at systemonebench.dev: the domain's DNS (Vercel) points the apex at GitHub Pages' A records and
`www` at `alilibx.github.io`. To preview locally:

```bash
mkdir -p _site && cp site/* results.json _site/ && python3 -m http.server 8062 -d _site
```

## Roadmap

The roadmap lives in `site/meta.json` and is published at [systemonebench.dev/#roadmap](https://systemonebench.dev/#roadmap).
As of 2026-10-07 it lists 8 models that run today on OpenRouter's Decisions API, 6 that need a small adapter, and 2 entries
on a watch list. Several vendors publish their own scores; none go on the site until they've run here like every other model.

## Live race

```bash
OPENROUTER_API_KEY=<your key> .venv/bin/python server.py
# or, with the key in Kerstel:
OPENROUTER_API_KEY=kerstel://global/OPENROUTER_API_KEY ks run -- .venv/bin/python server.py
```

Then open http://localhost:8061, pick a dataset and a challenger, and press **Start race**. Every API model on the
leaderboard is available as a challenger; the default is OpenAI Decisions (`openai/gpt-6-luna-decisions`). Jev runs as `typesafe/jev-1.13` on the same OpenRouter endpoint, so both lanes pay the
same gateway overhead. Each lane shows accuracy, latency and the summed `usage.cost` of its calls. The key stays in the
server process and never reaches the browser; keep it in a secrets manager, never in this repo. "API parallel 1" is the
fair setting: one request in flight per model.

To race Laya instead, start the server with `LAYA=1 OMP_NUM_THREADS=1 HF_HUB_OFFLINE=1` and pick Laya Router, laya-ara,
or laya-ara-rag. Laya runs on the Apple GPU (MPS); `OMP_NUM_THREADS=1` avoids a libomp deadlock when several
checkpoints load in one process. Without `OPENROUTER_API_KEY`, the Jev lane falls back to the TypeSafe API
(`TYPESAFE_API_KEY`).

## Full benchmark

```bash
.venv/bin/python build_suites.py                  # -> cases.jsonl (2,050 cases, 3,650 decisions)
.venv/bin/python build_arabic.py                  # appends 1,200 Arabic cases (4,850 decisions in all)
export OPENROUTER_API_KEY=<your key>
.venv/bin/python run_decisions.py --model openai/gpt-6-luna-decisions --name luna   # -> preds_luna.jsonl
.venv/bin/python run_decisions.py --model typesafe/jev-1.13 --name or_jev           # -> preds_or_jev.jsonl
.venv/bin/python score.py preds_or_jev.jsonl preds_luna.jsonl                       # -> results.json
```

`run_decisions.py` is resumable: re-running it retries only the cases that failed. Laya and direct-TypeSafe runs:

```bash
.venv/bin/python run_laya.py --variant router     # -> preds_laya_router.jsonl
.venv/bin/python run_laya.py --variant ara        # Wouze/laya-ara, Arabic suites only
.venv/bin/python run_laya.py --variant ara-rag    # Wouze/laya-ara-rag, Arabic suites only
TYPESAFE_API_KEY=<your key> .venv/bin/python run_jev.py
```

## Round 2: eleven models (2026-10-07)

Eight more models from OpenRouter's Decisions API, run the same day as Jev and GPT-6 Luna Decisions, all 3,250 cases.
The index is the mean accuracy across the 16 datasets, every dataset weighted equally. API latency is the median per call
through OpenRouter (8 calls in flight for Jev and GPT-6 Luna Decisions, 6 for the rest, Solar Decide finishing at 24);
Laya ran locally on an Apple M5 GPU.

| Model | Maker | Index | Calibration error | Median latency | Cost per 1,000 calls |
|---|---|---|---|---|---|
| **Decider V1 27B** | Perplexity | 83.3 | 5.6% | 528 ms | $0.019 |
| d1 | Liquid AI | 82.7 | 2.9% | 659 ms | $0.015 |
| Mercury Decide | Inception | 82.4 | 13.1% | 490 ms | free tier |
| Clef | Cloudflare | 81.6 | 2.2% | 706 ms | $0.116 |
| Clef Flash | Cloudflare | 80.3 | 3.5% | 534 ms | $0.044 |
| Jev 1.13 | TypeSafe | 79.0 | 6.1% | 372 ms | $0.028 |
| Kev 4B | Jared Palmer | 76.8 | 5.1% | 899 ms | $0.010 |
| GPT-6 Luna Decisions | OpenAI | 76.0 | 9.8% | 363 ms | $0.040 |
| Tev1 4B Experimental | Together AI | 71.5 | 5.3% | 518 ms | $0.019 |
| Solar Decide | Upstage | 68.8 | 20.3% | 1,231 ms | $0.042 |
| Laya Router | Convai Innovations | 60.4 | 20.0% | 75 ms | local |

- Five models beat Jev 1.13 on the index, led by Perplexity Decider V1 27B (83.3) and Liquid AI d1 (82.7). The top five
  are within 3 points of each other.
- Cloudflare Clef and Liquid AI d1 are the best calibrated (2.2% and 2.9% calibration error); Mercury Decide is accurate
  but overconfident (13.1%).
- Jev 1.13 and GPT-6 Luna Decisions remain the fastest API models (about 370 ms median). Kev 4B is the cheapest paid model.
- Together Tev1 accepts at most 20 options per question and Upstage Solar Decide at most 26, so each fails all 200
  Banking77 cases (77 options); failures count as wrong. Respan Span-01 only accepts conversations and wasn't run.
- Every price above is what OpenRouter billed; Mercury Decide ran on the free tier.

## Jev vs OpenAI Decisions (2026-10-07)

All 3,250 cases (4,850 decisions), both models through OpenRouter on the same day, 8 requests in flight. Models as served:
`typesafe/jev-1.13-20260917` and `openai/gpt-6-luna-decisions-20261006`. Nothing was tuned. OpenAI returned a 502 on
the same 2 cases on every retry (`ar_massive_intent-46`, `ar_miracl_rerank-177`); they count as wrong.

| Suite | n | Jev acc | OpenAI acc | Jev ECE | OpenAI ECE |
|---|---|---|---|---|---|
| AG News (4-way topic) | 200 | **0.885** | 0.875 | **0.086** | 0.112 |
| XNLI English | 150 | **0.860** | 0.667 | **0.082** | 0.207 |
| XNLI French | 150 | **0.767** | 0.620 | **0.122** | 0.215 |
| XNLI Arabic | 150 | **0.693** | 0.540 | **0.140** | 0.298 |
| DAIR Emotion (6-way) | 200 | 0.560 | **0.585** | 0.304 | **0.261** |
| SST-2 (noul) | 200 | **0.945** | 0.935 | 0.064 | **0.048** |
| BoolQ (noul) | 200 | **0.895** | 0.855 | **0.039** | 0.086 |
| SST-5 (5-level score) | 200 | **0.620** | 0.615 | 0.161 | **0.160** |
| Banking77 (77-way) | 200 | **0.790** | 0.775 | 0.101 | **0.087** |
| typed-decisions (5 q/case) | 400 | **0.727** | 0.711 | **0.040** | 0.084 |
| MASSIVE-ar scenario (18-way) | 200 | 0.630 | **0.660** | 0.210 | **0.168** |
| MASSIVE-ar intent (20 options) | 200 | 0.840 | **0.845** | **0.035** | 0.051 |
| OSACT4 offensive (noul) | 200 | 0.900 | **0.935** | 0.055 | **0.022** |
| Arabic tweet sentiment (3-way) | 200 | 0.760 | **0.800** | **0.097** | 0.107 |
| MIRACL-ar rerank (1 of 8) | 200 | **0.865** | 0.860 | 0.061 | **0.045** |
| MIRACL-ar pairwise (noul) | 200 | **0.905** | 0.875 | **0.027** | 0.107 |
| **All decisions** | 4,850 | **0.767** | 0.746 | **0.061** | 0.098 |

Log loss over all decisions: Jev 0.795, OpenAI 0.985. Brier: Jev 0.332, OpenAI 0.376.

Latency and cost:

| | Jev (OpenRouter) | OpenAI Decisions |
|---|---|---|
| 1 question, one in flight, p50 / p95 | 360 / 472 ms | **356 / 412 ms** |
| 5 questions, long state, one in flight, p50 / p95 | 376 / 585 ms | **363 / 428 ms** |
| 8 in flight, all cases, p50 / p95 | 372 / 469 ms | **363 / 460 ms** |
| Billed input tokens, whole run | 2.16 M | 1.31 M |
| Cost, whole run (`usage.cost`) | **$0.091** | $0.131 |
| Cost per 1,000 calls | **$0.028** | $0.040 |

- Jev is more accurate overall (+2.1 points) and better calibrated (ECE 0.061 vs 0.098). The gap is mostly NLI: XNLI
  English, French and Arabic are 15–19 points apart. Jev also leads on BoolQ, MIRACL pairwise and typed-decisions.
- OpenAI Decisions wins five suites: Arabic offensive, Arabic sentiment, MASSIVE-ar scenario and intent, and DAIR
  Emotion. Its biggest lead is Arabic sentiment (+4 points).
- Latency is a tie at p50, both about 360 ms through OpenRouter. OpenAI has a tighter tail, most visibly on 5-question
  calls (p95 428 vs 585 ms).
- OpenAI costs about 1.4× more for the same work. Its tokenizer counts 40% fewer input tokens, but it bills $0.10 per
  million against Jev's $0.042. Neither bills output tokens.
- Sanity check: Jev through OpenRouter scores 0.767 overall against 0.770 from the TypeSafe API on 2026-09-23. Same
  model, so the gateway doesn't change answers. OpenRouter was faster at p95 (469 vs 851 ms at 8 in flight).

## Laya vs Jev (2026-09-23, Apple M5, 24 GB)

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

## Laya Arabic fine-tunes vs Jev (2026-09-25)

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

## Laya takeaways

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
