# System One Bench

A benchmark for System One models: models that read a `state` and answer named questions with typed, probabilistic
answers (`noul` yes/no, `choice`, and ordered `score`) in one fast call. Every model gets byte-identical `state` +
`questions` bodies, and is scored on accuracy, calibration, latency, and cost across 16 labelled datasets (3,250 cases,
4,850 decisions) in English, French, and Arabic. A [vision track](#vision-track) adds 6 image datasets (1,200 cases) for
the models that read images, including Box questions. A live race runs two models side by side on the same cases.

Models covered so far:

| Model | Maker | How it runs | Round |
|---|---|---|---|
| Jev 1.14 (`typesafe/jev-1.14`) | TypeSafe | OpenRouter Decisions API | 2026-10-15, text and vision |
| Jev 1.13 (`typesafe/jev-1.13`) | TypeSafe | OpenRouter Decisions API, or the TypeSafe API | all |
| GPT-6 Luna Decisions (`openai/gpt-6-luna-decisions`) | OpenAI | OpenRouter Decisions API | 2026-10-07 |
| Decider V1.1 27B (`perplexity/pplx-decider-v1.1-27b`) | Perplexity | OpenRouter Decisions API | 2026-10-12 |
| Decider V1 27B (`perplexity/pplx-decider-v1-27b`), retired | Perplexity | OpenRouter Decisions API, no longer served | 2026-10-07 |
| d1 (`liquid/d1`) | Liquid AI | OpenRouter Decisions API | 2026-10-07 |
| Clef and Clef Flash (`cloudflare/clef`, `cloudflare/clef-flash`) | Cloudflare | OpenRouter Decisions API | 2026-10-07 |
| Clef Omni (`cloudflare/clef-omni`) | Cloudflare | OpenRouter Decisions API | 2026-10-12 |
| Solar Decide (`upstage/solar-decide`) | Upstage | OpenRouter Decisions API | 2026-10-07 |
| Solar Decide Flash (`upstage/solar-decide-flash`) | Upstage | OpenRouter Decisions API | 2026-10-12 |
| Mercury Decide (`inception/mercury-decide:free`) | Inception | OpenRouter Decisions API, free tier | 2026-10-07 |
| Kev 4B (`jaredpalmer/kev-4b`) | Jared Palmer | OpenRouter Decisions API | 2026-10-07 |
| Tev1 4B Experimental (`togethercomputer/tev1-4b-experimental`) | Together AI | OpenRouter Decisions API | 2026-10-07 |
| Microsoft-Decision-1 (`microsoft/microsoft-decision-1`) | Microsoft | OpenRouter Decisions API | 2026-10-12 |
| Drex v1.5 (`nace-ai/drex-v1.5`) | Nace AI | OpenRouter Decisions API | 2026-10-12 |
| [Laya](https://huggingface.co/convaiinnovations/laya) and its Arabic fine-tunes | Convai Innovations, Wouze | open weights, locally on Apple MPS | 2026-09-23, 2026-09-25 |

Every API model is called at `POST https://openrouter.ai/api/alpha/decisions`, so they all share one gateway.

**Latest results: [systemonebench.dev](https://systemonebench.dev/)**

System One Bench is a living benchmark: we keep adding models as they're released, and every new model runs the same
cases through the same scorer. The [roadmap](#roadmap) lists what's next.

## Results site

`site/` is a static page that reads `results.json`, `results_vision.json` and `site/meta.json` (display names, makers, run
dates, dataset descriptions, and the vision track's datasets). Its layout follows [Artificial Analysis](https://artificialanalysis.ai/): a headline **System One Index** (mean
accuracy across datasets, equal weight per dataset, 0 to 100, with English and Arabic sub-indexes), index-against-latency
and index-against-cost scatters with a Pareto line, calibration, latency and cost bar charts, and one bar chart per dataset. `.github/workflows/pages.yml` publishes it to GitHub Pages on every push to `main` that touches
`results.json`, `results_vision.json` or `site/`, so re-running `score.py` and merging updates it. Each run needs an entry in `site/meta.json` with its run
date; the newest run date becomes the page's "Updated" date, and runs without an entry are listed as not shown.
The site is served at systemonebench.dev: the domain's DNS (Vercel) points the apex at GitHub Pages' A records and
`www` at `alilibx.github.io`. To preview locally:

```bash
mkdir -p _site && cp site/* results.json results_vision.json _site/ && python3 -m http.server 8062 -d _site
```

## Roadmap

The roadmap lives in `site/meta.json` and is published at [systemonebench.dev/#roadmap](https://systemonebench.dev/#roadmap).
As of 2026-10-12 it lists 4 hosted models on their own APIs with the same request shape (Fastino GLiDE, Celeris-1
Decision, Hanzo Kai 1, meraGPT Decider 1), 6 open-weight families to run locally (including Liquid AI's d1-3B and
d1-omni-600M, vLLM's d3 and the Kev size sweep), 2 entries that need a track of their own, and 3 on a watch list. Everything that
runs as-is on OpenRouter's Decisions API has been benchmarked. Several vendors publish their own scores; none go on the
site until they've run here like every other model.

## Live race

```bash
OPENROUTER_API_KEY=<your key> .venv/bin/python server.py
# or, with the key in Kerstel:
OPENROUTER_API_KEY=kerstel://global/OPENROUTER_API_KEY ks run -- .venv/bin/python server.py
```

Then open http://localhost:8061, pick any set of models with the chips (or **All**), choose a dataset, and press **Start
race**. Every selected model gets its own lane and they all run at the same time on the same cases: everything vs
everything. Each lane shows accuracy, median latency, finish position and the summed `usage.cost` of its calls; the
standings table ranks lanes by accuracy as answers arrive, and the verdict names the fastest, most accurate and
cheapest model. The key stays in the server process and never reaches the browser; keep it in a secrets manager, never
in this repo. "Parallel per model 1" is the fair setting: one request in flight per model.

To add Laya lanes, start the server with `LAYA=1 OMP_NUM_THREADS=1 HF_HUB_OFFLINE=1`. Laya runs on the Apple GPU (MPS);
`OMP_NUM_THREADS=1` avoids a libomp deadlock when several checkpoints load in one process. Without
`OPENROUTER_API_KEY`, only Jev is available, through the TypeSafe API (`TYPESAFE_API_KEY`).

## Full benchmark

```bash
.venv/bin/python build_suites.py                  # -> cases.jsonl (2,050 cases, 3,650 decisions)
.venv/bin/python build_arabic.py                  # appends 1,200 Arabic cases (4,850 decisions in all)
export OPENROUTER_API_KEY=<your key>
.venv/bin/python run_decisions.py --model openai/gpt-6-luna-decisions --name luna   # -> preds_luna.jsonl
.venv/bin/python run_decisions.py --model typesafe/jev-1.13 --name or_jev           # -> preds_or_jev.jsonl
.venv/bin/python score.py preds_or_jev.jsonl preds_luna.jsonl                       # -> results.json
```

`score.py` rewrites `results.json` from only the files you pass, in that order, and the site lists models in that order.
`--merge` instead keeps the runs already there and adds or replaces just the ones you pass, which is how Jev 1.14 joined
(`score.py --merge preds_or_jev_14.jsonl`). To rebuild the published file, pass every run:

```bash
.venv/bin/python score.py preds_laya_router.jsonl preds_laya_ara.jsonl preds_laya_ara-rag.jsonl preds_jev.jsonl \
  preds_or_jev.jsonl preds_luna.jsonl preds_pplx.jsonl preds_liquid_d1.jsonl preds_clef.jsonl preds_clef_flash.jsonl \
  preds_solar.jsonl preds_mercury.jsonl preds_kev.jsonl preds_tev1.jsonl preds_msft_d1.jsonl preds_pplx_v11.jsonl \
  preds_clef_omni.jsonl preds_drex.jsonl preds_solar_flash.jsonl preds_or_jev_14.jsonl
```

Models on their vendor's own API take the same request body. `--provider fastino|celeris|hanzo|meragpt` sets the
endpoint, model id and auth header, and reads the key from `FASTINO_API_KEY`, `CELERIS_API_KEY`, `HANZO_API_KEY` or
`MERAGPT_API_KEY`. `--url` points the runner at any other endpoint that takes this body, such as a local server
(`--key-env ""` when it needs no key):

```bash
CELERIS_API_KEY=kerstel://global/CELERIS_API_KEY ks run -- .venv/bin/python run_decisions.py --provider celeris --name celeris
.venv/bin/python run_decisions.py --url http://localhost:8000/v1/systemone --key-env "" --model kev-9b --name kev_9b
```

`run_decisions.py` is resumable: re-running it retries only the cases that failed. Laya and direct-TypeSafe runs:

```bash
.venv/bin/python run_laya.py --variant router     # -> preds_laya_router.jsonl
.venv/bin/python run_laya.py --variant ara        # Wouze/laya-ara, Arabic suites only
.venv/bin/python run_laya.py --variant ara-rag    # Wouze/laya-ara-rag, Arabic suites only
TYPESAFE_API_KEY=<your key> .venv/bin/python run_jev.py
```

## Vision track

`build_images.py` builds 1,200 image cases (200 per dataset) into `image_cases.jsonl`, with the images under `images/`.
A case's `state` holds `{"image": "images/<suite>/<n>.jpg"}`, and `run_decisions.py` sends it as an `image_url` part with a
base64 data URL, the shape OpenRouter's Decisions API reads. Every image is rotated by its EXIF tag and resized to fit
every model's limits: each side 280 to 1,568 px, at least 78,400 pixels, at most about 300 KB.

| Dataset | Question | Source |
|---|---|---|
| SST-2, as an image | yes/no | The 200 SST-2 reviews from `cases.jsonl`, rendered in Noto Sans |
| Arabic sentiment, as an image | 3 options | The 200 Arabic tweets from `cases.jsonl`, rendered in Noto Naskh Arabic |
| Oxford-IIIT Pet | 37 options | `timm/oxford-iiit-pet` test, CC BY-SA 4.0 |
| VQAv2 yes/no | yes/no | `lmms-lab/VQAv2` validation, first shard, one question per image, CC BY 4.0 |
| CountBench | score, 2 to 10 | `vikhyatk/CountBenchQA` |
| RefCOCO | box | `lmms-lab/RefCOCO` testA (people) and testB (objects), 100 each, one phrase per image |

The two rendered datasets reuse the exact items of the text datasets, so the gap between them is what reading from
pixels costs. Rendering Arabic needs Pillow with libraqm (`brew install libraqm` on macOS); the fonts are downloaded from
Google Fonts into `fonts/`. The Box dataset only runs on models that answer Box questions, so pass `--suites` to the others:

```bash
.venv/bin/python build_images.py                  # needs cases.jsonl; -> image_cases.jsonl, images/
V=img_sst2,img_ar_sentiment,img_pets,img_vqa_yesno,img_count
.venv/bin/python run_decisions.py --cases image_cases.jsonl --out-dir vision --model typesafe/jev-1.14 --name or_jev_14
.venv/bin/python run_decisions.py --cases image_cases.jsonl --out-dir vision --model openai/gpt-6-luna-decisions --name luna --suites $V
.venv/bin/python score.py --cases image_cases.jsonl --out results_vision.json vision/preds_or_jev_14.jsonl vision/preds_or_jev.jsonl \
  vision/preds_luna.jsonl vision/preds_pplx_v11.jsonl vision/preds_clef.jsonl vision/preds_clef_flash.jsonl \
  vision/preds_clef_omni.jsonl vision/preds_mercury.jsonl
```

Box answers are scored by intersection over union with the labelled box: right when IoU is 0.5 or more, plus mean IoU,
and calibration of the answer's `confidence` against being right. They're reported apart and stay out of the Vision
Index, since only Jev 1.14 answers them. Run names match the text runs, so the site shows both under one name.

## Open-weight models, run locally

Most open decision models now run under llama.cpp, whose `llama-server` answers `POST /v1/systemone` with the same
request and response shape as the APIs above, so `run_decisions.py --url` runs them unchanged. Kev and Laya need
llama.cpp 0.6.0; Liquid AI's d1 needs build b11480 or later, and vLLM's d3 needs b11570 or later (12 October 2026), from
the [GitHub releases](https://github.com/ggml-org/llama.cpp/releases) or `brew install --HEAD llama.cpp`.

```bash
llama-server -hf LiquidAI/d1-3B-GGUF:Q8_0 --no-mmproj --port 8080
.venv/bin/python run_decisions.py --url http://localhost:8080/v1/systemone --key-env "" --model d1-3b --name d1_3b --concurrency 1
```

| Model | GGUF (Q8_0) | Size | Notes |
|---|---|---|---|
| Liquid AI d1-3B | `LiquidAI/d1-3B-GGUF` | 2.9 GB | LFM Open License; pass `--no-mmproj` for text only |
| Liquid AI d1-omni-600M | `LiquidAI/d1-omni-600M-GGUF` | 0.4 GB | Add `-b 4096 -ub 4096` so a whole prompt fits in one batch |
| Kev 0.8B, 4B, 9B | `ggml-org/Kev-{0.8B,4B,9B}-GGUF` | 0.8, 4.5, 9.5 GB | English only; Kev 4B is already on the board through OpenRouter |
| vLLM d3 edge, lite, nano, mini, flash | `ggml-org/vllm-d3-{edge,lite,nano,mini,flash}-GGUF` | 0.8 to 9.5 GB | Decision 3.0, released 10 October 2026; `-ub` must hold the whole prompt |

The other open models need their own runners:

- **Hugging Face `system_one`:** d1 and the vLLM Decision-1.0 and 2.0 models load with `trust_remote_code` and answer
  through `model.system_one(state, questions)`, the same shape. The project's `.venv` already has the transformers
  version they need.
- **`kev.serve`:** Kev 0.5B, 0.6B and 8B ship as LoRA adapters, served by [jaredpalmer/kev](https://github.com/jaredpalmer/kev)
  on `/v1/systemone`. It needs Python 3.12 or 3.13.
- **Adapters still to write:** GLiNER2.5-Decide and GLiClass answer label questions only, so noul and score map onto
  labels. bekko-system-one takes its own input format, and its weights have no licence yet.

A 24 GB Mac fits everything up to 9B at Q8. The 27B models (Kev 27B, d3 27B, Open-Jev 27B) and Open-Jev 9B, whose base
model alone is 19 GB, need a bigger machine. Laya Multilingual is already covered: Laya Router sends every non-English
state, Arabic and French included, to those same weights.

## Round 5: Jev 1.14 and the vision track (2026-10-15)

TypeSafe released Jev 1.14 on 2026-10-14. It adds images, a Box question type that returns a bounding box, and a 262k
context; TypeSafe marks all three as early features that aren't production-ready. This round runs Jev 1.14 on the text
benchmark and starts a vision track for every model on OpenRouter's Decisions API that accepts images. Everything ran
from a cloud machine, 8 calls in flight per model.

**Text.** Jev 1.14 ran all 3,250 cases, served as `typesafe/jev-1.14`. It billed exactly the 2,157,160 input tokens of
the Jev 1.13 run, which confirms the cases were rebuilt byte for byte.

| Model | Index | English | Arabic | Calibration error | Cost per 1,000 calls |
|---|---|---|---|---|---|
| Jev 1.14 | **79.4** | **78.8** | **80.4** | **5.9%** | $0.028 |
| Jev 1.13 | 79.0 | 78.5 | 79.9 | 6.1% | $0.028 |

- Jev 1.14 is 0.4 points ahead of Jev 1.13 and places 8th of 16, just behind Clef Omni (79.7). Its gains are spread
  thin (+2.5 on MASSIVE-ar scenario, +2.0 on XNLI Arabic); it drops 3 points on Arabic offensive language.
- Same price as Jev 1.13, and the same speed: one call at a time from the cloud machine, Jev 1.14 took 136 ms median
  and Jev 1.13 140 ms. Its 134 ms on the site is lower than earlier runs because of that machine's network, not the model.

**Vision.** 8 models accept images; d1, Microsoft-Decision-1, Kev 4B, Tev1, Solar Decide, Solar Decide Flash and Drex
return 400 ("does not accept image_url parts"). The Vision Index is the mean accuracy over 5 image datasets, kept apart
from the System One Index.

| Model | Vision Index | SST-2 image | Arabic sentiment image | Pets (37) | VQAv2 yes/no | CountBench | Calibration error | Median latency | Cost per 1,000 calls |
|---|---|---|---|---|---|---|---|---|---|
| **Jev 1.14** | **85.1** | 93.5 | 63.5 | **88.5** | 86.5 | 93.5 | 2.9% | **183 ms** | $0.046 |
| Jev 1.13 | 83.9 | 94.5 | 59.0 | 83.5 | 88.0 | **94.5** | **2.8%** | 192 ms | $0.046 |
| Decider V1.1 27B | 83.0 | **96.0** | 51.0 | 84.0 | **89.5** | **94.5** | 5.6% | 302 ms | $0.009 |
| GPT-6 Luna Decisions | 81.6 | 88.5 | **67.0** | 87.0 | 80.5 | 85.0 | 5.3% | 221 ms | $0.047 |
| Mercury Decide | 79.4 | 93.0 | 39.0 | 84.5 | 87.0 | 93.5 | 9.9% | 305 ms | free tier |
| Clef | 78.1 | 95.5 | 35.0 | 82.5 | 87.5 | 90.0 | 3.6% | 502 ms | $0.138 |
| Clef Flash | 78.0 | 95.5 | 33.5 | 85.0 | 88.0 | 88.0 | 9.4% | 429 ms | $0.022 |
| Clef Omni | 72.2 | 89.5 | 28.5 | 88.0 | 87.0 | 68.0 | 20.6% | 274 ms | $0.086 |

- Jev 1.14 leads the Vision Index by 1.2 points over Jev 1.13, mostly on pet breeds (+5.0) and the Arabic tweets
  (+4.5). Both Jevs are the best calibrated on images.
- Reading English from a picture costs little: every model is within 7.5 points of its text SST-2 score, and Decider,
  Jev 1.13 and Clef Flash lose nothing. Reading Arabic from a picture costs a lot: 13 to 23.5 points on the same 200
  tweets. GPT-6 Luna Decisions is best at it (67.0); the Clef models fall to near the 3-way floor.
- Jev bills about 1,100 input tokens per image call, twice what the others count, but its price per token is lower, so
  it costs the same as Luna. Decider V1.1 is again the cheapest paid model.
- GPT-6 Luna Decisions returned a 502 on the same 2 cases on every retry (`img_sst2-187`, `img_vqa_yesno-105`); they
  count as wrong.

**Box questions** (Jev 1.14 only; Jev 1.13 and the others reject them). On 200 RefCOCO phrases, 100 about people
(testA) and 100 about objects (testB):

| | Right (IoU ≥ 0.5) | Mean IoU | Mean confidence | Calibration error | Confidence AUROC |
|---|---|---|---|---|---|
| People (testA) | 83.0% | 0.74 | 52.9% | 30.7% | 0.82 |
| Objects (testB) | 74.0% | 0.67 | 43.4% | 30.6% | 0.77 |
| **All** | **78.5%** | **0.70** | 48.2% | 30.3% | 0.80 |

- Box answers are good but underconfident: right 78.5% of the time at an average stated confidence of 48%. The
  confidence still ranks answers well (AUROC 0.80), so it works as a threshold, not as a probability.
- A Box call takes 194 ms median, about the same as Jev's other image calls (182 ms).
- Jev's image docs give 256 px as the shortest side it accepts, but it answered "Invalid image dimensions" for every
  image with a side under 280 px. `build_images.py` keeps every side at 280 px or more.

## Round 4: four more models from OpenRouter (2026-10-12)

Four decision models that appeared on OpenRouter's Decisions API since round 2, run on all 3,250 cases with 8 calls in
flight: Perplexity Decider V1.1 27B, Cloudflare Clef Omni, Nace AI Drex v1.5 and Upstage Solar Decide Flash. Perplexity
retired Decider V1 27B (its slug now returns 404), so V1.1 replaces it on the charts and in the live race; the V1 run is
kept for reference.

| Model | Maker | Index | English | Arabic | Calibration error | Median latency | Cost per 1,000 calls |
|---|---|---|---|---|---|---|---|
| **Decider V1.1 27B** | Perplexity | **83.4** | 81.1 | **85.0** | 5.7% | 678 ms | **$0.009** |
| Clef Omni | Cloudflare | 79.7 | 78.6 | 80.5 | 3.4% | 544 ms | $0.074 |
| Drex v1.5 | Nace AI | 77.6 | 77.6 | 77.0 | 10.8% | 608 ms | $0.010 |
| Solar Decide Flash | Upstage | 63.6 | 61.3 | 66.5 | 14.4% | 908 ms | $0.042 |

- Decider V1.1 matches V1 on accuracy (83.4 against 83.3) and still leads the index, at half V1's price per token. It's
  now also the cheapest paid model. It's slower: 785 ms one call at a time, against 352 ms for V1.
- Clef Omni, Cloudflare's model that also reads audio and video, places 7th, between Clef Flash and Jev, and is well
  calibrated (3.4%).
- Drex v1.5 costs about the same as Microsoft-Decision-1 and Kev 4B and places 9th, but its confidence runs high
  (10.8% calibration error).
- Solar Decide Flash has Solar Decide's 26-option limit, so it fails all 200 Banking77 cases; those count as wrong. It
  bills the same as Solar Decide.

## Round 3: Microsoft-Decision-1 (2026-10-12)

Microsoft released Microsoft-Decision-1 on 2026-10-09: a decision model post-trained from Qwen3.5-9B, served on
OpenRouter's Decisions API at $0.042 per 1M input tokens. It ran all 3,250 cases through the same gateway and scorer as
round 2, 8 calls in flight, served as `microsoft/microsoft-decision-1-20261009`. Every call succeeded.

| Model | Maker | Index | English | Arabic | Calibration error | Median latency | Cost per 1,000 calls |
|---|---|---|---|---|---|---|---|
| Decider V1 27B | Perplexity | **83.3** | 81.1 | **85.0** | 5.6% | 528 ms | $0.019 |
| d1 | Liquid AI | 82.7 | 81.3 | 83.9 | 2.9% | 659 ms | $0.015 |
| Mercury Decide | Inception | 82.4 | **83.7** | 80.8 | 13.1% | 490 ms | free tier |
| **Microsoft-Decision-1** | Microsoft | 82.0 | 81.0 | 83.2 | 2.8% | 679 ms | $0.010 |
| Clef | Cloudflare | 81.6 | 80.6 | 83.0 | **2.2%** | 706 ms | $0.116 |

- Microsoft-Decision-1 places 4th of 12 on the System One Index, 1.3 points behind Perplexity Decider V1 27B. It's 3rd
  on the Arabic index.
- It's the second-best calibrated model (2.8% calibration error), behind Cloudflare Clef (2.2%).
- It's the cheapest paid model at its accuracy: $0.034 for all 3,250 cases, about the same as Kev 4B, which scores
  5.2 points lower.
- It's the most accurate model on SST-5 (62.5%), typed-decisions (79.0%) and Arabic MIRACL pairs (91.0%). Its weakest
  dataset is Emotion (57.0%, against Mercury Decide's 91.5%).
- Median latency was 679 ms with 8 calls in flight and 613 ms one call at a time, both through OpenRouter and including
  the network. Jev measured 365 ms one at a time from the same machine on 2026-10-07, so part of that gap may be the
  day's network. Microsoft reports 85 ms p50 in its own test.

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
