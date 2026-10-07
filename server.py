"""System One Bench live race: any set of decision models, racing on the same labelled cases at the same time.

Racers: every model on OpenRouter's Decisions API (TypeSafe Jev, OpenAI GPT-6 Luna Decisions, Perplexity Decider,
Liquid AI d1, Cloudflare Clef and Clef Flash, Inception Mercury, Upstage Solar, Kev 4B, Together Tev1), and, with
LAYA=1, Laya running locally.

Run:  OPENROUTER_API_KEY=<your key> .venv/bin/python server.py
      OPENROUTER_API_KEY=kerstel://global/OPENROUTER_API_KEY ks run -- .venv/bin/python server.py
Open: http://localhost:8061

Without OPENROUTER_API_KEY, Jev falls back to the TypeSafe API (TYPESAFE_API_KEY) and the other API racers are unavailable.

GET /api/race?racers=jev,openai,pplx&suite=ag_news&n=50&api_concurrency=1 streams Server-Sent Events:
  start  {n, suite, racers:[{key, label, where, series}], cases:[{id, suite, text}]}
  item   {model, i, id, ms, t, correct, total, pred, gold, cost}   (model = racer key; t = ms since the race began;
                                                                   cost in USD, null when local)
  done   {model, t}
API keys stay in this process; the browser never sees them.
"""
import asyncio
import json
import os
import random
import threading
import time

os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("HF_HUB_OFFLINE", "1")  # checkpoints are already in the HF cache

import httpx  # noqa: E402
import uvicorn  # noqa: E402
from fastapi import FastAPI, Query  # noqa: E402
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse  # noqa: E402

from score import gold_idx, options, pred_dist  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
DECISIONS_URL = os.environ.get("OPENROUTER_BASE_URL", "https://openrouter.ai") + "/api/alpha/decisions"
MAX_RACERS = 16


def key(name):
    """The env var's value, or "" when it is unset or is a Kerstel reference that was never resolved."""
    v = os.environ.get(name, "").strip()
    return "" if v.startswith("kerstel://") else v


# Every API racer sends the same request body to OpenRouter's Decisions API. `series` is the maker's color slot,
# the same one systemonebench.dev uses.
API_RACERS = [
    ("jev", "Jev 1.13", "TypeSafe", os.environ.get("JEV_MODEL", "typesafe/jev-1.13"), 1),
    ("openai", "GPT-6 Luna Decisions", "OpenAI", os.environ.get("OPENAI_DECISIONS_MODEL", "openai/gpt-6-luna-decisions"), 2),
    ("pplx", "Decider V1 27B", "Perplexity", "perplexity/pplx-decider-v1-27b", 4),
    ("liquid", "d1", "Liquid AI", "liquid/d1", 6),
    ("clef", "Clef", "Cloudflare", "cloudflare/clef", 5),
    ("clef-flash", "Clef Flash", "Cloudflare", "cloudflare/clef-flash", 5),
    ("mercury", "Mercury Decide", "Inception", "inception/mercury-decide:free", 7),
    ("solar", "Solar Decide", "Upstage", "upstage/solar-decide", 8),
    ("kev", "Kev 4B", "Jared Palmer", "jaredpalmer/kev-4b", 9),
    ("tev1", "Tev1 4B", "Together AI", "togethercomputer/tev1-4b-experimental", 9),
]
RACERS = {k: {"label": label, "maker": maker, "where": f"openrouter.ai · {model}", "series": series, "api": True,
              "url": DECISIONS_URL, "model": model, "key": "OPENROUTER_API_KEY"}
          for k, label, maker, model, series in API_RACERS}
if not key("OPENROUTER_API_KEY") and key("TYPESAFE_API_KEY"):
    jev_model = os.environ.get("JEV_MODEL", "jev-1.13.0")
    RACERS["jev"].update(url=os.environ.get("TYPESAFE_BASE_URL", "https://api.typesafe.ai") + "/v1/systemone",
                         model=jev_model, key="TYPESAFE_API_KEY", where=f"api.typesafe.ai · {jev_model}")

CASES = [json.loads(l) for l in open(os.path.join(HERE, "cases.jsonl"))]
SUITES = sorted({c["suite"] for c in CASES})
ARABIC = [c for c in CASES if c["suite"] == "xnli_ar" or c["suite"].startswith("ar_")]

# Laya is opt-in: it needs torch and the checkpoints in the HF cache, and takes a while to load.
router, agents = None, {}
if os.environ.get("LAYA") == "1":
    import laya
    from laya import Router

    DEVICE = os.environ.get("LAYA_DEVICE", "mps")
    router = Router(device=DEVICE, max_loaded=3)
    router.preload(["english", "multilingual"])
    for c in CASES[:2] + ARABIC[:2]:  # warm up GPU kernels on both checkpoints
        router.predict(c["state"], c["questions"])
    RACERS["router"] = {"label": "Laya Router", "maker": "Convai Innovations", "series": 3,
                        "where": "local · Apple M5 GPU · Router (English + multilingual)"}
    # The Arabic fine-tunes are fixed checkpoints with no routing.
    for name, repo in {"laya-ara": "Wouze/laya-ara", "laya-ara-rag": "Wouze/laya-ara-rag"}.items():
        try:
            agents[name] = laya.load(repo, device=DEVICE)
            agents[name].system_one(ARABIC[0]["state"], ARABIC[0]["questions"])
            RACERS[name] = {"label": name, "maker": "Wouze", "series": 3, "where": f"local · Apple M5 GPU · {repo}"}
        except Exception as e:  # not downloaded yet: the race still works with the Router
            print(f"skipping {repo}: {e!r}"[:200])
gpu_lock = threading.Lock()  # one Laya forward pass at a time, like one local inference process

app = FastAPI()


def preview(state):
    if isinstance(state, dict):
        texts = [v for v in state.values() if isinstance(v, str)]
        if texts:
            return max(texts, key=len)[:160]
    return json.dumps(state, ensure_ascii=False)[:160]


def grade(case, answers):
    right, total, preds, golds = 0, 0, [], []
    for qid, q in case["questions"].items():
        p, _ = pred_dist(q, (answers or {}).get(qid))
        y, yhat = gold_idx(q, case["gold"][qid]), int(p.argmax())
        k = options(q)
        total += 1
        right += int(y == yhat)
        preds.append(k[yhat])
        golds.append(k[y])
    return right, total, preds, golds


def laya_call(case, racer):
    with gpu_lock:
        t0 = time.perf_counter()
        if racer == "router":
            r = router.predict(case["state"], case["questions"])
            via = r["routing"]["model"]
        else:
            r = agents[racer].system_one(case["state"], case["questions"])
            via = racer
        return r["answers"], (time.perf_counter() - t0) * 1000, via, None  # local: no per-call price


async def api_call(client, lane, case):
    body = {"model": lane["model"], "state": case["state"], "questions": case["questions"]}
    for attempt in range(6):
        t0 = time.perf_counter()
        try:
            r = await client.post(lane["url"], json=body)
        except httpx.TransportError:  # dropped connection or timeout: retry like a 5xx
            await asyncio.sleep(2 ** attempt)
            continue
        ms = (time.perf_counter() - t0) * 1000
        if r.status_code == 200:
            d = r.json()
            return d["answers"], ms, d.get("model", lane["model"]), (d.get("usage") or {}).get("cost")
        if r.status_code not in (408, 429, 500, 502, 503, 529):
            raise RuntimeError(f"{lane['model']} returned {r.status_code}: {r.text[:200]}")
        await asyncio.sleep(2 ** attempt)
    raise RuntimeError(f"{lane['model']} kept failing with transport, rate-limit or overload errors")


def public(k):
    r = RACERS[k]
    return {"key": k, "label": r["label"], "maker": r["maker"], "where": r["where"], "series": r["series"],
            "ready": bool(key(r["key"])) if r.get("api") else True}


@app.get("/")
def index():
    return FileResponse(os.path.join(HERE, "race.html"))


@app.get("/api/suites")
def suites():
    counts = {s: sum(c["suite"] == s for c in CASES) for s in SUITES}
    return JSONResponse({"suites": counts, "arabic": len(ARABIC), "racers": [public(k) for k in RACERS]})


@app.get("/api/race")
async def race(racers: str = "jev,openai", suite: str = "mixed", n: int = Query(50, ge=1, le=5000),
               api_concurrency: int = Query(1, ge=1, le=16), seed: int = 0):
    keys = list(dict.fromkeys(k for k in racers.split(",") if k))
    unknown = [k for k in keys if k not in RACERS]
    if unknown or not keys or len(keys) > MAX_RACERS:
        return JSONResponse({"error": f"pick 1 to {MAX_RACERS} racers from {sorted(RACERS)}; unknown: {unknown}"}, status_code=422)
    no_key = [k for k in keys if not public(k)["ready"]]
    if no_key:
        return JSONResponse({"error": f"no API key in the server environment for {no_key}"}, status_code=422)
    pool = {"mixed": CASES, "arabic": ARABIC}.get(suite) or [c for c in CASES if c["suite"] == suite]
    pool = pool[:]
    random.Random(seed).shuffle(pool)
    picked = pool[:n]
    q: asyncio.Queue = asyncio.Queue()
    t_start = time.perf_counter()

    def emit(kind, data):
        q.put_nowait(f"event: {kind}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n")

    def emit_item(model, i, c, ans, ms, via, cost, err):
        right, total, preds, golds = grade(c, ans)
        emit("item", {"model": model, "i": i, "id": c["id"], "ms": round(ms, 1), "t": round((time.perf_counter() - t_start) * 1000),
                      "correct": right, "total": total, "pred": preds, "gold": golds, "via": via, "cost": cost, "error": err})

    async def run_laya(model):
        for i, c in enumerate(picked):
            try:
                ans, ms, via, cost = await asyncio.to_thread(laya_call, c, model)
                err = None
            except Exception as e:  # a failed call is graded as all-wrong
                ans, ms, via, cost, err = None, 0.0, "error", None, repr(e)[:200]
            emit_item(model, i, c, ans, ms, via, cost, err)
        emit("done", {"model": model, "t": round((time.perf_counter() - t_start) * 1000)})

    async def run_api(model, lane):
        sem = asyncio.Semaphore(api_concurrency)
        headers = {"Authorization": f"Bearer {key(lane['key'])}"}
        async with httpx.AsyncClient(headers=headers, timeout=60) as client:
            async def one(i, c):
                async with sem:
                    try:
                        ans, ms, via, cost = await api_call(client, lane, c)
                        err = None
                    except Exception as e:
                        ans, ms, via, cost, err = None, 0.0, "error", None, repr(e)[:200]
                    emit_item(model, i, c, ans, ms, via, cost, err)
            await asyncio.gather(*(one(i, c) for i, c in enumerate(picked)))
        emit("done", {"model": model, "t": round((time.perf_counter() - t_start) * 1000)})

    async def stream():
        start = {"n": len(picked), "suite": suite, "api_concurrency": api_concurrency, "racers": [public(k) for k in keys],
                 "cases": [{"id": c["id"], "suite": c["suite"], "text": preview(c["state"])} for c in picked]}
        yield f"event: start\ndata: {json.dumps(start, ensure_ascii=False)}\n\n"
        tasks = [asyncio.create_task(run_api(k, RACERS[k]) if RACERS[k].get("api") else run_laya(k)) for k in keys]
        finished = 0
        try:
            while finished < len(keys):
                msg = await q.get()
                if msg.startswith("event: done"):
                    finished += 1
                yield msg
        finally:
            for t in tasks:
                t.cancel()

    return StreamingResponse(stream(), media_type="text/event-stream", headers={"Cache-Control": "no-cache"})


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=int(os.environ.get("PORT", 8061)))
