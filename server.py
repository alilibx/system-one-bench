"""Live race: TypeSafe Jev vs a challenger on the same labelled cases.

Challengers: OpenAI's GPT-6 Luna Decisions (OpenRouter Decisions API), and, with LAYA=1, Laya running locally.

Run:  OPENROUTER_API_KEY=<your key> .venv/bin/python server.py
      OPENROUTER_API_KEY=kerstel://global/OPENROUTER_API_KEY ks run -- .venv/bin/python server.py
Open: http://localhost:8061

Jev goes through OpenRouter too when OPENROUTER_API_KEY is set, so both API lanes share one gateway. Without it, Jev
falls back to the TypeSafe API (TYPESAFE_API_KEY).

GET /api/race?suite=ag_news&n=50&api_concurrency=1&challenger=openai streams Server-Sent Events:
  start  {n, suite, cases:[{id, suite, text}]}
  item   {model, i, id, ms, t, correct, total, pred, gold, cost}   (t = ms since the race began; cost in USD, null when local)
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
OPENAI_MODEL = os.environ.get("OPENAI_DECISIONS_MODEL", "openai/gpt-6-luna-decisions")


def key(name):
    """The env var's value, or "" when it is unset or is a Kerstel reference that was never resolved."""
    v = os.environ.get(name, "").strip()
    return "" if v.startswith("kerstel://") else v


if key("OPENROUTER_API_KEY"):
    JEV = {"url": DECISIONS_URL, "model": os.environ.get("JEV_MODEL", "typesafe/jev-1.13"), "key": "OPENROUTER_API_KEY"}
    JEV["where"] = f"openrouter.ai · {JEV['model']}"
else:
    JEV = {"url": os.environ.get("TYPESAFE_BASE_URL", "https://api.typesafe.ai") + "/v1/systemone",
           "model": os.environ.get("JEV_MODEL", "jev-1.13.0"), "key": "TYPESAFE_API_KEY"}
    JEV["where"] = f"api.typesafe.ai · {JEV['model']}"

CASES = [json.loads(l) for l in open(os.path.join(HERE, "cases.jsonl"))]
SUITES = sorted({c["suite"] for c in CASES})
ARABIC = [c for c in CASES if c["suite"] == "xnli_ar" or c["suite"].startswith("ar_")]

# API challengers answer over HTTP with the same request body as Jev.
CHALLENGERS = {"openai": {"label": "OpenAI Decisions", "where": f"openrouter.ai · {OPENAI_MODEL}", "api": True,
                          "url": DECISIONS_URL, "model": OPENAI_MODEL, "key": "OPENROUTER_API_KEY"}}

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
    CHALLENGERS["router"] = {"label": "Laya Router", "where": "local · Apple M5 GPU · Router (English + multilingual)"}
    # The Arabic fine-tunes are fixed checkpoints with no routing.
    for name, repo in {"laya-ara": "Wouze/laya-ara", "laya-ara-rag": "Wouze/laya-ara-rag"}.items():
        try:
            agents[name] = laya.load(repo, device=DEVICE)
            agents[name].system_one(ARABIC[0]["state"], ARABIC[0]["questions"])
            CHALLENGERS[name] = {"label": name, "where": f"local · Apple M5 GPU · {repo}"}
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


def laya_call(case, challenger):
    with gpu_lock:
        t0 = time.perf_counter()
        if challenger == "router":
            r = router.predict(case["state"], case["questions"])
            via = r["routing"]["model"]
        else:
            r = agents[challenger].system_one(case["state"], case["questions"])
            via = challenger
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


@app.get("/")
def index():
    return FileResponse(os.path.join(HERE, "race.html"))


@app.get("/api/suites")
def suites():
    counts = {s: sum(c["suite"] == s for c in CASES) for s in SUITES}
    public = {k: {"label": c["label"], "where": c["where"], "ready": bool(key(c["key"])) if c.get("api") else True}
              for k, c in CHALLENGERS.items()}
    return JSONResponse({"suites": counts, "arabic": len(ARABIC), "challengers": public,
                         "jev": {"where": JEV["where"], "ready": bool(key(JEV["key"]))}})


@app.get("/api/race")
async def race(suite: str = "mixed", n: int = Query(50, ge=1, le=5000), api_concurrency: int = Query(1, ge=1, le=16),
               seed: int = 0, challenger: str = "openai"):
    if challenger not in CHALLENGERS:
        return JSONResponse({"error": f"unknown challenger {challenger!r}; pick one of {sorted(CHALLENGERS)}"}, status_code=422)
    ch = CHALLENGERS[challenger]
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
                ans, ms, via, cost = await asyncio.to_thread(laya_call, c, challenger)
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
        start = {"n": len(picked), "suite": suite, "api_concurrency": api_concurrency,
                 "challenger": {"label": ch["label"], "where": ch["where"], "local": not ch.get("api")},
                 "cases": [{"id": c["id"], "suite": c["suite"], "text": preview(c["state"])} for c in picked]}
        yield f"event: start\ndata: {json.dumps(start, ensure_ascii=False)}\n\n"
        lanes = [run_api("challenger", ch) if ch.get("api") else run_laya("challenger"), run_api("jev", JEV)]
        tasks = [asyncio.create_task(lane) for lane in lanes]
        finished = 0
        try:
            while finished < 2:
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
