"""Live race: Laya (local, this Mac) vs TypeSafe Jev (API) on the same labelled cases.

Run:  TYPESAFE_API_KEY=<your key> .venv/bin/python server.py
Open: http://localhost:8765

GET /api/race?suite=ag_news&n=50&jev_concurrency=1 streams Server-Sent Events:
  start  {n, suite, cases:[{id, suite, text}]}
  item   {model, i, id, ms, t, correct, total, pred, gold}   (t = ms since the race began)
  done   {model, t}
The API key stays in this process; the browser never sees it.
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
import laya  # noqa: E402
from laya import Router  # noqa: E402

from score import gold_idx, options, pred_dist  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
JEV_URL = os.environ.get("TYPESAFE_BASE_URL", "https://api.typesafe.ai") + "/v1/systemone"
JEV_MODEL = os.environ.get("JEV_MODEL", "jev-1.13.0")

CASES = [json.loads(l) for l in open(os.path.join(HERE, "cases.jsonl"))]
SUITES = sorted({c["suite"] for c in CASES})

ARABIC = [c for c in CASES if c["suite"] == "xnli_ar" or c["suite"].startswith("ar_")]
DEVICE = os.environ.get("LAYA_DEVICE", "mps")

router = Router(device=DEVICE, max_loaded=3)
router.preload(["english", "multilingual"])
for c in CASES[:2] + ARABIC[:2]:  # warm up GPU kernels on both checkpoints
    router.predict(c["state"], c["questions"])

# Challengers for the Laya lane. The Arabic fine-tunes are fixed checkpoints with no routing.
CHALLENGERS = {"router": {"label": "Laya Router", "where": "local · Apple M5 GPU · Router (English + multilingual)"}}
ARA = {"laya-ara": "Wouze/laya-ara", "laya-ara-rag": "Wouze/laya-ara-rag"}
agents = {}
for key, repo in ARA.items():
    try:
        agents[key] = laya.load(repo, device=DEVICE)
        agents[key].system_one(ARABIC[0]["state"], ARABIC[0]["questions"])
        CHALLENGERS[key] = {"label": key, "where": f"local · Apple M5 GPU · {repo}"}
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
        return r["answers"], (time.perf_counter() - t0) * 1000, via


async def jev_call(client, case):
    body = {"model": JEV_MODEL, "state": case["state"], "questions": case["questions"]}
    for attempt in range(6):
        t0 = time.perf_counter()
        r = await client.post(JEV_URL, json=body)
        ms = (time.perf_counter() - t0) * 1000
        if r.status_code == 200:
            return r.json()["answers"], ms, JEV_MODEL
        if r.status_code not in (429, 500, 502, 503, 529):
            raise RuntimeError(f"Jev returned {r.status_code}: {r.text[:200]}")
        await asyncio.sleep(2 ** attempt)
    raise RuntimeError("Jev kept returning rate-limit or overload errors")


@app.get("/")
def index():
    return FileResponse(os.path.join(HERE, "race.html"))


@app.get("/api/suites")
def suites():
    counts = {s: sum(c["suite"] == s for c in CASES) for s in SUITES}
    return JSONResponse({"suites": counts, "arabic": len(ARABIC), "challengers": CHALLENGERS, "jev_ready": bool(os.environ.get("TYPESAFE_API_KEY", "").strip()) and
                         not os.environ["TYPESAFE_API_KEY"].startswith("kerstel://")})


@app.get("/api/race")
async def race(suite: str = "mixed", n: int = Query(50, ge=1, le=5000), jev_concurrency: int = Query(1, ge=1, le=16),
               seed: int = 0, challenger: str = "router"):
    if challenger not in CHALLENGERS:
        return JSONResponse({"error": f"unknown challenger {challenger!r}; pick one of {sorted(CHALLENGERS)}"}, status_code=422)
    pool = {"mixed": CASES, "arabic": ARABIC}.get(suite) or [c for c in CASES if c["suite"] == suite]
    pool = pool[:]
    random.Random(seed).shuffle(pool)
    picked = pool[:n]
    q: asyncio.Queue = asyncio.Queue()
    t_start = time.perf_counter()

    def emit(kind, data):
        q.put_nowait(f"event: {kind}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n")

    async def run_laya():
        for i, c in enumerate(picked):
            try:
                ans, ms, via = await asyncio.to_thread(laya_call, c, challenger)
                err = None
            except Exception as e:  # a failed call is graded as all-wrong
                ans, ms, via, err = None, 0.0, "error", repr(e)[:200]
            right, total, preds, golds = grade(c, ans)
            emit("item", {"model": "laya", "i": i, "id": c["id"], "ms": round(ms, 1), "t": round((time.perf_counter() - t_start) * 1000),
                          "correct": right, "total": total, "pred": preds, "gold": golds, "via": via, "error": err})
        emit("done", {"model": "laya", "t": round((time.perf_counter() - t_start) * 1000)})

    async def run_jev():
        sem = asyncio.Semaphore(jev_concurrency)
        headers = {"Authorization": f"Bearer {os.environ.get('TYPESAFE_API_KEY', '').strip()}"}
        async with httpx.AsyncClient(headers=headers, timeout=60) as client:
            async def one(i, c):
                async with sem:
                    try:
                        ans, ms, via = await jev_call(client, c)
                        err = None
                    except Exception as e:
                        ans, ms, via, err = None, 0.0, "error", repr(e)[:200]
                    right, total, preds, golds = grade(c, ans)
                    emit("item", {"model": "jev", "i": i, "id": c["id"], "ms": round(ms, 1),
                                  "t": round((time.perf_counter() - t_start) * 1000), "correct": right, "total": total,
                                  "pred": preds, "gold": golds, "via": via, "error": err})
            await asyncio.gather(*(one(i, c) for i, c in enumerate(picked)))
        emit("done", {"model": "jev", "t": round((time.perf_counter() - t_start) * 1000)})

    async def stream():
        yield f"event: start\ndata: {json.dumps({'n': len(picked), 'suite': suite, 'jev_concurrency': jev_concurrency, 'challenger': CHALLENGERS[challenger], 'cases': [{'id': c['id'], 'suite': c['suite'], 'text': preview(c['state'])} for c in picked]}, ensure_ascii=False)}\n\n"
        tasks = [asyncio.create_task(run_laya()), asyncio.create_task(run_jev())]
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
