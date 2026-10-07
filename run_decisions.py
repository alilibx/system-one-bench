"""Send every case to a model on OpenRouter's Decisions API. Resumable: skips ids already in the output file.

The Decisions API serves TypeSafe Jev and OpenAI's GPT-6 Luna Decisions behind one endpoint with one request and
response shape, so both models go through the same gateway.

Usage: OPENROUTER_API_KEY=... python run_decisions.py --model openai/gpt-6-luna-decisions --name luna
       OPENROUTER_API_KEY=kerstel://global/OPENROUTER_API_KEY ks run -- python run_decisions.py --model typesafe/jev-1.13 --name or_jev
Writes preds_<name>.jsonl and, once, latency_<name>.jsonl (one request in flight at a time).
"""
import argparse
import asyncio
import json
import os
import random
import time

import httpx

URL = os.environ.get("OPENROUTER_BASE_URL", "https://openrouter.ai") + "/api/alpha/decisions"


async def call(client, model, case):
    body = {"model": model, "state": case["state"], "questions": case["questions"]}
    for attempt in range(8):
        t0 = time.perf_counter()
        try:
            r = await client.post(URL, json=body)
        except httpx.TransportError as e:
            err = repr(e)
        else:
            ms = (time.perf_counter() - t0) * 1000
            if r.status_code == 200:
                d = r.json()
                return {"id": case["id"], "answers": d["answers"], "usage": d.get("usage"), "model": d.get("model"),
                        "provider": d.get("provider"), "ms": ms}
            if r.status_code not in (408, 429, 500, 502, 503, 529):
                return {"id": case["id"], "error": f"{r.status_code} {r.text[:300]}", "ms": ms}
            err = f"{r.status_code}"
        await asyncio.sleep(min(30, 2 ** attempt) + random.random())
    return {"id": case["id"], "error": f"gave up: {err}"}


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="OpenRouter slug, e.g. openai/gpt-6-luna-decisions or typesafe/jev-1.13")
    ap.add_argument("--name", required=True, help="short name for the output files")
    ap.add_argument("--concurrency", type=int, default=8)
    ap.add_argument("--sequential-latency", type=int, default=60, help="single-question cases timed one at a time")
    a = ap.parse_args()
    out, lat_path = f"preds_{a.name}.jsonl", f"latency_{a.name}.jsonl"
    cases = [json.loads(l) for l in open("cases.jsonl")]
    done = set()
    if os.path.exists(out):
        done = {json.loads(l)["id"] for l in open(out) if "error" not in json.loads(l)}
    todo = [c for c in cases if c["id"] not in done]
    headers = {"Authorization": f"Bearer {os.environ['OPENROUTER_API_KEY'].strip()}"}
    async with httpx.AsyncClient(headers=headers, timeout=60) as client:
        # Latency probe: one request in flight at a time, like a single interactive caller.
        if a.sequential_latency and not os.path.exists(lat_path):
            probe = [c for c in cases if c["suite"] == "ag_news"][: a.sequential_latency]
            probe += [c for c in cases if c["suite"] == "typed-decisions"][: a.sequential_latency // 2]
            await call(client, a.model, probe[0])  # warm the connection
            with open(lat_path, "w") as f:
                for c in probe:
                    r = await call(client, a.model, c)
                    f.write(json.dumps({"id": c["id"], "suite": c["suite"], "nq": len(c["questions"]), "ms": r.get("ms"),
                                        "error": r.get("error")}) + "\n")
                    f.flush()
            print("latency probe done")
        sem = asyncio.Semaphore(a.concurrency)
        n = 0

        async def one(c):
            async with sem:
                return await call(client, a.model, c)

        with open(out, "a") as f:
            for fut in asyncio.as_completed([one(c) for c in todo]):
                r = await fut
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
                f.flush()
                n += 1
                if n % 100 == 0:
                    print(n, "/", len(todo), flush=True)
    print("done", n)


asyncio.run(main())
