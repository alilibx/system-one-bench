"""Send every case to TypeSafe Jev. Resumable: skips ids already in the output file.

Usage: TYPESAFE_API_KEY=... python run_jev.py [--concurrency 8] [--sequential-latency 60]
The key is read from TYPESAFE_API_KEY, or from the file named by TYPESAFE_API_KEY_FILE.
"""
import argparse
import asyncio
import json
import os
import random
import time

import httpx

URL = os.environ.get("TYPESAFE_BASE_URL", "https://api.typesafe.ai") + "/v1/systemone"
MODEL = os.environ.get("JEV_MODEL", "jev-1.13.0")


def api_key():
    if os.environ.get("TYPESAFE_API_KEY"):
        return os.environ["TYPESAFE_API_KEY"].strip()
    with open(os.path.expanduser(os.environ["TYPESAFE_API_KEY_FILE"])) as f:
        return f.read().strip()


async def call(client, case):
    body = {"model": MODEL, "state": case["state"], "questions": case["questions"]}
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
                return {"id": case["id"], "answers": d["answers"], "usage": d.get("usage"), "model": d.get("model"), "ms": ms}
            if r.status_code not in (429, 500, 502, 503, 529):
                return {"id": case["id"], "error": f"{r.status_code} {r.text[:300]}", "ms": ms}
            err = f"{r.status_code}"
        await asyncio.sleep(min(30, 2 ** attempt) + random.random())
    return {"id": case["id"], "error": f"gave up: {err}"}


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--concurrency", type=int, default=8)
    ap.add_argument("--sequential-latency", type=int, default=60, help="single-question cases timed one at a time")
    ap.add_argument("--out", default="preds_jev.jsonl")
    a = ap.parse_args()
    cases = [json.loads(l) for l in open("cases.jsonl")]
    done = set()
    if os.path.exists(a.out):
        done = {json.loads(l)["id"] for l in open(a.out) if "error" not in json.loads(l)}
    todo = [c for c in cases if c["id"] not in done]
    headers = {"Authorization": f"Bearer {api_key()}"}
    async with httpx.AsyncClient(headers=headers, timeout=60) as client:
        # Latency probe: one request in flight at a time, like a single interactive caller.
        if a.sequential_latency and not os.path.exists("latency_jev.jsonl"):
            probe = [c for c in cases if c["suite"] == "ag_news"][: a.sequential_latency]
            probe += [c for c in cases if c["suite"] == "typed-decisions"][: a.sequential_latency // 2]
            await call(client, probe[0])  # warm the connection
            with open("latency_jev.jsonl", "w") as f:
                for c in probe:
                    r = await call(client, c)
                    f.write(json.dumps({"id": c["id"], "suite": c["suite"], "nq": len(c["questions"]), "ms": r.get("ms"),
                                        "error": r.get("error")}) + "\n")
            print("latency probe done")
        sem = asyncio.Semaphore(a.concurrency)
        n = 0

        async def one(c):
            async with sem:
                return await call(client, c)

        with open(a.out, "a") as f:
            for fut in asyncio.as_completed([one(c) for c in todo]):
                r = await fut
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
                f.flush()
                n += 1
                if n % 100 == 0:
                    print(n, "/", len(todo), flush=True)
    print("done", n)


asyncio.run(main())
