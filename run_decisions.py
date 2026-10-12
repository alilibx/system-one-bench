"""Send every case to a decision model. Resumable: skips ids already in the output file.

By default it calls OpenRouter's Decisions API, which serves most decision models behind one endpoint with one request
and response shape. --provider points it at a vendor's own System One endpoint instead (same request body), and --url
at any other one, such as a local server.

Usage: OPENROUTER_API_KEY=... python run_decisions.py --model openai/gpt-6-luna-decisions --name luna
       OPENROUTER_API_KEY=kerstel://global/OPENROUTER_API_KEY ks run -- python run_decisions.py --model typesafe/jev-1.13 --name or_jev
       CELERIS_API_KEY=kerstel://global/CELERIS_API_KEY ks run -- python run_decisions.py --provider celeris --name celeris
       python run_decisions.py --url http://localhost:8000/v1/systemone --key-env "" --model kev-9b --name kev_9b
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
# Vendors' own System One endpoints: (url, model, env var holding the key, auth header). Same request body as above.
PROVIDERS = {
    "fastino": ("https://api.fastino.ai/v1/systemone", "fastino/GLiDE", "FASTINO_API_KEY", "x-api-key"),
    "celeris": ("https://inference.celeris.ai/celeris-1-decision/v1/systemone", "celeris-1-decision", "CELERIS_API_KEY", "bearer"),
    "hanzo": ("https://api.hanzo.ai/v1/decisions", "kai", "HANZO_API_KEY", "bearer"),
    "meragpt": ("https://meragpt.com/v1/systemone", "decider-1", "MERAGPT_API_KEY", "bearer"),
}


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
    global URL
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", help="model id, e.g. openai/gpt-6-luna-decisions; defaults to the provider's model")
    ap.add_argument("--provider", choices=PROVIDERS, help="call a vendor's own endpoint instead of OpenRouter")
    ap.add_argument("--url", help="any endpoint that takes the System One request body, e.g. a local server")
    ap.add_argument("--key-env", help="env var holding the API key; empty for none (default OPENROUTER_API_KEY)")
    ap.add_argument("--auth", choices=["bearer", "x-api-key"], help="how the key is sent (default bearer)")
    ap.add_argument("--name", required=True, help="short name for the output files")
    ap.add_argument("--concurrency", type=int, default=8)
    ap.add_argument("--sequential-latency", type=int, default=60, help="single-question cases timed one at a time")
    a = ap.parse_args()
    url, model, key_env, auth = PROVIDERS.get(a.provider, (URL, None, "OPENROUTER_API_KEY", "bearer"))
    URL, a.model, key_env, auth = a.url or url, a.model or model, a.key_env if a.key_env is not None else key_env, a.auth or auth
    if not a.model:
        ap.error("--model is required unless --provider sets it")
    out, lat_path = f"preds_{a.name}.jsonl", f"latency_{a.name}.jsonl"
    cases = [json.loads(l) for l in open("cases.jsonl")]
    done = set()
    if os.path.exists(out):
        done = {json.loads(l)["id"] for l in open(out) if "error" not in json.loads(l)}
    todo = [c for c in cases if c["id"] not in done]
    headers = {}
    if key_env:
        k = os.environ[key_env].strip()
        headers = {"X-API-Key": k} if auth == "x-api-key" else {"Authorization": f"Bearer {k}"}
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
