"""Send every extraction case to an extraction model, or to an LLM prompted for the same fields. Resumable.

Extraction models get the case's {state, fields} body as it is, byte for byte the same for every one of them:
  --provider cloudflare   Clairic on Workers AI (CLOUDFLARE_ACCOUNT_ID and CLOUDFLARE_API_TOKEN)
  --url                   any other endpoint that takes the body, such as a local server (adds "model" with --model)
LLMs get the same {state, fields} rendered into one fixed prompt (PROMPT below), with a JSON schema built from the
fields, through OpenRouter's chat completions API (OPENROUTER_API_KEY):
  --llm openai/gpt-6-luna

Usage: CLOUDFLARE_API_TOKEN=... CLOUDFLARE_ACCOUNT_ID=... python run_extraction.py --provider cloudflare --name clairic
       OPENROUTER_API_KEY=... python run_extraction.py --llm qwen/qwen3.5-9b --name llm_qwen35_9b
Writes extraction/preds_<name>.jsonl. Every answer is kept as the model returned it; score_extraction.py reads them.

The cases must match data.lock.json; --unverified skips that check for a run that isn't for the board.
"""
import argparse
import asyncio
import hashlib
import json
import os
import random
import re
import time

import httpx

import data_lock

# Vendors' extraction endpoints: (url, model, env vars holding the key and the URL's fields, USD per 1M input tokens).
PROVIDERS = {
    "cloudflare": ("https://api.cloudflare.com/client/v4/accounts/{CLOUDFLARE_ACCOUNT_ID}/ai/run/@cf/cloudflare/clairic",
                   "@cf/cloudflare/clairic", "CLOUDFLARE_API_TOKEN", ["CLOUDFLARE_ACCOUNT_ID"], 0.10),
}
CHAT_URL = os.environ.get("OPENROUTER_BASE_URL", "https://openrouter.ai") + "/api/v1/chat/completions"

# The one prompt every LLM gets. Changing it changes what every LLM score means: rerun every LLM.
PROMPT = """You read an input and fill in typed fields from it. Answer every field in the JSON schema you are given.

Field types:
- text: copy the shortest span of the input that answers it. If the input doesn't contain the answer, value is null.
- number: a plain number, with no units, currency signs or thousands separators. Work it out from the input if you have to.
- date: YYYY-MM-DD.
- list: every item that answers it, each copied from the input. An empty list if there are none.
- estimate: your best guess of a value the input doesn't state, between the field's min and max: median, and p10 and p90
  so the true value has an 80% chance of falling between them.
- forecast: the next `steps` values of the series in the input, each with a median and an 80% interval (p10, p90).

For text, number, date and list fields, p_null is the probability that the input doesn't contain the value, and
confidence is the probability that your value is right."""
PROMPT_SHA = hashlib.sha256(PROMPT.encode()).hexdigest()


def user_message(case):
    return f"Input:\n{case['state']}\n\nFields:\n{json.dumps(case['fields'], ensure_ascii=False, indent=1)}"


def answer_schema(field):
    t = field["type"]
    if t == "estimate":
        props = {k: {"type": "number"} for k in ("p10", "median", "p90")}
    elif t == "forecast":
        props = {k: {"type": "array", "items": {"type": "number"}} for k in ("p10", "median", "p90")}
    else:
        value = {"text": {"type": ["string", "null"]}, "number": {"type": ["number", "null"]},
                 "date": {"type": ["string", "null"]}, "list": {"type": "array", "items": {"type": "string"}}}[t]
        props = {"value": value, "p_null": {"type": "number"}, "confidence": {"type": "number"}}
    return {"type": "object", "properties": props, "required": list(props), "additionalProperties": False}


def schema(case):
    f = case["fields"]
    return {"type": "object", "properties": {k: answer_schema(v) for k, v in f.items()}, "required": list(f),
            "additionalProperties": False}


def parse_json(text):
    """The JSON object in a reply, also when a model wraps it in a code fence or adds a sentence around it."""
    try:
        return json.loads(text)
    except (TypeError, json.JSONDecodeError):
        m = re.search(r"\{.*\}", text or "", re.S)
        return json.loads(m.group()) if m else None


def priced(usage, per_m):
    if not usage or usage.get("cost") is not None:
        return usage
    tokens = usage.get("input_tokens", usage.get("prompt_tokens"))
    if per_m and tokens is not None:
        return {**usage, "cost": tokens * per_m / 1e6, "cost_from_list_price": True}
    return usage


async def post(client, url, body, case):
    """POST with retries on transient errors; (response JSON, ms) or (None, error record)."""
    for attempt in range(8):
        t0 = time.perf_counter()
        try:
            r = await client.post(url, json=body)
        except httpx.TransportError as e:
            err = repr(e)
        else:
            ms = (time.perf_counter() - t0) * 1000
            if r.status_code == 200:
                return r.json(), ms
            if r.status_code not in (408, 429, 500, 502, 503, 504, 529):
                return None, {"id": case["id"], "error": f"{r.status_code} {r.text[:300]}", "ms": ms}
            err = f"{r.status_code}"
        await asyncio.sleep(min(30, 2 ** attempt) + random.random())
    return None, {"id": case["id"], "error": f"gave up: {err}"}


async def extract(client, url, model, case, per_m):
    body = {"state": case["state"], "fields": case["fields"]}
    if model:
        body = {"model": model, **body}
    d, ms = await post(client, url, body, case)
    if d is None:
        return ms
    d = d.get("result", d)  # Workers AI wraps the model's reply in {"result": ..., "success": true}
    return {"id": case["id"], "answers": d.get("answers", d.get("fields")), "usage": priced(d.get("usage"), per_m), "ms": ms}


async def chat(client, model, case, a):
    body = {"model": model, "temperature": 0, "max_tokens": a.max_tokens, "usage": {"include": True},
            "messages": [{"role": "system", "content": PROMPT}, {"role": "user", "content": user_message(case)}],
            "response_format": {"type": "json_schema", "json_schema": {"name": "fields", "strict": True, "schema": schema(case)}}}
    if a.reasoning_effort:
        body["reasoning"] = {"effort": a.reasoning_effort}
    d, ms = await post(client, CHAT_URL, body, case)
    if d is None:
        return ms
    msg = (d.get("choices") or [{}])[0].get("message") or {}
    answers = parse_json(msg.get("content"))
    rec = {"id": case["id"], "answers": answers, "usage": d.get("usage"), "model": d.get("model"),
           "provider": d.get("provider"), "ms": ms, "prompt_sha256": PROMPT_SHA}
    if not isinstance(answers, dict):
        rec["error"] = f"unreadable reply: {str(msg.get('content'))[:300]}"
    return rec


async def main():
    ap = argparse.ArgumentParser()
    target = ap.add_mutually_exclusive_group(required=True)
    target.add_argument("--provider", choices=PROVIDERS, help="an extraction model on its vendor's API")
    target.add_argument("--url", help="any endpoint that takes the {state, fields} body, e.g. a local server")
    target.add_argument("--llm", metavar="MODEL", help="an OpenRouter chat model, prompted with PROMPT")
    ap.add_argument("--model", help="model name sent in the body with --url")
    ap.add_argument("--key-env", help="env var holding the API key with --url; empty for none")
    ap.add_argument("--name", required=True, help="short name for the output file")
    ap.add_argument("--cases", default="extraction_cases.jsonl")
    ap.add_argument("--suites", help="comma-separated suites to run (default: all)")
    ap.add_argument("--out-dir", default="extraction")
    ap.add_argument("--unverified", action="store_true", help="run even if the cases don't match data.lock.json")
    ap.add_argument("--concurrency", type=int, default=8)
    ap.add_argument("--max-tokens", type=int, default=4096, help="LLM output limit; raise it for reasoning models")
    ap.add_argument("--reasoning-effort", help="LLM reasoning effort, passed as OpenRouter's reasoning.effort")
    a = ap.parse_args()

    per_m, url, model = None, None, None
    if a.provider:
        url, _, key_env, url_envs, per_m = PROVIDERS[a.provider]  # the model is in the URL
        missing = [e for e in [key_env, *url_envs] if not os.environ.get(e)]
        if missing:
            ap.error(f"set {', '.join(missing)}")
        url = url.format(**{e: os.environ[e].strip() for e in url_envs})
    elif a.url:
        url, model, key_env = a.url, a.model, a.key_env or ""
    else:
        key_env = "OPENROUTER_API_KEY"
    if key_env and not os.environ.get(key_env):
        ap.error(f"{key_env} is not set: export it, or wrap the command in `ks run --` with a kerstel:// reference")

    cases = [json.loads(l) for l in open(a.cases)]
    if a.suites:
        cases = [c for c in cases if c["suite"] in a.suites.split(",")]
    problems = [] if a.unverified else data_lock.check(a.cases, {c["suite"] for c in cases})
    if problems:
        ap.error(f"{a.cases} doesn't match data.lock.json, so results wouldn't compare with the board:\n  " + "\n  ".join(problems)
                 + "\nRebuild it with build_extraction.py, or pass --unverified for a run that isn't for the board.")
    os.makedirs(a.out_dir, exist_ok=True)
    out = os.path.join(a.out_dir, f"preds_{a.name}.jsonl")
    done = set()
    if os.path.exists(out):
        done = {json.loads(l)["id"] for l in open(out) if "error" not in json.loads(l)}
    todo = [c for c in cases if c["id"] not in done]
    headers = {"Authorization": f"Bearer {os.environ[key_env].strip()}"} if key_env else {}
    sem = asyncio.Semaphore(a.concurrency)

    async with httpx.AsyncClient(headers=headers, timeout=120) as client:
        async def one(c):
            async with sem:
                return await (chat(client, a.llm, c, a) if a.llm else extract(client, url, model, c, per_m))

        n = 0
        with open(out, "a") as f:
            for fut in asyncio.as_completed([one(c) for c in todo]):
                f.write(json.dumps(await fut, ensure_ascii=False) + "\n")
                f.flush()
                n += 1
                if n % 100 == 0:
                    print(n, "/", len(todo), flush=True)
    print("done", n)


if __name__ == "__main__":
    asyncio.run(main())
