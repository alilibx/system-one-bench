"""Run every case through Laya locally. Resumable: skips ids already in the output file.

Variants:
  router  Router() as the model card recommends: auto language routing, shipped temperatures.
  tuned   Same, with head_max_len/max_len raised as the card advises for many-option questions.
  td      The typed-decisions checkpoint, typed-decisions suite only. A specialist fine-tuned
          on that benchmark's train split, so it is not comparable to a zero-shot generalist.
  ara     Wouze/laya-ara (laya-multilingual fine-tuned for Arabic NLU), Arabic suites only.
  ara-rag Wouze/laya-ara-rag (laya-multilingual fine-tuned for Arabic reranking), Arabic suites only.

Usage: python run_laya.py --variant router [--device mps]
"""
import argparse
import json
import os
import time

os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import laya  # noqa: E402
from laya import Router  # noqa: E402

ARABIC = {"xnli_ar"}  # plus every ar_* suite
SINGLE = {"ara": "Wouze/laya-ara", "ara-rag": "Wouze/laya-ara-rag"}

ap = argparse.ArgumentParser()
ap.add_argument("--variant", choices=["router", "tuned", "td", *SINGLE], default="router")
ap.add_argument("--device", default="mps")
a = ap.parse_args()
out = f"preds_laya_{a.variant}.jsonl"

cases = [json.loads(l) for l in open("cases.jsonl")]
if a.variant == "td":
    cases = [c for c in cases if c["suite"] == "typed-decisions"]
if a.variant in SINGLE:
    cases = [c for c in cases if c["suite"] in ARABIC or c["suite"].startswith("ar_")]
done = set()
if os.path.exists(out):
    done = {json.loads(l)["id"] for l in open(out)}

if a.variant in SINGLE:
    # One fixed checkpoint, no routing. Shaped like Router.predict so the loop below is shared.
    class _Single:
        def __init__(self, repo):
            self.agent = laya.load(repo, device=a.device)

        def predict(self, state, questions):
            r = self.agent.system_one(state, questions)
            r["routing"] = {"model": a.variant}
            return r

    router, names = _Single(SINGLE[a.variant]), []
else:
    router = Router(device=a.device, max_loaded=3)
    names = ["typed-decisions"] if a.variant == "td" else ["english", "multilingual"]
    router.preload(names)
if a.variant == "tuned":
    for n in names:
        router.load(n).cfg.update(head_max_len=512, max_len=1024)
kw = {"model": "typed-decisions"} if a.variant == "td" else {}

for c in cases[:3]:  # warm up kernels on both checkpoints before timing anything
    router.predict(c["state"], c["questions"], **kw)
router.predict(cases[-1]["state"], cases[-1]["questions"], **kw)

with open(out, "a") as f:
    for i, c in enumerate(cases):
        if c["id"] in done:
            continue
        t0 = time.perf_counter()
        try:
            r = router.predict(c["state"], c["questions"], **kw)
            rec = {"id": c["id"], "answers": r["answers"], "usage": r.get("usage"), "routed_to": r["routing"]["model"]}
        except Exception as e:  # counted as a failed decision by the scorer
            rec = {"id": c["id"], "error": repr(e)[:300]}
        rec["ms"] = (time.perf_counter() - t0) * 1000
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        f.flush()
        if i % 200 == 0:
            print(f"{i}/{len(cases)}", flush=True)
print("done")
