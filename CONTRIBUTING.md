# Rules for System One Bench

System One Bench is only useful if every model is measured the same way, including models added months from now.
These rules apply to every run that goes on [systemonebench.dev](https://systemonebench.dev/), whoever adds it.
The [PR template](.github/pull_request_template.md) asks you to confirm each one.

## 1. Every model runs on the frozen data

- Build the data, then check it before any run:

  ```bash
  .venv/bin/pip install -r requirements-data.txt
  .venv/bin/python build_suites.py && .venv/bin/python build_arabic.py && .venv/bin/python build_images.py
  .venv/bin/python data_lock.py        # both files must say "matches"
  ```

- `run_decisions.py` and `score.py` refuse cases that don't match `data.lock.json`. Don't work around that:
  `--unverified` is for experiments only, and an unverified run never goes on the board.
- Every scored run records the data it ran on as `data_sha256` in `results.json` or `results_vision.json`.
- Don't edit `cases.jsonl`, `image_cases.jsonl` or the images by hand. They're rebuilt from pinned sources, and the
  400 rendered-text images under `images/img_sst2` and `images/img_ar_sentiment` are committed because they can't be
  redrawn byte for byte.

## 2. Changing the data makes a new benchmark version

Changing a case, a prompt, a label, a dataset, a source revision or an image changes what every score means. Only do it
on purpose, as a new version:

1. Change the build script, and pin any new source to an exact commit in its `REVISIONS`.
2. Rebuild, then freeze: `.venv/bin/python data_lock.py --write cases.jsonl image_cases.jsonl`.
3. Rerun every model on the board, since scores on different data don't compare.
4. Say so in the README round notes.

`.github/workflows/data.yml` rebuilds the data from scratch on every pull request that touches it, and every Monday,
and fails if anything no longer matches the lock. A red run there means the data moved: find out why before running
any model.

## 3. Every model gets the same request

- Every model receives byte-identical `state` and `questions` bodies. No per-model prompts, instructions, option
  orders or retries that change the content.
- Nothing is tuned on the test data.
- A failed call or unreadable answer counts as wrong. Retry only transient errors (408, 429, 5xx), as
  `run_decisions.py` does; if a model rejects a case outright (too many options, an unsupported input), it stays wrong
  and the model's `note` in `site/meta.json` says why.
- A model that doesn't accept images has no vision score. List it under `vision.no_image` in `site/meta.json`; don't
  score it as zero. Box questions run only on models that support them (`--suites` leaves them out for the rest).

## 4. Latency is timed from one machine

Latency includes the network between the caller and the API, so it only compares between runs from the same machine.
The board's latency comes from one re-run of every API model on 15 October 2026 (see the README). Time a new model
from that same kind of machine, 8 calls in flight, and set it with `set_latency.py`. If you can't, say so in its
`note` and in the round notes.

## 5. Results go in without disturbing other runs

- Add a run with `score.py --merge preds_<name>.jsonl`, which keeps every other run in `results.json` unchanged. For
  the vision track, score into `results_vision.json` with `--cases image_cases.jsonl`.
- Add the model to `site/meta.json` with its label, maker, slug, run date and any note. Runs without an entry don't
  show on the site.
- Add a round section to the README with what ran, how, and what failed.

## 6. Only scores measured here go on the site

Vendors publish their own numbers for several models. None go on the site until the model has run here under these
rules.

## 7. Keys stay out of the repo

API keys live in the environment or a secrets manager (`kerstel://` references with `ks run`), never in a file in
this repo.
