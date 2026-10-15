## What this changes

<!-- The models run, the data or code changed, and why. -->

## Benchmark rules

See [CONTRIBUTING.md](../blob/main/CONTRIBUTING.md). Tick what applies; explain anything you can't tick.

- [ ] `python data_lock.py` says both case files match, and every new run in `results.json` / `results_vision.json` carries the frozen `data_sha256`.
- [ ] No run used `--unverified`.
- [ ] Every model got the same request bodies: no per-model prompts or tuning on test data.
- [ ] Failed calls count as wrong, and any model that rejected cases has a `note` in `site/meta.json` saying why.
- [ ] Latency was timed from the same machine as the board's (or the note says it wasn't).
- [ ] New runs were added with `score.py --merge`, so other runs are unchanged, and each has a `site/meta.json` entry.
- [ ] The README has a round section for this change.
- [ ] If the data changed: this is a new benchmark version, `data.lock.json` was rewritten, and every model was rerun.
