# System One Bench

A benchmark of decision models: `build_*.py` build the cases, `run_decisions.py` runs a model, `score.py` scores it into
`results.json` (text) or `results_vision.json` (images), and `site/` publishes them. See README.md for everything else.

Follow [CONTRIBUTING.md](CONTRIBUTING.md) for any run, data or results change. In short:

- Check the data with `python data_lock.py` before running a model. Never bypass a mismatch with `--unverified` for a
  run that goes on the board, and never edit the case files or committed images by hand.
- Changing any case, prompt, label, source or image is a new benchmark version: rewrite `data.lock.json` and rerun every
  model.
- Every model gets identical request bodies; failures count as wrong; add runs with `score.py --merge`.
- Keep API keys out of the repo.
