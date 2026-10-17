# Security

## Reporting a vulnerability

Don't open a public issue. Report it privately through
[GitHub's private vulnerability reporting](https://github.com/alilibx/system-one-bench/security/advisories/new), with
what's affected and how to reproduce it. You'll get an answer within a week.

## What's in scope

- `server.py`, the live race server, which holds an API key in its process. The key must never reach the browser.
- The results site (`site/` and the pages `.github/scripts/build_pages.py` writes), including anything in
  `results.json`, `results_vision.json` or `site/meta.json` that could inject markup into it.
- The GitHub workflows in `.github/workflows/`.
- Any API key or other secret committed to the repo. If you find one, report it; don't use it.

Only the latest commit on `main` and the live site at [systemonebench.dev](https://systemonebench.dev/) are supported.

## Benchmark integrity

A score that looks wrong, a mislabelled case, or a run that broke the [benchmark rules](CONTRIBUTING.md) isn't a
security issue: [report it as a problem](https://github.com/alilibx/system-one-bench/issues/new?template=report-a-problem.yml)
so the fix is public.
