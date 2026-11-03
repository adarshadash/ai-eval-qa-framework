# .github/ — CI/CD

`workflows/evals.yml` runs two jobs:

| Job | Trigger | Does |
|---|---|---|
| `mock-evals` | Every pull request and push to `main` | Unit tests, per-case eval gates, runner with baseline check. Blocks the merge on failure. Free, needs no secrets. |
| `live-evals` | Nightly 01:30 UTC and manual "Run workflow" | Same suites against the real Claude API to catch model drift and prompt regressions. |

Both publish `report.md` to the job summary and upload `reports/` as an artifact.

## Enable live evals

Add `ANTHROPIC_API_KEY` under **Settings → Secrets and variables → Actions → New repository secret**. Before the first scheduled live run, create a live baseline locally with `EVAL_MODE=live python -m evals.runner --update-baseline` and commit it.
