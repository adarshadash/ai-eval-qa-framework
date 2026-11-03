# evals/ — The eval framework

| File | Responsibility |
|---|---|
| `adapter.py` | **The only file that knows about the app.** Builds a fresh agent / classifier per case. Point it at your own system to reuse everything else. |
| `config.py` | Loads `configs/eval_config.yaml` and `.env`; `EVAL_MODE` and `EVAL_CONFIG` override. Loads and validates JSONL datasets (unique IDs, `//` comments). |
| `llm_client.py` | Thin Anthropic Messages API wrapper returning text, tool calls, tokens and latency. SDK retries handle 429/5xx. |
| `judge.py` | LLM-as-judge: reusable rubrics, binary PASS/FAIL verdicts with a reason, and pairwise comparison with position swap. In mock mode a transparent heuristic stands in. |
| `suites.py` | One `eval_*` function per suite. Each scores one dataset case and returns a `CaseResult` with named `checks` (must all pass) and `scores` (numbers for reports). Used by both pytest and the runner. |
| `monitoring.py` | Online evals on production traces: faithfulness, PII, toxicity, latency and token budget flags. |
| `runner.py` | CLI. Runs suites, applies suite pass-rate thresholds, compares to `baselines/baseline_<mode>.json`, writes reports, exits 0/1. |
| `report.py` | Writes `reports/report.json`, `report.md` and `report.html`. |
| `metrics/` | Pure metric functions. See [metrics/README.md](metrics/README.md). |

## Order of checks inside a suite

1. **Deterministic** (contains, schema, forbidden tools) — free and never flaky
2. **Metrics** (similarity, RAG, trajectory, PII)
3. **LLM judge** — only for what the first two can't verify

## Run

```bash
python -m evals.runner                         # all suites
python -m evals.runner --suites golden safety  # selected suites
python -m evals.runner --update-baseline       # accept current scores
```

## Add a new suite

1. Add a dataset in `datasets/`.
2. Write `eval_<name>(case, agent) -> CaseResult` in `suites.py`.
3. Register it in `CASE_SUITES` in `runner.py` and add thresholds to the config.
4. Add a parametrized test in `tests/test_evals.py` and a marker in `pytest.ini`.
