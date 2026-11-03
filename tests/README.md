# tests/ — pytest suites

| File | What it tests |
|---|---|
| `test_evals.py` | One test per dataset row for every suite: the per-case release gates. |
| `test_metrics_unit.py` | "Testing the tester": each metric must pass good output and **fail known-bad output** (wrong numbers, invented claims, PII, wrong tool order). |
| `test_datasets.py` | Eval data quality: required fields, unique IDs, references to real docs and tools, balanced calibration labels. |
| `conftest.py` | Fixtures (fresh agent per test so refunds never leak between cases) and the run header showing mode and models. |

## Run

```bash
pytest                    # all
pytest -m unit            # metrics + datasets only (no SUT calls)
pytest -m safety          # one suite
pytest -m "not slow"      # skip repeated-run suites (cheaper in live mode)
pytest -k refund -v       # by name
```

Markers: `unit`, `golden`, `rag`, `agent`, `multi_turn`, `safety`, `structured`, `consistency`, `judge`, `performance`, `monitoring`, `slow` (defined in `pytest.ini`).

A failing case prints its failed checks, scores and the actual answer, so you can debug without re-running.
