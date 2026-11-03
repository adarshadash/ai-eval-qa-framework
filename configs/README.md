# configs/ — Eval configuration

`eval_config.yaml` holds every model choice, threshold and budget that gates a release, versioned with the code so a threshold change is reviewed like any other change.

| Section | Controls |
|---|---|
| `mode` | `mock` (offline) or `live` (real API). Overridden by `EVAL_MODE`. |
| `models` | SUT model, judge model, temperature, max tokens, timeout, retries |
| `pricing_per_million_tokens` | Cost calculation. **Example values** — update from the provider's pricing page. |
| `agent.max_steps` | Hard cap on the tool loop |
| `retrieval` | `top_k` and `min_score` (below it the bot must abstain) |
| `thresholds.<suite>` | Per-case pass criteria and `suite_pass_rate` |
| `thresholds.regression.tolerance` | Allowed score drop vs baseline |
| `monitoring` | Alert thresholds for production traces |

## Multiple configs

Use another file for a different environment, for example stricter thresholds before a release:

```bash
EVAL_CONFIG=configs/release.yaml python -m evals.runner
```
