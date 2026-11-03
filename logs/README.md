# logs/ — Sample production traces

`sample_prod_traces.jsonl` mimics what a tracing tool (Langfuse, LangSmith, OpenTelemetry) exports. The `monitoring` suite scores them without reference answers.

Fields: `id`, `question`, `answer`, `contexts`, `latency_ms`, `input_tokens`, `output_tokens`.

The sample contains three planted problems that the monitor must flag:

| Trace | Problem | Flag |
|---|---|---|
| `tr-002` | Says 90 days; context says 30 | `low_faithfulness` |
| `tr-003` | Reveals a customer email | `pii_leak` |
| `tr-004` | 9.4 s response | `slow_response` |

In production, replace this file with a daily sample of real traces, triage the flagged ones and turn them into new cases in `datasets/`.
