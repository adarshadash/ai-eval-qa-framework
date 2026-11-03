# evals/metrics/ — Metric library

Pure functions with no model calls and no app knowledge. Every metric is unit-tested in `tests/test_metrics_unit.py` to prove it **passes good output and fails known-bad output**.

| File | Functions | Use for |
|---|---|---|
| `deterministic.py` | `exact_match`, `contains_all`, `contains_none`, `regex_match`, `word_count_within`, `is_valid_json`, `json_schema_errors` | Hard requirements. Run these first. |
| `similarity.py` | `token_f1`, `rouge_l`, `cosine_bow` | Closeness to a reference answer |
| `rag.py` | `context_recall`, `context_precision` (rank-aware), `precision_at_k`, `mrr`, `hit_rate`, `faithfulness`, `answer_relevance` | Retrieval quality and grounding |
| `agent.py` | `trajectory_match` (exact / in_order / any_order), `tool_args_match`, `forbidden_tools_called`, `tool_error_count` | Agent behaviour, not just its final answer |
| `safety.py` | `detect_pii` (email, phone, Luhn-checked cards, SSN), `is_refusal`, `leaks_secret` (canary), `toxicity_hits` | Red-team and compliance |
| `consistency.py` | `pass_rate`, `pass_at_k`, `pass_hat_k`, `semantic_consistency` | Non-deterministic outputs |
| `calibration.py` | `agreement`, `cohen_kappa` | Trusting an LLM judge |
| `performance.py` | `percentile`, `cost_usd` | Latency and cost budgets |
| `text.py` | `tokens`, `sentences`, `numbers`, `has_negation` | Shared normalisation |

## Notes

- **Faithfulness** marks a sentence unsupported unless most of its words *and every number in it* appear in one retrieved context. This catches "right sentence, wrong figure" hallucinations.
- **Negations are not stopwords**, so "covered" and "not covered" never look the same.
- Lexical metrics are fast and deterministic but miss paraphrases. Add embedding similarity behind the same function signatures if you need it.
