# datasets/ — Versioned eval data

JSONL, one case per line. Lines starting with `//` are comments. IDs must be unique. `tests/test_datasets.py` checks required fields, valid doc and tool references, and label balance.

| File | Suite | Key fields |
|---|---|---|
| `golden_qa.jsonl` | golden | `question`, `reference`, `must_include`, `must_not_include`, `tags` (`abstention` = bot must decline) |
| `rag_cases.jsonl` | rag | `question`, `expected_doc_ids`, `reference` |
| `agent_tool_cases.jsonl` | agent | `question`, `expected_tools` (ordered), `expected_args`, `forbidden_tools`, `max_steps`, `must_include` |
| `multi_turn.jsonl` | multi_turn | `turns[]` each with `user`, `must_include`, optional `expected_tools` / `forbidden_tools` |
| `safety_redteam.jsonl` | safety | `prompt`, `category`, `checks` (`refuse`, `no_secret_leak`, `no_pii`, `no_forbidden_tools`, `no_toxicity`) |
| `structured_output.jsonl` | structured_output | `ticket`, `expected` field labels |
| `consistency.jsonl` | consistency | `question`, `must_include`, `reference` (run N times) |
| `judge_calibration.jsonl` | judge_calibration | `question`, `reference`, `answer`, `human_label` (PASS/FAIL) |

## Good practice

- **Every production bug becomes a case.** That's how the golden set stays relevant.
- Include **negative cases**: questions the bot shouldn't answer, tools it mustn't call.
- Keep the calibration set **balanced** (30–70% PASS) or agreement is meaningless.
- Review dataset changes in pull requests like code. A wrong reference answer silently passes bad output.

## Example

```json
{"id": "gq-return-fee", "question": "Do I pay for return shipping?", "reference": "Return shipping is free for damaged or incorrect items.", "must_include": ["free"], "must_not_include": [], "tags": ["returns"]}
```
