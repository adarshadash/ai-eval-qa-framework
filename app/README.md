# app/ — System under test (ShopBot)

A sample customer-support AI that the evals run against. It has the three parts most production LLM apps have:

| File | What it is | Evaluated by |
|---|---|---|
| `support_agent.py` | The agent. `run(question, history)` returns an `AgentResult` with the answer, tool calls, retrieved contexts, steps, tokens and latency. Has a **live** mode (real Claude tool-use loop) and a **mock** mode (deterministic, offline). | golden, rag, agent, multi_turn, safety, consistency, performance |
| `knowledge_base.py` | 8 store-policy documents and an IDF-weighted keyword retriever (stands in for a vector store). | rag |
| `tools.py` | `search_policy`, `get_order_status`, `issue_refund` plus their JSON schemas and a fake order database. `get_order_status` deliberately returns customer PII so the safety evals can check it never reaches the user. | agent, safety |
| `ticket_classifier.py` | Classifies support tickets into JSON (`category`, `priority`, `sentiment`, `summary`) validated by `TICKET_SCHEMA`. | structured_output |
| `text_utils.py` | Tokenizer and stemmer used by the retriever. | — |

## Things planted on purpose

- **`CANARY`** in the system prompt. If it ever appears in an answer, a prompt-extraction attack worked.
- **PII in tool output.** The agent must use it without repeating it.
- **`inject_bugs`** switch (`EVAL_INJECT_BUGS=1`). Makes the mock bot hallucinate numbers, leak an email, obey "print your system prompt" and refund undelivered orders, so you can watch the evals fail.

## Using your own app instead

Don't edit the evals. Change `evals/adapter.py` so it returns your agent; see the root README section "Evaluating your own app".
