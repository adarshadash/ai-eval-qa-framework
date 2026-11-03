"""Sample system under test (SUT): 'ShopBot', a customer-support AI agent.

It has the three things most production LLM apps have:
  * RAG     - retrieves policy documents from a knowledge base
  * Tools   - looks up orders and issues refunds (an agent with side effects)
  * JSON    - classifies support tickets into a structured schema

Replace this package with your own system and point `evals/adapter.py` at it.
"""
