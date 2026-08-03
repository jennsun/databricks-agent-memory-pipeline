# agent-stateless-example

Stateless LangGraph agent on Databricks with web search and code execution. It answers each
question accurately but treats every conversation as brand new — it has no memory of past sessions,
user preferences, or organizational knowledge.

This is the baseline counterpart to [`agent-stateful-example`](../agent-stateful-example), which adds
dual-layer [Lakebase](https://docs.databricks.com/en/lakebase/index.html) memory. Compare the two to
see how durable memory changes the agent's behavior.

See the [repository README](../README.md) for the full architecture overview.

## Run locally

```bash
uv run quickstart   # set up auth and MLflow experiment
uv run start-app    # start backend + chat UI
```
