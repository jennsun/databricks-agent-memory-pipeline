# agent-stateful-example

Stateful LangGraph agent with dual-layer memory on Databricks:

- **User memory** (per-user, read/write) — preferences, facts, and context that persist across sessions.
- **Agent memory** (shared, read-only at runtime) — curated knowledge that applies to all users.

Both stores are backed by [Lakebase](https://docs.databricks.com/en/lakebase/index.html) (managed
Postgres): `AsyncDatabricksStore` for vector-searchable long-term memory and `AsyncCheckpointSaver`
for conversation state. The agent also has web search, Python execution, and a Genie space tool.

See the [repository README](../README.md) for the full architecture and the story behind the demo.

## Run locally

```bash
uv run quickstart   # set up auth, MLflow experiment, and Lakebase
uv run start-app    # start backend + chat UI
```
