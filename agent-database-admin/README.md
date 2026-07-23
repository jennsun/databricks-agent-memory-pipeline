# agent-database-admin

Admin interface for curating the **agent-scoped (shared) memory** store used by
[`agent-stateful-example`](../agent-stateful-example). An agent developer or admin uses this app to
add, update, or delete the shared knowledge entries that all users benefit from — the read-only
"agent memory" layer.

Backed by the same [Lakebase](https://docs.databricks.com/en/lakebase/index.html) (managed Postgres)
memory store as the stateful agent.

See the [repository README](../README.md) for the full architecture overview.

## Run locally

```bash
uv run quickstart   # set up auth, MLflow experiment, and Lakebase
uv run start-app    # start backend + admin UI
```
