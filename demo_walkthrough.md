
Prior to running the curl commands below, set these environment variables:

```bash
export OAUTH_TOKEN=$(databricks auth token --host <your-workspace-url> | jq -r .access_token)
export STATELESS_AGENT_URL=<your-stateless-app-url>           # e.g. https://agent-stateless-example-<workspace-id>.<cloud>.databricksapps.com
export STATEFUL_AGENT_URL=<your-stateful-app-url>             # e.g. https://agent-stateful-example-<workspace-id>.<cloud>.databricksapps.com
```

## First — stateless agent

```bash
curl -X POST "$STATELESS_AGENT_URL/invocations" \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $OAUTH_TOKEN" \
  -d '{
    "input": [
      {"role": "user", "content": "Can you look through my company expense data and figure out the total employee spending on fitness reimbursements in 2026?"}
    ],
    "stream": false
  }'
```

The stateless agent will answer the question but it has no memory — re-running the same question in a new conversation produces an identical answer because nothing is remembered between sessions.

## Then — stateful agent

```bash
curl -X POST "$STATEFUL_AGENT_URL/invocations" \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $OAUTH_TOKEN" \
  -d '{
    "input": [
      {"role": "user", "content": "Can you look through my company expense data and figure out the total employee spending on fitness reimbursements in 2026?"}
    ],
    "custom_inputs": {"user_id": "<your-test-user-id>"},
    "stream": false
  }'
```

The stateful agent calls `search_agent_memories` and `search_memories` first, so its answer is shaped by any rules/preferences saved for the agent or the user (e.g. money-formatting rules, preferred response style). Drive a second turn that says "from now on, format all money totals with a 💵 emoji" and re-ask the same question — the second answer will pick up the new rule because the agent saved it via `write_memory`.
