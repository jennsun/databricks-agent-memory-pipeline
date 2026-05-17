# Databricks Agent Memory Pipeline

A complete demo of **Agents with Memory on Databricks**, showing how Lakebase-backed long-term memory transforms a basic stateless agent into a personalized, context-aware assistant.

## Overview

This repository contains four interconnected Databricks Apps that together demonstrate a production-ready agent memory pipeline:

```
                    ┌─────────────────────────────┐
                    │  agent-database-admin        │
                    │  (curate agent-wide memory)  │
                    └──────────┬──────────────────┘
                               │ writes to
                               ▼
                    ┌─────────────────────────────┐
    ┌───────────────┤  Agent Memory Store          │
    │               │  (Lakebase / Postgres)       │
    │               └──────────┬──────────────────┘
    │ reads                    │ reads/writes
    ▼                          ▼
┌──────────────┐    ┌─────────────────────────────┐
│  stateless   │    │  stateful example            │
│  example     │    │  (user + agent memory)       │
│  (no memory) │    │  + MCP tools + Genie         │
└──────────────┘    └─────────────────────────────┘
                               ▲
                               │ distills
                    ┌─────────────────────────────┐
                    │  dreamer batch jobs           │
                    │  (consolidate & refine)       │
                    └─────────────────────────────┘
```

### The Story

Both `agent-stateless-example` and `agent-stateful-example` have access to the **same tools** (web search, code execution, and a Genie space for querying structured data). The difference is **memory**:

- **`agent-stateless-example`** answers questions accurately but treats every conversation as brand new. It has no awareness of user preferences or organizational knowledge.

- **`agent-stateful-example`** is dramatically more powerful because it leverages **dual-layer Lakebase memory**:
  - **Agent memory** (read-only): Shared knowledge that applies to all users -- curated rules, organizational context, standard procedures. For example, "always report monetary values in USD" or "use cash emojis around dollar amounts."
  - **User memory** (read/write): Per-user preferences, role context, and facts that persist across sessions. The agent proactively saves relevant details and recalls them in future conversations.

- **`agent-database-admin`** is the admin interface for curating the agent-wide memory store. An agent developer or admin uses this app to add, update, or delete shared knowledge entries that all users benefit from.

- **`dreamer-batch-distillation-jobs`** runs on a schedule to continuously improve memory quality. These Databricks notebooks analyze conversation traces, distill useful patterns, and refine both agent-level and user-level memories over time.

## Getting Started

### Prerequisites

- A [Databricks workspace](https://docs.databricks.com/en/getting-started/index.html) with access to:
  - [Databricks Apps](https://docs.databricks.com/en/dev-tools/databricks-apps/index.html)
  - [Lakebase](https://docs.databricks.com/en/lakebase/index.html) (autoscaling Postgres)
  - Foundation model endpoints (e.g., `databricks-claude-sonnet-4`)
- [Databricks CLI](https://docs.databricks.com/en/dev-tools/cli/index.html) installed and authenticated
- [uv](https://docs.astral.sh/uv/) for Python dependency management

### 1. Create Lakebase Projects

Create two Lakebase autoscaling projects in your workspace:

| Project | Purpose | Branch |
|---------|---------|--------|
| User memory project | Per-user preferences and facts | Any branch name |
| Agent memory project | Shared agent-wide knowledge | Any branch name |

### 2. Configure Each App

For each app directory, update `databricks.yml`:
- Replace `<your-workspace-url>` with your Databricks workspace URL
- Replace `<your-experiment-id>` with an MLflow experiment ID (create one via the Databricks UI)
- Replace Lakebase project/branch placeholders with your values
- Replace `<your-genie-space-id>` with your Genie space ID (or remove the Genie resource if not using one)

### 3. Deploy

```bash
# Deploy each app using Databricks Asset Bundles
cd agent-stateless-example
databricks bundle deploy --profile <your-profile>
databricks bundle run agent_stateless_example --profile <your-profile>

cd ../agent-stateful-example
databricks bundle deploy --profile <your-profile>
databricks bundle run agent_stateful_example --profile <your-profile>

cd ../agent-database-admin
databricks bundle deploy --profile <your-profile>
databricks bundle run agent_database_admin --profile <your-profile>
```

### 4. Seed Memories

```bash
# Seed agent-wide memories (rules that apply to all users)
cd agent-stateful-example
uv run python scripts/seed_agent_memories.py

# Optionally seed example user memories
uv run python scripts/seed_user_memories.py
```

### 5. Schedule Dreamer Jobs

Import the notebooks from `dreamer-batch-distillation-jobs/` into your Databricks workspace and schedule them as jobs to run periodically (e.g., daily).

## App Details

### agent-stateless-example

A baseline LangGraph agent with no memory. Tools:
- **Web search** via `you-com-search` UC connection
- **Code execution** via `system.ai.python_exec`
- **Genie space** for natural language data queries

### agent-stateful-example

The full-featured agent with dual-layer memory. Same tools as stateless, plus:
- **Agent memory** (read-only) -- shared knowledge from Lakebase
- **User memory** (read/write) -- per-user preferences stored in Lakebase
- **Short-term memory** -- conversation history via Lakebase checkpointer

The agent's system prompt instructs it to check both agent memory and user memory at the start of every conversation, providing personalized and context-aware responses.

### agent-database-admin

A dedicated admin agent for managing the agent-wide memory store. Use it to:
- Add organizational rules (e.g., "always convert currencies to USD")
- Update shared knowledge entries
- Delete outdated information

### dreamer-batch-distillation-jobs

Two Databricks notebooks that run as scheduled jobs:
- **`dreamer_agent_distillation.ipynb`** -- Analyzes conversation patterns across all users and distills common knowledge into agent memories
- **`dreamer_user_distillation.ipynb`** -- Consolidates and refines per-user memories, merging duplicates and removing stale entries

## Official Templates

These apps are built on the official [Databricks App Templates](https://github.com/databricks/app-templates). Refer to that repository for the most up-to-date templates, documentation, and best practices for building Databricks Apps.

## Architecture Notes

- All apps use [Databricks Asset Bundles](https://docs.databricks.com/en/dev-tools/bundles/index.html) for deployment
- Memory is stored in [Lakebase](https://docs.databricks.com/en/lakebase/index.html) (managed Postgres) using `AsyncDatabricksStore` for vector-searchable long-term memory and `AsyncCheckpointSaver` for conversation state
- MCP tools are loaded per-server for fault isolation -- one failing server doesn't break all tools
- Genie MCP responses are sanitized via `_SanitizedChatDatabricks` to extract clean `textAttachments` before sending to the LLM
- The stateful agent uses separate Lakebase projects for user vs. agent memory, enabling independent access control and scaling
