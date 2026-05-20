# Databricks Agents <> Lakebase Memory Pipeline

Code corresponding to the demo of **Agents with Memory on Databricks**, showing how [Lakebase-backed](https://docs.databricks.com/en/lakebase/index.html) long-term memory transforms a basic stateless agent into a personalized, context-aware assistant that compounds and improves over time due to [memory scaling](https://www.databricks.com/blog/memory-scaling-ai-agents)

## Overview

This repository contains an example memory setup for a production-ready agent memory pipeline across your Databricks workspace:

<img width="1026" height="439" alt="image" src="https://github.com/user-attachments/assets/14680ddb-1a7b-48cb-8c5f-5681d96b5efe" />

### The Story

Both `agent-stateless-example` and `agent-stateful-example` have access to the **same tools** (web search, code execution, and a Genie space for querying structured data). The difference is **memory**:

- **`agent-stateless-example`** answers questions accurately but treats every conversation as brand new. It has no awareness of user preferences or organizational knowledge. There is no improvement or compounding (memory scaling) over time.

- **`agent-stateful-example`** is dramatically more powerful because it leverages **dual-layer Lakebase memory**:
  - **Agent memory** (read-only): Shared knowledge that applies to all users like curated skills, organizational context, standard procedures that your users commonly run into. For example, "always report monetary values in USD". Every user can read and benefit from the agent memory, but not every user can write from it (this is limited to Agent Developers/Admins who have permissions to edit the Agent-scoped Memory).
  - **User memory** (read/write): Per-user preferences, role context, and facts that persist across sessions. The agent proactively saves relevant details and recalls them in future conversations. Every user can read and write from their personal memory, which is scoped at the user level.
 - When instructions conflict, instructions will follow priority: system > developer > user > memory
- Memory is stored in [Lakebase](https://docs.databricks.com/en/lakebase/index.html) (managed Postgres) using `AsyncDatabricksStore` for vector-searchable long-term memory and `AsyncCheckpointSaver` for conversation state

- **`dreamer-batch-distillation-jobs`** example notebooks that can be set up as Databricks jobs to run on a schedule to continuously improve memory quality. These Databricks notebooks analyze raw conversation data from your agent and distill useful patterns. For user-scoped memories, it will review conversation threads, distill memories into semantic/episodic/procedural memory, and update the user's memory store. For agent-scoped memories, it will review all conversation threads but distill anonymized/general insights and patterns over time. The insights will be presented to the agent developer, who can either continue to create an automated pipeline to update the agent memories/preference or do so on their own after review using the agent-database-admin app. 

Both distillation pipeline results are logged into a UC Volume so you have a safe, governed store for your logs.

- **`agent-database-admin`** is the admin interface for curating the agent-wide memory store. An agent developer or admin uses this app to add, update, or delete shared knowledge entries that all users benefit from (agent-scoped memory). 

## Official Templates Reference

These apps are built on the official [Databricks App Templates](https://github.com/databricks/app-templates). Refer to that repository for the most up-to-date templates, documentation, and best practices for building Databricks Apps. We also provide a set of skills to make agentic development even easier: just point your favorite coding agent to the repository and let it cook B)
