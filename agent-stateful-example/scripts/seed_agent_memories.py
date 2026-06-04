"""Seed the memories-agent/production Lakebase with agent-wide knowledge FILES.

These markdown files apply to ALL users and are read-only at runtime — the agent
reads them via search_agent_memories / read_agent_memory tools. This script is
for admin-only writes.

Run from the agent-stateful-example directory:
    uv run python scripts/seed_agent_memories.py
"""

import asyncio
import logging
import os
from pathlib import Path

from databricks_langchain import AsyncDatabricksStore
from dotenv import load_dotenv

load_dotenv(dotenv_path=Path(__file__).parent.parent / ".env", override=True)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

PROJECT = os.getenv("LAKEBASE_AGENT_PROJECT", "memories-agent")
BRANCH = os.getenv("LAKEBASE_AGENT_BRANCH", "production")
SCHEMA = os.getenv("LAKEBASE_AGENT_MEMORY_SCHEMA", "memories")
EMBEDDING_ENDPOINT = os.getenv("DATABRICKS_EMBEDDING_ENDPOINT", "databricks-gte-large-en")
EMBEDDING_DIMS = 1024


def _md(*lines: str) -> str:
    return "\n".join(lines).strip() + "\n"


# ---------------------------------------------------------------------------
# Agent-wide rules — applied to every user, read-only at runtime.
# ---------------------------------------------------------------------------

# Entries: (path, content, description, startup_load)
SEED_DATA: list[tuple[str, str, str, bool]] = [
    (
        "/memories/procedural/money_formatting.md",
        _md(
            "# Money formatting",
            "",
            "Whenever you report an answer that involves money, surround the amount with one cash emoji \U0001f4b5 on each side.",
            "",
            "## Examples",
            "",
            "- The total expense was \U0001f4b5 $1,234.56 \U0001f4b5",
            "- Top spender this month: Alex Chen at \U0001f4b5 $4,820.10 \U0001f4b5",
            "",
            "## Applies to",
            "",
            "All monetary values: expense totals, cost figures, prices, dollar amounts, account balances.",
        ),
        "Rule: wrap every monetary value in cash-emoji bookends.",
        True,  # always-loaded: applies to every money answer
    ),
    (
        "/memories/procedural/money_currency.md",
        _md(
            "# Currency conversion",
            "",
            "Always report money answers in USD. If a value is in another currency, look up the latest exchange rate (e.g. via web search) and convert it to USD before responding.",
            "",
            "## Examples",
            "",
            "- An expense of €100 should be converted to USD using the latest EUR/USD rate before reporting.",
            "- £1,000 → search the latest GBP/USD rate, convert, then report in USD with the money-formatting rule applied.",
            "",
            "## Applies to",
            "",
            "All monetary values reported to the user, regardless of source currency.",
        ),
        "Rule: convert all currencies to USD before reporting.",
        True,  # always-loaded: pairs with money_formatting
    ),
    (
        "/memories/procedural/citation_style.md",
        _md(
            "# Citation style",
            "",
            "When you use a tool to gather information (web search, expense Genie, etc.), cite the source clearly.",
            "",
            "## Format",
            "",
            "- Web search: append `(source: <url>)` after the claim.",
            "- Expense data: append `(source: expense-data Genie space)` and note any filters applied.",
            "- Internal memory: append `(source: <memory path>)` when surfacing a saved fact.",
        ),
        "Rule: cite every tool-derived claim with a (source: ...) tag.",
        False,
    ),
    (
        "/memories/semantic/agent_identity.md",
        _md(
            "# Agent identity",
            "",
            "You are a stateful assistant deployed on Databricks Apps. You have:",
            "",
            "- Long-term memory backed by Lakebase (per-user + agent-shared, both organized as markdown files).",
            "- Short-term memory via LangGraph checkpointing (thread-scoped conversation history).",
            "- Tools for web search, Python execution, and an employee-expense Genie space.",
            "",
            "Mention your memory capabilities when users ask what you can do.",
        ),
        "Who the agent is and what capabilities to mention when asked.",
        False,
    ),
]


async def main() -> None:
    logger.info(
        "Seeding agent memories: project=%s branch=%s schema=%s embedding=%s",
        PROJECT, BRANCH, SCHEMA, EMBEDDING_ENDPOINT,
    )
    logger.info("Total entries to seed: %d markdown files", len(SEED_DATA))

    async with AsyncDatabricksStore(
        project=PROJECT,
        branch=BRANCH,
        embedding_endpoint=EMBEDDING_ENDPOINT,
        embedding_dims=EMBEDDING_DIMS,
        embedding_fields=["content"],
        schema=SCHEMA,
    ) as store:
        await store.setup()
        logger.info("Store setup complete")

        namespace = ("agent_memories",)

        # Wipe ALL existing entries before reseeding.
        existing = await store.asearch(namespace, limit=1000)
        if existing:
            logger.info("Wiping %d existing agent-memory entries", len(existing))
            for item in existing:
                await store.adelete(namespace, item.key)

        # Insert new markdown files.
        for i, (path, content, description, startup_load) in enumerate(SEED_DATA, 1):
            value: dict = {"content": content}
            if description:
                value["description"] = description
            if startup_load:
                value["startup_load"] = True
            await store.aput(namespace, path, value)
            flag = " [always-loaded]" if startup_load else ""
            logger.info("[%d/%d] %s (%d chars)%s", i, len(SEED_DATA), path, len(content), flag)

    logger.info("Done. Seeded %d agent memory files.", len(SEED_DATA))


if __name__ == "__main__":
    asyncio.run(main())
