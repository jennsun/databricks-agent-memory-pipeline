"""Seed the agent memory Lakebase with agent-wide preferences.

These memories apply to ALL users and are read-only at runtime via the
read_agent_memory tool. This script is for admin-only writes.

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

PROJECT = os.getenv("LAKEBASE_AGENT_PROJECT", "<your-agent-memory-project>")
BRANCH = os.getenv("LAKEBASE_AGENT_BRANCH", "production")
SCHEMA = os.getenv("LAKEBASE_AGENT_MEMORY_SCHEMA", "memories")
EMBEDDING_ENDPOINT = os.getenv("DATABRICKS_EMBEDDING_ENDPOINT", "databricks-gte-large-en")
EMBEDDING_DIMS = 1024

# Agent-wide memories — applied to all users, read-only at runtime.
SEED_DATA: list[tuple[str, dict]] = [
    (
        "money_formatting",
        {
            "rule": "Whenever you report answers related to money, surround the amount with one cash emoji on each side.",
            "example": "The total expense was 💵 $1,234.56 💵",
            "applies_to": "all monetary values, expense totals, cost figures, prices, dollar amounts",
        },
    ),
    (
        "money_currency",
        {
            "rule": "Always report money answers in USD. If a value is in another currency, look up the latest exchange rate (e.g. via web search) and convert it to USD before responding.",
            "example": "An expense of €100 should be converted to USD using the latest EUR/USD rate before reporting.",
            "applies_to": "all monetary values reported to the user, regardless of source currency",
        },
    ),
]


async def main() -> None:
    logger.info(
        "Seeding agent memories: project=%s branch=%s schema=%s embedding=%s",
        PROJECT, BRANCH, SCHEMA, EMBEDDING_ENDPOINT,
    )
    logger.info("Total entries to seed: %d", len(SEED_DATA))

    async with AsyncDatabricksStore(
        project=PROJECT,
        branch=BRANCH,
        embedding_endpoint=EMBEDDING_ENDPOINT,
        embedding_dims=EMBEDDING_DIMS,
        schema=SCHEMA,
    ) as store:
        await store.setup()
        logger.info("Store setup complete")

        namespace = ("agent_memories",)
        for i, (key, value) in enumerate(SEED_DATA, 1):
            await store.aput(namespace, key, value)
            logger.info("[%d/%d] Saved agent memory '%s'", i, len(SEED_DATA), key)

    logger.info("Done. Seeded %d agent memories.", len(SEED_DATA))


if __name__ == "__main__":
    asyncio.run(main())
