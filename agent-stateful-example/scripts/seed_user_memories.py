"""Seed the user memory Lakebase with example user preferences.

Run from the agent-stateful-example directory:
    uv run python scripts/seed_user_memories.py
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

PROJECT = os.getenv("LAKEBASE_USER_PROJECT", "<your-user-memory-project>")
BRANCH = os.getenv("LAKEBASE_USER_BRANCH", "production")
SCHEMA = os.getenv("LAKEBASE_AGENT_MEMORY_SCHEMA", "memories")
EMBEDDING_ENDPOINT = os.getenv("DATABRICKS_EMBEDDING_ENDPOINT", "databricks-gte-large-en")
EMBEDDING_DIMS = 1024

USERS = [
    "9182746501837465@4729103847561029",
    "1039485762019384@6657382910456721",
    "7777000011112222@3333444455556666",
    "4829103756102948@9182736455463728",
    "1234567890123456@6543210987654321",
    "9000111122223333@4444555566667777",
]

# 50 memory entries spread across 6 users
SEED_DATA: list[tuple[str, str, dict]] = [
    # User 1 — Software Engineer
    (USERS[0], "profession", {"profession": "software engineer", "specialty": "backend distributed systems"}),
    (USERS[0], "response_preference", {"style": "concise", "format": "code-first with brief explanations"}),
    (USERS[0], "preferred_language", {"language": "Python", "secondary": "Go"}),
    (USERS[0], "tool_preference", {"editor": "VS Code", "version_control": "git", "frameworks": ["FastAPI", "asyncio"]}),
    (USERS[0], "communication_style", {"tone": "technical", "verbosity": "low"}),
    (USERS[0], "timezone", {"timezone": "America/Los_Angeles", "working_hours": "9-6 PT"}),
    (USERS[0], "favorite_topics", {"topics": ["microservices", "event-driven architecture", "Kubernetes"]}),
    (USERS[0], "learning_style", {"prefers": "hands-on examples and code", "avoids": "long theoretical explanations"}),

    # User 2 — Data Analyst
    (USERS[1], "profession", {"profession": "data analyst", "industry": "e-commerce"}),
    (USERS[1], "response_preference", {"style": "detailed", "format": "with charts and SQL examples"}),
    (USERS[1], "preferred_language", {"language": "SQL", "secondary": "Python (pandas)"}),
    (USERS[1], "tool_preference", {"viz": "Tableau", "notebook": "Databricks notebooks", "frameworks": ["pandas", "matplotlib"]}),
    (USERS[1], "communication_style", {"tone": "professional", "verbosity": "medium", "uses_jargon": True}),
    (USERS[1], "timezone", {"timezone": "America/New_York", "working_hours": "8-5 ET"}),
    (USERS[1], "favorite_topics", {"topics": ["cohort analysis", "A/B testing", "customer segmentation"]}),
    (USERS[1], "data_sources", {"primary": ["Snowflake", "Unity Catalog"], "freshness": "daily refresh"}),
    (USERS[1], "report_cadence", {"daily": ["sales dashboard"], "weekly": ["exec summary"], "monthly": ["retention deep-dive"]}),

    # User 3 — Solutions Engineer
    (USERS[2], "profession", {"profession": "solutions engineer", "focus": "pre-sales technical demos"}),
    (USERS[2], "response_preference", {"style": "balanced", "format": "with examples and customer stories"}),
    (USERS[2], "tool_preference", {"demo": "Databricks notebooks", "diagramming": "Lucidchart", "presentations": "Google Slides"}),
    (USERS[2], "communication_style", {"tone": "friendly and consultative", "verbosity": "medium"}),
    (USERS[2], "timezone", {"timezone": "America/Chicago", "working_hours": "flexible, frequent travel"}),
    (USERS[2], "favorite_topics", {"topics": ["agent orchestration", "ROI calculations", "customer onboarding"]}),
    (USERS[2], "industries_served", {"verticals": ["financial services", "retail", "healthcare"]}),
    (USERS[2], "preferred_demo_style", {"style": "live coding with narration", "avoids": "static slide decks"}),

    # User 4 — Marketing
    (USERS[3], "profession", {"profession": "marketing manager", "specialty": "product marketing"}),
    (USERS[3], "response_preference", {"style": "story-driven", "format": "narrative with concrete examples"}),
    (USERS[3], "tool_preference", {"crm": "Salesforce", "analytics": "Google Analytics", "writing": "Notion"}),
    (USERS[3], "communication_style", {"tone": "warm and approachable", "verbosity": "high", "uses_jargon": False}),
    (USERS[3], "timezone", {"timezone": "Europe/London", "working_hours": "9-5 GMT"}),
    (USERS[3], "favorite_topics", {"topics": ["positioning", "competitive analysis", "go-to-market strategy"]}),
    (USERS[3], "content_formats", {"prefers": ["blog posts", "case studies", "webinars"], "avoids": ["whitepapers"]}),
    (USERS[3], "kpis_tracked", {"primary": ["MQLs", "pipeline influence", "content engagement"]}),

    # User 5 — ML Engineer
    (USERS[4], "profession", {"profession": "ML engineer", "specialty": "LLM fine-tuning and evaluation"}),
    (USERS[4], "response_preference", {"style": "rigorous", "format": "with citations and benchmark numbers"}),
    (USERS[4], "preferred_language", {"language": "Python", "secondary": "Rust (for inference)"}),
    (USERS[4], "tool_preference", {"experiment_tracking": "MLflow", "frameworks": ["PyTorch", "vLLM", "Transformers"]}),
    (USERS[4], "communication_style", {"tone": "academic", "verbosity": "medium", "uses_jargon": True}),
    (USERS[4], "timezone", {"timezone": "America/Los_Angeles", "working_hours": "10-7 PT"}),
    (USERS[4], "favorite_topics", {"topics": ["RAG", "evaluation harnesses", "GPU optimization", "model distillation"]}),
    (USERS[4], "model_preferences", {"production": "Claude Sonnet", "experimentation": ["GPT-4o", "Llama 3"]}),

    # User 6 — Product Manager
    (USERS[5], "profession", {"profession": "product manager", "specialty": "developer tools"}),
    (USERS[5], "response_preference", {"style": "structured", "format": "bulleted lists with tradeoffs"}),
    (USERS[5], "tool_preference", {"roadmap": "Linear", "specs": "Notion", "metrics": "Amplitude"}),
    (USERS[5], "communication_style", {"tone": "direct and decisive", "verbosity": "low"}),
    (USERS[5], "timezone", {"timezone": "America/New_York", "working_hours": "9-7 ET"}),
    (USERS[5], "favorite_topics", {"topics": ["developer experience", "API design", "onboarding funnels"]}),
    (USERS[5], "decision_framework", {"prefers": "RICE prioritization", "data_sources": ["user interviews", "usage metrics"]}),
    (USERS[5], "meeting_preference", {"max_per_day": 4, "prefers": "async over sync when possible"}),
]


async def main() -> None:
    logger.info(
        "Seeding user memories: project=%s branch=%s schema=%s embedding=%s",
        PROJECT, BRANCH, SCHEMA, EMBEDDING_ENDPOINT,
    )
    logger.info("Total entries to seed: %d across %d users", len(SEED_DATA), len(USERS))

    async with AsyncDatabricksStore(
        project=PROJECT,
        branch=BRANCH,
        embedding_endpoint=EMBEDDING_ENDPOINT,
        embedding_dims=EMBEDDING_DIMS,
        schema=SCHEMA,
    ) as store:
        await store.setup()
        logger.info("Store setup complete")

        for i, (user_id, key, value) in enumerate(SEED_DATA, 1):
            namespace = ("user_memories", user_id)
            await store.aput(namespace, key, value)
            logger.info("[%d/%d] Saved %s for %s", i, len(SEED_DATA), key, user_id)

    logger.info("Done. Seeded %d memories.", len(SEED_DATA))


if __name__ == "__main__":
    asyncio.run(main())
