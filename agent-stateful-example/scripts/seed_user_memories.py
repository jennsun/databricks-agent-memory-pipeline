"""Seed the memories-user/production Lakebase with example user memory FILES.

Each user is represented as a small set of markdown files stored at paths
under /memories/{episodic,semantic,procedural}/. This matches the filesystem
shape that the agent's memory tools expose at runtime.

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

PROJECT = os.getenv("LAKEBASE_USER_PROJECT", "memories-user")
BRANCH = os.getenv("LAKEBASE_USER_BRANCH", "production")
SCHEMA = os.getenv("LAKEBASE_AGENT_MEMORY_SCHEMA", "memories")
EMBEDDING_ENDPOINT = os.getenv("DATABRICKS_EMBEDDING_ENDPOINT", "databricks-gte-large-en")
EMBEDDING_DIMS = 1024

USERS = [
    "9182746501837465@4729103847561029",  # software engineer
    "1039485762019384@6657382910456721",  # data analyst
    "7777000011112222@3333444455556666",  # solutions engineer
    "4829103756102948@9182736455463728",  # marketing manager
    "1234567890123456@6543210987654321",  # ML engineer
    "9000111122223333@4444555566667777",  # product manager
]


def _md(*lines: str) -> str:
    return "\n".join(lines).strip() + "\n"


# ---------------------------------------------------------------------------
# Per-filename metadata: (description, startup_load).
# Files whose basename is in this table get a description in the memory map.
# Files marked startup_load=True have their full content injected into the
# system prompt at session start.
# ---------------------------------------------------------------------------

PATH_METADATA: dict[str, tuple[str, bool]] = {
    "profile.md":              ("Who the user is — role, timezone, communication baseline.", True),
    "answer_format.md":        ("How to format every answer for this user.", True),
    "response_format.md":      ("How to format every answer for this user.", True),
    "coding_preferences.md":   ("Programming languages, frameworks, and editor.", False),
    "data_stack.md":           ("SQL/Python data tooling and data sources.", False),
    "demo_style.md":           ("How the user prefers to run customer demos.", False),
    "content_preferences.md":  ("Content formats this user prefers vs. avoids.", False),
    "ml_stack.md":             ("ML frameworks and languages used.", False),
    "tooling.md":              ("Day-to-day product tools (roadmap, specs, metrics).", False),
    "interests.md":            ("Topics the user is most interested in.", False),
    "kpis.md":                 ("KPIs the user is measured on.", False),
    "industries.md":           ("Customer verticals the user covers.", False),
    "model_preferences.md":    ("Preferred models for production vs. experimentation.", False),
    "report_cadence.md":       ("Recurring reports the user owns and their cadence.", False),
    "decision_framework.md":   ("How the user makes prioritization decisions.", False),
    "meeting_preferences.md":  ("How the user wants meetings handled (sync vs. async).", False),
}


# ---------------------------------------------------------------------------
# Per-user memory files. Each entry: (user_id, path, content)
# ---------------------------------------------------------------------------

SEED_DATA: list[tuple[str, str, str]] = [
    # ------------------------- User 1 — Software Engineer -------------------------
    (USERS[0], "/memories/semantic/profile.md", _md(
        "# Profile",
        "",
        "- Role: Software engineer",
        "- Specialty: Backend distributed systems",
        "- Timezone: America/Los_Angeles (PT), working hours 9-6",
        "- Communication style: Technical tone, low verbosity. Prefers code-first answers with brief explanations.",
    )),
    (USERS[0], "/memories/semantic/coding_preferences.md", _md(
        "# Coding preferences",
        "",
        "- Primary language: Python",
        "- Secondary language: Go",
        "- Editor: VS Code",
        "- Version control: git",
        "- Frameworks: FastAPI, asyncio",
        "- Learning style: Prefers hands-on examples and code over long theoretical explanations.",
    )),
    (USERS[0], "/memories/semantic/interests.md", _md(
        "# Technical interests",
        "",
        "- Microservices architecture",
        "- Event-driven systems",
        "- Kubernetes / container orchestration",
    )),
    (USERS[0], "/memories/procedural/answer_format.md", _md(
        "# How to answer this user",
        "",
        "1. Lead with a working code snippet.",
        "2. Add at most 2-3 lines of explanation under the snippet.",
        "3. Skip the conceptual background — they want the code first and will ask if they need more.",
        "4. Use Python type hints in examples.",
    )),

    # ------------------------- User 2 — Data Analyst ------------------------------
    (USERS[1], "/memories/semantic/profile.md", _md(
        "# Profile",
        "",
        "- Role: Data analyst",
        "- Industry: E-commerce",
        "- Timezone: America/New_York (ET), working hours 8-5",
        "- Communication style: Professional tone, medium verbosity, comfortable with jargon.",
    )),
    (USERS[1], "/memories/semantic/data_stack.md", _md(
        "# Data stack",
        "",
        "- Primary language: SQL",
        "- Secondary: Python (pandas)",
        "- Viz tools: Tableau",
        "- Notebook: Databricks notebooks",
        "- Python libraries: pandas, matplotlib",
        "- Primary data sources: Snowflake, Unity Catalog (daily refresh)",
    )),
    (USERS[1], "/memories/semantic/response_format.md", _md(
        "# How to format data answers",
        "",
        "- Include a SQL example whenever the answer involves data.",
        "- Add a small chart (ASCII or markdown table) when visualization helps.",
        "- Spell out assumptions about column names and freshness.",
    )),
    (USERS[1], "/memories/procedural/report_cadence.md", _md(
        "# Recurring reports the user owns",
        "",
        "- **Daily:** sales dashboard",
        "- **Weekly:** exec summary",
        "- **Monthly:** retention deep-dive",
        "",
        "When the user mentions one of these, default to the cadence above unless they say otherwise.",
    )),
    (USERS[1], "/memories/semantic/interests.md", _md(
        "# Analytical interests",
        "",
        "- Cohort analysis",
        "- A/B testing",
        "- Customer segmentation",
    )),

    # ------------------------- User 3 — Solutions Engineer ------------------------
    (USERS[2], "/memories/semantic/profile.md", _md(
        "# Profile",
        "",
        "- Role: Solutions engineer",
        "- Focus: Pre-sales technical demos",
        "- Timezone: America/Chicago (CT), flexible hours, frequent travel",
        "- Communication style: Friendly and consultative, medium verbosity.",
    )),
    (USERS[2], "/memories/semantic/demo_style.md", _md(
        "# Preferred demo style",
        "",
        "- Live coding with narration > static slide decks.",
        "- Tools: Databricks notebooks for demos, Lucidchart for architecture diagrams, Google Slides for deck-style content.",
        "- Always frame demos with a customer story.",
    )),
    (USERS[2], "/memories/semantic/industries.md", _md(
        "# Industries this user covers",
        "",
        "- Financial services",
        "- Retail",
        "- Healthcare",
    )),
    (USERS[2], "/memories/procedural/answer_format.md", _md(
        "# How to answer this user",
        "",
        "- Lead with the use case or customer story, then the technical detail.",
        "- Include ROI / business-value framing when possible.",
        "- Skip dry technical specs; pivot to 'what would a demo of this look like?'.",
    )),

    # ------------------------- User 4 — Marketing Manager -------------------------
    (USERS[3], "/memories/semantic/profile.md", _md(
        "# Profile",
        "",
        "- Role: Marketing manager",
        "- Specialty: Product marketing",
        "- Timezone: Europe/London (GMT), working hours 9-5",
        "- Communication style: Warm and approachable, high verbosity, avoids jargon.",
    )),
    (USERS[3], "/memories/semantic/content_preferences.md", _md(
        "# Content format preferences",
        "",
        "- Prefers: blog posts, case studies, webinars.",
        "- Avoids: dense whitepapers.",
        "- Writing tool: Notion. CRM: Salesforce. Analytics: Google Analytics.",
    )),
    (USERS[3], "/memories/semantic/kpis.md", _md(
        "# KPIs the user tracks",
        "",
        "- MQLs (marketing-qualified leads)",
        "- Pipeline influence",
        "- Content engagement",
    )),
    (USERS[3], "/memories/procedural/answer_format.md", _md(
        "# How to answer this user",
        "",
        "- Use a story-driven narrative with concrete customer examples.",
        "- Avoid acronyms or technical jargon unless you define them inline.",
        "- Frame everything in terms of positioning, competitive analysis, or go-to-market impact.",
    )),

    # ------------------------- User 5 — ML Engineer -------------------------------
    (USERS[4], "/memories/semantic/profile.md", _md(
        "# Profile",
        "",
        "- Role: ML engineer",
        "- Specialty: LLM fine-tuning and evaluation",
        "- Timezone: America/Los_Angeles (PT), working hours 10-7",
        "- Communication style: Academic tone, medium verbosity, comfortable with jargon.",
    )),
    (USERS[4], "/memories/semantic/ml_stack.md", _md(
        "# ML stack",
        "",
        "- Primary language: Python",
        "- Secondary: Rust (for inference)",
        "- Frameworks: PyTorch, vLLM, Transformers",
        "- Experiment tracking: MLflow",
    )),
    (USERS[4], "/memories/semantic/model_preferences.md", _md(
        "# Model preferences",
        "",
        "- **Production:** Claude Sonnet",
        "- **Experimentation:** GPT-4o, Llama 3",
    )),
    (USERS[4], "/memories/procedural/answer_format.md", _md(
        "# How to answer this user",
        "",
        "- Be rigorous. Include citations or paper references where relevant.",
        "- Show benchmark numbers, not just qualitative claims.",
        "- It's fine to use technical jargon — they're comfortable with it.",
    )),
    (USERS[4], "/memories/semantic/interests.md", _md(
        "# Research interests",
        "",
        "- RAG (retrieval-augmented generation)",
        "- Evaluation harnesses",
        "- GPU optimization",
        "- Model distillation",
    )),

    # ------------------------- User 6 — Product Manager ---------------------------
    (USERS[5], "/memories/semantic/profile.md", _md(
        "# Profile",
        "",
        "- Role: Product manager",
        "- Specialty: Developer tools",
        "- Timezone: America/New_York (ET), working hours 9-7",
        "- Communication style: Direct and decisive, low verbosity.",
    )),
    (USERS[5], "/memories/semantic/tooling.md", _md(
        "# Tools",
        "",
        "- Roadmap: Linear",
        "- Specs: Notion",
        "- Product metrics: Amplitude",
    )),
    (USERS[5], "/memories/procedural/decision_framework.md", _md(
        "# Decision-making framework",
        "",
        "- Prioritization: RICE (Reach, Impact, Confidence, Effort).",
        "- Data inputs: user interviews + usage metrics from Amplitude.",
        "- Trade-offs should be made explicit in any recommendation.",
    )),
    (USERS[5], "/memories/procedural/meeting_preferences.md", _md(
        "# Meeting preferences",
        "",
        "- Max 4 sync meetings per day.",
        "- Prefer async (Linear / Notion comments) over sync wherever possible.",
        "- When suggesting a meeting, propose async first.",
    )),
    (USERS[5], "/memories/procedural/answer_format.md", _md(
        "# How to answer this user",
        "",
        "- Structured bulleted lists with explicit trade-offs.",
        "- Recommend a single option but show what was considered and why.",
        "- Keep things short — they will ask follow-ups if they want more.",
    )),
]


async def main() -> None:
    logger.info(
        "Seeding user memories: project=%s branch=%s schema=%s embedding=%s",
        PROJECT, BRANCH, SCHEMA, EMBEDDING_ENDPOINT,
    )
    logger.info(
        "Total entries to seed: %d markdown files across %d users",
        len(SEED_DATA), len(USERS),
    )

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

        # Wipe ALL existing entries in each user's namespace before reseeding.
        for user_id in USERS:
            namespace = ("user_memories", user_id)
            existing = await store.asearch(namespace, limit=1000)
            if existing:
                logger.info("Wiping %d existing entries for user %s", len(existing), user_id)
                for item in existing:
                    await store.adelete(namespace, item.key)

        # Insert new markdown files with description + startup_load metadata
        # derived from the filename.
        for i, (user_id, path, content) in enumerate(SEED_DATA, 1):
            namespace = ("user_memories", user_id)
            basename = path.rsplit("/", 1)[-1]
            description, startup_load = PATH_METADATA.get(basename, ("", False))
            value: dict = {"content": content}
            if description:
                value["description"] = description
            if startup_load:
                value["startup_load"] = True
            await store.aput(namespace, path, value)
            flag = " [always-loaded]" if startup_load else ""
            logger.info(
                "[%d/%d] %s :: %s (%d chars)%s",
                i, len(SEED_DATA), user_id, path, len(content), flag,
            )

    logger.info("Done. Seeded %d memory files.", len(SEED_DATA))


if __name__ == "__main__":
    asyncio.run(main())
