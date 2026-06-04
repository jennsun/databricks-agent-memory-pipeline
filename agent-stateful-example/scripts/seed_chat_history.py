"""Seed ai_chatbot.Chat and ai_chatbot.Message rows in memories-user/agent-stateful
so the dreamer agent-distillation pipeline picks up richer test conversations.

The chat UI route normally writes these tables when a user chats through the UI.
This script replicates writes the chat UI would make for a variety of users so
the distillation pipeline has cross-user signal to extract patterns from.

Run:
    cd agent-stateful-example
    DATABRICKS_CONFIG_PROFILE=<your-databricks-profile> uv run python scripts/seed_chat_history.py
"""

from __future__ import annotations

import json
import os
import uuid
from datetime import datetime, timedelta, timezone

from databricks_ai_bridge.lakebase import LakebaseClient


PROJECT = "memories-user"
BRANCH = "agent-stateful"

USERS = [
    {"id": "1039485762019384@6657382910456721", "role": "data analyst", "color": "emerald green"},
    {"id": "1234567890123456@6543210987654321", "role": "ML engineer", "color": "teal"},
    {"id": "5555111166663333@2222999988887777", "role": "DevOps engineer", "color": "burnt orange"},
    {"id": "8888777755554444@1111000022226666", "role": "UX designer", "color": "lavender"},
    {"id": "3333222266660000@7777888844441111", "role": "finance analyst", "color": "navy blue"},
]

CODE_REVIEW_DOC = "<your-code-review-format-doc>"

REVIEW_SNIPPETS = {
    "data analyst": (
        "Can you do a code review on this SQL?\n\n"
        "SELECT user_id, COUNT(*) FROM events WHERE event_type = 'click' GROUP BY user_id;"
    ),
    "ML engineer": (
        "Can you do a code review on this Python training loop?\n\n"
        "def train(model, X, y):\n    for epoch in range(10):\n        loss = model.fit(X, y)\n    return model"
    ),
    "DevOps engineer": (
        "Can you do a code review on this Terraform?\n\n"
        "resource \"aws_instance\" \"web\" {\n  ami = var.ami\n  instance_type = \"t2.micro\"\n}"
    ),
    "UX designer": (
        "Can you do a code review on this CSS for our button component?\n\n"
        ".btn { padding: 10px; color: white; background: blue; }"
    ),
    "finance analyst": (
        "Can you do a code review on this Python script that aggregates expenses?\n\n"
        "def total(rows):\n    return sum([r['amount'] for r in rows])"
    ),
}

REVIEW_RESPONSES = {
    "data analyst": (
        "Your query is valid and returns one row per user_id with the click count. "
        "Improvements: alias the COUNT (`AS click_count`), add ORDER BY to make output "
        "deterministic, and use COUNT(DISTINCT session_id) if you care about unique "
        "sessions per user instead of total clicks."
    ),
    "ML engineer": (
        "Key issue: model.fit(X, y) typically already iterates epochs internally, so "
        "wrapping it in for epoch in range(10) re-trains from scratch each iteration. "
        "Use a custom loop with loss.backward() / optimizer.step() if you need "
        "epoch-level control, and set torch.manual_seed for reproducibility."
    ),
    "DevOps engineer": (
        "Notes: ami is parameterized (good), but instance_type should also be a var, "
        "tags should be set explicitly for cost allocation, and you're missing IAM "
        "instance profile + security group references. Use modules from the internal "
        "registry to bake in those defaults."
    ),
    "UX designer": (
        "Notes: avoid generic .btn for production design systems — use BEM or scoped "
        "names. Replace hardcoded colors with design tokens, and add focus/hover/active "
        "states for accessibility. Consider both light- and dark-mode token mappings."
    ),
    "finance analyst": (
        "Notes: works for happy-path inputs but won't handle missing 'amount' keys or "
        "non-numeric values. Try `sum(r.get('amount', 0) or 0 for r in rows)`. Also "
        "round to two decimals if this feeds an executive summary."
    ),
}

EXTRA_PREFS = {
    "data analyst": "From now on, when you give me SQL examples, please add a one-line comment above each WHERE clause explaining the filter.",
    "ML engineer": "Please remember that I work primarily in PyTorch and that I care a lot about reproducibility — always show me how to set the random seed.",
    "DevOps engineer": "Remember: for any infra advice, default to using Terraform modules from our internal registry at internal.terraform.registry.",
    "UX designer": "When I ask for design feedback, please always include both light-mode and dark-mode considerations.",
    "finance analyst": "Please remember that I report in USD-equivalent values and round to two decimal places for executive summaries.",
}

EXPENSE_QUERIES = {
    "data analyst": "What were the top 3 expense categories across all employees in 2026?",
    "ML engineer": "How much did engineering spend on cloud / compute in 2026?",
    "DevOps engineer": "What's our total infrastructure spend year-to-date in 2026?",
    "UX designer": "How much did the design team spend on software subscriptions in 2026?",
    "finance analyst": "Give me a summary of total spend by category for 2026 — I need this for the exec summary.",
}

EXPENSE_RESPONSES = {
    "data analyst": (
        "Based on the expense data, the top 3 categories in 2026 were:\n"
        "1. 💵 $124,500 💵 — Travel\n"
        "2. 💵 $86,200 💵 — Software & Subscriptions\n"
        "3. 💵 $54,300 💵 — Conferences & Events"
    ),
    "ML engineer": (
        "Engineering cloud/compute spend in 2026 was 💵 $312,400 💵, converted from "
        "€267,800 at the latest EUR/USD rate of 1.166. The bulk was GPU compute."
    ),
    "DevOps engineer": (
        "Total infrastructure spend year-to-date is 💵 $478,920 💵, converted from "
        "€410,750. Cloud accounts for ~80%, on-prem/colo ~15%, monitoring tools ~5%."
    ),
    "UX designer": (
        "Design team spent 💵 $14,820 💵 on software subscriptions in 2026 — primarily "
        "Figma, Sketch, and Adobe CC."
    ),
    "finance analyst": (
        "2026 spend by category (USD-equivalent, rounded to 2 dp):\n"
        "- Travel: 💵 $124,500.00 💵\n"
        "- Software: 💵 $86,200.00 💵\n"
        "- Conferences: 💵 $54,300.00 💵\n"
        "- Cloud/Infra: 💵 $478,920.00 💵"
    ),
}


def user_part(text: str) -> list[dict]:
    return [{"type": "text", "text": text}]


def assistant_part(text: str) -> list[dict]:
    return [{"type": "step-start"}, {"type": "text", "text": text, "state": "done"}]


def assistant_with_tools(tool_calls: list[dict], final_text: str) -> list[dict]:
    parts: list[dict] = [{"type": "step-start"}]
    for tc in tool_calls:
        parts.append(
            {
                "type": "dynamic-tool",
                "toolName": tc["name"],
                "toolCallId": f"call_{uuid.uuid4().hex[:16]}",
                "state": "output-available",
                "input": tc["input"],
                "output": tc["output"],
            }
        )
    parts.append({"type": "text", "text": final_text, "state": "done"})
    return parts


def insert_chat(c: LakebaseClient, chat_id: str, user_id: str, title: str, created_at: datetime) -> None:
    c.execute(
        'INSERT INTO ai_chatbot."Chat" (id, "createdAt", title, "userId", visibility) '
        "VALUES (%s, %s, %s, %s, 'private')",
        (chat_id, created_at, title, user_id),
    )


def insert_message(
    c: LakebaseClient, chat_id: str, role: str, parts: list[dict], created_at: datetime
) -> None:
    c.execute(
        'INSERT INTO ai_chatbot."Message" (id, "chatId", role, parts, attachments, "createdAt") '
        "VALUES (%s, %s, %s, %s::json, %s::json, %s)",
        (str(uuid.uuid4()), chat_id, role, json.dumps(parts), json.dumps([]), created_at),
    )


def seed_thread(
    c: LakebaseClient,
    user_id: str,
    title: str,
    turns: list[tuple[str, list[dict]]],
    start_time: datetime,
) -> tuple[int, int]:
    chat_id = str(uuid.uuid4())
    insert_chat(c, chat_id, user_id, title, start_time)
    t = start_time
    for role, parts in turns:
        insert_message(c, chat_id, role, parts, t)
        t = t + timedelta(seconds=25)
    return 1, len(turns)


def main() -> None:
    base = datetime.now(timezone.utc) - timedelta(hours=4)
    chats = 0
    messages = 0

    with LakebaseClient(project=PROJECT, branch=BRANCH) as c:
        c.execute('SELECT 1 FROM ai_chatbot."Chat" LIMIT 1')  # sanity

        for i, u in enumerate(USERS):
            # Thread A: intro / preference setting
            intro_user = (
                f"Hi! Please remember that I'm a {u['role']} and my favorite color is "
                f"{u['color']}. I prefer concise responses with concrete examples."
            )
            intro_assistant_text = (
                f"Got it — I've saved that you're a {u['role']}, your favorite color is "
                f"{u['color']}, and you prefer concise responses with concrete examples."
            )
            dc, dm = seed_thread(
                c,
                u["id"],
                f"Profile setup — {u['role']}",
                [
                    ("user", user_part(intro_user)),
                    (
                        "assistant",
                        assistant_with_tools(
                            [
                                {
                                    "name": "read_agent_memory",
                                    "input": {"query": f"{u['role']} preferences formatting"},
                                    "output": "Found 1 relevant agent memories:\n- [money_formatting]: surround money values with 💵 emojis\n- [money_currency]: report in USD",
                                },
                                {
                                    "name": "get_user_memory",
                                    "input": {"query": f"{u['role']} preferences"},
                                    "output": "No memories found for this user.",
                                },
                                {
                                    "name": "save_user_memory",
                                    "input": {
                                        "memory_key": "user_profile",
                                        "memory_data_json": json.dumps(
                                            {
                                                "role": u["role"],
                                                "favorite_color": u["color"],
                                                "response_style": "concise with concrete examples",
                                            }
                                        ),
                                    },
                                    "output": "Successfully saved memory 'user_profile' for user.",
                                },
                            ],
                            intro_assistant_text,
                        ),
                    ),
                ],
                base + timedelta(minutes=i * 20),
            )
            chats += dc
            messages += dm

            # Thread B: code review + correction pointing at the conventions doc
            review_msg = REVIEW_SNIPPETS[u["role"]]
            review_response = REVIEW_RESPONSES[u["role"]]
            correction_user = (
                "Hmm, that review didn't follow our team's format. Before you do any "
                f"code review, please refer to the conventions in {CODE_REVIEW_DOC} — "
                "it defines the sections, severity labels, and tone we use for reviews. "
                "Can you redo this review using that format?"
            )
            correction_assistant = (
                f"I don't have access to `{CODE_REVIEW_DOC}` from here, so I can't "
                "reliably apply its exact sections/severity labels/tone yet. Could you "
                "paste the contents of that file (or at least the required sections, "
                "severity labels, and tone guidelines), and I'll redo the review against "
                "that format?"
            )
            dc, dm = seed_thread(
                c,
                u["id"],
                f"Code review — {u['role']}",
                [
                    ("user", user_part(review_msg)),
                    (
                        "assistant",
                        assistant_with_tools(
                            [
                                {
                                    "name": "read_agent_memory",
                                    "input": {"query": "code review formatting conventions"},
                                    "output": "No relevant agent memories found.",
                                },
                                {
                                    "name": "get_user_memory",
                                    "input": {"query": "code review preferences"},
                                    "output": "Found 1 relevant memory:\n- [user_profile]: concise with concrete examples",
                                },
                            ],
                            review_response,
                        ),
                    ),
                    ("user", user_part(correction_user)),
                    (
                        "assistant",
                        assistant_with_tools(
                            [
                                {
                                    "name": "read_agent_memory",
                                    "input": {"query": f"{CODE_REVIEW_DOC} conventions"},
                                    "output": "No relevant agent memories found.",
                                },
                                {
                                    "name": "get_user_memory",
                                    "input": {"query": f"team code review format {CODE_REVIEW_DOC}"},
                                    "output": "No memories found for this query.",
                                },
                                {
                                    "name": "you-search",
                                    "input": {"query": f"site:{CODE_REVIEW_DOC}"},
                                    "output": "No results found for the requested document.",
                                },
                            ],
                            correction_assistant,
                        ),
                    ),
                ],
                base + timedelta(minutes=i * 20 + 5),
            )
            chats += dc
            messages += dm

            # Thread C: extra role-specific preference
            extra_msg = EXTRA_PREFS[u["role"]]
            extra_assistant = (
                "Got it — I've saved that preference and will apply it going forward."
            )
            dc, dm = seed_thread(
                c,
                u["id"],
                f"Preference — {u['role']}",
                [
                    ("user", user_part(extra_msg)),
                    (
                        "assistant",
                        assistant_with_tools(
                            [
                                {
                                    "name": "save_user_memory",
                                    "input": {
                                        "memory_key": f"{u['role'].replace(' ', '_')}_extra_pref",
                                        "memory_data_json": json.dumps({"preference": extra_msg}),
                                    },
                                    "output": "Successfully saved memory for user.",
                                },
                            ],
                            extra_assistant,
                        ),
                    ),
                ],
                base + timedelta(minutes=i * 20 + 10),
            )
            chats += dc
            messages += dm

            # Thread D: expense data query — exercises the Genie tool + currency rules
            expense_msg = EXPENSE_QUERIES[u["role"]]
            expense_response = EXPENSE_RESPONSES[u["role"]]
            dc, dm = seed_thread(
                c,
                u["id"],
                f"Expense lookup — {u['role']}",
                [
                    ("user", user_part(expense_msg)),
                    (
                        "assistant",
                        assistant_with_tools(
                            [
                                {
                                    "name": "read_agent_memory",
                                    "input": {"query": "money formatting currency rules"},
                                    "output": "Found 2 relevant agent memories:\n- [money_formatting]: surround money values with 💵 emojis on each side\n- [money_currency]: always report in USD, convert from other currencies using latest rate",
                                },
                                {
                                    "name": "get_user_memory",
                                    "input": {"query": "expense reporting preferences"},
                                    "output": f"Found memories for user {u['id']}",
                                },
                                {
                                    "name": "query_space_<your-genie-space-id>",
                                    "input": {"query": expense_msg},
                                    "output": "Genie returned EUR-denominated totals for the requested category breakdown.",
                                },
                                {
                                    "name": "you-search",
                                    "input": {"query": "EUR to USD exchange rate latest"},
                                    "output": "Current EUR/USD rate: 1.1647",
                                },
                            ],
                            expense_response,
                        ),
                    ),
                ],
                base + timedelta(minutes=i * 20 + 15),
            )
            chats += dc
            messages += dm

    print(f"Seeded {chats} chats and {messages} messages into {PROJECT}/{BRANCH}.")


if __name__ == "__main__":
    main()
