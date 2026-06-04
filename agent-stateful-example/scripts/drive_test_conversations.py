"""Drive realistic test conversations against the local stateful agent.

Sends a mix of:
  1. Preference-setting messages (different professions / colors / styles per user)
  2. Code-review threads where multiple users mention referring to
     <your-code-review-format-doc>

The local agent writes save_user_memory results + checkpointer messages to the
same memories-user/production Lakebase that the deployed app uses, so an
overnight job reading from those tables will see all of this data.

Run from agent-stateful-example/ while `uvicorn agent_server.start_server:app` is up:
    uv run python scripts/drive_test_conversations.py
"""

from __future__ import annotations

import json
import logging
import time
import uuid
from typing import Optional

import httpx

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

BASE_URL = "http://localhost:8000/invocations"

# Mix of pre-seeded users and brand-new ones so the dream job sees variety.
USERS = [
    {
        "id": "1039485762019384@6657382910456721",
        "role": "data analyst",
        "favorite_color": "emerald green",
    },
    {
        "id": "1234567890123456@6543210987654321",
        "role": "ML engineer",
        "favorite_color": "teal",
    },
    {
        "id": "5555111166663333@2222999988887777",
        "role": "DevOps engineer",
        "favorite_color": "burnt orange",
    },
    {
        "id": "8888777755554444@1111000022226666",
        "role": "UX designer",
        "favorite_color": "lavender",
    },
    {
        "id": "3333222266660000@7777888844441111",
        "role": "finance analyst",
        "favorite_color": "navy blue",
    },
]

CODE_REVIEW_CONVENTIONS_DOC = "<your-code-review-format-doc>"

# Snippets to send for code review by role.
CODE_SNIPPETS = {
    "data analyst": (
        "Can you do a code review on this SQL?\n\n"
        "SELECT user_id, COUNT(*) FROM events WHERE event_type = 'click' GROUP BY user_id;"
    ),
    "ML engineer": (
        "Can you do a code review on this Python training loop?\n\n"
        "def train(model, X, y):\n"
        "    for epoch in range(10):\n"
        "        loss = model.fit(X, y)\n"
        "    return model"
    ),
    "DevOps engineer": (
        "Can you do a code review on this Terraform?\n\n"
        "resource \"aws_instance\" \"web\" {\n"
        "  ami = var.ami\n"
        "  instance_type = \"t2.micro\"\n"
        "}"
    ),
    "UX designer": (
        "Can you do a code review on this CSS for our button component?\n\n"
        ".btn { padding: 10px; color: white; background: blue; }"
    ),
    "finance analyst": (
        "Can you do a code review on this Python script that aggregates expenses?\n\n"
        "def total(rows):\n"
        "    return sum([r['amount'] for r in rows])"
    ),
}


def send(
    client: httpx.Client,
    user_id: str,
    thread_id: str,
    message: str,
) -> dict:
    payload = {
        "input": [{"role": "user", "content": message}],
        "custom_inputs": {"user_id": user_id, "thread_id": thread_id},
    }
    r = client.post(BASE_URL, json=payload, timeout=180.0)
    r.raise_for_status()
    return r.json()


def summarize_response(resp: dict) -> str:
    """Pull out tool calls and the final assistant text for compact logging."""
    parts = []
    for item in resp.get("output", []):
        t = item.get("type")
        if t == "function_call":
            parts.append(f"  -> tool {item.get('name')}({item.get('arguments','')[:60]}...)")
        elif t == "message":
            text = item.get("content", [{}])[0].get("text", "")[:180]
            parts.append(f"  <- {text}")
    return "\n".join(parts)


def main() -> None:
    with httpx.Client() as client:
        # Phase 1: each user introduces themselves and sets preferences
        for u in USERS:
            thread = str(uuid.uuid4())
            intro = (
                f"Hi! Please remember that I'm a {u['role']} and my favorite color is "
                f"{u['favorite_color']}. I prefer concise responses with concrete examples."
            )
            logger.info("[%s] intro thread=%s", u["id"], thread[:8])
            resp = send(client, u["id"], thread, intro)
            print(f"\n=== USER {u['id']} (intro) ===\n{summarize_response(resp)}")

        # Phase 2: each user asks for a code review and then corrects the agent,
        # pointing at the shared code-review-format doc. This is the pattern the
        # agent-wide memory job should detect.
        for u in USERS:
            thread = str(uuid.uuid4())
            snippet = CODE_SNIPPETS[u["role"]]

            logger.info("[%s] code-review request thread=%s", u["id"], thread[:8])
            resp = send(client, u["id"], thread, snippet)
            print(f"\n=== USER {u['id']} (review request) ===\n{summarize_response(resp)}")

            # Follow-up: tell the agent to use the shared conventions doc
            followup = (
                f"Hmm, that review didn't follow our team's format. Before you do any "
                f"code review, please refer to the conventions in "
                f"{CODE_REVIEW_CONVENTIONS_DOC} — it defines the sections, severity "
                f"labels, and tone we use for reviews. Can you redo this review using "
                f"that format?"
            )
            logger.info("[%s] code-review correction", u["id"])
            resp = send(client, u["id"], thread, followup)
            print(f"\n=== USER {u['id']} (correction) ===\n{summarize_response(resp)}")

        # Phase 3: a few extra varied queries so the user_memories table fills
        # with more than just "favorite color + profession" rows.
        extras = [
            (USERS[0]["id"], "From now on, when you give me SQL examples, please add a one-line comment above each WHERE clause explaining the filter."),
            (USERS[1]["id"], "Please remember that I work primarily in PyTorch and that I care a lot about reproducibility — always show me how to set the random seed."),
            (USERS[2]["id"], "Remember: for any infra advice, default to using Terraform modules from our internal registry at internal.terraform.registry."),
            (USERS[3]["id"], "When I ask for design feedback, please always include both light-mode and dark-mode considerations."),
            (USERS[4]["id"], "Please remember that I report in USD-equivalent values and round to two decimal places for executive summaries."),
        ]
        for user_id, msg in extras:
            thread = str(uuid.uuid4())
            logger.info("[%s] extra preference", user_id)
            resp = send(client, user_id, thread, msg)
            print(f"\n=== USER {user_id} (extra) ===\n{summarize_response(resp)}")

    print("\n\nAll conversations sent.")


if __name__ == "__main__":
    main()
