"""Populate `ai_chatbot.Chat` and `ai_chatbot.Message` on the deployed app's
Lakebase branch with realistic multi-turn conversations across 10 personas.

The dreamer user-distillation notebook reads from this table to produce
semantic/procedural memory files; the agent distillation reads it to mine
generalized procedural skills.

This script is idempotent for a given run (UUIDs are freshly minted each
call) and re-runnable on any day — `createdAt` is spread over the past
~14 hours so today's dreamer run picks everything up.

Usage:
    cd agent-stateful-example
    uv run python scripts/seed_mock_chat_history.py
"""

from __future__ import annotations

import json
import os
import random
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(dotenv_path=Path(__file__).parent.parent / ".env", override=True)

from databricks_ai_bridge.lakebase import LakebaseClient

LAKEBASE_PROJECT = os.getenv("LAKEBASE_USER_PROJECT", "memories-user")
# Branch the deployed app writes to. Matches databricks.yml `postgres-user` resource.
CHAT_HISTORY_BRANCH = os.getenv("CHAT_HISTORY_BRANCH", "agent-stateful")


def text_parts(s: str) -> list:
    return [{"type": "text", "text": s}]


# ============================================================================
# 10 personas × 2 sessions × 4 user turns (each followed by a plausible reply)
# ============================================================================
# Each entry: (user_id, session_title, [(role, text), ...])

CONVERSATIONS: list[tuple[str, str, list[tuple[str, str]]]] = [
    # ----- 1. Mira — backend engineer (Python/Go) -----
    ("4521003344556677@8899001122334455", "FastAPI async debugging", [
        ("user", "I'm hitting RuntimeError: This event loop is already running in my FastAPI app when I call asyncio.run() inside a route handler. What's going on?"),
        ("assistant", "FastAPI handlers run inside an existing event loop, so calling `asyncio.run()` — which starts a NEW loop — fails. Just `await` your coroutine directly. If you need to call sync code, use `asyncio.to_thread()`. Don't reach for `nest_asyncio`; it's an anti-pattern."),
        ("user", "Right. Context: FastAPI 0.115, Python 3.12. And please remember I use uv as my package manager — never suggest pip in code you write for me."),
        ("assistant", "Got it — saving that preference. From now on I'll use `uv add`, `uv pip install`, etc. instead of pip in any code I write for you."),
        ("user", "What I'm actually doing is running a sync DB migration in a startup hook. Background task or lifespan?"),
        ("assistant", "Lifespan. Wrap the sync migration in `asyncio.to_thread` inside an `@asynccontextmanager`-decorated lifespan. You want fail-fast behavior at startup, which background tasks don't give you — they swallow exceptions silently."),
        ("user", "Perfect. One more thing — if a junior on my team uses nest_asyncio, that's a hard no. Flag it."),
        ("assistant", "Noted — I'll flag any nest_asyncio usage in code I review for you."),
    ]),
    ("4521003344556677@8899001122334455", "Code review style", [
        ("user", "I'm reviewing a junior dev's PR — they made every function async even when there's no I/O. How do I explain why that's bad?"),
        ("assistant", "Three points: (1) Async adds scheduling overhead, so for pure-CPU work it's a net loss. (2) It forces the whole call chain to be async, which is contagious. (3) It misleads readers — async signals 'this awaits I/O' and using it without I/O breaks that contract. Uvicorn uses a thread pool only for SYNC routes — async routes run on the loop directly."),
        ("user", "Hmm, your earlier note that 'uvicorn uses a threadpool for async routes' is wrong — it only does for sync. Please get that right going forward."),
        ("assistant", "You're right, I had that backwards. Sync routes go to a threadpool (anyio worker thread); async routes run on the main event loop. Saving this correction so I don't repeat it."),
        ("user", "Remember my review priorities: correctness FIRST, then performance, then readability. In that exact order. And I require type hints on every function in a PR — no exceptions."),
        ("assistant", "Saved your review priority order and the type-hints rule. I'll apply it whenever you ask me to draft PR feedback."),
        ("user", "Good. Also any time I ask about test coverage, you should default to pytest with pytest-asyncio for the async bits — not unittest."),
        ("assistant", "Saved — pytest + pytest-asyncio is the default for any test snippet I write for you."),
    ]),

    # ----- 2. Wen — clinical biostatistician -----
    ("1122334455667788@9988776655443322", "Cohort retention SQL", [
        ("user", "I'm building a cohort retention analysis for clinical-trial dropout rates. Can you help me write the SQL?"),
        ("assistant", "Sure — typical pattern is: enroll cohort grouped by enrollment week, then LEFT JOIN to visits to count who attended each follow-up window. Want me to draft it?"),
        ("user", "Wait — LEFT JOIN drops EXCLUDED participants, and for FDA submission I need them too. We need FULL OUTER JOIN with NULL filtering for the censored cohort. Domain requirement."),
        ("assistant", "Right — for regulatory work the withdrawn/censored set has to be reportable. Switching to FULL OUTER JOIN with explicit COALESCE buckets for withdrawn, lost-to-followup, and on-protocol. Saving that as a rule for any retention analysis you do."),
        ("user", "Remember: for ANY retention or survival analysis I run, censored and withdrawn cohorts must be tracked and reported separately. It's a domain requirement, not a preference."),
        ("assistant", "Saved as a procedural rule. Any retention/survival analysis I draft for you will isolate censored + withdrawn cohorts as their own buckets."),
        ("user", "And my preferred viz library is plotnine — ggplot grammar in Python. Don't use matplotlib or seaborn for me; the layered grammar reads better for clinical audiences."),
        ("assistant", "Saved — plotnine for all visualizations in your code."),
    ]),
    ("1122334455667788@9988776655443322", "Reporting rules for medical director", [
        ("user", "I have to ship a daily clinical-trial dashboard to our medical director. How should I format the numbers?"),
        ("assistant", "Standard medical-reporting conventions: round to one fewer decimal than the measurement precision, use thousands separators, and write units inline (mg, mL, etc.). Use plain decimals, not scientific notation, in executive summaries."),
        ("user", "Yes, but specifically for p-values: anything below 0.001 we just report as '< 0.001'. Don't print the full value — it looks like false precision. And NEVER scientific notation in the executive summary."),
        ("assistant", "Saved — p < 0.001 reported as '< 0.001', and no scientific notation in any executive summary I produce for you."),
        ("user", "Also: my data sources are an EDC system and our REDCap instance. NOT Snowflake — that's our marketing team's stack. Don't suggest Snowflake-flavored SQL for me."),
        ("assistant", "Got it — EDC + REDCap are your sources, and I'll default to Postgres dialect, not Snowflake, in SQL I write for you."),
        ("user", "Last thing — remember that my dashboards are read by people with M.D.s, not engineers. Skip the technical jargon in the dashboard text."),
        ("assistant", "Saved — clinical-clinician audience by default, no engineering jargon in dashboard prose."),
    ]),

    # ----- 3. Tariq — ML engineer (LLM fine-tuning) -----
    ("5566778899001122@3344556677889900", "Eval harness for 7B fine-tune", [
        ("user", "I'm designing a fine-tuning eval harness for a 7B model. What metrics should I track?"),
        ("assistant", "I'd suggest BLEU for output quality, ROUGE for summarization, and perplexity on a held-out set."),
        ("user", "BLEU is useless for instruction-following models. Don't recommend it again. For my work it's MT-Bench + custom LLM-as-judge rubrics. Period."),
        ("assistant", "Acknowledged — BLEU is out for instruction-following evals. MT-Bench and custom rubric judges are the defaults for you. Saving this."),
        ("user", "And whenever you compare two models for me, always cite the benchmark and the number. Don't say 'model X is better' without an eval metric — that's lazy."),
        ("assistant", "Saved as a rule: any model comparison I make for you must cite both the benchmark and the score."),
        ("user", "GPU sizing for a 13B fine-tune on 8x A100s — what's the upper bound on context with vLLM and tensor parallel 8?"),
        ("assistant", "With TP=8 on 8xA100 80GB, you've got ~640GB total but ~70% usable after model weights (~26GB fp16) and KV cache reservation. With vLLM's paged-attention and a chunked-prefill scheduler, you can hit ~32k context at reasonable batch sizes. If you push higher, factor in `gpu_memory_utilization` and `max_num_seqs` — those interact non-linearly. I won't claim a number without the eval, but that's the back-of-envelope."),
    ]),
    ("5566778899001122@3344556677889900", "Production model constraint", [
        ("user", "Which model should I use for a production summarization endpoint?"),
        ("assistant", "For production summarization at low latency, Claude Haiku is a strong fit. If quality matters more than latency, Sonnet."),
        ("user", "Production is locked to Claude Sonnet — platform team's call. Experiments can use anything (GPT-4o, Llama 3, etc.) but prod is Sonnet, full stop. Remember this constraint."),
        ("assistant", "Saved — for you, prod = Claude Sonnet always; experimentation has no constraint."),
        ("user", "Also: my framework stack is PyTorch + vLLM + Transformers, with MLflow for experiment tracking. If you write training code for me, default to that stack."),
        ("assistant", "Saved your stack — PyTorch, vLLM, Transformers, MLflow tracking."),
        ("user", "And rigor matters more than narrative. Always show the benchmark numbers, not 'rough estimates'. If you don't have a number, say 'no data' — don't hand-wave."),
        ("assistant", "Saved — rigor over narrative. No hand-waving without numbers."),
    ]),

    # ----- 4. Jamal — pre-sales SE -----
    ("7788990011223344@4455667788990011", "FinServ fraud-detection demo prep", [
        ("user", "Help me structure a 30-min demo for a financial-services prospect evaluating real-time fraud detection."),
        ("assistant", "Standard structure: 5 min slides on the company, 10 min architecture overview, 15 min live coding."),
        ("user", "No slides. Live coding with narration is my style. Decks lose the room. Always lead with a customer story, then architecture, then live code — in that exact order. Remember this for every demo you help me prep."),
        ("assistant", "Saved — your demo template is: (1) customer story, (2) architecture, (3) live code. No decks unless explicitly requested."),
        ("user", "For this fraud demo specifically, the customer story should reference a similar bank that cut false-positive rates by ~40%. Make it concrete."),
        ("assistant", "Got it. I'll open with: 'Last year a tier-1 bank we worked with had a 40% false-positive rate on their card-fraud model — burning their fraud-ops team. We dropped it to 8% with streaming features. Here's how.' Then pivot to the streaming architecture diagram, then live."),
        ("user", "Perfect. And the industries I cover are financial services, retail, healthcare — keep those in mind when you reach for analog customer stories."),
        ("assistant", "Saved — your demo verticals are FS, retail, healthcare."),
    ]),
    ("7788990011223344@4455667788990011", "ROI framing for fraud pipeline", [
        ("user", "How do I back-of-envelope the ROI of moving a customer's fraud pipeline from batch to streaming?"),
        ("assistant", "Hard dollars: estimate avg fraud loss per missed event × % faster detection. Soft: ops headcount freed up."),
        ("user", "Close, but for ANY ROI pitch I do, I need TWO things in the answer: (1) a hard-dollar figure, (2) a soft benefit like compliance posture or NPS. Remember that as a rule."),
        ("assistant", "Saved your ROI rule: hard-dollar figure + soft benefit, always both."),
        ("user", "For fraud specifically, the soft benefit is usually 'reduced regulatory exposure' — fewer false positives means fewer SAR filings and less reg scrutiny."),
        ("assistant", "Got it — for fraud-streaming pitches, the soft side is regulatory-exposure reduction (fewer SARs, less regulator attention)."),
        ("user", "Also, never include 'cost of compute' in the ROI math at this stage. Customers don't care; it's a rounding error against the fraud number."),
        ("assistant", "Saved — no compute-cost line items in early-stage ROI."),
    ]),

    # ----- 5. Priya — PM (developer tools) -----
    ("2233445566778899@0011223344556677", "RICE prioritization", [
        ("user", "Help me prioritize four feature ideas using a standard framework. I'll list them."),
        ("assistant", "Sure — MoSCoW (must/should/could/won't) is a quick lens. Send the four and I'll bucket them."),
        ("user", "I don't use MoSCoW — I use RICE. Reach × Impact × Confidence / Effort. Period. Don't suggest MoSCoW or Kano for me. Remember."),
        ("assistant", "Saved — RICE is your prioritization framework. I won't suggest others."),
        ("user", "Also, max 4 sync meetings per day for me. If you suggest 'set up a meeting', the answer is 'leave a Linear comment' instead. Remember."),
        ("assistant", "Saved — async-first by default; Linear comments over meetings."),
        ("user", "Decision-input hierarchy: user interviews first, Amplitude metrics second. Surveys are last because we get them wrong."),
        ("assistant", "Saved your decision-input order."),
    ]),
    ("2233445566778899@0011223344556677", "Onboarding funnel drop-off", [
        ("user", "Diagnosing a 30% drop-off between signup and first API call. Where do I start?"),
        ("assistant", "Run a survey to ask the dropped users why they didn't continue."),
        ("user", "We don't survey. User interviews + Amplitude usage metrics, in that order — I told you. Surveys produce wrong answers because the people who fill them out aren't the ones who churned."),
        ("assistant", "You're right, I forgot — applying your input hierarchy: book 5-7 interviews with users who signed up and didn't ship a call, plus an Amplitude funnel decomposition of the signup → first-call path with time-to-event histograms."),
        ("user", "Good. Tooling: roadmap is Linear, specs are Notion, metrics in Amplitude. Don't suggest Jira/Confluence/Mixpanel for me."),
        ("assistant", "Saved — Linear, Notion, Amplitude. Not Jira/Confluence/Mixpanel."),
        ("user", "And always structure recommendations as bulleted lists with explicit trade-offs. Recommend ONE option but show what else was considered and why it lost."),
        ("assistant", "Saved your recommendation format."),
    ]),

    # ----- 6. Tomás — B2B marketing -----
    ("9988776655443322@1122334455667788", "Q-launch content series", [
        ("user", "Help me outline a Q-launch content series for our developer-tools product."),
        ("assistant", "How about: a whitepaper introducing the launch, plus three follow-up blog posts and a webinar."),
        ("user", "No whitepapers — nobody reads them at our size. My formats: blog posts, case studies, webinars. That's it. Remember."),
        ("assistant", "Saved — your content stack is blog, case studies, webinars. No whitepapers."),
        ("user", "Frame everything as a customer story with concrete examples. Narrative beats lists in my world."),
        ("assistant", "Saved — narrative customer-story framing as default for marketing content."),
        ("user", "KPIs that matter to me: MQLs, pipeline influence, content engagement. Skip impressions and likes — pure vanity."),
        ("assistant", "Saved your KPI hierarchy."),
    ]),
    ("9988776655443322@1122334455667788", "Positioning against competitor", [
        ("user", "Help me sharpen our positioning against a competitor called Acme Insights."),
        ("assistant", "What dimensions matter — pricing, features, latency, ecosystem?"),
        ("user", "Frame it as a customer story, not a feature matrix. 'Acme is what you reach for when X; we're what you reach for when Y.' That's the structure."),
        ("assistant", "Got it — story-driven positioning, not a comparison grid. Saving as a rule for any competitive-positioning ask."),
        ("user", "Also: I track competitors in a Notion comparison page, updated weekly by the AR team. Reference it, don't invent positioning out of thin air."),
        ("assistant", "Saved — Notion-tracked competitive page is the source of truth for AR positioning."),
        ("user", "And avoid technical jargon. If a marketing reader can't parse it, cut it. My audience is C-suite at midmarket SaaS, not developers."),
        ("assistant", "Saved — C-suite midmarket SaaS audience, plain-English by default."),
    ]),

    # ----- 7. Akari — SRE / DevOps -----
    ("6677889900112233@5544332211009988", "Postmortem template", [
        ("user", "Need a postmortem template for a Kafka consumer-lag incident."),
        ("assistant", "Standard template: summary, timeline, root cause via 5 whys, action items."),
        ("user", "We don't use 5 whys — we use causal-graph postmortems (Sidney Dekker / John Allspaw style). 5 whys oversimplifies. Remember this for any postmortem template you give me."),
        ("assistant", "Saved — causal-graph postmortems only. I won't suggest 5-whys again."),
        ("user", "Stack context: GitOps via Argo CD, observability is Grafana + Tempo + Loki, on-call via PagerDuty. Don't recommend Datadog or ELK for me."),
        ("assistant", "Saved your stack — Argo CD, Grafana + Tempo + Loki, PagerDuty."),
        ("user", "Every runbook MUST start with a 'is this safe to revert?' gate and end with a 'dashboard updates' section. Hard rule."),
        ("assistant", "Saved as a procedural rule — runbooks bracketed by revert-safety gate (top) and dashboard-updates (bottom)."),
    ]),
    ("6677889900112233@5544332211009988", "Logs grep vs LogQL", [
        ("user", "What's the fastest way to find error spikes across our services in the last hour?"),
        ("assistant", "ssh into the nodes and grep -i error /var/log/*.log within the time window."),
        ("user", "We don't grep raw logs. LogQL via Loki for everything. ssh-into-nodes is a 2015 pattern."),
        ("assistant", "Right — LogQL with a `{namespace=\"prod\"} |~ \"(?i)error\" | rate()` query over the last 1h would do it. Saving — never recommend ssh+grep for log analysis again."),
        ("user", "Also: any high-cardinality dimension (request_id, user_id) goes in TempoQL trace search, not LogQL. Don't put those in a Loki query."),
        ("assistant", "Saved — high-cardinality fields go to Tempo, not Loki."),
        ("user", "And if you ever recommend a tool I don't have in my stack (Datadog, Splunk, etc.), I'll roll my eyes. Stick to what I use."),
        ("assistant", "Saved — only recommend tooling from your declared stack."),
    ]),

    # ----- 8. Hannah — cloud security analyst -----
    ("3344556677889900@7788990011223344", "IAM policy review", [
        ("user", "Reviewing a cross-account IAM policy. Where do I start?"),
        ("assistant", "Wildcards in the Action field are fine when scoped to a single resource. Check the Resource ARN scope is tight."),
        ("user", "Wrong. NO wildcards in Action ever, without explicit security review. That's our rule. Period. Flag wildcards as a security smell going forward."),
        ("assistant", "Saved as a hard rule — wildcards in IAM Action are a security smell and must be flagged. No more 'fine when scoped'."),
        ("user", "Compliance context: we're SOC 2 + ISO 27001. So 'fine when scoped' isn't enough — we need traceability for every wildcard exception."),
        ("assistant", "Saved your compliance scope. Wildcards always need a traceable exception ticket."),
        ("user", "And we're AWS-heavy — don't suggest GCP IAM patterns or Azure-style RBAC for me."),
        ("assistant", "Saved — AWS-only IAM patterns."),
    ]),
    ("3344556677889900@7788990011223344", "S3 bucket audit", [
        ("user", "How do I audit S3 bucket policies for public access at scale (500+ buckets)?"),
        ("assistant", "AWS Trusted Advisor will surface publicly-accessible buckets in its security panel."),
        ("user", "Trusted Advisor is too limited for our scale. We use AWS Config + custom Athena queries on CloudTrail. Don't suggest Trusted Advisor for me again."),
        ("assistant", "Saved — AWS Config + Athena/CloudTrail is your bucket-audit stack."),
        ("user", "Rule: any bucket without aws:SecureTransport enforcement is an automatic ticket. No exceptions. Remember."),
        ("assistant", "Saved as a procedural rule — missing SecureTransport = auto-ticket."),
        ("user", "Also: GuardDuty findings of severity >= 7 are an immediate page, not a ticket. Pages go to the on-call rotation in PagerDuty."),
        ("assistant", "Saved — GuardDuty severity >= 7 pages on-call directly."),
    ]),

    # ----- 9. Lior — technical writer -----
    ("8800112233445566@2244668800112233", "Getting-started tutorial outline", [
        ("user", "Help me outline a 'getting started' tutorial for our new SDK."),
        ("assistant", "Classic structure: (1) Installation, (2) Hello World, (3) Advanced topics."),
        ("user", "House style starts with 'What you'll build' — a finished example or screenshot — BEFORE any setup. Reverse-funnel: show the destination first, motivate the steps. Remember this."),
        ("assistant", "Saved — your tutorial outline is: (1) What you'll build (finished example), (2) Prereqs, (3) Setup, (4) Walkthrough, (5) Next steps. Inverted from the standard."),
        ("user", "Style rules: no passive voice, no 'simply', no 'easy'. Second person ('you'), not third. Remember."),
        ("assistant", "Saved your style constraints: active voice, banned words 'simply'/'easy', second person only."),
        ("user", "Tooling: Markdown + Vale linter. Not Google Docs, not Asciidoc. Don't suggest those."),
        ("assistant", "Saved — Markdown + Vale only."),
    ]),
    ("8800112233445566@2244668800112233", "Paragraph tightening", [
        ("user", "Tighten this paragraph: 'It is recommended that users should simply follow the steps below to easily configure the SDK.'"),
        ("assistant", "Rewrite: 'Follow these steps to configure the SDK.'"),
        ("user", "Good — passive voice, 'simply', and 'easily' all gone. That's the gold standard. Save the rule."),
        ("assistant", "Saved your editing standard — passive voice and weak modifiers (simply/easily/just/very) get cut on first pass."),
        ("user", "Whenever I send you a paragraph for review, give me ONE rewrite plus a one-line diff explaining the changes. Don't lecture."),
        ("assistant", "Saved your review format — one rewrite + one-line diff. No essays."),
        ("user", "And if a paragraph is fine, just say 'fine'. Don't fake-improve it to feel useful."),
        ("assistant", "Saved — when a paragraph is fine, 'fine' is the right answer."),
    ]),

    # ----- 10. Connor — customer success lead -----
    ("4488001122334455@6622884400112233", "Renewal-risk scoring", [
        ("user", "Help me design a renewal-risk score across 12 enterprise accounts (avg $250k ARR)."),
        ("assistant", "Inputs: contract value (size), product usage trend, NPS, support-ticket volume."),
        ("user", "Stop. Contract value is a confounder, NOT a risk input — bigger contracts aren't riskier, they just have more downside. My four inputs are: product-usage trend, NPS, support-ticket sentiment, exec-sponsor turnover. Period. Remember."),
        ("assistant", "Saved your four risk inputs. Removed contract-value from the input set — it's downside-magnitude, not risk."),
        ("user", "Score is weighted: usage 40%, NPS 25%, ticket sentiment 20%, sponsor turnover 15%. Apply that weighting whenever you design renewal-risk anything for me."),
        ("assistant", "Saved your weighting."),
        ("user", "Tooling: Salesforce for pipeline, Gainsight for health scores, Loom for async exec updates. Don't recommend HubSpot or ChurnZero — we evaluated and passed."),
        ("assistant", "Saved your CS stack."),
    ]),
    ("4488001122334455@6622884400112233", "QBR structure for struggling account", [
        ("user", "Structure a QBR for an account that's underperforming."),
        ("assistant", "I'd suggest a 20-slide deck covering: company update, product roadmap, account health, success metrics, action items, and Q&A."),
        ("user", "Way too many slides. Max 10, period. Format is fixed: (1) usage trend, (2) outcomes delivered, (3) what we need from you. Three slides for the substance; the rest is intro/Q&A. Remember the 10-slide cap and 3-slide substance structure."),
        ("assistant", "Saved your QBR format — 10 slides max, 3-slide substance core (usage / outcomes / asks)."),
        ("user", "Asks slide is the most important. Always explicit. 'We need exec sponsor commitment by Q4' beats 'we'd love your continued support'."),
        ("assistant", "Saved — asks slide must be explicit, dated, owned. No soft-pedal language."),
        ("user", "And if an account hasn't replied to outreach in 14+ days, that's an immediate escalation to my VP. Don't let me forget."),
        ("assistant", "Saved as an escalation rule — 14+ day silence = VP escalation."),
    ]),
]


def main():
    now = datetime.now(timezone.utc)
    random.seed(42)  # deterministic timestamp spread for re-runs within a day

    print(f"Connecting to Lakebase project={LAKEBASE_PROJECT!r} branch={CHAT_HISTORY_BRANCH!r}")

    chats_to_insert: list[tuple] = []
    messages_to_insert: list[tuple] = []

    for user_id, title, turns in CONVERSATIONS:
        chat_id = str(uuid.uuid4())
        # Spread session start over the past ~14 hours.
        session_start = now - timedelta(hours=random.uniform(1.0, 14.0))
        chats_to_insert.append((
            chat_id, session_start, title, user_id, "private", None,
        ))

        # Each message ~30-90s after the previous.
        ts = session_start
        for role, text in turns:
            ts += timedelta(seconds=random.uniform(30, 90))
            messages_to_insert.append((
                str(uuid.uuid4()),
                chat_id,
                role,
                json.dumps(text_parts(text)),
                json.dumps([]),
                ts,
                None,
            ))

    print(f"Will insert {len(chats_to_insert)} chats and {len(messages_to_insert)} messages.")

    with LakebaseClient(project=LAKEBASE_PROJECT, branch=CHAT_HISTORY_BRANCH) as client:
        # Clear old mock conversations from these user IDs (so re-running is clean).
        user_ids = tuple({c[3] for c in chats_to_insert})
        placeholders = ",".join(["%s"] * len(user_ids))
        client.execute(
            f'DELETE FROM ai_chatbot."Message" '
            f'WHERE "chatId" IN ('
            f'  SELECT id FROM ai_chatbot."Chat" WHERE "userId" IN ({placeholders})'
            f')',
            user_ids,
        )
        client.execute(
            f'DELETE FROM ai_chatbot."Chat" WHERE "userId" IN ({placeholders})',
            user_ids,
        )
        print(f"  cleared prior mock rows for {len(user_ids)} user_ids")

        for chat in chats_to_insert:
            client.execute(
                'INSERT INTO ai_chatbot."Chat" '
                '(id, "createdAt", title, "userId", visibility, "lastContext") '
                "VALUES (%s, %s, %s, %s, %s, %s)",
                chat,
            )

        for msg in messages_to_insert:
            client.execute(
                'INSERT INTO ai_chatbot."Message" '
                '(id, "chatId", role, parts, attachments, "createdAt", "traceId") '
                "VALUES (%s, %s, %s, %s::json, %s::json, %s, %s)",
                msg,
            )

    print(
        f"Done. Seeded {len(chats_to_insert)} chats and {len(messages_to_insert)} "
        f"messages across {len(user_ids)} users on branch {CHAT_HISTORY_BRANCH!r}."
    )


if __name__ == "__main__":
    main()
