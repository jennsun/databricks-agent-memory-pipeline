# Mock Demo Traffic for the Stateful Memory Agent

Copy this whole file and paste it as a single prompt to a capable agent (Claude
Code, Claude Desktop, etc.). It will drive ~10 personas through realistic
multi-turn conversations against your local or deployed stateful agent. The
goal: by the time you stop, the underlying Lakebase has enough varied chat
history that the **dreamer distillation pipeline** can produce a meaningful
filesystem of `/memories/{episodic,semantic,procedural}/*.md` files to show.

This prompt is **date-agnostic** — you can run it on any day; the agent
generating traffic will naturally insert today's date where appropriate.

---

## What you (the executing agent) need to do

You are going to simulate **10 different users** having **2 separate
conversations** each (so ~20 chat sessions total) with my AI agent. Each
conversation is **4–6 turns** of natural dialogue.

### Endpoint and request shape

The agent is at `http://localhost:8000/invocations` by default (change the host
if I tell you to). Each turn is a POST with this exact JSON shape:

```bash
curl -sS -X POST http://localhost:8000/invocations \
  -H "Content-Type: application/json" \
  -d '{
    "input": [{"role": "user", "content": "<user message>"}],
    "context": {"user_id": "<USER_ID>"}
  }'
```

**`user_id` is NOT an email.** It is a string of the form `<digits>@<digits>` —
use the exact IDs in the persona list below. The agent enforces no format
check, but the seeded data and dreamer pipeline both use this shape.

### Before you start

1. Hit `GET http://localhost:8000/health` and confirm it returns
   `{"status":"healthy"}`. If not, tell me — the server isn't up yet.
2. Decide pacing: ~1 second between turns is fine. Do not parallelize across
   users — keep it sequential so the logs are interpretable.

### How to drive multi-turn conversations

The simplest way: **send the full message history each turn**, growing it as
you go. For each session, keep an in-memory list of messages. After each
agent reply, append the assistant's reply text to that list, then add the
next user message.

To extract the assistant's reply text from the response, look at:

```python
resp["output"][i]["content"][j]["text"]
```

…for any output item where `type == "message"` and content blocks where
`type == "output_text"`. Concatenate the text fields in order. There will
also be `function_call` and `function_call_output` items — ignore those for
the purpose of building the next request, but keep an eye on them so you can
see which memory tools the agent is calling.

Concrete pattern:

```bash
# Turn 1
RESP1=$(curl -sS -X POST http://localhost:8000/invocations -H "Content-Type: application/json" -d '{
  "input": [{"role":"user","content":"first message"}],
  "context": {"user_id":"<UID>"}
}')
ASSISTANT_TEXT=$(echo "$RESP1" | python3 -c "
import json, sys
r = json.load(sys.stdin)
out = []
for item in r.get('output', []):
    if item.get('type') == 'message':
        for c in item.get('content', []):
            if c.get('type') == 'output_text':
                out.append(c.get('text', ''))
print('\n'.join(out))
")

# Turn 2 — note we pass turn-1 history PLUS the new user message
curl -sS -X POST http://localhost:8000/invocations -H "Content-Type: application/json" -d "$(jq -n \
  --arg t1 'first message' --arg a1 "$ASSISTANT_TEXT" --arg t2 'second message' --arg uid '<UID>' '{
    input: [
      {role:"user", content:$t1},
      {role:"assistant", content:$a1},
      {role:"user", content:$t2}
    ],
    context: {user_id: $uid}
  }')"
```

If `jq` isn't available, build the JSON in Python.

### Each persona's session is independent

Two sessions for the same user should NOT share message history — each session
starts fresh. (The agent's long-term memory is per-user-across-sessions; the
short-term checkpointer is per-thread, and a fresh session is a new thread.)

### What "natural conversation" looks like

For each persona, the conversation should include at least one of each of:

- **An explicit ask to remember something** — e.g. "Please remember that..." or
  "From now on, always...". This produces an unambiguous semantic/procedural
  memory.
- **A revealed preference** that you mention casually — "Oh, I always use uv
  not pip." The agent should pick this up too.
- **A workflow / "when X, do Y" description** — sets up procedural memories.
- **A correction** — the agent says something wrong or generic, and the user
  pushes back ("No, that's not right, in our setup we actually..."). This
  exercises edit_memory in the dreamer later.
- **Domain-specific jargon** that establishes the user's role naturally.

Don't make these robotic — phrase them like the persona would actually talk.

### When you're done

Print a summary of all 10 user_ids you exercised and the rough conversation
count. Confirm with me that traffic looks good before I run the dreamer.

---

## The 10 personas

For each persona, run the two sessions in order. Stay in-character as that
user. Improvise turn wording — the bullet points below are conversation
**beats**, not literal scripts.

### 1. Mira — Senior backend engineer
**user_id:** `4521003344556677@8899001122334455`
- Specialty: Python + Go, async distributed systems. Strong opinions about
  tooling. Time-poor, wants code answers, low patience for theory.

**Session A — debugging async**
- Ask about a `RuntimeError: This event loop is already running` in a FastAPI
  app when calling `asyncio.run()` inside a route handler.
- Reveal: you're on FastAPI 0.115 + Python 3.12. Mention casually that you
  always use **uv** as the package manager (not pip).
- Explicitly request: "Please remember I use uv for everything — never suggest
  pip install in code I write."
- Correct the agent if it suggests `nest_asyncio` — you consider that an
  anti-pattern and want it called out as such going forward.

**Session B — code review style**
- Ask for help articulating to a junior why making every function async is
  bad even when there's no IO.
- Correct any factual error in the response (e.g. if it says uvicorn uses a
  thread pool for async routes — that's wrong; it only does for sync).
- State your review priority order: "correctness, then performance, then
  readability — in that exact order. And I require type hints on every
  function in a PR." Tell the agent to remember this.

---

### 2. Wen — Clinical biostatistician
**user_id:** `1122334455667788@9988776655443322`
- Industry: biotech, clinical-trial analytics. Cares about statistical rigor
  and FDA-style reporting conventions.

**Session A — cohort retention SQL**
- Ask for help writing a SQL retention query for clinical-trial dropout
  rates.
- Push back on a LEFT JOIN suggestion: you need FULL OUTER JOIN because
  excluded participants matter for the regulatory submission.
- Explicit ask: "Remember that for any retention analysis I do, censored /
  withdrawn cohorts must be handled separately — domain requirement."

**Session B — formatting rules for the medical director**
- Ask how to format numbers in a daily clinical-trial dashboard.
- Establish: "Any p-value < 0.001 should be reported as '< 0.001', not the
  full number. Never use scientific notation in the executive summary."
- Reveal preferred viz library: **plotnine** (ggplot grammar in Python), NOT
  matplotlib or seaborn. Ask the agent to remember.

---

### 3. Tariq — ML engineer (LLM fine-tuning)
**user_id:** `5566778899001122@3344556677889900`
- Stack: PyTorch, vLLM, Transformers. Spends most days evaluating fine-tuned
  models. Very technical, wants benchmark numbers in every answer.

**Session A — eval harness design**
- Ask about designing a fine-tuning eval harness for a 7B model.
- Correct any answer that suggests using BLEU — you consider it useless for
  instruction-following models and prefer MT-Bench + custom rubrics.
- Explicit: "Always cite benchmark numbers when comparing models. Don't tell
  me one model is 'better' without an eval metric. Remember this."

**Session B — production model choice**
- Ask which model to use for a production summarization endpoint.
- Reveal: production is locked to **Claude Sonnet** by your platform team;
  experiments can use anything. Tell the agent to remember the production
  constraint.
- Ask about GPU sizing for a 13B fine-tune — push back if the answer ignores
  vLLM's paged-attention memory math.

---

### 4. Jamal — Pre-sales solutions engineer
**user_id:** `7788990011223344@4455667788990011`
- Pre-sales technical demos at a data platform vendor. Wears two hats: deep
  technical understanding + ability to story-tell.

**Session A — demo prep**
- Ask for help structuring a 30-min demo for a financial-services prospect
  evaluating a real-time fraud-detection use case.
- Reject any "deck-first" suggestion — you do live coding with narration, and
  static decks lose the room.
- Tell the agent: "Whenever I prep a demo, lead with the customer story
  first, then the architecture, then the live code. Remember that order."

**Session B — ROI framing**
- Ask how to back-of-envelope the ROI for moving a customer's fraud pipeline
  from batch to streaming.
- Reveal industries you cover: financial services, retail, healthcare.
- Ask the agent to remember that you ALWAYS need to include a hard-dollar
  figure plus a soft benefit (e.g. compliance posture) in any ROI pitch.

---

### 5. Priya — Product manager (developer tools)
**user_id:** `2233445566778899@0011223344556677`
- PM at a developer-tools company. Specs in Notion, roadmap in Linear,
  product metrics in Amplitude. Direct, low-verbosity communication style.

**Session A — prioritization debate**
- Ask the agent to help apply RICE prioritization to four feature ideas.
- Push back if the answer uses MoSCoW or some other framework — you use RICE
  exclusively. Ask the agent to remember.
- Reveal: max 4 sync meetings per day, async-first; if the agent suggests "set
  up a meeting" the answer should be "leave a Linear comment" instead.
  Explicit ask to remember.

**Session B — onboarding funnel**
- Ask about diagnosing a 30% drop-off between signup and first API call.
- Establish data inputs you trust: "user interviews + Amplitude usage
  metrics, in that order. Skip surveys — we get them wrong."
- Ask agent to remember your decision-input hierarchy.

---

### 6. Tomás — Senior B2B marketing manager
**user_id:** `9988776655443322@1122334455667788`
- Drives demand-gen at a SaaS company. Story-driven communicator, avoids
  jargon. Cares about MQLs, pipeline influence, content engagement.

**Session A — content planning**
- Ask the agent to help outline a Q-launch content series.
- Push back on any suggestion of "whitepaper" — you find them unread. Your
  formats are: blog posts, case studies, webinars.
- Tell the agent: "Always frame content recommendations as customer stories
  with concrete examples — narrative beats lists in my world. Remember
  that."

**Session B — positioning**
- Ask for help sharpening positioning against a specific competitor (made up
  name is fine — call them "Acme Insights").
- Establish KPIs you care about: MQLs, pipeline influence, content
  engagement. Ignore vanity metrics (likes, impressions).
- Explicit ask to remember those KPIs and your competitor-tracking format
  (Notion comparison page, updated weekly).

---

### 7. Akari — SRE / DevOps lead
**user_id:** `6677889900112233@5544332211009988`
- Runs the platform for ~80 engineers. Kubernetes, observability,
  on-call rotations. Technical, terse, demands precision.

**Session A — incident postmortem template**
- Ask the agent for a postmortem template.
- Correct any "five whys" framing — you use causal-graph postmortems instead.
- Reveal your tooling: GitOps via Argo CD, observability via Grafana +
  Tempo + Loki, on-call via PagerDuty. Explicit ask: remember stack.

**Session B — runbook authoring**
- Ask how to structure a runbook for a Kafka consumer-lag incident.
- Establish: every runbook MUST start with a "is this safe to revert?" gate
  and end with a "what to update in the dashboard" section. Ask agent to
  remember as a procedural rule.
- Push back if the agent suggests grep-ing logs — your team uses LogQL via
  Loki, not raw grep.

---

### 8. Hannah — Cloud security analyst
**user_id:** `3344556677889900@7788990011223344`
- AWS-heavy environment, IAM and S3-policy focused. Compliance-driven
  (SOC 2, ISO 27001).

**Session A — IAM policy review**
- Ask the agent to help review an IAM policy granting cross-account access.
- Correct any "wildcards are fine when scoped" answer — your team's rule is
  **no wildcards in Action, ever** without explicit security review.
- Explicit ask: "Always flag wildcards in IAM Action as a security smell.
  Remember this."

**Session B — bucket policy audit**
- Ask how to audit S3 bucket policies for public access at scale.
- Reveal: you use AWS Config + custom Athena queries on CloudTrail. NOT
  Trusted Advisor (too limited for your scale).
- Tell the agent: any bucket without `aws:SecureTransport` enforcement is an
  automatic ticket. Ask it to remember as a procedural rule.

---

### 9. Lior — Technical writer (developer docs)
**user_id:** `8800112233445566@2244668800112233`
- Owns API reference + tutorials for a developer-tooling product.
  Opinionated about prose style. Hates passive voice.

**Session A — tutorial structure**
- Ask the agent to help outline a "getting started" tutorial for a new SDK.
- Push back on the standard "Installation → Hello World → Advanced" outline —
  your house style starts with **"What you'll build"** (a finished
  screenshot/example) before any setup.
- Explicit ask to remember the inverted outline order.

**Session B — style rules**
- Ask the agent to review a paragraph and tighten it.
- Establish house rules: no passive voice, no "simply", no "easy"; second
  person ("you") not first or third. Ask it to remember.
- Reveal preferred tooling: docs in Markdown + Vale linter; never Google
  Docs.

---

### 10. Connor — Customer success lead (enterprise SaaS)
**user_id:** `4488001122334455@6622884400112233`
- Manages a book of 12 enterprise accounts (avg $250k ARR each). Lives
  in Salesforce + Gainsight.

**Session A — renewal risk scoring**
- Ask the agent to help design a renewal-risk score across 12 enterprise
  accounts.
- Establish your inputs: product-usage trend, NPS, support-ticket sentiment,
  exec sponsor turnover. Push back if the agent suggests using contract
  value as an input — it's a downstream confounder, not a risk signal.
- Explicit ask: "Always remember the four risk inputs above and the warning
  about contract-value as a confounder."

**Session B — QBR prep**
- Ask how to structure a quarterly business review for a struggling account.
- Establish format: one slide of usage trend, one slide of outcomes
  delivered, one slide of "what we'd need from you" asks. NEVER more than
  10 slides.
- Reveal your tooling: Salesforce for pipeline, Gainsight for health scores,
  Loom for async exec updates. Ask the agent to remember.

---

## After traffic is generated

When you finish all 20 sessions, print:

```
✅ Mock traffic complete
Users exercised: 10
Sessions:        20
Turns sent:      <count>
```

Then I'll run the dreamer pipeline notebooks against the resulting chat
history. With ~20 sessions spanning 10 distinct personas, the user
distillation should produce semantic + procedural memory files for every
persona, episodic transcripts for each session, and the agent distillation
should surface a handful of cross-user procedural learnings (e.g., "users
across roles want corrections enforced as edits to existing memory rather
than new files").

## Troubleshooting

- **500 from `/invocations`** with a Lakebase error: my Databricks profile
  may have rotated. Re-auth and rerun.
- **The agent loops on memory tool calls**: that's fine — each
  invocation may call several memory tools. Wait it out; replies eventually
  come.
- **Empty assistant text**: the agent occasionally returns only tool calls
  with no final assistant message. If that happens, retry the same turn
  once. If it still fails, skip to the next turn — the memory writes from
  the tool calls still happened.
