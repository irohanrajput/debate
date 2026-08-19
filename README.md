# Investment Committee

Multi-agent debate over an investment thesis under a **hard token budget**. Four analyst agents with genuinely different reasoning lenses research a thesis (RAG over a document corpus + live yfinance data), argue across rounds, and a non-voting Chair synthesizes a committee memo: `BUY`, `BUY_LIMITED` (with position guidance), `WAIT`, or `DO_NOT_BUY`. The orchestrator makes meta-decisions about how to spend reasoning tokens: research vs argument, breadth vs depth, flash vs pro model, who speaks in each round.

Jnaara take-home, Problem A. Python 3.12, LangGraph, Gemini (`2.5-flash` / `2.5-pro`), Chroma, FastAPI.

## Quickstart

```bash
poetry install
cp .env.example .env             # put your GEMINI_API_KEY in it
poetry run python -m committee ingest data/                       # build the corpus index (one-time)
poetry run python -m committee debate \
  --thesis "NVIDIA looks overextended. Buy now, wait, or avoid?" \
  --ticker NVDA --budget 60000
poetry run pytest                # no network needed; runs on a fake LLM
poetry run python -m committee serve --port 8000                  # FastAPI + SSE
```

API: `POST /debate {thesis, ticker?, entity?, budget?, policy?}` → `{run_id}`; `GET /debate/{id}` → full trace; `GET /debate/{id}/stream` → SSE (`curl -N localhost:8000/debate/<id>/stream`).

Every run writes `runs/<run_id>/`: `trace.json` (full structured trace incl. every budget decision), `events.jsonl` (live event stream), `memo.md`.

## Example runs (local `runs/`, not committed)

| Run | What it shows |
|---|---|
| `runs/9f24767416` | **Live NVDA debate** → `BUY_LIMITED` 0.70 with sizing guidance ("one-third now, add toward the 50DMA"), grounded in real yfinance stats (margins, z-score vs 200DMA, relative strength vs peers). Verdicts split BUY/HOLD/BUY/HOLD. |
| `runs/216bf3dac3` | **Corpus (RAG) debate** on NovaTech (fictional, from the provided dataset) → explore→exploit transition, an analyst flipping SELL→BUY mid-debate, tie-breakers adjudicating claims, memo `WAIT`. |
| `runs/e4d805e0d3` | **Uniform-policy baseline**, same thesis: disagreement stays flat (0.52) across all rounds — nothing gets adjudicated. The control for the explore-exploit policy. |
| `runs/85b7a6b852` | **Tiny budget (25k)**: agents skipped with logged reasons, synthesis pool survives, memo still produced. Graceful degradation. |

## Architecture

```
                    ┌─ ingest (one-time) ────────────────────────────────┐
                    │ datasets/docs → 1 record = 1 chunk (+metadata)     │
                    │ → gemini-embedding-001 → Chroma                    │
                    └────────────────────────────────────────────────────┘
CLI / POST /debate
  → init: parse thesis, build Ledger {research|debate|synthesis}, EAGER-fetch MarketSnapshot (frozen)
  → LangGraph loop:
      allocate   BudgetPolicy(state, ledger) → BudgetDecision (mode, lenses, tokens, model tier)
      run_round  fan-out: each analyst = plan research → shared tools → structured argue
      assess     stance entropy + claim embeddings → opposite pairs → flash judge → contested claims
      (loop until converged / budget floor / max rounds)
  → resolve: tie-breaker agent per top contested claim (or flag unresolved) — never averages
  → synthesize: Chair (pro tier, reserved pool) → CommitteeMemo
  → trace.json + events.jsonl + memo.md; EventBus feeds CLI live view and SSE
```

### The four lenses (`committee/agents/lenses.py`)
Fundamentalist ("what is this business actually worth?"), Momentum/Trend ("what is the market telling us?"), Quality/Moat ("can this company keep winning?" — explicitly forbidden from arguing on multiples), Risk/Macro ("what can go wrong?" — must quantify downside and give sizing guidance). Each `LensSpec` has priorities, `does_not_weigh` (enforced blind spots), buy/sell triggers, personality, and preferred tools. Adding a fifth analyst = one registry entry; no orchestrator changes.

### Rounds
- **R1 DISCOVER** — research-heavy, findings only (observations, open questions, falsifiers). No stances: prevents anchoring before evidence.
- **R2 EXPLORE** — first positions: claims with direction, importance, evidence ids.
- **R3+ EXPLOIT** — when contested claims exist, the policy funds the parties to those claims at the pro tier with `must_address` obligations: each agent must CONCEDE / PARTIAL / REBUT / INCORPORATE every contested claim it owns or disputes (a validator rejects silence). The explore→exploit transition is a function of debate state, never a hardcoded round number.

### Budget (the core meta-decision)
Total tokens (input + output) across every LLM call, split into three pools: research / debate / synthesis (synthesis is untouchable by analysts — the Chair always gets to speak). The `Ledger` requires a reservation before every call and reconciles against the provider's real `usage_metadata` after. Reservations account for estimated prompt input (chars/3, deliberately conservative) plus a schema/thinking overhead, so a call that can't fund a minimal useful output is skipped and logged rather than truncated into garbage. Policies are deterministic, pluggable code (`policies/` registry): `uniform` (baseline) and `explore_exploit` (mode from disagreement state; exploit funds the top contested parties at realistic allocations and folds leftover research budget into debate). Every `BudgetDecision` — allocations, tier choices, transfers, rationale, state snapshot — lands in the trace.

### Disagreement (claim-level, never averaged)
Two signals: stance entropy across the committee, and claim-level contradiction — embeddings pre-filter opposite-direction cross-lens claim pairs (capped, most-similar first), a flash judge confirms genuine contradiction. Contested claims drive exploit-round funding and `must_address` lists. Convergence = low weighted disagreement score with non-increasing delta. After the final round, surviving contested claims go to a tie-breaker agent (pro tier, top-k by score, debate-pool leftovers only); the rest are flagged unresolved with reasoning — visible in the memo, not hidden.

### Data plane: numbers never come from the model
Two stores. **Chroma** holds text only (the two provided datasets — 84 facts + 52 decisions, one record per chunk with entity/source/reliability/date metadata — plus any md/txt docs, chunked header-aware). **MarketSnapshot** holds numbers: full daily OHLCV history, financial statements, macro series — fetched once at debate start, then frozen, so every agent in every round sees the same data (no mid-debate drift, reproducible replays). Tools compute statistics in pandas and return typed summaries; the LLM reads computed facts, it never does arithmetic on raw rows. All five tools (`search_corpus`, `company_snapshot`, `price_stats`, `peer_compare`, `macro_context`) are a shared registry — agents choose which to call with what args, never define their own; results are per-run cached and registered as `Evidence` with provenance, so claims cite `EV-n` ids that trace back to a source.

## Key design decisions & tradeoffs

- **Deterministic meta-control, not an LLM supervisor.** Mode selection, speaker selection, token allocation, and model tiering are plain Python reading `DebateState`. Rationale: the assignment's core question is about allocating reasoning resources — that logic should be inspectable, testable, and reproducible. An `LLMSupervisorPolicy` would be one more registry entry if wanted. Tradeoff: less "agentic" flavor; much stronger auditability.
- **Rounds are parallel fan-out/fan-in, not sequential turns.** 4x less wall-clock, no first-speaker bias. Tradeoff: agents react to the previous round, not to each other mid-round.
- **Total-token budgeting (input + output).** Honest but hard: input grows with debate context, so the system must actively compress (evidence render caps, claim-only summaries of other analysts) and estimate input before reserving. Output-only budgeting would have been simpler but gameable by context stuffing. Residual estimate error is possible; the ledger records actual vs reserved for audit.
- **R1 has no stances.** Forcing a stance before research invites anchoring; discovery-then-position produced visibly better debates. Tradeoff: one extra round of budget before any position exists (which is why the minimum is 3 rounds, within the assignment's 2–3 spirit).
- **Frozen market snapshot.** Consistency across agents and reproducibility beat freshness — investment debate, not trading. `--offline` replays without network.
- **In-process asyncio, no Celery/Redis.** The workload is I/O-bound LLM calls; a queue adds ops burden for a take-home reviewers must run. The API's background-task runner is behind a small seam that a real deployment would swap for a distributed queue.
- **Considered and rejected**: LLM supervisor for allocation (unauditable), a devil's-advocate 5th lens (cut for scope), sequential turn-taking (latency/bias), embedding raw price history (numbers via similarity search is how hallucinations happen), Next.js frontend (CLI live view + SSE covers the streaming story).

## What broke and what I learned (honest log)

1. **Gemini structured output can't take free-form dicts.** `ToolRequest.args: dict[str, Any]` failed 100% of the time; flattened to typed optional fields.
2. **Gemini 2.5 thinking tokens silently eat `max_output_tokens`.** Small caps → truncated JSON → schema failures. Fixed with explicit `thinking_budget` (0 for flash, small for pro) and output caps that account for it.
3. **Budget enforcement is an economics problem.** First version reserved only output → pools overshot on input; second starved plans by capping totals below prompt size; third had judges drain the debate pool mid-assessment and tie-breakers drain the synthesis pool. Final shape: input-aware reservations, per-kind schema overheads, judge-pair caps, tie-break caps, pool isolation for synthesis, and a stop rule when the remaining budget can't fund a realistic round. Every one of those failures is visible in the git history.
4. **Exact-match metadata filters are a silent RAG killer.** `entity="NovaTech"` vs stored `"NovaTech Inc."` returned zero results and the committee concluded "no data exists" with high confidence — a convincing, wrong debate. Normalized entity keys fixed it; the lesson is that retrieval failures masquerade as confident conclusions.

## Prompts

**System/agent prompts** (all committed, templated per lens): `committee/agents/prompts/` — `analyst_system.md`, `plan_user.md`, `findings_user.md`, `position_user.md`, `chair_system.md`, `chair_user.md`, `tiebreak_user.md`. The judge prompt is inline in `committee/debate/disagreement.py`.

**AI-coding prompts**: this project was built with Claude Code; the full prompt log and generated-vs-refactored breakdown are in [`docs/ai-log.md`](docs/ai-log.md).

## Tests

`poetry run pytest` — 17 tests, no network (scripted `FakeProvider` + deterministic `FakeEmbedder`): structured-output validation (schema bounds, `BUY_LIMITED` requires guidance, `must_address` completeness), budget enforcement (reservation invariants, synthesis-pool lockout, transfer rules, truncation), policy logic (discover/explore/exploit transitions, party selection, tier upgrades, research fold-in), and a full end-to-end orchestrator run asserting round modes, contested-claim detection, exploit targeting, convergence stop, trace/memo/event outputs, and total spend ≤ budget.
