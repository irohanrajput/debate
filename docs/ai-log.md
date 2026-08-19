# AI-coding log

Built with AI coding tools, as encouraged by the assignment. What follows is the honest breakdown: the prompts I gave, what the AI generated, and what I designed, decided, or corrected myself.

## How the collaboration worked

I did not prompt for code first. The build was preceded by a long design conversation in which I made the structural decisions and the AI proposed/challenged alternatives. Code generation started only after the design was locked.

## My design decisions (before any code)

- Choice of Problem A; LangGraph + Gemini as the stack; Poetry for deps
- The four analyst personas (Fundamentalist, Momentum/Trend, Quality/Moat, Risk/Macro): I wrote the persona definitions (philosophy, priorities, buy/sell triggers, personalities, typical arguments) that became `lenses.py` and the system prompts, and rejected an earlier AI-proposed set (Value/Growth/Macro/Quant) plus a devil's-advocate lens
- Round 1 must be open-ended discovery: no stances before research (anchoring concern)
- Shared tool registry: agents call tools, never define them; one tool may wrap several API calls
- Fetch-once-then-freeze market data (consistency and reproducibility over freshness), decided after explicitly weighing eager vs on-demand fetching
- Historical price data must NOT be embedded; numbers computed in code, text in the vector store
- Both corpus (provided datasets) and live yfinance must work in one system
- Drop the frontend; CLI live view + SSE endpoint instead. Drop Typer for argparse
- No magic numbers in code: all constants in `config.py` (pydantic-settings), secrets in `.env`
- Commit discipline: one commit per concern (tools, pub/sub, etc.)
- The final answer vocabulary (BUY / BUY_LIMITED with sizing / WAIT / DO_NOT_BUY)

## Representative prompts I gave the AI

(Paraphrased where the original was conversational; the sequence is faithful. Bracketed notes record where the direction came from me rather than the tool.)

1. "It should work for a dataset as well as live real-world data from yfinance. Agents should decide what data to fetch, analyze it through their own lens, then debate." [The AI's first proposal scoped the evidence layer to the provided datasets only. I rejected that: a committee that can't look at a real ticker isn't useful, so live data became a requirement, not a stretch goal.]
2. "We'll use LangGraph for sure; Gemini for the LLM." [Stack was my call. The AI had recommended a different SDK with a hand-rolled orchestrator; I overruled it because I wanted the debate to be an explicit graph.]
3. "How do we give a turn to each agent? I'm thinking workers and queues, since four agents will be talking and we need streaming. And is RAG here overengineering? If used, why?" [I raised the queue question; after arguing it through I agreed in-process asyncio was the right size for this and asked for a seam where a real queue could replace it. On RAG I accepted it only once there was a concrete reason: the corpus doesn't fit in every prompt under a token budget, so retrieval is a budget mechanism here, not decoration.]
4. "Dry-run the model: I give 'Should I buy Nvidia now?' and I want to see exactly how one agent listens to the other three and modifies its answer." [In this exchange I defined the final answer vocabulary myself: do not buy / not safe, only this much because of this / yes, buy / wait, it might swing. That became the memo's recommendation enum, and I made "show me the update mechanism" a hard requirement, which is where must_address came from.]
5. "The tools are shared. Agents call them on demand; they don't create their own. One tool can wrap multiple API calls. And round 1 should stay open-ended, no debate yet, there is still much to learn from the other findings. Correct me if I'm wrong. Also: how many states per agent, what goes into embeddings, who decides chunks, what chunking strategy?" [Both rules were mine and both survived into the final design.]
6. "We also need complete historical stock data. Think very hard about whether we fetch everything at once or on demand; this is decision-based, we cannot rely on hallucinated data. It's not trading, so a few hours of staleness doesn't matter." [I set those constraints; we went back and forth on whether price history belongs in the vector store and landed on computing every number in code instead, with a snapshot frozen at debate start.]
7. "First define what the four analysts are; generally, who are the ones with different opinions?" [I then wrote the four persona definitions myself: philosophies, time horizons, buy/sell triggers, personalities, typical arguments. The AI had proposed a Value/Growth/Macro/Quant set; I replaced it with mine because Growth and Quant overlapped, and I dropped its devil's-advocate suggestion as noise.]
8. "Drop the frontend. Do everything in the CLI. Can we still stream there?" [My scope cut. The AI pushed back once, correctly, when I also wanted to drop the CLI framework entirely: a CLI entrypoint is a required core item, so we kept a minimal argparse entrypoint and put the streaming effort into the terminal live view plus the SSE endpoint.]
9. "Use Poetry, pydantic-settings for env, and no magic numbers or strings in code. Constants in config.py, secrets in dotenv. One commit per concern: tools get their own commit, pub/sub gets its own commit. Build it end to end according to the commits." [House rules, non-negotiable, and they're visible in the git history.]

## What the AI wrote vs what got reworked

The AI wrote the first pass of most modules: the Pydantic models, the ledger, the policy classes, disagreement detection, the LangGraph wiring, the CLI renderer, the FastAPI endpoints, and the tests. Almost none of it survived first contact with real runs unchanged. The build went through repeated cycles of run it, watch it fail or behave badly, decide what the behavior should be, and change it. Each cycle is its own commit:

- `ToolRequest.args: dict` became flat typed fields after every structured research plan failed at runtime (a Gemini structured-output limitation)
- Thinking-token accounting (`thinking_budget`) was added after silent JSON truncation broke argue calls
- The budget model went through four iterations: output-only reservations, then input-aware reservations, then per-kind schema overheads, then judge and tie-break caps plus synthesis-pool isolation plus a realistic stop rule. Each step came from watching a real run overshoot a pool or starve a round
- The corpus entity filter matched exactly and silently returned nothing; fixed with normalized entity keys after a run confidently concluded "no data exists" about a company the corpus covers in detail
- The exploit allocator originally spread budget across every party and starved them all; changed to fund the top contested parties at realistic allocations and drop the rest explicitly
- An event-ordering bug (`debate_finished` published after the trace file closed) was caught by the orchestrator test

## What was hardest

The budget economics. Making "fixed token budget" real, with input tokens, schema overhead, and hidden thinking tokens all counting, turned out to be the actual engineering problem of this assignment. Every naive version either overshot silently or starved agents into producing truncated JSON. The final design (reserve-before-call with conservative input estimates, per-kind overheads, skip-and-log instead of truncate, pool isolation for synthesis) came from watching six real runs fail in six different ways.

Second hardest: making disagreement real rather than cosmetic. The lens `does_not_weigh` lists, the no-stance discovery round, and claim-level `must_address` obligations are what stopped the committee from converging into four flavors of the same opinion.
