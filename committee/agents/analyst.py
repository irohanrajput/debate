from __future__ import annotations

from pathlib import Path
from typing import Awaitable, Callable

from committee.config import settings
from committee.agents.lenses import LensSpec
from committee.evidence.tools import ToolBox, tool_catalog
from committee.debate.budget import InsufficientBudget, estimate_tokens
from committee.llm.client import LLMProvider, SchemaError
from committee.models import (
    Analysis,
    AnalystMemory,
    AnalystPosition,
    Evidence,
    Findings,
    Mode,
    ResearchPlan,
    Thesis,
    Tier,
    Usage,
)

_PROMPTS = Path(__file__).parent / "prompts"

EmitFn = Callable[[str, dict], Awaitable[None]]


# load a prompt template from agents/prompts/
def _prompt(name: str) -> str:
    return (_PROMPTS / f"{name}.md").read_text()


# render the persona system prompt from the lens spec
def _system_for(lens: LensSpec) -> str:
    return _prompt("analyst_system").format(
        title=lens.title, core_question=lens.core_question, philosophy=lens.philosophy,
        prioritizes="; ".join(lens.prioritizes), does_not_weigh="; ".join(lens.does_not_weigh),
        horizon=lens.horizon, buy_when=lens.buy_when, sell_when="; ".join(lens.sell_when),
        personality=lens.personality, typical_argument=lens.typical_argument,
    )


def _render_evidence(evidence: list[Evidence], cap: int | None = None) -> str:
    """Most recent evidence first (targeted fetches), bounded by a char cap.
    When an analysis already distilled the evidence, a smaller cap suffices."""
    if not evidence:
        return "(no evidence gathered)"
    cap = cap or settings.evidence_render_char_cap
    lines: list[str] = []
    used = 0
    for ev in reversed(evidence):
        line = f"{ev.id} [{ev.source}:{ev.ref}] {ev.snippet}"
        if used + len(line) > cap:
            break
        lines.append(line)
        used += len(line)
    return "\n".join(reversed(lines))


# once an analysis exists the raw evidence can render smaller (already distilled)
def _evidence_cap(analysis: Analysis | None) -> int:
    return settings.evidence_render_with_analysis_cap if analysis and analysis.insights else settings.evidence_render_char_cap


# append the analyst's own distilled insights to its prompt
def _render_analysis(analysis: Analysis | None) -> str:
    if analysis is None or not analysis.insights:
        return ""
    lines = "\n".join(f"- {i.text} [ev: {', '.join(i.evidence_ids) or '-'}]" for i in analysis.insights)
    return f"\n\nYour prior analysis of the evidence:\n{lines}"


# compressed view of the other analysts' stances and claims (ids kept for rebuttals)
def render_others(positions: dict[str, AnalystPosition], exclude: str) -> str:
    lines: list[str] = []
    for lens, pos in positions.items():
        if lens == exclude:
            continue
        claims = "; ".join(f"{c.id}({c.direction},imp{c.importance}): {c.text}" for c in pos.claims)
        lines.append(f"{lens}: {pos.stance} (conf {pos.confidence:.2f}). {claims}")
    return "\n".join(lines)[: settings.others_summary_char_cap] or "(none yet)"


# compressed view of the other analysts' R1 findings
def render_findings(findings: dict[str, Findings], exclude: str) -> str:
    lines: list[str] = []
    for lens, f in findings.items():
        if lens == exclude:
            continue
        obs = "; ".join(o.text for o in f.observations)
        lean = f" lean={f.provisional_lean}" if f.provisional_lean else ""
        lines.append(f"{lens}:{lean} {obs} | open: {'; '.join(f.open_questions)}")
    return "\n".join(lines)[: settings.others_summary_char_cap] or "(none yet)"


class Analyst:
    """One committee member: plan research -> execute shared tools -> argue."""

    def __init__(self, lens: LensSpec, provider: LLMProvider, toolbox: ToolBox, emit: EmitFn) -> None:
        self.lens = lens
        self._provider = provider
        self._tools = toolbox
        self._emit = emit

    # one plan LLM call -> execute the chosen shared tools -> evidence with ids
    async def research(self, thesis: Thesis, tier: Tier, max_tokens: int, focus: str,
                       round: int) -> tuple[list[Evidence], Usage]:
        user = _prompt("plan_user").format(
            thesis=thesis.statement, tool_catalog=tool_catalog(),
            preferred_tools=", ".join(self.lens.preferred_tools),
            focus=focus, max_queries=settings.max_research_queries,
        )
        system = _system_for(self.lens)
        out_cap = self._output_cap(system, user, max_tokens, settings.plan_min_output_tokens, settings.plan_tokens,
                                   overhead=settings.plan_schema_overhead_tokens)
        plan, usage = await self._provider.structured(
            schema=ResearchPlan, system=system, user=user,
            tier=tier, max_tokens=out_cap, kind="plan",
        )
        evidence: list[Evidence] = []
        for req in plan.queries[: settings.max_research_queries]:
            args = req.to_args()
            await self._emit("research_query", {"tool": req.tool, "args": args})
            evidence.extend(self._tools.call(req.tool, self.lens.name, **args))
        await self._emit("evidence_fetched", {"count": len(evidence), "rationale": plan.rationale})
        return evidence, usage

    # distill evidence into lens-specific derived judgments before arguing
    async def analyze(self, thesis: Thesis, evidence: list[Evidence], tier: Tier,
                      max_tokens: int, round: int) -> Analysis:
        user = _prompt("analyze_user").format(thesis=thesis.statement, evidence=_render_evidence(evidence))
        system = _system_for(self.lens)
        out_cap = self._output_cap(system, user, max_tokens, settings.analyze_min_output_tokens,
                                   settings.analyze_output_tokens)
        analysis, usage = await self._provider.structured(
            schema=Analysis, system=system, user=user,
            tier=tier, max_tokens=out_cap, kind="analyze",
        )
        analysis.lens, analysis.round, analysis.usage = self.lens.name, round, usage
        return analysis

    # R1 output: observations and open questions, deliberately no stance
    async def discover(self, thesis: Thesis, evidence: list[Evidence], tier: Tier,
                       max_tokens: int, round: int, analysis: Analysis | None = None) -> Findings:
        user = _prompt("findings_user").format(
            thesis=thesis.statement, evidence=_render_evidence(evidence, _evidence_cap(analysis)))
        user += _render_analysis(analysis)
        system = _system_for(self.lens)
        out_cap = self._output_cap(system, user, max_tokens, settings.min_output_tokens)
        findings, usage = await self._provider.structured(
            schema=Findings, system=system, user=user,
            tier=tier, max_tokens=out_cap, kind="findings",
        )
        findings.lens, findings.round, findings.usage = self.lens.name, round, usage
        return findings

    # the structured position call; retries once if obligations or comparisons are missing
    async def argue(self, thesis: Thesis, evidence: list[Evidence], others: str,
                    must_address: list[str], mode: Mode, tier: Tier, max_tokens: int,
                    round: int, analysis: Analysis | None = None) -> AnalystPosition:
        obligations = (
            f"You MUST respond to each of these claim ids: {', '.join(must_address)}."
            if must_address else "No mandatory claims to address."
        )
        user = _prompt("position_user").format(
            thesis=thesis.statement, evidence=_render_evidence(evidence, _evidence_cap(analysis)), others=others,
            obligations=obligations, round=round, mode=mode.value,
        )
        user += _render_analysis(analysis)
        system = _system_for(self.lens)
        out_cap = self._output_cap(system, user, max_tokens, settings.min_output_tokens)
        position, usage = await self._provider.structured(
            schema=AnalystPosition, system=system, user=user,
            tier=tier, max_tokens=out_cap, kind="argue",
        )
        missing = position.missing_responses(must_address)
        shallow = self._comparison_gap(position)
        if missing or shallow:
            complaints = []
            if missing:
                complaints.append(f"you ignored claim ids {missing}; respond to ALL required claims")
            if shallow:
                complaints.append(
                    f"claims {shallow} state raw levels without any comparison; rewrite each to compare "
                    "against the company's own history, peers, the index, or a scenario")
            retry_user = user + "\n\nYour previous answer had problems: " + "; ".join(complaints) + "."
            position, usage2 = await self._provider.structured(
                schema=AnalystPosition, system=system, user=retry_user,
                tier=tier, max_tokens=out_cap, kind="argue",
            )
            usage.input_tokens += usage2.input_tokens
            usage.output_tokens += usage2.output_tokens
            still_missing = position.missing_responses(must_address)
            if still_missing:
                raise SchemaError(f"{self.lens.name} failed to address {still_missing}")
            # a still-shallow retry is kept: prompt pressure, not a hard gate
        position.lens, position.round, position.usage = self.lens.name, round, usage
        self._assign_claim_ids(position, round)
        return position

    @staticmethod
    def _comparison_gap(position: AnalystPosition) -> list[str]:
        """Claim ids whose text contains no comparison cue. Soft check: used to
        trigger one rewrite, never to reject a position outright."""
        shallow = [c.id or f"#{i}" for i, c in enumerate(position.claims, 1)
                   if not any(cue in c.text.lower() for cue in settings.comparison_cues)]
        if len(position.claims) == 0:
            return []
        ratio_ok = (len(position.claims) - len(shallow)) / len(position.claims) >= settings.comparison_min_ratio
        return [] if ratio_ok else shallow

    @staticmethod
    def _output_cap(system: str, user: str, total_budget: int, min_out: int, hard_cap: int | None = None,
                    overhead: int | None = None) -> int:
        cap = total_budget - estimate_tokens(system + user) - (overhead if overhead is not None else settings.schema_overhead_tokens)
        if hard_cap is not None:
            cap = min(cap, hard_cap)
        if cap < min_out:
            raise InsufficientBudget(f"allocation {total_budget} cannot cover prompt + {min_out} output tokens")
        return cap

    # canonical claim ids like FUND-R2-1 so rebuttals and tie-breaks can reference them
    def _assign_claim_ids(self, position: AnalystPosition, round: int) -> None:
        for i, claim in enumerate(position.claims, start=1):
            if not claim.id or not claim.id.startswith(f"{self.lens.name[:4].upper()}"):
                claim.id = f"{self.lens.name[:4].upper()}-R{round}-{i}"
