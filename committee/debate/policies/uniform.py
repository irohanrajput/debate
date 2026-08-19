from __future__ import annotations

from committee.config import settings
from committee.debate.budget import Ledger, Pool
from committee.debate.policies.allocation import share_of_remaining, split_evenly, uniform_tiers
from committee.debate.policies.base import BudgetPolicy, register_policy
from committee.models import BudgetDecision, DebateState, Mode, Tier


@register_policy("uniform")
class UniformPolicy(BudgetPolicy):
    """Baseline: every lens, equal split, flash tier, every round."""

    def allocate(self, state: DebateState | None, ledger: Ledger, round: int, lenses: list[str]) -> BudgetDecision:
        mode = Mode.DISCOVER if round == 1 else Mode.EXPLORE
        research = split_evenly(share_of_remaining(ledger, Pool.RESEARCH, settings.explore_research_frac), lenses)
        argue = split_evenly(share_of_remaining(ledger, Pool.DEBATE, settings.explore_argue_frac), lenses, floor=settings.argue_floor_tokens)
        return BudgetDecision(
            round=round, mode=mode, selected_lenses=lenses,
            tier_by_lens=uniform_tiers(lenses, Tier.FLASH),
            research_by_lens=research, argue_by_lens=argue,
            rationale=f"uniform baseline: {len(lenses)} lenses, equal split, flash tier",
            state_snapshot=state,
        )
