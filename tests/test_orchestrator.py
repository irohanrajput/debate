import json
from pathlib import Path

from committee.debate.graph import run_debate
from committee.evidence.corpus import NullCorpus
from committee.llm.fake import FakeEmbedder, FakeProvider
from committee.models import Mode, Thesis
from tests.conftest import full_scripts

BUDGET = 60000


async def test_full_debate_with_fake_llm(tmp_path: Path):
    provider = FakeProvider(full_scripts())
    trace = await run_debate(
        thesis=Thesis(statement="Should we buy TestCorp?"),
        budget=BUDGET, policy="explore_exploit", offline=True,
        run_dir=tmp_path, provider=provider, embedder=FakeEmbedder(), corpus=NullCorpus(),
    )

    assert len(trace.rounds) == 3
    assert [r.mode for r in trace.rounds] == [Mode.DISCOVER, Mode.EXPLORE, Mode.EXPLOIT]
    assert len(trace.rounds[0].findings) == 4 and not trace.rounds[0].positions
    assert len(trace.rounds[1].positions) == 4

    r2_state = trace.rounds[1].state
    assert r2_state and r2_state.contested, "opposing claims must be detected as contested"
    exploit = trace.rounds[2].budget_decision
    assert set(exploit.selected_lenses) == {c.owner for c in r2_state.contested} | {
        lens for c in r2_state.contested for lens in c.against_lenses}
    assert exploit.transfers

    assert trace.memo is not None
    assert trace.totals["spent"] <= BUDGET
    assert trace.rounds[2].state and trace.rounds[2].state.converged

    run_dir = tmp_path / trace.run_id
    assert (run_dir / "trace.json").exists() and (run_dir / "memo.md").exists()
    events = [json.loads(line) for line in (run_dir / "events.jsonl").read_text().splitlines()]
    types = {e["type"] for e in events}
    assert {"budget_decision", "contested_claim", "disagreement_update", "memo_ready"} <= types


async def test_budget_enforced_across_run(tmp_path: Path):
    provider = FakeProvider(full_scripts(), tokens_per_call=50)
    trace = await run_debate(
        thesis=Thesis(statement="Budget check"), budget=BUDGET, policy="explore_exploit",
        offline=True, run_dir=tmp_path, provider=provider, embedder=FakeEmbedder(), corpus=NullCorpus(),
    )
    assert trace.totals["spent"] <= BUDGET
    pools = trace.totals["pools"]
    for pool in pools.values():
        assert pool["spent"] <= pool["size"]
