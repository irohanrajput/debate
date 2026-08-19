from __future__ import annotations

import sys

from rich.console import Console

from committee.models import TraceEvent

_MODE_STYLE = {"DISCOVER": "cyan", "EXPLORE": "yellow", "EXPLOIT": "red"}


class ConsoleRenderer:
    """Bus subscriber rendering the debate live in the terminal."""

    def __init__(self, plain: bool | None = None) -> None:
        self.plain = plain if plain is not None else not sys.stdout.isatty()
        self.console = Console(highlight=False)

    def __call__(self, event: TraceEvent) -> None:
        line = self._format(event)
        if line is None:
            return
        if self.plain:
            print(line)
        else:
            self.console.print(line)

    def _format(self, e: TraceEvent) -> str | None:
        p = e.payload
        match e.type:
            case "debate_started":
                return f"[bold]debate[/bold] {p['thesis']} | budget={p['budget']} policy={p['policy']}"
            case "budget_decision":
                mode = p.get("mode", "")
                style = _MODE_STYLE.get(mode, "white")
                return (f"[{style}]R{e.round} {mode}[/{style}] lenses={','.join(p.get('selected', []))} — {p.get('rationale', '')}")
            case "agent_started":
                return f"  R{e.round} [dim]{e.lens}[/dim] {p.get('phase')}…"
            case "research_query":
                return f"  R{e.round} [dim]{e.lens}[/dim] → {p.get('tool')}({p.get('args')})"
            case "agent_done":
                stance = p.get("stance") or p.get("lean") or ""
                return f"  R{e.round} [bold]{e.lens}[/bold] done: {stance} — {p.get('summary', '')[:120]}"
            case "agent_skipped":
                return f"  R{e.round} [red]{e.lens} SKIPPED[/red]: {p.get('reason')}"
            case "disagreement_update":
                return (f"[magenta]R{e.round} disagreement={p.get('score')} delta={p.get('delta')} "
                        f"contested={len(p.get('contested', []))} converged={p.get('converged')}[/magenta]")
            case "contested_claim":
                return f"  [red]! contested[/red] {p.get('claim_id')} ({p.get('owner')} vs {','.join(p.get('against', []))}) score={p.get('score')}"
            case "convergence":
                return f"[green]R{e.round} converged (score={p.get('score')})[/green]"
            case "budget_update":
                pools = p.get("pools", {})
                bits = " | ".join(f"{k}: {v['spent']}/{v['size']}" for k, v in pools.items())
                return f"  [dim]budget {p.get('spent')}/{p.get('total')} ({bits})[/dim]"
            case "resolution":
                return f"[blue]resolve[/blue] {p.get('claim_id')} [{p.get('method')}] {p.get('verdict', '')[:100]}"
            case "synthesis_started":
                return "[bold]chair synthesizing…[/bold]"
            case "memo_ready":
                return f"[bold green]MEMO: {p.get('recommendation')}[/bold green] ({p.get('confidence')}) — {p.get('headline')}"
            case "error":
                return f"  [red]error[/red] {e.lens or ''} {p.get('where', '')}: {p.get('detail', '')[:150]}"
            case _:
                return None
