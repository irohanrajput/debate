from __future__ import annotations

import argparse
import asyncio
from pathlib import Path

from committee.config import settings


def _cmd_ingest(args: argparse.Namespace) -> None:
    from committee.evidence.corpus import build_corpus

    count = build_corpus(Path(args.data_dir))
    print(f"ingested {count} chunks into {settings.chroma_dir}")


def _cmd_debate(args: argparse.Namespace) -> None:
    from committee.cli_render import ConsoleRenderer
    from committee.debate.graph import run_debate
    from committee.models import Thesis
    from committee.trace.writer import render_memo

    thesis = Thesis(statement=args.thesis, ticker=args.ticker, entity=args.entity)
    renderer = ConsoleRenderer(plain=args.no_tty or None)
    trace = asyncio.run(run_debate(
        thesis=thesis, budget=args.budget, policy=args.policy,
        offline=args.offline, run_dir=Path(args.out), subscribers=[renderer],
    ))
    print(f"\nrun saved: {Path(args.out) / trace.run_id}")
    if trace.memo:
        print(render_memo(trace.memo))


def _cmd_serve(args: argparse.Namespace) -> None:
    import uvicorn

    uvicorn.run("committee.api:app", host=settings.api_host, port=args.port)


def main() -> None:
    parser = argparse.ArgumentParser(prog="committee", description="Multi-agent investment committee")
    sub = parser.add_subparsers(dest="command", required=True)

    p_ingest = sub.add_parser("ingest", help="build the corpus index from datasets/docs")
    p_ingest.add_argument("data_dir", nargs="?", default=settings.data_dir)
    p_ingest.set_defaults(fn=_cmd_ingest)

    p_debate = sub.add_parser("debate", help="run a committee debate")
    p_debate.add_argument("--thesis", required=True)
    p_debate.add_argument("--ticker", default=None)
    p_debate.add_argument("--entity", default=None)
    p_debate.add_argument("--budget", type=int, default=settings.default_budget)
    p_debate.add_argument("--policy", default="explore_exploit")
    p_debate.add_argument("--offline", action="store_true")
    p_debate.add_argument("--no-tty", action="store_true")
    p_debate.add_argument("--out", default=settings.runs_dir)
    p_debate.set_defaults(fn=_cmd_debate)

    p_serve = sub.add_parser("serve", help="run the FastAPI server")
    p_serve.add_argument("--port", type=int, default=settings.api_port)
    p_serve.set_defaults(fn=_cmd_serve)

    args = parser.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
