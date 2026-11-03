"""Offline extraction, one-pass experience judging, deterministic scores and HTML."""

import argparse
import json
from pathlib import Path

from .common import DATA, load_gold, read, write


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("catalog", help="List frozen environments and scoring schedules")
    p = sub.add_parser("performance", help="Compute B/W/D from a complete paired run, no API calls")
    p.add_argument("--run", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p = sub.add_parser("prepare", help="Freeze full-corpus unit-judge requests, no API calls")
    p.add_argument("--run", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--judge-model", required=True)
    p.add_argument("--judge-base-url", required=True)
    p.add_argument("--judge-api", choices=("responses", "chat_completions"), default="responses")
    p.add_argument("--reasoning-effort", default="medium", help="Use 'omit' for models without this parameter")
    p.add_argument("--max-output-tokens", type=int, default=14000)
    p = sub.add_parser("judge", help="Call your configured judge; all other commands are offline")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--workers", type=int, default=16)
    p = sub.add_parser("score", help="Replay judgments and compute all seven scores")
    p.add_argument("--output", type=Path, required=True)
    p = sub.add_parser("report", help="Build static HTML/CSV from version-matched score files")
    p.add_argument("--inputs", type=Path, nargs="+", required=True)
    p.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "catalog":
        value = read(DATA / "manifest.json")["cases"]
    elif args.command == "performance":
        from .extract import from_run
        from .scores import score_pair
        pair, _ = from_run(args.run)
        value = score_pair(pair, load_gold(pair["case_id"]))
        write(args.output, value)
        value = value["metrics"]
    elif args.command == "prepare":
        from .judge import configuration, prepare
        cfg = configuration(args.judge_model, args.judge_base_url, args.judge_api,
                            None if args.reasoning_effort == "omit" else args.reasoning_effort, args.max_output_tokens)
        value = prepare(args.run, args.output, cfg)
    elif args.command == "judge":
        from .judge import run
        value = run(args.output, args.workers)
    elif args.command == "score":
        from .judge import score
        value = score(args.output)
    else:
        from .report import generate
        value = generate([read(p) for p in args.inputs], args.output)
    print(json.dumps(value, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
