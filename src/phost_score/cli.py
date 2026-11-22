"""``phost-score``: record lab results in a local ledger, rank them, check the ledger, run an entry again."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .board import leaderboard
from .entry import SELF_REPORTED, Entry, make_entries
from .ledger import Head, Ledger, LedgerError
from .paths import default_ledger
from .reverify import reverify
from .rules import CLASSES


def _fail(message: str) -> int:
    print(f"error: {message}", file=sys.stderr)
    return 2


def _submit(ledger: Ledger, args) -> int:
    try:
        result = json.loads(Path(args.result).read_text())
    except (OSError, json.JSONDecodeError) as error:
        return _fail(f"cannot read {args.result}: {error}")
    entries = make_entries(result, label=args.label, trust=SELF_REPORTED)
    if not entries:
        return _fail(f"{args.result} has no cells to record" + (f" ({result['error']})" if "error" in result else ""))
    for entry in entries:
        record = ledger.append(entry.to_dict())
        verdict = ("ranked: " + ", ".join(f"{k}={v:.4g}" for k, v in entry.objectives.items())) if entry.eligible \
            else "not ranked: " + "; ".join(entry.reasons)
        print(f"#{record.seq} worlds={entry.worlds} {entry.klass}: {verdict}" + ("" if entry.reliable else " (timing unreliable)"))
    return 0


def _board(ledger: Ledger, args) -> int:
    groups = leaderboard(ledger.records(), scene=args.scene, device=args.device, worlds=args.worlds, klass=args.klass,
                         trust=args.trust, label=args.label, include_unreliable=args.include_unreliable)
    if args.json:
        print(json.dumps(groups, indent=2))
        return 0
    if not groups:
        print("nothing recorded yet")
    for group in groups:
        names = [o["name"] for o in group["objectives"]]
        print(f"\n{group['scene']}  {group['device']}  worlds={group['worlds']}  {group['klass']}"
              + (f"  stages={','.join(group['stages'])}" if group["stages"] else ""))
        print("  " + "  ".join(["rank", f"{'label':<20}", f"{'trust':<13}", *[f"{n:>14}" for n in names], "seq"]))
        for row in group["rows"]:
            print("  " + "  ".join([f"{row['rank']:>4}", f"{row['label'][:20]:<20}", f"{row['trust']:<13}",
                                    *[f"{row['objectives'][n]:>14.4g}" for n in names], f"#{row['seq']}"]))
        skipped = {k: v for k, v in group["skipped"].items() if v}
        if skipped:
            print("  left out: " + ", ".join(f"{v} {k}" for k, v in skipped.items()))
    return 0


def _check(ledger: Ledger, args) -> int:
    try:
        expected = Head.parse(args.head) if args.head else None
    except ValueError as error:
        return _fail(str(error))
    verification = ledger.verify(expected)
    for problem in verification.problems:
        print(problem, file=sys.stderr)
    print(f"{len(verification.records)} records, head {verification.head}" + ("" if verification.ok else "  DAMAGED"))
    return 0 if verification.ok else 1


def _reverify(ledger: Ledger, args) -> int:
    try:
        from phost_lab import GraphSpec
        from phost_lab.isolate import run_isolated
        from phost_lab.runner import RunConfig
    except ImportError:
        return _fail("reverify needs phost-lab (and a GPU): pip install phost-score[lab]")
    record = next((r for r in ledger.records() if r.seq == args.seq), None)
    if record is None:
        return _fail(f"no record #{args.seq}")
    entry = Entry.from_dict(record.entry)

    def run(graph, scene, config):
        return run_isolated(GraphSpec.from_dict(dict(graph)), scene, RunConfig.from_recorded(config),
                            scene_dirs=args.scene_dir)

    outcome = reverify(entry, run)
    word = {True: "agrees", False: "DISAGREES", None: "cannot say"}[outcome.agrees]
    print(f"#{record.seq}: {word}")
    for difference in outcome.differences:
        print("  " + difference)
    return {True: 0, False: 1, None: 3}[outcome.agrees]


def _repair(ledger: Ledger, args) -> int:
    moved = ledger.repair_tail()
    print(f"moved {moved} bytes of an interrupted write aside" if moved else "nothing to repair")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="phost-score", allow_abbrev=False)
    ap.add_argument("--ledger", type=Path, default=None, help=f"ledger file (default {default_ledger()})")
    sub = ap.add_subparsers(dest="command", required=True)
    s = sub.add_parser("submit", allow_abbrev=False, help="record a lab result (phost-lab run --out result.json)")
    s.add_argument("result")
    s.add_argument("--label", default="", help="a name for this submission")
    b = sub.add_parser("board", allow_abbrev=False, help="rank what is recorded")
    b.add_argument("--scene")
    b.add_argument("--device")
    b.add_argument("--worlds", type=int)
    b.add_argument("--class", dest="klass", choices=CLASSES)
    b.add_argument("--trust")
    b.add_argument("--label")
    b.add_argument("--include-unreliable", action="store_true")
    b.add_argument("--json", action="store_true")
    c = sub.add_parser("check", allow_abbrev=False, help="verify the ledger's hash chain")
    c.add_argument("--head", help="a head (<seq>:<hash>) saved earlier; detects removed newest records")
    r = sub.add_parser("reverify", allow_abbrev=False, help="run a recorded entry again and compare")
    r.add_argument("seq", type=int)
    r.add_argument("--scene-dir", action="append", default=[])
    sub.add_parser("repair", allow_abbrev=False, help="move an interrupted last write aside")
    args = ap.parse_args(argv)
    ledger = Ledger(args.ledger or default_ledger())
    handlers = dict(submit=_submit, board=_board, check=_check, reverify=_reverify, repair=_repair)
    try:
        return handlers[args.command](ledger, args)
    except LedgerError as error:
        return _fail(str(error))


if __name__ == "__main__":
    raise SystemExit(main())
