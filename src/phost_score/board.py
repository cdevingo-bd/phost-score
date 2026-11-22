"""Leaderboards: runs ranked against comparable runs only.

Two runs are comparable when they were measured on the same scene, device and world count, and judged under the
same class (a model-changing run is also only comparable with runs of the same stages, because step time between
different physics says nothing). Within such a group the eligible runs are ranked on the class's objectives, by
Pareto front where there is more than one.

Only the latest run of a given label and graph counts. Taking the best of several runs would reward noise: run
anything enough times and one of the runs looks fast.
"""

from __future__ import annotations

from typing import Any, Iterable

from .entry import Entry
from .ledger import Record
from .pareto import ranks
from .rules import MODEL_CHANGING, OBJECTIVES

GroupKey = tuple


def group_key(entry: Entry) -> GroupKey:
    return (entry.scene, entry.device, entry.worlds, entry.klass,
            entry.stages if entry.klass == MODEL_CHANGING else ())


def leaderboard(records: Iterable[Record], *, scene: str | None = None, device: str | None = None,
                worlds: int | None = None, klass: str | None = None, trust: str | None = None,
                label: str | None = None, include_unreliable: bool = False) -> list[dict[str, Any]]:
    """Groups of ranked rows, most competitive group first by size. JSON-ready.

    Each row: ``rank`` (0 is the Pareto front), ``seq`` and ``id`` of the record, the entry's label, trust, graph
    hash, objectives and notes. ``skipped`` counts what was left out of the group and why.
    """
    groups: dict[GroupKey, dict[str, Any]] = {}
    latest: dict[tuple, Record] = {}
    for record in records:
        entry = Entry.from_dict(record.entry)
        if not _matches(entry, scene, device, worlds, klass, trust, label):
            continue
        group = groups.setdefault(group_key(entry), dict(skipped=dict(ineligible=0, unreliable=0, superseded=0)))
        identity = (group_key(entry), entry.label, entry.graph_hash, entry.trust)
        if identity in latest:
            group["skipped"]["superseded"] += 1
        latest[identity] = record                     # later records replace earlier ones
    rows_by_group: dict[GroupKey, list[tuple[Record, Entry]]] = {}
    for (key, *_), record in latest.items():
        entry = Entry.from_dict(record.entry)
        skipped = groups[key]["skipped"]
        if not entry.eligible:
            skipped["ineligible"] += 1
        elif not entry.reliable and not include_unreliable:
            skipped["unreliable"] += 1
        else:
            rows_by_group.setdefault(key, []).append((record, entry))
    out = []
    for key, group in groups.items():
        rows = rows_by_group.get(key, [])
        klass_ = key[3]
        objectives = OBJECTIVES.get(klass_, ())
        ranked = ranks([e.objectives for _, e in rows], objectives) if rows else []
        primary = objectives[0] if objectives else None
        listing = [dict(rank=rank, seq=record.seq, id=entry.id, label=entry.label, trust=entry.trust,
                        graph_hash=entry.graph_hash, objectives=dict(entry.objectives), reliable=entry.reliable,
                        time=entry.time, notes=list(entry.notes))
                   for rank, (record, entry) in zip(ranked, rows)]
        if primary:
            sign = -1 if primary.sense == "max" else 1
            listing.sort(key=lambda r: (r["rank"], sign * r["objectives"][primary.name], r["seq"]))
        out.append(dict(scene=key[0], device=key[1], worlds=key[2], klass=klass_, stages=list(key[4]),
                        objectives=[vars(o).copy() for o in objectives], rows=listing, skipped=group["skipped"]))
    out.sort(key=lambda g: (-len(g["rows"]), g["scene"], g["device"], g["worlds"], g["klass"]))
    return out


def wins(records: Iterable[Record], *, scene: str | None = None, device: str | None = None,
         min_speedup: float = 1.0) -> list[dict[str, Any]]:
    """The entries that beat the host's own step, strongest first. JSON-ready.

    A win keeps the answer (a model-changing entry is a different simulation, so its step time says nothing), passed its checks,
    has a timing that can be trusted, and is faster than ``min_speedup`` at every world count the same graph was measured at:
    one that wins at one world count and loses at another is not a win. As on the leaderboard, only the latest run of a graph
    under a label counts at each world count. ``speedups`` is by world count and ``lowest`` is the smallest of them, which is
    what the list is ordered on; ``seq`` is the newest record behind it and ``graph`` is the recorded graph, to run it again."""
    latest: dict[tuple, dict[int, tuple[Record, Entry]]] = {}
    for record in records:
        entry = Entry.from_dict(record.entry)
        if entry.klass == MODEL_CHANGING or not _matches(entry, scene, device, None, None, None, None):
            continue
        key = (entry.scene, entry.device, entry.graph_hash, entry.label, entry.trust, entry.klass)
        latest.setdefault(key, {})[entry.worlds] = (record, entry)             # later records replace earlier ones
    out = []
    for (scene_, device_, graph_hash, label, trust, klass), cells in latest.items():
        speeds = {worlds: entry.objectives.get("speedup") for worlds, (_, entry) in cells.items()}
        if not all(e.eligible and e.reliable and speeds[w] is not None and speeds[w] > min_speedup for w, (_, e) in cells.items()):
            continue
        newest = max((r for r, _ in cells.values()), key=lambda r: r.seq)
        entry = Entry.from_dict(newest.entry)
        out.append(dict(scene=scene_, device=device_, label=label, trust=trust, klass=klass, graph_hash=graph_hash,
                        speedups=dict(sorted(speeds.items())), lowest=min(speeds.values()), seq=newest.seq, time=entry.time,
                        graph=entry.graph, notes=list(entry.notes)))
    out.sort(key=lambda w: (-w["lowest"], w["seq"]))
    return out


def _matches(entry: Entry, scene, device, worlds, klass, trust, label) -> bool:
    return ((scene is None or entry.scene == scene) and (device is None or entry.device == device)
            and (worlds is None or entry.worlds == worlds) and (klass is None or entry.klass == klass)
            and (trust is None or entry.trust == trust) and (label is None or entry.label == label))
