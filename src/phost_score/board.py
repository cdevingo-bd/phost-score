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


def _matches(entry: Entry, scene, device, worlds, klass, trust, label) -> bool:
    return ((scene is None or entry.scene == scene) and (device is None or entry.device == device)
            and (worlds is None or entry.worlds == worlds) and (klass is None or entry.klass == klass)
            and (trust is None or entry.trust == trust) and (label is None or entry.label == label))
