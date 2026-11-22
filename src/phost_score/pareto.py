"""Ranking on several objectives at once, without inventing a weighted sum.

A point dominates another when it is at least as good on every objective and strictly better on one. Rank 0 is
the Pareto front: nothing beats those points on all objectives together. Rank 1 is the front once rank 0 is
removed, and so on.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal, Mapping, Sequence


@dataclass(frozen=True)
class Objective:
    name: str
    sense: Literal["max", "min"]

    def __post_init__(self) -> None:
        if self.sense not in ("max", "min"):
            raise ValueError(f"objective {self.name!r}: sense must be 'max' or 'min', not {self.sense!r}")

    def better(self, a: float, b: float) -> bool:
        return a > b if self.sense == "max" else a < b


def _check(point: Mapping[str, float], objectives: Sequence[Objective]) -> None:
    for objective in objectives:
        value = point.get(objective.name)
        if value is None or not isinstance(value, (int, float)) or not math.isfinite(value):
            raise ValueError(f"objective {objective.name!r} is missing or not a finite number: {value!r}")


def dominates(a: Mapping[str, float], b: Mapping[str, float], objectives: Sequence[Objective]) -> bool:
    """True when ``a`` is at least as good as ``b`` everywhere and strictly better somewhere."""
    _check(a, objectives)
    _check(b, objectives)
    strictly = False
    for objective in objectives:
        if objective.better(b[objective.name], a[objective.name]):
            return False
        strictly = strictly or objective.better(a[objective.name], b[objective.name])
    return strictly


def ranks(points: Sequence[Mapping[str, float]], objectives: Sequence[Objective]) -> list[int]:
    """Front index of each point (0 is the Pareto front). Equal points share a rank."""
    if not objectives:
        raise ValueError("at least one objective is needed to rank anything")
    for point in points:
        _check(point, objectives)
    remaining = set(range(len(points)))
    result = [0] * len(points)
    level = 0
    while remaining:
        front = {i for i in remaining
                 if not any(dominates(points[j], points[i], objectives) for j in remaining if j != i)}
        for i in front:
            result[i] = level
        remaining -= front
        level += 1
    return result
