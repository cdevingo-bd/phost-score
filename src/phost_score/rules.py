"""How a run is judged. The swap class of what was swapped decides the rules, because a different kind of
replacement is allowed to be different in different ways.

    exact           same math, different algorithm: gated on correctness, ranked on speedup
    approximate     same math, bounded error: gated on correctness, ranked on speedup AND accuracy (a front)
    model_changing  different physics: there is no native answer to match, so it is gated on its own reference law
                    and ranked on step time only, among runs with the same stages
    observer        reads state and changes nothing: gated on leaving the trajectory alone, ranked on overhead

A run that selects no stage is the control and is never ranked.

The correctness gates themselves (finite state, accuracy against the native control, each stage's metrics against
its contract) were applied by the lab and are read here from the result's ``verdict.gates``. The scorer adds the
rules that depend on the class, and decides whether the timing can be trusted.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping

from .pareto import Objective

EXACT, APPROXIMATE, MODEL_CHANGING, OBSERVER, CONTROL = "exact", "approximate", "model_changing", "observer", "control"
CLASSES = (EXACT, APPROXIMATE, MODEL_CHANGING, OBSERVER)

# A graph is judged by the strictest thing in it: one model-changing stage means there is no native answer at all.
_STRICTEST_FIRST = (MODEL_CHANGING, APPROXIMATE, EXACT, OBSERVER)

OBJECTIVES: dict[str, tuple[Objective, ...]] = {
    EXACT: (Objective("speedup", "max"),),
    APPROXIMATE: (Objective("speedup", "max"), Objective("accuracy_ratio", "min")),
    MODEL_CHANGING: (Objective("step_ms", "min"),),
    OBSERVER: (Objective("overhead", "min"),),
}


@dataclass(frozen=True)
class Rules:
    max_native_spread: float = 0.10     # timing is unreliable when the two native runs differ by more than this
    idle_gpu: bool = True               # timing is unreliable when other processes used the GPU (or this is unknown)


@dataclass(frozen=True)
class Judgement:
    klass: str
    eligible: bool                      # may it be ranked at all
    reliable: bool                      # can its timing be trusted
    reasons: tuple[str, ...]            # why it is not eligible
    notes: tuple[str, ...]              # things worth knowing about an eligible run
    objectives: Mapping[str, float]     # empty unless eligible


def graph_class(stage_classes: Mapping[str, str]) -> str:
    """The class a graph is judged under. Raises ValueError on a class this version does not know."""
    unknown = sorted(set(stage_classes.values()) - set(CLASSES))
    if unknown:
        raise ValueError(f"unknown swap class {unknown[0]!r}")
    present = set(stage_classes.values())
    return next((c for c in _STRICTEST_FIRST if c in present), CONTROL)


def _number(value: Any) -> float | None:
    return float(value) if isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value) \
        else None


def judge_cell(result: Mapping[str, Any], cell: Mapping[str, Any], rules: Rules = Rules()) -> Judgement:
    """Judge one cell (one world count) of a lab result."""
    stage_classes = result.get("stage_classes")
    if stage_classes is None:
        return Judgement(CONTROL, False, False, ("the result does not say how its stages are judged "
                                                  "(no stage_classes); it predates scoring",), (), {})
    try:
        klass = graph_class(stage_classes)
    except ValueError as error:
        return Judgement(CONTROL, False, False, (str(error),), (), {})
    if klass == CONTROL:
        return Judgement(CONTROL, False, False, ("no stage is selected: this is the control, and is not ranked",), (), {})

    reliable, notes = _reliability(cell, rules)
    verdict = cell.get("verdict")
    if not isinstance(verdict, Mapping):
        return Judgement(klass, False, reliable, ("the cell has no verdict",), tuple(notes), {})
    if not verdict.get("ok"):
        problems = tuple(verdict.get("problems") or ("the lab's verdict is not ok",))
        return Judgement(klass, False, reliable, problems, tuple(notes), {})

    speedup, step_ms, native_ms = _number(cell.get("speedup")), _number(cell.get("step_ms")), _number(cell.get("native_ms"))
    if not all(v is not None and v > 0 for v in (speedup, step_ms, native_ms)):
        return Judgement(klass, False, reliable, ("the timings are missing or not positive numbers",), tuple(notes), {})

    if klass != MODEL_CHANGING and verdict.get("gates", {}).get("accuracy") is None:
        notes.append("accuracy against native could not be checked (the native control is at rounding-noise level); "
                     "the stage's own checks still applied")
    objectives: dict[str, float] = {}
    if klass in (EXACT, APPROXIMATE):
        objectives["speedup"] = speedup
    if klass == APPROXIMATE:
        ratio = _number(cell.get("accuracy_ratio"))
        if ratio is None:
            return Judgement(klass, False, reliable, ("accuracy cannot be measured on this scene: the native control "
                                                      "is at rounding-noise level, so an approximation cannot be ranked",),
                             tuple(notes), {})
        objectives["accuracy_ratio"] = ratio
    elif klass == MODEL_CHANGING:
        objectives["step_ms"] = step_ms
        notes.append("ranked on step time only, against runs with the same stages; whether two runs implement the "
                     "same law is not checked here")
    elif klass == OBSERVER:
        objectives["overhead"] = step_ms / native_ms - 1.0
    notes += [str(a) for a in cell.get("advisories") or ()]
    return Judgement(klass, True, reliable, (), tuple(notes), objectives)


def _reliability(cell: Mapping[str, Any], rules: Rules) -> tuple[bool, list[str]]:
    notes: list[str] = []
    spread = _number(cell.get("native_spread"))
    if spread is None:
        notes.append("timing reliability unknown: the native control spread is missing")
    elif spread > rules.max_native_spread:
        notes.append(f"timing unreliable: the two native runs differ by {spread:.1%}")
    others = cell.get("other_gpu_processes")
    if rules.idle_gpu and others != 0:
        notes.append(f"{others} other GPU processes were running (-1: unknown)")
    return not notes, notes
