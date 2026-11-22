"""Run a recorded entry again and say whether the new run agrees with it.

``run`` is whatever executes a graph on a scene and returns a lab result (``phost_lab.isolate.run_isolated`` is
one); this module does not import the lab or touch a GPU itself. A result from a different device is not a
disagreement, it is a result that cannot confirm or refute: ``agrees`` is then ``None``.

Timing varies from run to run, so timing objectives agree within a relative tolerance, the observer overhead (a
number near zero) within an absolute one, and the accuracy ratio (itself a ratio against a noisy control) within a
factor.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Mapping

from .entry import Entry, make_entries
from .rules import Rules


@dataclass(frozen=True)
class Tolerances:
    timing: float = 0.25            # relative: speedup and step_ms
    overhead: float = 0.05          # absolute: observer overhead
    accuracy_factor: float = 3.0    # accuracy_ratio may differ by this factor either way


@dataclass(frozen=True)
class Reverification:
    agrees: bool | None             # None: this run cannot say (different device, or the run failed to produce a cell)
    differences: tuple[str, ...]
    entry: Entry | None             # the fresh entry, when there is one


Runner = Callable[[Mapping[str, Any], str, Mapping[str, Any]], Mapping[str, Any]]


def reverify(recorded: Entry, run: Runner, *, tolerances: Tolerances = Tolerances(),
             rules: Rules = Rules()) -> Reverification:
    config = dict(recorded.config)
    config["worlds"] = [recorded.worlds]
    result = run(recorded.graph, recorded.scene, config)
    if "error" in result:
        return Reverification(None, (f"the run failed: {result['error']}",), None)
    fresh = next((e for e in make_entries(result, rules=rules) if e.worlds == recorded.worlds), None)
    if fresh is None:
        return Reverification(None, ("the run produced no cell for this world count",), None)
    if fresh.device != recorded.device:
        return Reverification(None, (f"measured on {fresh.device!r}, recorded on {recorded.device!r}: not comparable",),
                              fresh)
    differences = []
    if fresh.klass != recorded.klass:
        differences.append(f"judged as {fresh.klass}, recorded as {recorded.klass}")
    if fresh.eligible != recorded.eligible:
        differences.append(f"eligible is {fresh.eligible}, recorded {recorded.eligible}"
                           + (f" ({'; '.join(fresh.reasons)})" if fresh.reasons else ""))
    if fresh.eligible and recorded.eligible:
        for name, old in recorded.objectives.items():
            new = fresh.objectives.get(name)
            problem = _compare(name, old, new, tolerances)
            if problem:
                differences.append(problem)
    return Reverification(not differences, tuple(differences), fresh)


def _compare(name: str, old: float, new: float | None, tol: Tolerances) -> str | None:
    if new is None:
        return f"{name}: not measured now (recorded {old:.4g})"
    if name == "overhead":
        return None if abs(new - old) <= tol.overhead else f"overhead {new:+.1%} now, {old:+.1%} recorded"
    if name == "accuracy_ratio":
        low, high = sorted((old, new))
        if high <= tol.accuracy_factor * max(low, 1e-12) or high == low:
            return None
        return f"accuracy_ratio {new:.3g} now, {old:.3g} recorded (more than {tol.accuracy_factor:g}x apart)"
    return None if abs(new - old) <= tol.timing * abs(old) else f"{name} {new:.4g} now, {old:.4g} recorded"
