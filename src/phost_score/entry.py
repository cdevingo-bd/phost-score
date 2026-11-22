"""One scored run, as it is stored: what was run, where, how it was judged, and how far to trust it."""

from __future__ import annotations

import datetime
import hashlib
import math
from dataclasses import dataclass, field
from typing import Any, Mapping

from .ledger import canonical
from .rules import Judgement, Rules, judge_cell

SELF_REPORTED = "self-reported"      # produced on the submitter's machine; nothing here can check that
JUDGED = "judged"                    # produced by a judge that ran the submission itself (see phost-author)
TRUST_LEVELS = (SELF_REPORTED, JUDGED)

_METRICS = ("speedup", "step_ms", "native_ms", "native_spread", "accuracy_ratio", "rms_dqpos", "control_rms_dqpos",
            "other_gpu_processes")
_META = ("device", "mujoco", "warp", "nv", "timestep", "date")


@dataclass(frozen=True)
class Entry:
    label: str
    trust: str
    time: str
    scene: str
    scene_hidden: bool
    device: str
    worlds: int
    klass: str
    stages: tuple[str, ...]
    graph: Mapping[str, Any]
    graph_hash: str
    config: Mapping[str, Any]
    meta: Mapping[str, Any]
    eligible: bool
    reliable: bool
    reasons: tuple[str, ...]
    notes: tuple[str, ...]
    objectives: Mapping[str, float]
    metrics: Mapping[str, Any]
    stage_metrics: Mapping[str, Any] = field(default_factory=dict)
    artifacts: Mapping[str, str] = field(default_factory=dict)      # name -> sha256 of what was submitted

    def to_dict(self) -> dict:
        d = {k: getattr(self, k) for k in self.__dataclass_fields__}
        d["stages"], d["reasons"], d["notes"] = list(self.stages), list(self.reasons), list(self.notes)
        return json_safe(d)

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "Entry":
        fields = set(cls.__dataclass_fields__)
        if set(d) - fields or fields - set(d):
            raise ValueError(f"not an entry: keys differ by {sorted(set(d) ^ fields)}")
        return cls(**{**d, "stages": tuple(d["stages"]), "reasons": tuple(d["reasons"]), "notes": tuple(d["notes"])})

    @property
    def id(self) -> str:
        return hashlib.sha256(canonical(self.to_dict()).encode()).hexdigest()[:16]


def json_safe(value: Any) -> Any:
    """The value with every non-finite float replaced by None, so it can be hashed and stored exactly."""
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, Mapping):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    return value


def graph_hash(graph: Mapping[str, Any]) -> str:
    """Identifies what was selected, not what it was called: the same stages, parameters, host and plugins hash alike."""
    identity = {k: graph.get(k) for k in ("stages", "host", "plugins")}
    return hashlib.sha256(canonical(json_safe(identity)).encode()).hexdigest()


def make_entries(result: Mapping[str, Any], *, label: str = "", trust: str = SELF_REPORTED,
                 artifacts: Mapping[str, str] | None = None, rules: Rules = Rules(),
                 when: datetime.datetime | None = None) -> list[Entry]:
    """One entry per cell of a lab result (per world count), eligible or not. A result that failed outright
    (``error``) has no cells and produces none."""
    if trust not in TRUST_LEVELS:
        raise ValueError(f"trust must be one of {TRUST_LEVELS}, not {trust!r}")
    if "error" in result:
        return []
    stamp = (when or datetime.datetime.now(datetime.timezone.utc)).replace(microsecond=0).isoformat()
    meta = result.get("meta", {})
    entries = []
    for cell in result.get("cells", ()):
        judgement: Judgement = judge_cell(result, cell, rules)
        graph = result["graph"]
        entries.append(Entry(
            label=label, trust=trust, time=stamp, scene=result["scene"], scene_hidden=bool(result.get("scene_hidden")),
            device=str(meta.get("device", "unknown")), worlds=int(cell["worlds"]), klass=judgement.klass,
            stages=tuple(sorted(result.get("stage_classes", {}))), graph=graph, graph_hash=graph_hash(graph),
            config=result.get("config", {}), meta={k: meta[k] for k in _META if k in meta},
            eligible=judgement.eligible, reliable=judgement.reliable, reasons=judgement.reasons,
            notes=judgement.notes, objectives=dict(judgement.objectives),
            metrics={k: cell[k] for k in _METRICS if k in cell}, stage_metrics=cell.get("stage_metrics", {}),
            artifacts=dict(artifacts or {})))
    return entries

