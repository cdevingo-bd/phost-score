"""Lab results built by hand, shaped like what ``phost_lab.runner.run_graph`` returns."""

import copy

import pytest

STAGE = "newton.linear_solve"


def make_cell(worlds=8, speedup=2.0, step_ms=1.0, accuracy_ratio=1.0, native_spread=0.01, other=0, ok=True,
              problems=(), accuracy_gate=True, **extra):
    cell = dict(worlds=worlds, step_ms=step_ms, native_ms=step_ms * speedup, speedup=speedup,
                native_spread=native_spread, rms_dqpos=1e-6, control_rms_dqpos=1e-6, accuracy_ratio=accuracy_ratio,
                stage_metrics={STAGE: dict(calls=10)}, advisories=[], other_gpu_processes=other,
                verdict=dict(ok=ok, problems=list(problems), notes=[],
                             gates=dict(finite=True, accuracy=accuracy_gate, stage=ok)))
    cell.update(extra)
    return cell


def make_result(classes=None, cells=None, scene="trampoline12:impact", device="RTX Test", impl="a/b", hidden=False,
                params=None):
    classes = {STAGE: "exact"} if classes is None else classes
    graph = dict(name="g", host="mjwarp", plugins=[], note="",
                 stages={s: dict(impl=impl, params=params or {}) for s in classes})
    return dict(graph=graph, scene=scene, scene_hidden=hidden, stage_classes=dict(classes),
                config=dict(worlds=[8], steps=20),
                meta=dict(device=device, mujoco="3.17.0", warp="1.16.0", nv=330, timestep=0.002, date="2026-01-01"),
                cells=cells if cells is not None else [make_cell()], comparable_to_native=True, **{"pass": True})


@pytest.fixture
def result():
    return make_result()


@pytest.fixture
def clone():
    return copy.deepcopy
