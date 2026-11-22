"""A real lab run, scored, recorded, ranked and run again. Skipped without phost-lab, mujoco_warp and a CUDA device."""

import json
from pathlib import Path

import pytest

pytest.importorskip("mujoco_warp")
pytest.importorskip("phost_lab")
wp = pytest.importorskip("warp")
try:
    wp.init()
    HAVE_CUDA = wp.is_cuda_available()
except Exception:                                   # pragma: no cover
    HAVE_CUDA = False
pytestmark = pytest.mark.skipif(not HAVE_CUDA, reason="no CUDA device")

from phost_lab import GraphSpec
from phost_lab.isolate import run_isolated
from phost_lab.runner import RunConfig

from phost_score import Entry, Ledger, leaderboard, make_entries, reverify

CONFIG = RunConfig(worlds=(8,), steps=10)
PLUGIN = "phost_coupledchol.phost_impl"


def lab_run(graph, scene="trampoline12:impact", config=CONFIG):
    return run_isolated(graph, scene, config, plugins=[PLUGIN], timeout=600)


@pytest.fixture(scope="module")
def coupled_graph():
    examples = Path(__import__("phost_lab").__file__).parents[2] / "examples"
    return GraphSpec.from_dict(json.loads((examples / "coupled.json").read_text()))


@pytest.fixture(scope="module")
def coupled_result(coupled_graph):
    return lab_run(coupled_graph)


def test_a_real_exact_swap_is_judged_exact_and_ranked(coupled_result, tmp_path):
    assert coupled_result["pass"], coupled_result.get("error")
    entry, = make_entries(coupled_result, label="coupled")
    assert entry.klass == "exact" and entry.eligible and entry.objectives["speedup"] > 1.0
    ledger = Ledger(tmp_path / "ledger.jsonl")
    ledger.append(entry.to_dict())
    group, = leaderboard(ledger.records(), include_unreliable=True)
    assert group["rows"][0]["label"] == "coupled" and group["scene"] == "trampoline12:impact"


def test_the_control_graph_is_recorded_but_not_ranked():
    from phost_lab import NATIVE
    entry, = make_entries(lab_run(NATIVE))
    assert entry.klass == "control" and not entry.eligible


def test_a_real_run_can_be_run_again_and_agrees_with_itself(coupled_result, coupled_graph):
    entry, = make_entries(coupled_result)
    if not entry.reliable:
        pytest.skip("this GPU was busy or noisy: timing cannot be compared reliably")

    def run(graph, scene, config):
        return lab_run(GraphSpec.from_dict(dict(graph)), scene, RunConfig(**{**config, "worlds": tuple(config["worlds"])}))

    outcome = reverify(Entry.from_dict(entry.to_dict()), run)
    assert outcome.agrees is True, outcome.differences
