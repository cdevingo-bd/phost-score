import pytest

from phost_score import MODEL_CHANGING, Tolerances, make_entries, reverify

from conftest import STAGE, make_cell, make_result


def recorded(classes=None, **cell):
    return make_entries(make_result(classes, cells=[make_cell(**cell)]))[0]


def runner(**cell):
    return lambda graph, scene, config: make_result(cells=[make_cell(worlds=config["worlds"][0], **cell)])


def test_the_run_is_asked_for_what_was_recorded():
    seen = []

    def run(graph, scene, config):
        seen.append((graph, scene, config))
        return make_result()

    entry = recorded()
    reverify(entry, run)
    (graph, scene, config), = seen
    assert graph == entry.graph and scene == entry.scene and config["worlds"] == [8] and config["steps"] == 20


def test_a_fresh_run_within_tolerance_agrees():
    out = reverify(recorded(speedup=2.0), runner(speedup=2.3))
    assert out.agrees is True and out.differences == () and out.entry.objectives["speedup"] == 2.3


def test_a_speedup_that_does_not_reproduce_disagrees():
    out = reverify(recorded(speedup=3.0), runner(speedup=1.5))
    assert out.agrees is False and "speedup 1.5 now, 3 recorded" in out.differences[0]


def test_the_timing_tolerance_is_adjustable():
    assert reverify(recorded(speedup=2.0), runner(speedup=2.3), tolerances=Tolerances(timing=0.05)).agrees is False


def test_a_run_that_now_fails_its_checks_disagrees_with_the_reason():
    out = reverify(recorded(), runner(ok=False, problems=["non-finite qpos"]))
    assert out.agrees is False and "eligible is False" in out.differences[0] and "non-finite qpos" in out.differences[0]


def test_accuracy_ratio_agrees_within_a_factor_in_either_direction():
    old = recorded({STAGE: "approximate"}, accuracy_ratio=2.0)
    run = lambda ratio: (lambda g, s, c: make_result({STAGE: "approximate"}, cells=[make_cell(worlds=8, accuracy_ratio=ratio)]))
    assert reverify(old, run(5.0)).agrees is True and reverify(old, run(0.8)).agrees is True
    out = reverify(old, run(9.0))
    assert out.agrees is False and "accuracy_ratio" in out.differences[0]


def test_observer_overhead_is_compared_in_absolute_terms():
    old = recorded({STAGE: "observer"}, speedup=1 / 1.01)
    run = lambda overhead: (lambda g, s, c: make_result({STAGE: "observer"}, cells=[make_cell(worlds=8, speedup=1 / (1 + overhead))]))
    assert reverify(old, run(0.03)).agrees is True           # 1% against 3% is noise on a near-zero number
    assert reverify(old, run(0.30)).agrees is False


def test_a_different_device_cannot_confirm_or_refute():
    other = lambda g, s, c: make_result(device="Another GPU", cells=[make_cell(worlds=8, speedup=0.1)])
    out = reverify(recorded(), other)
    assert out.agrees is None and "not comparable" in out.differences[0]


def test_a_failed_run_cannot_confirm_or_refute():
    out = reverify(recorded(), lambda g, s, c: dict(error="timed out after 900 s", **{"pass": False}))
    assert out.agrees is None and "timed out" in out.differences[0] and out.entry is None


def test_a_result_without_the_cell_cannot_confirm_or_refute():
    out = reverify(recorded(), lambda g, s, c: make_result(cells=[make_cell(worlds=1)]))
    assert out.agrees is None and "no cell" in out.differences[0]


def test_a_change_of_class_is_a_difference():
    out = reverify(recorded({STAGE: "exact"}), lambda g, s, c: make_result({STAGE: "model_changing"}, cells=[make_cell(worlds=8)]))
    assert out.agrees is False and any("judged as model_changing" in d for d in out.differences)
    assert out.entry.klass == MODEL_CHANGING


def test_an_unranked_entry_that_stays_unranked_agrees_on_that():
    bad = recorded(ok=False, problems=["x"])
    assert reverify(bad, runner(ok=False, problems=["x"])).agrees is True
