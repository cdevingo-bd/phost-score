import pytest

from phost_score import APPROXIMATE, CONTROL, EXACT, MODEL_CHANGING, OBSERVER, Rules, graph_class, judge_cell

from conftest import STAGE, make_cell, make_result


def judge(classes=None, cell=None, **kw):
    result = make_result(classes, cells=[cell or make_cell()])
    return judge_cell(result, result["cells"][0], **kw)


# ---- which class a graph is judged under ----

@pytest.mark.parametrize("classes,expected", [
    ({}, CONTROL),
    ({"a": "exact"}, EXACT),
    ({"a": "observer"}, OBSERVER),
    ({"a": "observer", "b": "exact"}, EXACT),
    ({"a": "exact", "b": "approximate"}, APPROXIMATE),
    ({"a": "approximate", "b": "model_changing", "c": "observer"}, MODEL_CHANGING),
])
def test_a_graph_is_judged_by_the_strictest_class_in_it(classes, expected):
    assert graph_class(classes) == expected


def test_an_unknown_class_is_refused_not_guessed():
    with pytest.raises(ValueError, match="unknown swap class 'magic'"):
        graph_class({"a": "magic"})
    j = judge({"a": "magic"})
    assert not j.eligible and "magic" in j.reasons[0]


# ---- eligibility ----

def test_the_control_is_never_ranked():
    j = judge({})
    assert j.klass == CONTROL and not j.eligible and "control" in j.reasons[0] and j.objectives == {}


def test_a_result_without_stage_classes_is_refused():
    result = make_result()
    del result["stage_classes"]
    j = judge_cell(result, result["cells"][0])
    assert not j.eligible and "stage_classes" in j.reasons[0]


def test_a_failed_verdict_is_not_ranked_and_carries_the_labs_reasons():
    j = judge(cell=make_cell(ok=False, problems=["accuracy ratio 12.00 exceeds 10.0"]))
    assert not j.eligible and j.reasons == ("accuracy ratio 12.00 exceeds 10.0",) and j.objectives == {}


def test_a_cell_without_a_verdict_is_not_ranked():
    cell = make_cell()
    del cell["verdict"]
    assert not judge(cell=cell).eligible


@pytest.mark.parametrize("field,value", [("speedup", 0.0), ("speedup", -1.0), ("step_ms", float("nan")),
                                         ("native_ms", None), ("speedup", True)])
def test_timings_that_are_not_positive_numbers_are_not_ranked(field, value):
    j = judge(cell=make_cell(**{field: value}))
    assert not j.eligible and "timings" in j.reasons[0]


# ---- objectives per class ----

def test_exact_is_ranked_on_speedup():
    j = judge({STAGE: "exact"}, make_cell(speedup=2.4))
    assert j.eligible and j.objectives == {"speedup": 2.4}


def test_approximate_is_ranked_on_speedup_and_accuracy():
    j = judge({STAGE: "approximate"}, make_cell(speedup=3.0, accuracy_ratio=2.5))
    assert j.eligible and j.objectives == {"speedup": 3.0, "accuracy_ratio": 2.5}


def test_approximate_with_an_unmeasurable_accuracy_cannot_be_ranked():
    j = judge({STAGE: "approximate"}, make_cell(accuracy_ratio=None, accuracy_gate=None))
    assert not j.eligible and "cannot be measured" in j.reasons[0]


def test_exact_with_an_unmeasurable_accuracy_is_ranked_with_a_note():
    j = judge({STAGE: "exact"}, make_cell(accuracy_ratio=None, accuracy_gate=None))
    assert j.eligible and any("could not be checked" in n for n in j.notes)


def test_model_changing_is_ranked_on_step_time_never_on_speedup_over_native():
    j = judge({STAGE: "model_changing"}, make_cell(speedup=0.5, step_ms=4.0, accuracy_ratio=40.0, accuracy_gate=None))
    assert j.eligible and j.objectives == {"step_ms": 4.0}
    assert any("same stages" in n for n in j.notes) and not any("could not be checked" in n for n in j.notes)


def test_observer_is_ranked_on_overhead():
    j = judge({STAGE: "observer"}, make_cell(speedup=1 / 1.02, step_ms=1.0))
    assert j.eligible and j.objectives["overhead"] == pytest.approx(0.02)
    assert judge({STAGE: "observer"}, make_cell(speedup=1.0)).objectives["overhead"] == 0.0


def test_advisories_from_the_host_are_carried_as_notes():
    j = judge(cell=make_cell(advisories=["implicit damping needs the Euler integrator"]))
    assert "implicit damping needs the Euler integrator" in j.notes


# ---- reliability of the timing ----

def test_a_quiet_idle_run_is_reliable():
    assert judge(cell=make_cell(native_spread=0.02, other=0)).reliable


def test_a_noisy_native_control_makes_timing_unreliable_but_still_eligible():
    j = judge(cell=make_cell(native_spread=0.5))
    assert j.eligible and not j.reliable and any("differ by 50.0%" in n for n in j.notes)


def test_other_gpu_processes_make_timing_unreliable_and_unknown_counts_too():
    assert not judge(cell=make_cell(other=2)).reliable
    assert not judge(cell=make_cell(other=-1)).reliable


def test_the_rules_set_the_thresholds():
    cell = make_cell(native_spread=0.2, other=1)
    assert judge(cell=cell, rules=Rules(max_native_spread=0.3, idle_gpu=False)).reliable
    assert not judge(cell=cell, rules=Rules(max_native_spread=0.3, idle_gpu=True)).reliable


def test_a_missing_spread_is_unreliable_not_assumed_fine():
    cell = make_cell()
    del cell["native_spread"]
    j = judge(cell=cell)
    assert not j.reliable and any("unknown" in n for n in j.notes)
