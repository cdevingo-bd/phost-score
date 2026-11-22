import datetime
import json

import pytest

from phost_score import JUDGED, SELF_REPORTED, Entry, Ledger, leaderboard, make_entries
from phost_score.entry import graph_hash, json_safe

from conftest import STAGE, make_cell, make_result

WHEN = datetime.datetime(2026, 3, 1, 12, 0, 0, 999, tzinfo=datetime.timezone.utc)


@pytest.fixture
def ledger(tmp_path):
    return Ledger(tmp_path / "ledger.jsonl")


# ---- entries ----

def test_one_entry_per_cell_with_everything_needed_to_rerun_it():
    result = make_result(cells=[make_cell(worlds=1), make_cell(worlds=8, speedup=3.0)])
    entries = make_entries(result, label="mine", when=WHEN)
    assert [e.worlds for e in entries] == [1, 8]
    e = entries[1]
    assert (e.label, e.trust, e.scene, e.device, e.klass, e.stages) == ("mine", SELF_REPORTED, "trampoline12:impact",
                                                                       "RTX Test", "exact", (STAGE,))
    assert e.graph == result["graph"] and e.config == result["config"] and e.time == "2026-03-01T12:00:00+00:00"
    assert e.metrics["speedup"] == 3.0 and e.meta["warp"] == "1.16.0" and e.objectives == {"speedup": 3.0}


def test_an_entry_survives_storage_unchanged(ledger):
    entry = make_entries(make_result(cells=[make_cell(accuracy_ratio=None)]), when=WHEN)[0]
    stored = ledger.append(entry.to_dict()).entry
    assert Entry.from_dict(stored) == entry and Entry.from_dict(stored).id == entry.id


def test_the_id_changes_with_any_content():
    a, = make_entries(make_result(), when=WHEN)
    b, = make_entries(make_result(cells=[make_cell(speedup=2.1)]), when=WHEN)
    assert a.id != b.id and len(a.id) == 16


def test_a_result_that_failed_outright_makes_no_entries():
    assert make_entries(dict(error="timed out after 900 s", pass_=False)) == []


def test_ineligible_runs_are_still_recorded_with_their_reasons():
    e, = make_entries(make_result(cells=[make_cell(ok=False, problems=["non-finite qpos"])]), when=WHEN)
    assert not e.eligible and e.reasons == ("non-finite qpos",)


def test_trust_must_be_a_known_level():
    assert make_entries(make_result(), trust=JUDGED, when=WHEN)[0].trust == JUDGED
    with pytest.raises(ValueError, match="trust"):
        make_entries(make_result(), trust="verified")


def test_from_dict_rejects_a_dict_that_is_not_an_entry():
    d = make_entries(make_result(), when=WHEN)[0].to_dict()
    with pytest.raises(ValueError, match="not an entry"):
        Entry.from_dict({**d, "extra": 1})
    del d["klass"]
    with pytest.raises(ValueError, match="not an entry"):
        Entry.from_dict(d)


def test_the_graph_hash_ignores_its_name_but_not_its_choices():
    a, b = make_result()["graph"], make_result()["graph"]
    b["name"], b["note"] = "renamed", "different note"
    assert graph_hash(a) == graph_hash(b)
    assert graph_hash(a) != graph_hash(make_result(impl="other/impl")["graph"])
    assert graph_hash(a) != graph_hash(make_result(params=dict(k=1))["graph"])


def test_json_safe_replaces_only_non_finite_numbers():
    assert json_safe(dict(a=float("nan"), b=[float("inf"), 1.5, (2, None)])) == dict(a=None, b=[None, 1.5, [2, None]])


def test_a_hidden_scene_is_recorded_as_hidden_without_describing_it():
    e, = make_entries(make_result(scene="held-7", hidden=True), when=WHEN)
    assert e.scene == "held-7" and e.scene_hidden
    assert "description" not in json.dumps(e.to_dict())


# ---- leaderboards ----

def run(ledger, label, **kw):
    result = make_result(impl=f"impl/{label}", **kw)
    for entry in make_entries(result, label=label, when=WHEN):
        ledger.append(entry.to_dict())


def test_exact_runs_are_ranked_by_speedup_best_first(ledger):
    for label, s in (("slow", 1.2), ("fast", 3.1), ("mid", 2.0)):
        run(ledger, label, cells=[make_cell(speedup=s)])
    group, = leaderboard(ledger.records())
    assert [(r["label"], r["rank"]) for r in group["rows"]] == [("fast", 0), ("mid", 1), ("slow", 2)]
    assert group["klass"] == "exact" and group["objectives"] == [dict(name="speedup", sense="max")]


def test_approximate_runs_form_a_front_not_a_single_order(ledger):
    for label, s, err in (("fast-rough", 4.0, 8.0), ("slow-exactish", 1.5, 1.0), ("middle", 2.5, 3.0), ("worse", 2.0, 5.0)):
        run(ledger, label, classes={STAGE: "approximate"}, cells=[make_cell(speedup=s, accuracy_ratio=err)])
    group, = leaderboard(ledger.records())
    ranks = {r["label"]: r["rank"] for r in group["rows"]}
    assert ranks == {"fast-rough": 0, "slow-exactish": 0, "middle": 0, "worse": 1}


def test_only_the_latest_run_of_a_label_and_graph_counts(ledger):
    """Best-of-N would reward noise: a lucky fast run must not hide a later, ordinary one."""
    run(ledger, "me", cells=[make_cell(speedup=5.0)])        # lucky
    run(ledger, "me", cells=[make_cell(speedup=2.0)])        # typical, and later
    group, = leaderboard(ledger.records())
    assert [r["objectives"]["speedup"] for r in group["rows"]] == [2.0] and group["skipped"]["superseded"] == 1


def test_a_different_graph_or_label_is_not_superseded(ledger):
    run(ledger, "me", cells=[make_cell(speedup=2.0)])
    run(ledger, "you", cells=[make_cell(speedup=3.0)])
    assert len(leaderboard(ledger.records())[0]["rows"]) == 2


def test_runs_on_different_scenes_devices_or_world_counts_are_never_ranked_together(ledger):
    run(ledger, "a", scene="s1", cells=[make_cell(worlds=8)])
    run(ledger, "b", scene="s2", cells=[make_cell(worlds=8)])
    run(ledger, "c", scene="s1", device="Other GPU", cells=[make_cell(worlds=8)])
    run(ledger, "d", scene="s1", cells=[make_cell(worlds=32)])
    groups = leaderboard(ledger.records())
    assert len(groups) == 4 and all(len(g["rows"]) == 1 for g in groups)


def test_model_changing_runs_are_ranked_only_against_the_same_stages(ledger):
    cell = make_cell(step_ms=2.0)
    run(ledger, "pd", classes={"control.law": "model_changing"}, cells=[cell])
    run(ledger, "spring", classes={"dof.passive_force": "model_changing"}, cells=[make_cell(step_ms=1.0)])
    run(ledger, "pd2", classes={"control.law": "model_changing"}, cells=[make_cell(step_ms=1.5)])
    by_stage = {tuple(g["stages"]): [r["label"] for r in g["rows"]] for g in leaderboard(ledger.records())}
    assert by_stage == {("control.law",): ["pd2", "pd"], ("dof.passive_force",): ["spring"]}


def test_observers_are_ranked_on_overhead_lowest_first(ledger):
    run(ledger, "heavy", classes={STAGE: "observer"}, cells=[make_cell(speedup=1 / 1.10)])
    run(ledger, "light", classes={STAGE: "observer"}, cells=[make_cell(speedup=1 / 1.01)])
    group, = leaderboard(ledger.records())
    assert [r["label"] for r in group["rows"]] == ["light", "heavy"]


def test_ineligible_and_unreliable_runs_are_left_out_and_counted(ledger):
    run(ledger, "good", cells=[make_cell(speedup=2.0)])
    run(ledger, "wrong", cells=[make_cell(speedup=9.0, ok=False, problems=["bad"])])
    run(ledger, "noisy", cells=[make_cell(speedup=8.0, native_spread=0.5)])
    group, = leaderboard(ledger.records())
    assert [r["label"] for r in group["rows"]] == ["good"] and group["skipped"] == dict(ineligible=1, unreliable=1, superseded=0)
    shown, = leaderboard(ledger.records(), include_unreliable=True)
    assert [r["label"] for r in shown["rows"]] == ["noisy", "good"] and not shown["rows"][0]["reliable"]


def test_a_wrong_but_fast_run_never_outranks_a_correct_one(ledger):
    run(ledger, "right", cells=[make_cell(speedup=1.1)])
    run(ledger, "cheat", cells=[make_cell(speedup=100.0, ok=False, problems=["accuracy ratio 50 exceeds 10.0"])])
    group, = leaderboard(ledger.records())
    assert [r["label"] for r in group["rows"]] == ["right"]


def test_the_control_does_not_appear_on_the_board(ledger):
    run(ledger, "control", classes={})
    group, = leaderboard(ledger.records())
    assert group["rows"] == [] and group["skipped"]["ineligible"] == 1


def test_filters_narrow_the_board(ledger):
    run(ledger, "a", scene="s1")
    run(ledger, "b", scene="s2")
    for entry in make_entries(make_result(impl="j"), label="j", trust=JUDGED, when=WHEN):
        ledger.append(entry.to_dict())
    records = ledger.records()
    assert {g["scene"] for g in leaderboard(records, scene="s1")} == {"s1"}
    assert all(r["trust"] == JUDGED for g in leaderboard(records, trust=JUDGED) for r in g["rows"])
    assert leaderboard(records, label="nobody") == [] and leaderboard(records, worlds=999) == []
    assert leaderboard(records, klass="observer") == [] and leaderboard(records, device="nope") == []


def test_judged_and_self_reported_runs_are_both_shown_with_their_trust(ledger):
    run(ledger, "me", cells=[make_cell(speedup=2.0)])
    for entry in make_entries(make_result(impl="impl/me", cells=[make_cell(speedup=2.1)]), label="me", trust=JUDGED, when=WHEN):
        ledger.append(entry.to_dict())
    group, = leaderboard(ledger.records())
    assert sorted(r["trust"] for r in group["rows"]) == [JUDGED, SELF_REPORTED]


def test_the_board_is_json_ready(ledger):
    run(ledger, "a", classes={STAGE: "approximate"}, cells=[make_cell(accuracy_ratio=2.0)])
    json.dumps(leaderboard(ledger.records()))
