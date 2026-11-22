import json
import multiprocessing
import threading

import pytest

from phost_score import Head, Ledger, LedgerError
from phost_score.ledger import GENESIS, canonical, check_text


@pytest.fixture
def ledger(tmp_path):
    return Ledger(tmp_path / "state" / "ledger.jsonl")


def fill(ledger, n=3):
    return [ledger.append(dict(n=i, name=f"entry-{i}")) for i in range(1, n + 1)]


def lines(ledger):
    return ledger.path.read_text().splitlines()


def rewrite(ledger, new_lines):
    ledger.path.write_text("\n".join(new_lines) + "\n")


def test_an_empty_ledger_verifies_and_has_the_genesis_head(ledger):
    v = ledger.verify()
    assert v.ok and v.records == () and v.head == Head(0, GENESIS)


def test_appending_chains_the_records(ledger):
    a, b, c = fill(ledger)
    assert (a.seq, b.seq, c.seq) == (1, 2, 3)
    assert a.prev == GENESIS and b.prev == a.hash and c.prev == b.hash
    assert ledger.verify().ok and ledger.head == Head(3, c.hash)
    assert [r.entry["n"] for r in ledger.records()] == [1, 2, 3]


def test_the_directory_is_created_on_the_first_append_only(tmp_path):
    ledger = Ledger(tmp_path / "a" / "b" / "ledger.jsonl")
    assert not ledger.path.parent.exists() and ledger.verify().ok
    ledger.append(dict(x=1))
    assert ledger.path.exists()


def test_what_is_stored_is_what_was_hashed(ledger):
    record = ledger.append(dict(values=(1, 2), nested=dict(b=1.5, a=None)))
    assert record.entry == dict(values=[1, 2], nested=dict(b=1.5, a=None))
    assert ledger.verify().ok


def test_editing_an_entry_is_reported_at_that_record_only(ledger):
    fill(ledger)
    rows = lines(ledger)
    changed = json.loads(rows[1])
    changed["entry"]["n"] = 99
    rows[1] = json.dumps(changed)
    rewrite(ledger, rows)
    problems = ledger.verify().problems
    assert len(problems) == 1 and "line 2" in problems[0] and "changed" in problems[0]


def test_editing_an_entry_and_its_hash_breaks_the_next_record(ledger):
    fill(ledger)
    rows = lines(ledger)
    changed = json.loads(rows[0])
    changed["entry"]["n"] = 99
    from phost_score.ledger import digest
    changed["hash"] = digest(changed["seq"], changed["prev"], changed["entry"])
    rows[0] = json.dumps(changed)
    rewrite(ledger, rows)
    problems = ledger.verify().problems
    assert len(problems) == 1 and "line 2" in problems[0] and "does not follow" in problems[0]


def test_removing_a_middle_record_is_reported(ledger):
    fill(ledger, 4)
    rows = lines(ledger)
    del rows[1]
    rewrite(ledger, rows)
    problems = ledger.verify().problems
    assert any("line 2" in p and "sequence" in p for p in problems) and any("does not follow" in p for p in problems)


def test_reordering_records_is_reported(ledger):
    fill(ledger, 3)
    rows = lines(ledger)
    rows[1], rows[2] = rows[2], rows[1]
    rewrite(ledger, rows)
    assert not ledger.verify().ok


def test_renumbering_a_record_is_reported(ledger):
    fill(ledger, 2)
    rows = lines(ledger)
    moved = json.loads(rows[1])
    moved["seq"] = 7
    rows[1] = json.dumps(moved)
    rewrite(ledger, rows)
    assert any("sequence is 7" in p for p in ledger.verify().problems)


def test_a_line_that_is_not_json_or_not_a_record_is_reported(ledger):
    fill(ledger, 1)
    with open(ledger.path, "a") as f:
        f.write("{not json\n")
        f.write(json.dumps(dict(seq=2)) + "\n")
    problems = ledger.verify().problems
    assert any("line 2" in p and "not JSON" in p for p in problems) and any("line 3" in p and "not a ledger record" in p for p in problems)


def test_a_nan_smuggled_into_a_line_is_reported_not_a_crash(ledger):
    fill(ledger, 1)
    with open(ledger.path, "a") as f:
        f.write('{"seq": 2, "prev": "x", "entry": {"v": NaN}, "hash": "y"}\n')
    assert any("line 2" in p for p in ledger.verify().problems)


def test_dropping_the_newest_records_is_only_caught_with_a_saved_head(ledger):
    records = fill(ledger, 3)
    saved = Head(records[2].seq, records[2].hash)
    rewrite(ledger, lines(ledger)[:2])
    assert ledger.verify().ok                                                  # the chain alone cannot tell
    problems = ledger.verify(saved).problems
    assert len(problems) == 1 and "newest records were removed" in problems[0]


def test_a_saved_head_that_matches_passes_and_one_that_does_not_fails(ledger):
    records = fill(ledger, 3)
    assert ledger.verify(Head(2, records[1].hash)).ok          # an older head is still part of the chain
    assert "differs" in ledger.verify(Head(2, "f" * 64)).problems[0]
    assert "expected head is empty" in ledger.verify(Head(0, GENESIS)).problems[0]


def test_head_parse_round_trips_and_rejects_garbage():
    head = Head(3, "ab" * 32)
    assert Head.parse(str(head)) == head
    for bad in ("", "3", "x:" + "a" * 64, "3:short"):
        with pytest.raises(ValueError):
            Head.parse(bad)


def test_an_interrupted_write_is_reported_and_blocks_appending(ledger):
    fill(ledger, 2)
    with open(ledger.path, "a") as f:
        f.write('{"seq": 3, "prev": "abc')
    assert "incomplete" in ledger.verify().problems[0]
    with pytest.raises(LedgerError, match="incomplete"):
        ledger.append(dict(n=3))
    with pytest.raises(LedgerError):
        ledger.records()
    assert len(ledger.records(strict=False)) == 2


def test_repair_moves_the_partial_line_aside_and_keeps_the_rest(ledger):
    fill(ledger, 2)
    with open(ledger.path, "a") as f:
        f.write('{"seq": 3, "prev": "abc')
    assert ledger.repair_tail() == len('{"seq": 3, "prev": "abc')
    assert ledger.verify().ok and len(ledger.records()) == 2
    kept = ledger.path.with_name("ledger.jsonl.partial-1")
    assert kept.read_text() == '{"seq": 3, "prev": "abc'
    ledger.append(dict(n=3))
    assert ledger.verify().ok and ledger.repair_tail() == 0


def test_repair_never_overwrites_an_earlier_partial_file(ledger):
    fill(ledger, 1)
    for _ in range(2):
        with open(ledger.path, "a") as f:
            f.write("partial")
        ledger.repair_tail()
    assert sorted(p.name for p in ledger.path.parent.glob("*.partial-*")) == ["ledger.jsonl.partial-1", "ledger.jsonl.partial-2"]


def test_a_damaged_ledger_refuses_new_records(ledger):
    fill(ledger, 2)
    rows = lines(ledger)
    rows[0] = rows[0].replace('"n":1', '"n":5')
    rewrite(ledger, rows)
    with pytest.raises(LedgerError, match="damaged"):
        ledger.append(dict(n=3))
    assert len(lines(ledger)) == 2


def test_a_value_json_cannot_hold_is_refused_and_nothing_is_written(ledger):
    for bad in (float("nan"), float("inf")):
        with pytest.raises(LedgerError, match="cannot represent"):
            ledger.append(dict(v=bad))
    assert not ledger.path.exists() or ledger.path.read_text() == ""


def test_threads_appending_together_get_distinct_consecutive_numbers(ledger):
    workers = [threading.Thread(target=lambda i=i: [ledger.append(dict(w=i, k=k)) for k in range(10)]) for i in range(4)]
    for w in workers:
        w.start()
    for w in workers:
        w.join()
    v = ledger.verify()
    assert v.ok and [r.seq for r in v.records] == list(range(1, 41))


def _append_many(path, worker):
    ledger = Ledger(path)
    for k in range(15):
        ledger.append(dict(w=worker, k=k))


def test_processes_appending_together_keep_one_valid_chain(ledger):
    ctx = multiprocessing.get_context("spawn")
    procs = [ctx.Process(target=_append_many, args=(str(ledger.path), i)) for i in range(3)]
    for p in procs:
        p.start()
    for p in procs:
        p.join(60)
        assert p.exitcode == 0
    v = ledger.verify()
    assert v.ok and len(v.records) == 45
    assert sorted((r.entry["w"], r.entry["k"]) for r in v.records) == sorted((w, k) for w in range(3) for k in range(15))


def test_canonical_form_ignores_key_order_and_rejects_nan():
    assert canonical(dict(b=1, a=2)) == canonical(dict(a=2, b=1)) == '{"a":2,"b":1}'
    with pytest.raises(LedgerError):
        canonical(dict(v=float("nan")))


def test_reformatting_the_file_does_not_break_verification(ledger):
    fill(ledger, 2)
    rewrite(ledger, [json.dumps(json.loads(row), indent=None, sort_keys=False, separators=(", ", ": ")) for row in lines(ledger)])
    assert check_text(ledger.path.read_text()).ok
