import json

import pytest

from phost_score.cli import main

from conftest import STAGE, make_cell, make_result


@pytest.fixture
def ledger_path(tmp_path):
    return str(tmp_path / "ledger.jsonl")


def write(tmp_path, result, name="result.json"):
    path = tmp_path / name
    path.write_text(json.dumps(result))
    return str(path)


def cli(ledger_path, *args):
    return main(["--ledger", ledger_path, *args])


def test_submit_then_board_then_check(tmp_path, ledger_path, capsys):
    path = write(tmp_path, make_result(cells=[make_cell(worlds=1), make_cell(worlds=8, speedup=2.5)]))
    assert cli(ledger_path, "submit", path, "--label", "mine") == 0
    out = capsys.readouterr().out
    assert "#1 worlds=1 exact: ranked" in out and "#2 worlds=8 exact: ranked: speedup=2.5" in out
    assert cli(ledger_path, "board") == 0
    board = capsys.readouterr().out
    assert "trampoline12:impact" in board and "mine" in board and "self-reported" in board
    assert cli(ledger_path, "check") == 0 and "2 records" in capsys.readouterr().out


def test_submit_always_records_as_self_reported(tmp_path, ledger_path, capsys):
    with pytest.raises(SystemExit):
        cli(ledger_path, "submit", write(tmp_path, make_result()), "--trust", "judged")


def test_board_json_is_machine_readable(tmp_path, ledger_path, capsys):
    cli(ledger_path, "submit", write(tmp_path, make_result()))
    capsys.readouterr()
    assert cli(ledger_path, "board", "--json") == 0
    groups = json.loads(capsys.readouterr().out)
    assert groups[0]["rows"][0]["objectives"] == {"speedup": 2.0}


def test_board_on_an_empty_ledger_says_so(ledger_path, capsys):
    assert cli(ledger_path, "board") == 0 and "nothing recorded" in capsys.readouterr().out


def test_submit_reports_why_a_run_was_not_ranked(tmp_path, ledger_path, capsys):
    path = write(tmp_path, make_result(cells=[make_cell(ok=False, problems=["non-finite qpos"])]))
    assert cli(ledger_path, "submit", path) == 0
    assert "not ranked: non-finite qpos" in capsys.readouterr().out


def test_submit_flags_unreliable_timing(tmp_path, ledger_path, capsys):
    cli(ledger_path, "submit", write(tmp_path, make_result(cells=[make_cell(native_spread=0.6)])))
    assert "timing unreliable" in capsys.readouterr().out


def test_submit_errors_are_messages_not_tracebacks(tmp_path, ledger_path, capsys):
    assert cli(ledger_path, "submit", str(tmp_path / "missing.json")) == 2
    assert "cannot read" in capsys.readouterr().err
    assert cli(ledger_path, "submit", write(tmp_path, dict(error="timed out"))) == 2
    assert "no cells" in capsys.readouterr().err


def test_check_reports_damage_and_a_saved_head(tmp_path, ledger_path, capsys):
    cli(ledger_path, "submit", write(tmp_path, make_result(cells=[make_cell(worlds=1), make_cell(worlds=8)])))
    capsys.readouterr()
    cli(ledger_path, "check")
    head = capsys.readouterr().out.split("head ")[1].split()[0]
    assert cli(ledger_path, "check", "--head", head) == 0
    capsys.readouterr()
    lines = open(ledger_path).read().splitlines()
    open(ledger_path, "w").write(lines[0] + "\n")
    assert cli(ledger_path, "check", "--head", head) == 1
    captured = capsys.readouterr()
    assert "newest records were removed" in captured.err and "DAMAGED" in captured.out
    assert cli(ledger_path, "check", "--head", "garbage") == 2


def test_a_damaged_ledger_is_refused_by_board_with_a_message(tmp_path, ledger_path, capsys):
    cli(ledger_path, "submit", write(tmp_path, make_result()))
    text = open(ledger_path).read().replace('"worlds":8', '"worlds":9', 1)
    open(ledger_path, "w").write(text)
    capsys.readouterr()
    assert cli(ledger_path, "board") == 2 and "damaged" in capsys.readouterr().err


def test_repair_moves_an_interrupted_write_aside(tmp_path, ledger_path, capsys):
    cli(ledger_path, "submit", write(tmp_path, make_result()))
    partial = '{"seq": 2'
    with open(ledger_path, "a") as f:
        f.write(partial)
    capsys.readouterr()
    assert cli(ledger_path, "repair") == 0 and f"moved {len(partial)} bytes" in capsys.readouterr().out
    assert cli(ledger_path, "check") == 0 and cli(ledger_path, "repair") == 0


def test_options_cannot_be_abbreviated(ledger_path):
    with pytest.raises(SystemExit):
        cli(ledger_path, "board", "--incl")


def test_the_default_ledger_lives_under_the_state_dir(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("PHOST_HOME", str(tmp_path / "home"))
    assert main(["submit", write(tmp_path, make_result())]) == 0
    assert (tmp_path / "home" / "ledger.jsonl").exists()
