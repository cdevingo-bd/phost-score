from pathlib import Path

from phost_score import default_ledger, state_dir


def test_phost_home_wins(monkeypatch, tmp_path):
    monkeypatch.setenv("PHOST_HOME", str(tmp_path / "x"))
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "xdg"))
    assert state_dir() == tmp_path / "x" and default_ledger() == tmp_path / "x" / "ledger.jsonl"


def test_xdg_data_home_is_next(monkeypatch, tmp_path):
    monkeypatch.delenv("PHOST_HOME", raising=False)
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "xdg"))
    assert state_dir() == tmp_path / "xdg" / "phost"


def test_the_fallback_is_under_the_home_directory(monkeypatch):
    monkeypatch.delenv("PHOST_HOME", raising=False)
    monkeypatch.delenv("XDG_DATA_HOME", raising=False)
    assert state_dir() == Path.home() / ".local" / "share" / "phost"


def test_nothing_is_created_by_asking(monkeypatch, tmp_path):
    monkeypatch.setenv("PHOST_HOME", str(tmp_path / "never"))
    default_ledger()
    assert not (tmp_path / "never").exists()
