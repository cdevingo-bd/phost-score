"""Where phost keeps its local state. Nothing here is created until something writes to it."""

from __future__ import annotations

import os
from pathlib import Path


def state_dir() -> Path:
    """``$PHOST_HOME``, else ``$XDG_DATA_HOME/phost``, else ``~/.local/share/phost``."""
    explicit = os.environ.get("PHOST_HOME")
    if explicit:
        return Path(explicit).expanduser()
    base = os.environ.get("XDG_DATA_HOME")
    return (Path(base).expanduser() if base else Path.home() / ".local" / "share") / "phost"


def default_ledger() -> Path:
    return state_dir() / "ledger.jsonl"
