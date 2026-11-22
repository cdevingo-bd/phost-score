"""This repository carries nothing that must not be published: no committed generated files, no private paths."""

from pathlib import Path

import pytest

hygiene = pytest.importorskip("phost.testing.hygiene")
ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.skipif(not (ROOT / ".git").exists(), reason="not a git checkout")
def test_the_tree_and_every_commit_are_clean():
    """Checks the tracked files and the whole history for committed bytecode, caches and build output, and for the
    current user's home directory path. Names that must stay private are checked too when a deny list is configured
    outside the repository (see phost.testing.hygiene)."""
    found = hygiene.problems(hygiene.check(ROOT))
    assert not found, "\n".join(str(f) for f in found)
