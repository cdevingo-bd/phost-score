"""An append-only record of scored runs, with a hash chain.

Each line holds a record: its sequence number, the hash of the previous record, the entry, and its own hash over
all three. Editing an entry, deleting or reordering records in the middle, or numbering them again breaks the
chain at that point, and ``verify`` says where.

What this does not do: whoever can write the file can recompute the whole chain, so it detects accidents and
careless edits, not a determined forger. Dropping the newest records leaves a valid chain; to catch that, keep
the head (``Ledger.head``) somewhere else and pass it back to ``verify``.
"""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Mapping

GENESIS = "0" * 64


class LedgerError(Exception):
    """The ledger cannot be used as it is (damaged, or asked to hold something it cannot represent)."""


def canonical(value: Any) -> str:
    """One stable text form of a JSON value: sorted keys, no spaces, no NaN or infinity."""
    try:
        return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False)
    except ValueError as error:
        raise LedgerError(f"entry holds a value JSON cannot represent exactly ({error})") from None


def digest(seq: int, prev: str, entry: Mapping[str, Any]) -> str:
    return hashlib.sha256(f"{seq}\n{prev}\n{canonical(entry)}".encode()).hexdigest()


@dataclass(frozen=True)
class Record:
    seq: int
    prev: str
    entry: dict
    hash: str


@dataclass(frozen=True)
class Head:
    seq: int          # 0 for an empty ledger
    hash: str

    def __str__(self) -> str:
        return f"{self.seq}:{self.hash}"

    @classmethod
    def parse(cls, text: str) -> "Head":
        seq, _, hash_ = text.partition(":")
        if not seq.isdigit() or len(hash_) != 64:
            raise ValueError(f"a head looks like <seq>:<64 hex digits>, not {text!r}")
        return cls(int(seq), hash_)


@dataclass(frozen=True)
class Verification:
    records: tuple[Record, ...]
    problems: tuple[str, ...]

    @property
    def ok(self) -> bool:
        return not self.problems

    @property
    def head(self) -> Head:
        return Head(self.records[-1].seq, self.records[-1].hash) if self.records else Head(0, GENESIS)


def _parse(line: str, number: int) -> Record | str:
    """A record, or a sentence saying why the line is not one."""
    try:
        raw = json.loads(line)
    except json.JSONDecodeError as error:
        return f"line {number}: not JSON ({error.msg})"
    if not isinstance(raw, dict) or set(raw) != {"seq", "prev", "entry", "hash"} or not isinstance(raw["entry"], dict) \
            or not isinstance(raw["seq"], int) or isinstance(raw["seq"], bool) \
            or not isinstance(raw["prev"], str) or not isinstance(raw["hash"], str):
        return f"line {number}: not a ledger record"
    return Record(raw["seq"], raw["prev"], raw["entry"], raw["hash"])


def check_text(text: str, expected_head: Head | None = None) -> Verification:
    """Verify ledger text. Each record is checked against the previous record as stored, so one damaged record is
    reported once, not as a cascade."""
    problems: list[str] = []
    records: list[Record] = []
    if text and not text.endswith("\n"):
        problems.append("the last line is incomplete (a write was interrupted); see Ledger.repair_tail")
        text = text[:text.rfind("\n") + 1]
    prev_hash, expected_seq = GENESIS, 1
    for number, line in enumerate(text.splitlines(), start=1):
        record = _parse(line, number)
        if isinstance(record, str):
            problems.append(record)
            continue
        if record.seq != expected_seq:
            problems.append(f"line {number}: sequence is {record.seq}, expected {expected_seq} "
                            "(a record was removed, added or reordered)")
        if record.prev != prev_hash:
            problems.append(f"line {number}: record {record.seq} does not follow the previous record")
        try:
            wanted = digest(record.seq, record.prev, record.entry)
        except LedgerError as error:
            problems.append(f"line {number}: {error}")
        else:
            if record.hash != wanted:
                problems.append(f"line {number}: record {record.seq} was changed (hash mismatch)")
        records.append(record)
        prev_hash, expected_seq = record.hash, record.seq + 1
    if expected_head is not None:
        problems.extend(_against_head(records, expected_head))
    return Verification(tuple(records), tuple(problems))


def _against_head(records: list[Record], expected: Head) -> list[str]:
    """What is wrong with the ledger given a head recorded elsewhere. This is what catches dropped newest records."""
    if expected.seq == 0:
        return ["the ledger has records but the expected head is empty"] if records else []
    held = next((r for r in records if r.seq == expected.seq), None)
    if held is None:
        end = records[-1].seq if records else 0
        return [f"the ledger ends at {end}; the expected head is {expected.seq} (newest records were removed)"]
    if held.hash != expected.hash:
        return [f"record {expected.seq} differs from the expected head"]
    return []


class Ledger:
    """A JSONL file of chained records. Appends are serialised with a file lock, so several processes may share one."""

    def __init__(self, path: str | Path):
        self.path = Path(path)

    # ---- reading ----

    def _text(self) -> str:
        try:
            return self.path.read_text(encoding="utf-8")
        except FileNotFoundError:
            return ""

    def verify(self, expected_head: Head | None = None) -> Verification:
        return check_text(self._text(), expected_head)

    def records(self, *, strict: bool = True) -> tuple[Record, ...]:
        """All records. A damaged ledger raises unless ``strict=False``, which returns what parses."""
        verification = self.verify()
        if strict and not verification.ok:
            raise LedgerError(f"{self.path} is damaged: {verification.problems[0]}"
                              + (f" (+{len(verification.problems) - 1} more)" if len(verification.problems) > 1 else ""))
        return verification.records

    def __iter__(self) -> Iterator[Record]:
        return iter(self.records())

    @property
    def head(self) -> Head:
        return self.verify().head

    # ---- writing ----

    def append(self, entry: Mapping[str, Any]) -> Record:
        """Add one entry. Refuses to extend a damaged ledger: that would bury the damage under valid records."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.path, "a+b") as handle:
            fcntl.flock(handle, fcntl.LOCK_EX)
            try:
                handle.seek(0)
                verification = check_text(handle.read().decode("utf-8"))
                if not verification.ok:
                    raise LedgerError(f"{self.path} is damaged: {verification.problems[0]}; "
                                      "run `phost-score check`, and `Ledger.repair_tail` for an interrupted write")
                prev = verification.records[-1] if verification.records else None
                seq = prev.seq + 1 if prev else 1
                entry = json.loads(canonical(entry))           # what is stored is exactly what is hashed
                before = prev.hash if prev else GENESIS
                record = Record(seq, before, entry, digest(seq, before, entry))
                line = canonical(dict(seq=record.seq, prev=record.prev, entry=record.entry, hash=record.hash)) + "\n"
                handle.write(line.encode("utf-8"))
                handle.flush()
                os.fsync(handle.fileno())
            finally:
                fcntl.flock(handle, fcntl.LOCK_UN)
        return record

    def repair_tail(self) -> int:
        """If the last line was cut short, move it to ``<ledger>.partial-<n>`` and cut it from the ledger. Returns the
        number of bytes moved (0 when there was nothing to repair). Nothing is deleted."""
        if not self.path.exists():
            return 0
        with open(self.path, "r+b") as handle:
            fcntl.flock(handle, fcntl.LOCK_EX)
            try:
                data = handle.read()
                if not data or data.endswith(b"\n"):
                    return 0
                cut = data.rfind(b"\n") + 1
                n = 1
                while self.path.with_name(f"{self.path.name}.partial-{n}").exists():
                    n += 1
                self.path.with_name(f"{self.path.name}.partial-{n}").write_bytes(data[cut:])
                handle.truncate(cut)
                handle.flush()
                os.fsync(handle.fileno())
                return len(data) - cut
            finally:
                fcntl.flock(handle, fcntl.LOCK_UN)
