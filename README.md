# phost-score

Judging for [phost](https://github.com/cdevingo-bd/phost) run results. The gate a result must pass depends on the swap class of
what was swapped, results are ranked on a Pareto front, and a hash-chained local ledger records them so the record is hard to
alter by accident. It works on the JSON that [phost-lab](https://github.com/cdevingo-bd/phost-lab) produces and needs no GPU.

```
phost-lab run examples/coupled.json --scene trampoline12:impact --worlds 8 --out result.json
phost-score submit result.json --label coupled
phost-score board
phost-score check
phost-score reverify 1          # run entry #1 again here and compare (needs phost-lab and a GPU)
```

The ledger lives in `$PHOST_HOME`, else `$XDG_DATA_HOME/phost`, else `~/.local/share/phost`. Nothing leaves your
machine.

## How a run is judged

What was swapped decides the rules, because a different kind of replacement may differ in different ways:

| class | gate | ranked on |
| --- | --- | --- |
| exact | the lab's checks (finite state, accuracy against the native control, the stage's own metrics) | speedup |
| approximate | the same checks | speedup **and** accuracy ratio, as a Pareto front: no weighted sum |
| model changing | the stage's own reference law (there is no native answer to match) | step time, among runs with the same stages |
| observer | leaving the trajectory alone | overhead |

A graph is judged by the strictest class in it. A run with no stage is the control and is never ranked. A run that
fails its gates is recorded with the reasons and never ranked, however fast it was.

Two runs are compared only when they were measured on the same scene, device and world count. Only the latest run of
a label and graph counts, because the best of several runs rewards noise.

Timing is marked unreliable when the two native control runs differ by more than 10% or other processes used the GPU;
unreliable runs are left off the board unless asked for (`--include-unreliable`).

## What the ledger does and does not do

Each record carries the hash of the one before it. Editing an entry, removing or reordering records in the middle, or
renumbering them breaks the chain, and `phost-score check` says where. Appends are serialised with a file lock.

It does not stop someone who can write the file from recomputing the whole chain, and dropping the newest records
leaves a valid chain. Save the head (`phost-score check` prints it) somewhere else and pass it back with `--head` to
catch that.

Every entry has a **trust** level. `submit` records `self-reported`: the result was produced on your machine and
nothing here can confirm it. `judged` is for a judge that ran the submission itself; the board always shows which is
which.

## As a library

```python
from phost_score import Ledger, make_entries, leaderboard

ledger = Ledger("ledger.jsonl")
for entry in make_entries(result, label="coupled"):
    ledger.append(entry.to_dict())
groups = leaderboard(ledger.records())
```

`reverify(entry, run)` takes any function that runs a graph and returns a lab result, so the scorer never imports the
lab. A fresh result from a different device is "cannot say", not a disagreement.

MIT licensed.
