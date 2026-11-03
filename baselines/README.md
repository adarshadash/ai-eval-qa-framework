# baselines/ — Accepted scores

`baseline_<mode>.json` stores the last accepted score per suite. The runner fails if any suite drops more than `thresholds.regression.tolerance` below it, which catches slow drift that still passes the absolute thresholds.

Mock and live keep separate files because their scores aren't comparable.

```bash
python -m evals.runner --update-baseline   # after an intended behaviour change
```

Commit the updated baseline in the same pull request as the change, so reviewers see exactly which scores moved.
