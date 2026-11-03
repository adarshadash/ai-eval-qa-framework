# reports/ — Generated eval reports

Written by `python -m evals.runner`. Generated files are git-ignored.

| File | For |
|---|---|
| `report.html` | Humans: suite badges, metrics, every case with failed checks and details, flagged production traces |
| `report.md` | Pull-request comments and the CI job summary |
| `report.json` | Dashboards and tracking scores over time |

In CI, the reports are uploaded as a build artifact on every run.
