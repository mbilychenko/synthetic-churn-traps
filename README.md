# synthetic-churn-traps

A **fully synthetic B2B SaaS company** (~16,900 accounts, 2023-01 → 2026-10), plus an analysis pipeline that answers the question every SaaS company asks: *why are customers leaving, and who is next?*

The data comes first. A seeded simulation generates the company's raw exports. Analysis scripts then answer business questions using **only** those exports, and every number in `results/` traces back to a specific file and line of code.

The dataset is built for practising (and testing AI agents on) the things that make data science hard: ambiguous definitions, messy exports, convincing but wrong correlations, and models that look perfect for the wrong reason.

> **No real, commercial, personal or third-party data was used.** Every value comes from `config/simulation.toml` and a seeded random generator. See [DATASHEET.md](DATASHEET.md).

```
config/simulation.toml ──► src/churnsim ──► data/ ──► analysis/q00…q06 ──► results/ ──► notebooks/ (and slides)
   assumptions about         simulate +      4 tables     one script per         REPORT.md
   a fictional company        export         + SQLite     business question      metrics.json
```

## Quick start

```bash
pip install -r requirements.txt
python run_pipeline.py                # verify data/ checksums, run all analyses -> results/REPORT.md
python run_pipeline.py --regenerate   # rebuild data/ from the config first (byte-identical)
pytest                                # schema, determinism and cross-check tests
```

Then open [`notebooks/results_tour.ipynb`](notebooks/results_tour.ipynb) for a visual walk through the results.

## Why the data is committed

The generated data in [`data/`](data) is committed on purpose, even though the code can rebuild it:

- **Anyone can analyse it immediately**, with pandas, SQL, a BI tool or an AI agent, without installing or running anything.
- **Everyone analyses the same bytes.** `data/CHECKSUMS.sha256` pins the exact files that `results/` and the talk were computed from.
- **Provenance can be checked.** `python run_pipeline.py --regenerate` (and `pytest`) rebuilds the files from `config/simulation.toml` and confirms they are **byte-identical** to the committed ones. So the data is demonstrably the output of this code and nothing else.
- It's small: about 23 MB of CSV, plus the same tables in one SQLite file.

## The data

| Table | Rows | What it is |
|---|---|---|
| `customer_accounts` | 16,929 | CRM: one row per account, **current values at export (2026-10-31)** |
| `subscription_events` | 46,370 | Billing event log: trials, starts, renewals, seat and price changes, cancellations, refunds, reactivations |
| `support_tickets` | 8,908 | Help-desk tickets: category, channel, first response time, CSAT |
| `product_usage_monthly` | 121,910 | Product analytics: active users, sessions, reports and API calls per account-month |

Column definitions are in [DATASHEET.md](DATASHEET.md#4-tables). `data/saas_churn.sqlite` contains the same four tables.

## The questions

| Script | Business question | Answer from the data (see [`results/REPORT.md`](results/REPORT.md)) |
|---|---|---|
| [q00](analysis/q00_data_profile.py) | Is the export fit for analysis? | No, not as-is: duplicate rows, placeholder dates, internal QA accounts, legacy MRR in cents, an analytics outage, and CRM fields that hold export-time values |
| [q01](analysis/q01_q1_churn_rate.py) | What was Q1 2026 churn? | 6.3% from the straightforward SQL |
| [q02](analysis/q02_definition_sensitivity.py) | How much does "churn" depend on definitions? | Six defensible choices move it from 4.9% to 9.1% |
| [q03](analysis/q03_churn_drivers.py) | What distinguishes churned accounts? | Low seat utilisation, loss of active users, the Feb 2026 Starter price change, support contact (3.5×) |
| [q04](analysis/q04_support_and_churn.py) | Does contacting support cause churn? | No. The tickets come *after* the decision to leave; with product tickets only, the ratio is 1.4× |
| [q05](analysis/q05_churn_prediction.py) | Can we predict who leaves next? | Manual LR 0.77 · AutoML on all columns 0.99 (leakage) · Human + AI LightGBM 0.84 |
| [q06](analysis/q06_golden_set.py) | How do we check an agent's answers? | 20 questions with exact answers → `results/golden_set.csv` |

## Repository layout

```
config/simulation.toml   every assumption about the fictional company (prices, behaviour, export artefacts)
src/churnsim/            simulation + export          (numpy, pandas only)
src/churnlab/            shared analysis helpers (cohort, churn definitions, features)
sql/                     SQL used by the analysis
analysis/                q00…q06, one script per question, + build_report.py
data/                    generated dataset + CHECKSUMS.sha256
results/                 qNN.md / qNN.json per question, REPORT.md, metrics.json, golden_set.csv
notebooks/               results_tour.ipynb: displays results/, computes nothing
tests/                   determinism, schema and cross-checks
```

## Used in

*"Delegate the work, never delegate the responsibility"*, PyData Global 2026, Maksym Bilychenko. The numbers on the slides are taken from [`results/metrics.json`](results/metrics.json).

## Licence

Code: MIT. Data and documentation: CC BY 4.0. See [LICENSE](LICENSE).

```
Bilychenko, M. (2026). synthetic-churn-traps: a synthetic B2B SaaS churn dataset and analysis pipeline.
```
