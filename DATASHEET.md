# Datasheet: synthetic-churn-traps

*Version 1.0 · seed 8 · export date 2026-10-31*

## 1. Motivation

This is a teaching and benchmarking dataset for churn analysis. It simulates a fictional B2B SaaS company whose systems behave the way real ones do: definitions are ambiguous, exports are messy, and some of the most obvious correlations are misleading. It was designed so that an analysis, whether by a person or an AI agent, can be **checked**. Because the data comes from known rules, a careful analysis of the data alone should reach conclusions consistent with those rules.

## 2. Provenance: no commercial or personal data

| Question | Answer |
|---|---|
| Was any real company's data used, sampled, copied or used to calibrate distributions? | **No.** |
| Was any public or third-party dataset (Kaggle or similar) used as input? | **No.** No external file is read at any point. |
| Where do the numbers come from? | Every parameter is a hand-chosen assumption written in [`config/simulation.toml`](config/simulation.toml). None of them describes a real business, the author's employer or any client. |
| Does it contain personal data? | **No.** Accounts are sequential IDs (`A00001`...). Email domains use the reserved `.example` top-level domain (RFC 2606). Country is stored as an ISO-3166 code only. There are no names, addresses or IP addresses. |
| Are the company, products or brands real? | No. The company is unnamed. Plan names (Starter / Growth / Enterprise / Free) are generic, and prices are invented. |
| How is it produced? | By deterministic Python ([`src/churnsim`](src/churnsim)) using NumPy's seeded PCG64 generator. The same config and seed reproduce byte-identical files, which `tests/test_generation.py` checks. |
| Was AI involved? | The code was written with help from an AI coding assistant and reviewed by the author. No record was written by a language model; every value comes from the code's random draws. |

The dataset is self-contained and reproducible from code. It can be shared and reused under CC BY 4.0 with no confidentiality obligation to any third party.

## 3. How it was built

The simulation runs one account at a time, month by month, from trial signup to the export date.

1. **Signups.** About 205 trials per month in Jan 2023, rising linearly to about 530 per month in Oct 2026.
2. **Account traits.** Each account draws:
   - company size, which sets the seat count;
   - acquisition channel, industry and country;
   - an *unobserved* engagement trait;
   - onboarding completion and the number of integrations, both depending on engagement.
3. **Trial → paid.** A 14-day trial converts with a probability that depends on engagement, onboarding and channel. Converted accounts pick a plan (driven by size) and a billing cycle (annual is more common on bigger plans). Prices are per seat, with a 15% annual discount.
4. **Each month, for each customer:**
   - Engagement drifts. Occasionally the customer's internal champion leaves, and engagement drops sharply.
   - Product usage is drawn from engagement: active users, sessions, reports, API calls and minutes.
   - Product support tickets arrive at a small, steady rate.
   - The billing system records seat changes, annual renewals, failed payments and occasional same-day cancel-and-reactivate operations (system migrations).
   - The customer may **decide to leave**, according to a monthly logistic risk (see `[churn_hazard]` in the config). Annual customers can only leave near renewal.
5. **After deciding to leave**, some customers:
   - contact support about cancelling, billing or exporting their data;
   - then submit a cancel request, or downgrade to the Free plan.
   
   Cancellations take effect at the end of the term. Very new customers get a money-back refund and some others a goodwill refund. Some churned customers come back later.
6. **Price change.** On 2026-02-01 the Starter monthly price rises from $15 to $18 per seat.
7. **Export.** The records are written as a real company's systems would hold them on 2026-10-31 (see §5).

The simulation tracks some quantities a real company never sees, such as the engagement trait and the day a customer privately decided to leave. **These are not exported**, just as they would not exist in real data.

## 4. Tables

### `customer_accounts`: CRM, one row per account, values **as of the export date**
| Column | Description |
|---|---|
| account_id | `A00001`... |
| signup_date | Trial start date |
| billing_email_domain | `custNNNNN.example`; internal QA accounts use `qa-internal.example` |
| country · industry · company_size · acquisition_channel | Account traits (industry and company_size contain nulls) |
| onboarding_completed · integrations_count | Set during the trial |
| current_plan · current_seats · current_mrr | Current subscription at export |
| account_status | `active`, `cancelled`, `pending_cancellation`, `free`, `trial` or `trial_expired`, at export |
| cancellation_reason | Reason on the last cancel request, for accounts cancelled at export |
| last_cancellation_date | Most recent cancellation (the field is never cleared on reactivation) |
| last_login_date | Last login as of export |
| last_nps_score | Latest survey score; for churned accounts this is the exit survey |

### `subscription_events`: billing event log
| Column | Description |
|---|---|
| event_id · account_id · event_ts | Identifiers and timestamp (`YYYY-MM-DD HH:MM:SS`) |
| event_type | `trial_start`, `trial_expired`, `subscription_start`, `renewal`, `seat_change`, `price_change`, `cancel_request`, `cancellation`, `refund`, `reactivation`, `downgrade`, `upgrade`, `payment_failed`, `payment_recovered` |
| plan · billing_cycle · seats | Subscription after the event |
| mrr_before · mrr_after | Monthly recurring revenue before and after the event |
| reason_code | Cancellation or payment reason, where recorded |
| source_system | `billing_v1` (before 2024-06-01) or `billing_v2` |

### `support_tickets`
`ticket_id, account_id, created_ts, category, priority, channel, first_response_hours, csat_score`

### `product_usage_monthly`: one row per account per active month
`account_id, month, active_users, seats, sessions, reports_created, api_calls, avg_session_minutes`

## 5. Export artefacts (by design)

Real exports are never clean, and this one reproduces common artefacts. They are configured in `[export_artefacts]` and `[internal_accounts]`, and [`analysis/q00_data_profile.py`](analysis/q00_data_profile.py) shows how to detect each one from the data alone:

- a legacy billing system with different conventions (labels, units) before a 2024 migration;
- exact duplicate rows from a re-run ingestion job;
- a few placeholder timestamps;
- internal QA accounts on the production billing system;
- accounts deleted from the CRM whose billing history remains;
- a product-analytics outage month;
- missing CRM attributes;
- CRM fields that hold *current* values rather than history.

## 6. Intended use and limitations

- **Use it for:** teaching churn analysis, SQL and EDA practice, testing AI analysis agents, and demonstrating leakage, reverse causality and definition sensitivity.
- **It is not** a benchmark of real SaaS behaviour. Distributions are plausible but simplified: one currency, no taxes, single-product accounts, monthly usage granularity. Effects are cleaner than in real life.
- Another seed (`python -m churnsim --seed N`) gives an equally valid company. The qualitative behaviour is the same and the exact figures shift slightly.

## 7. Licence

Data and documentation: CC BY 4.0. Code: MIT. See [LICENSE](LICENSE).
