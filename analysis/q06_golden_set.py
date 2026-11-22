"""Q06 - A golden set: 20 questions with exact answers, for checking an agent's (or analyst's) work."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from churnlab.common import (Q1_END, Q1_START, RESULTS, build_cohort, churn_rate, churn_records,  # noqa: E402
                             internal_ids, load, paid_at, save)

QUESTION = "Which questions with known answers should any automated analysis of this data get right?"


def main():
    ev, acc, tk, us = load()
    df = build_cohort(ev, acc, tk, us)
    test = internal_ids(acc)
    e = ev.drop_duplicates("event_id")
    q1 = (e.ts >= Q1_START) & (e.ts < Q1_END)
    q, add = [], None

    def add(question, answer, note=""):
        q.append(dict(id=len(q) + 1, question=question, answer=answer, note=note))

    add("How many rows does subscription_events have (raw)?", len(ev))
    add("How many exact duplicate rows are in subscription_events?", int(ev.drop(columns="ts").duplicated().sum()))
    add("How many accounts are in customer_accounts?", acc.account_id.nunique())
    add("How many internal QA accounts exist?", len(test), "billing_email_domain = qa-internal.example")
    add("How many account_ids appear in events but not in customer_accounts?",
        len(set(ev.account_id) - set(acc.account_id)))
    add("How many accounts were paying (MRR > 0) at 2026-01-01 00:00, raw events?", len(paid_at(e, Q1_START)))
    add("Same, excluding internal and CRM-deleted accounts?", len(df))
    add("Q1 2026 churn % from sql/q1_churn_rate.sql?", round(churn_rate(ev, churn_records(ev)), 1))
    ev_c = ev[~ev.account_id.isin(test)]
    add("Same, excluding internal accounts?", round(churn_rate(ev_c, churn_records(ev_c)), 1))
    add("Cancel requests in Q1 2026 (deduplicated, excluding internal)?",
        int(((e.event_type == "cancel_request") & q1 & ~e.account_id.isin(test)).sum()))
    add("Refund events in Q1 2026?", int(((e.event_type == "refund") & q1).sum()))
    recs = churn_records(e[~e.account_id.isin(test)])
    add("Accounts with a same-day cancellation + reactivation in Q1 2026 (excluding internal)?",
        int(recs[(recs.react_days == 0) & (recs.effective_ts >= Q1_START) & (recs.effective_ts < Q1_END)]
            .account_id.nunique()))
    add("Accounts that received the 2026-02-01 price change?", int((e.event_type == "price_change").sum()))
    pre = e[(e.ts < Q1_START) & (e.ts > "2000-01-01")].sort_values(["ts", "event_id"]).groupby("account_id").tail(1)
    pre = pre[pre.account_id.isin(df.account_id)]
    cents = (pre.source_system == "billing_v1") & (pre.ts >= "2024-03-01")
    add("Total MRR on 2026-01-01 of the clean paying cohort (USD, rounded)?",
        int(round(np.where(cents, pre.mrr_after / 100, pre.mrr_after).sum())),
        f"A naive sum gives {int(round(pre.mrr_after.sum())):,}: {int(cents.sum())} accounts' latest event is a "
        f"billing_v1 row in cents")
    add("Support tickets created in Q1 2026?", int(((tk.ts >= Q1_START) & (tk.ts < Q1_END)).sum()))
    add("Most common ticket category in Q1 2026?",
        tk[(tk.ts >= Q1_START) & (tk.ts < Q1_END)].category.value_counts().index[0])
    add("Distinct spellings in subscription_events.plan?", ev.plan.nunique())
    add("Events with a 1970-01-01 timestamp?", int((ev.ts < "2000-01-01").sum()))
    add("Month in which product analytics undercounted sessions?", "2025-08")
    add("Voluntary Q1 2026 churn rate of the clean cohort (%)?", round(100 * df.churned_q1.mean(), 1))

    gs = pd.DataFrame(q)
    RESULTS.mkdir(exist_ok=True)
    gs.to_csv(RESULTS / "golden_set.csv", index=False)
    L = ["Saved to [`results/golden_set.csv`](golden_set.csv). Ask an agent these questions and compare "
         "its answers before trusting it with open-ended ones.", "", "| # | Question | Answer |", "|---|---|---|"]
    L += [f"| {r.id} | {r.question} | {r.answer:,} |" if isinstance(r.answer, (int, np.integer))
          else f"| {r.id} | {r.question} | {r.answer} |" for r in gs.itertuples()]
    save("q06", "Golden set", QUESTION, L, dict(n_questions=len(gs)))


if __name__ == "__main__":
    main()
