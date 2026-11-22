"""Q03 - What distinguishes accounts that churned in Q1 2026 from those that stayed?"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import pandas as pd  # noqa: E402
from scipy.stats import fisher_exact  # noqa: E402

from churnlab.common import Q1_END, Q1_START, build_cohort, load, save  # noqa: E402

QUESTION = ("What drove churn in Q1 2026? Compare churned vs retained accounts on every signal "
            "and rank the drivers.")


def main():
    ev, acc, tk, us = load()
    df = build_cohort(ev, acc, tk, us)
    y = df["churned_q1"]
    q1t = tk[(tk.ts >= Q1_START) & (tk.ts < Q1_END)]
    signals = {
        "Contacted support in Q1": df.account_id.isin(q1t.account_id),
        "<35% of seats active in Dec 2025": df.ratio_last1 < 0.35,
        "Active users down >50% from 12-month peak": df.au_drop_from_peak > 0.5,
        "Paying for < 3 months": df.tenure_months < 3,
        "Starter plan, monthly billing (price change Feb 2026)": df.is_starter_monthly == 1,
        "Onboarding not completed": df.onboarding == 0,
        "No integrations": df.integrations_count == 0,
        "Company size 1-10": df.company_size == "1-10",
        "Annual contract NOT up for renewal in Q1": df.renewal_due_q1 == 0,
    }
    L = [f"Cohort: **{len(df):,} paying accounts on 2026-01-01** (internal and CRM-deleted accounts removed). "
         f"Label: voluntary churn decision in Q1 (a cancel request or a downgrade to Free), "
         f"**{y.mean():.1%}**.", "",
         "| Signal | Accounts | Churn with | Churn without | Lift |", "|---|---|---|---|---|"]
    rows = []
    for name, mask in signals.items():
        mask = mask.fillna(False).astype(bool)
        a, b = y[mask].mean(), y[~mask].mean()
        rows.append((name, int(mask.sum()), a, b, a / b if b > 0 else float("inf")))
    rows.sort(key=lambda r: -r[4])
    for name, n, a, b, lift in rows:
        L.append(f"| {name} | {n:,} | {a:.1%} | {b:.1%} | **{lift:.1f}×** |")
    M = {"cohort": len(df), "churn_rate": round(float(y.mean()), 4),
         "lifts": {r[0]: round(r[4], 2) for r in rows}}

    # price change shows up in time
    e = ev.drop_duplicates("event_id")
    cr = e[(e.event_type == "cancel_request") & (e.ts >= "2025-10-01") & (e.ts < Q1_END)
           & e.account_id.isin(df.account_id)].copy()
    cr["month"] = cr.ts.dt.strftime("%Y-%m")
    cr["seg"] = cr.account_id.map(df.set_index("account_id").is_starter_monthly).map({1: "Starter monthly", 0: "Other"})
    trend = cr.groupby(["month", "seg"]).size().unstack(fill_value=0)
    L += ["", "Cancel requests per month from this cohort (the Starter monthly price rose on 2026-02-01):", "",
          "| Month | " + " | ".join(trend.columns) + " |", "|---" * (len(trend.columns) + 1) + "|"]
    L += [f"| {m} | " + " | ".join(str(v) for v in r) + " |" for m, r in trend.iterrows()]
    M["cancel_requests_by_month"] = trend.to_dict()

    # small segments
    seg = df.groupby(["industry", "country"])["churned_q1"].agg(["size", "sum", "mean"])
    small = seg[(seg["size"] >= 10) & (seg["size"] < 30)].sort_values(["mean", "size"], ascending=False)
    (ind, cty), top = small.index[0], small.iloc[0]
    rest = df[~((df.industry == ind) & (df.country == cty))]
    _, p = fisher_exact([[int(top["sum"]), int(top["size"] - top["sum"])],
                         [int(rest.churned_q1.sum()), int(len(rest) - rest.churned_q1.sum())]])
    L += ["", f"**Small samples:** {len(small)} industry × country segments have only 10–29 accounts. The highest "
              f"churn rate among them, **{ind} / {cty}**, is {int(top['sum'])} of {int(top['size'])} = "
              f"{top['mean']:.0%} (vs {y.mean():.1%} overall). Fisher exact p = {p:.2f}, and with {len(small)} "
              f"segments compared, a gap this size is expected by chance. **Not a finding.**"]
    gov = df[df.industry == "Government"]
    L.append(f"Government is the smallest industry: n = {len(gov)}, {int(gov.churned_q1.sum())} churned.")
    M["small_segment"] = dict(segment=f"{ind}/{cty}", n=int(top["size"]), churned=int(top["sum"]),
                              rate=round(float(top["mean"]), 3), fisher_p=round(float(p), 3), n_small_segments=len(small))
    L += ["", "*Lift is association, not cause. Q04 tests the top signal.*"]
    save("q03", "Churn drivers (EDA)", QUESTION, L, M)


if __name__ == "__main__":
    main()
