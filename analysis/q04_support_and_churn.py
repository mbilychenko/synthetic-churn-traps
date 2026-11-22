"""Q04 - Does contacting support make customers churn? Should we invest in support to cut churn?"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from churnlab.common import Q1_END, Q1_START, build_cohort, load, save  # noqa: E402

QUESTION = ("Customers who contacted support churn far more often. Is support causing churn, and would "
            "investing in support quality reduce it?")
PRODUCT = {"how_to", "bug", "integration", "feature_request"}


def main():
    ev, acc, tk, us = load()
    df = build_cohort(ev, acc, tk, us)
    y = df["churned_q1"]
    churners = set(df.loc[y == 1, "account_id"])
    q1t = tk[(tk.ts >= Q1_START) & (tk.ts < Q1_END)].copy()

    contacted = df.account_id.isin(q1t.account_id)
    pc, pn = y[contacted].mean(), y[~contacted].mean()
    L = ["**1. The association** (the headline an automated analysis reports):", "",
         f"- Contacted support in Q1: churn **{pc:.1%}** (n = {int(contacted.sum()):,})",
         f"- Did not contact support: churn **{pn:.1%}**",
         f"- **{pc / pn:.1f}× more likely to churn**", ""]

    # timing relative to the cancel request
    first_req = ev[(ev.event_type == "cancel_request") & (ev.ts >= Q1_START)].groupby("account_id")["ts"].min()
    q1t["req_ts"] = q1t.account_id.map(first_req)
    q1t["days_before_request"] = (q1t.req_ts - q1t.ts).dt.days
    ct = q1t[q1t.account_id.isin(churners) & q1t.req_ts.notna()]
    near = ct[ct.days_before_request.between(0, 14)]
    share_near = len(near) / len(ct)
    cats = near.category.value_counts(normalize=True).head(3)
    L += ["**2. When and why did churners contact support?**", "",
          f"- **{share_near:.0%}** of churners' Q1 tickets were filed **0–14 days before their cancel request**.",
          "- Those tickets are mostly about leaving: " + ", ".join(f"`{k}` {v:.0%}" for k, v in cats.items()) + ".",
          "- So the sequence is: *decide to leave → ask how to cancel or export data → cancel*. "
          "The ticket is a **symptom** of the decision, not its cause.", ""]

    # remove leaving-related tickets
    contacted_p = df.account_id.isin(q1t[q1t.category.isin(PRODUCT)].account_id)
    ppc, ppn = y[contacted_p].mean(), y[~contacted_p].mean()
    L += ["**3. Counting product tickets only** (how-to, bug, integration, feature request):", "",
          f"- Churn {ppc:.1%} vs {ppn:.1%} → **{ppc / ppn:.1f}×**. Most of the association disappears.", ""]

    # support quality
    t_ch, t_st = q1t[q1t.account_id.isin(churners)], q1t[q1t.account_id.isin(set(df.account_id) - churners)]
    L += ["**4. Is support quality worse for churners?**", "",
          f"- CSAT: churners {t_ch.csat_score.mean():.1f} vs retained {t_st.csat_score.mean():.1f}. That *looks* like bad support.",
          f"- Median first response: churners {t_ch.first_response_hours.median():.1f} h vs retained "
          f"{t_st.first_response_hours.median():.1f} h. Churners were **not served slower**; people on their way out "
          f"rate the interaction lower.", "",
          "**Conclusion:** the evidence points to reverse causality. Spending on support quality to reduce churn would "
          "act too late: by the time the ticket arrives, the decision has been made. Use a cancellation, billing or "
          "export ticket as an **early-warning trigger** for a retention conversation instead, and look upstream "
          "(Q03: usage, champion loss, early tenure, the price change) for causes."]
    save("q04", "Support contact and churn", QUESTION, L,
         dict(ratio_all_tickets=round(pc / pn, 2), churn_contacted=round(pc, 4), churn_not_contacted=round(pn, 4),
              n_contacted=int(contacted.sum()), share_tickets_0_14d_before_request=round(share_near, 3),
              ratio_product_tickets_only=round(ppc / ppn, 2),
              csat_churners=round(float(t_ch.csat_score.mean()), 2), csat_retained=round(float(t_st.csat_score.mean()), 2),
              median_frt_churners=float(t_ch.first_response_hours.median()),
              median_frt_retained=float(t_st.first_response_hours.median())))


if __name__ == "__main__":
    main()
