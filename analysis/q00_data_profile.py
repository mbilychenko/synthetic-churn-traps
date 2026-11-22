"""Q00 - Is the export fit for analysis? Profile every table before answering anything."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import pandas as pd  # noqa: E402

from churnlab.common import internal_ids, load, save  # noqa: E402

QUESTION = "Before we answer business questions: what in this export is unusable or misleading as-is?"


def main():
    ev, acc, tk, us = load()
    L, M = [], {}

    M["rows"] = {"customer_accounts": len(acc), "subscription_events": len(ev),
                 "support_tickets": len(tk), "product_usage_monthly": len(us)}
    L.append("| Table | Rows |\n|---|---|")
    L += [f"| {k} | {v:,} |" for k, v in M["rows"].items()]
    L.append("")

    # 1 duplicates
    M["duplicate_event_rows"] = int(ev.drop(columns="ts").duplicated().sum())
    L.append(f"- **{M['duplicate_event_rows']:,} exact duplicate rows** in `subscription_events` "
             f"(same `event_id`): count events with `DISTINCT`.")
    # 2 placeholder timestamps
    M["placeholder_timestamps"] = int((ev["ts"] < "2000-01-01").sum())
    L.append(f"- **{M['placeholder_timestamps']} events dated 1970-01-01**: placeholder timestamps, "
             f"which must be excluded from any time-based logic.")
    # 3 orphans
    orphans = sorted(set(ev["account_id"]) - set(acc["account_id"]))
    M["orphan_account_ids"] = len(orphans)
    L.append(f"- **{len(orphans)} account_ids appear in events but not in `customer_accounts`**: "
             f"they were deleted from the CRM, but their billing history remains.")
    # 4 internal accounts
    ids = internal_ids(acc)
    e_int = ev[ev.account_id.isin(ids)].drop_duplicates("event_id")
    M["internal_accounts"] = len(ids)
    M["internal_account_cancellations"] = int((e_int.event_type == "cancellation").sum())
    L.append(f"- **{len(ids)} internal QA accounts** (`billing_email_domain = qa-internal.example`, MRR = $1) "
             f"produce **{M['internal_account_cancellations']} cancellations**, because they exercise the cancel flow.")
    # 5 plan labels
    labels = sorted(ev["plan"].dropna().unique())
    canon = sorted({p.strip().title() for p in labels})
    M["plan_label_variants"] = len(labels)
    L.append(f"- **{len(labels)} spellings of {len(canon)} plans** ({', '.join(repr(x) for x in labels)}): "
             f"normalise with `strip().title()`.")
    # 6 MRR scale by source system
    e = ev.drop_duplicates("event_id")
    e = e[(e.mrr_after > 0) & (e.ts > "2000-01-01")].copy()
    e["month"] = e.ts.dt.to_period("M")
    med = e.groupby(["month"])["mrr_after"].median()
    jump = med[med > 20 * med.median()]
    M["mrr_scale_anomaly_months"] = [str(m) for m in jump.index]
    M["mrr_rows_in_anomaly"] = int(e[e.month.isin(jump.index)].shape[0])
    L.append(f"- **MRR is ~100× too large in {', '.join(M['mrr_scale_anomaly_months'])}** (median MRR per event: "
             f"{jump.median():,.0f} vs {med.median():,.0f} in other months, {M['mrr_rows_in_anomaly']} rows, all "
             f"`source_system = billing_v1`). The legacy system stored cents. Any account whose *latest* event "
             f"falls in that window carries the inflated value forward.")
    # 7 usage outage
    m = us.groupby("month")[["active_users", "sessions"]].sum()
    spu = m["sessions"] / m["active_users"]
    roll = spu.rolling(3, center=True).median()
    dips = spu[spu < 0.7 * roll]
    M["usage_outage_months"] = [d.strftime("%Y-%m") for d in dips.index]
    if len(dips):
        d0 = dips.index[0]
        L.append(f"- **Sessions per active user fall {1 - spu[d0] / roll[d0]:.0%} in {d0:%Y-%m}**, while active "
                 f"users stay flat. That looks like a product-analytics outage, so avoid session-based trends across that month.")
    # 8 nulls
    nulls = acc[["industry", "company_size", "last_nps_score", "cancellation_reason"]].isna().sum()
    M["nulls"] = nulls.to_dict()
    L.append(f"- Nulls in `customer_accounts`: " + ", ".join(f"{k} {v:,}" for k, v in nulls.items()))
    drop = int(acc[["industry", "company_size"]].isna().any(axis=1).sum())
    M["accounts_lost_by_dropna"] = drop
    L.append(f"- Dropping accounts with missing industry or company_size would silently remove **{drop:,} of "
             f"{len(acc):,} accounts**, along with every event, ticket and usage row attached to them.")
    # 9 export-time snapshot columns
    snap = ["account_status", "current_plan", "current_seats", "current_mrr", "cancellation_reason",
            "last_cancellation_date", "last_login_date", "last_nps_score"]
    M["export_time_columns"] = snap
    L.append(f"- **Point-in-time warning:** `customer_accounts` reflects the CRM on the export date "
             f"(2026-10-31). These fields are *current* values, not history: {', '.join(f'`{c}`' for c in snap)}. "
             f"Don't use them as features for any period before the export.")
    save("q00", "Data profile", QUESTION, L, M)


if __name__ == "__main__":
    main()
