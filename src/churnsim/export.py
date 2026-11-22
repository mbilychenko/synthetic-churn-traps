"""
Turn simulation records into the four exported tables.

This step also reproduces artefacts that real SaaS exports typically contain:
a legacy billing system with different conventions, a re-run ingestion job,
placeholder timestamps, internal QA accounts, a product-analytics outage and
CRM fields that hold their *current* value at export time. They are described
in `config/simulation.toml` under [export_artefacts] and [internal_accounts].
"""
from __future__ import annotations

import datetime as dt
import hashlib
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd

from .config import date_ord

TABLES = ["customer_accounts", "subscription_events", "support_tickets", "product_usage_monthly"]


def _iso(o):
    return dt.date.fromordinal(int(o)).isoformat() if o is not None and not pd.isna(o) else None


def build_tables(sim: dict) -> tuple[dict[str, pd.DataFrame], dict]:
    """Return (observable tables, latent simulation state)."""
    cfg, rng, MS = sim["cfg"], sim["rng"], sim["MS"]
    EXPORT = sim["cal"].export_ord
    N = sim["cal"].n_months
    X = cfg["export_artefacts"]
    A = cfg["accounts"]

    # ---------------- subscription_events -------------------------------------------
    ev = pd.DataFrame(sim["EV"], columns=["account_id", "day", "k", "event_type", "plan", "billing_cycle",
                                          "seats", "mrr_before", "mrr_after", "reason_code"])
    ev = ev[ev["day"] <= EXPORT].copy()
    ev["sec"] = rng.integers(7 * 3600, 21 * 3600, len(ev))          # time of day, order-preserving
    a = ev.sort_values(["account_id", "day", "k"]).copy()
    a["sec"] = ev.sort_values(["account_id", "day", "sec"])["sec"].values
    ev = a
    base = pd.to_datetime(ev["day"].map(lambda o: dt.date.fromordinal(int(o))))
    ev["event_ts"] = (base + pd.to_timedelta(ev["sec"], unit="s")).dt.strftime("%Y-%m-%d %H:%M:%S")
    ev = ev.sort_values(["event_ts", "account_id", "k"]).reset_index(drop=True)
    ev["event_id"] = [f"E{j + 1:07d}" for j in range(len(ev))]
    ev["source_system"] = np.where(ev["day"] < date_ord(X["legacy_billing_cutover"]), "billing_v1", "billing_v2")

    cents = (ev["source_system"] == "billing_v1") & (ev["day"] >= date_ord(X["legacy_mrr_in_cents_from"]))
    ev.loc[cents, ["mrr_before", "mrr_after"]] *= 100              # legacy system wrote cents
    v1 = ev["source_system"] == "billing_v1"
    ev.loc[v1, "plan"] = ev.loc[v1, "plan"].str.lower()             # legacy system lower-cased labels
    sp = (rng.random(len(ev)) < X["trailing_space_plan_prob"]) & (ev["source_system"] == "billing_v2").values
    ev.loc[sp, "plan"] = ev.loc[sp, "plan"] + " "

    cols = ["event_id", "account_id", "event_ts", "event_type", "plan", "billing_cycle", "seats",
            "mrr_before", "mrr_after", "reason_code", "source_system"]
    ev = ev[cols]
    sc = ev.index[ev["event_type"] == "seat_change"]
    bad = rng.choice(sc, X["placeholder_timestamp_rows"], replace=False)
    ev.loc[bad, "event_ts"] = "1970-01-01 00:00:00"                 # unset timestamp in a manual fix
    dup = ev.sample(frac=X["duplicate_ingestion_fraction"], random_state=int(rng.integers(1e9)))
    ev = pd.concat([ev, dup]).sort_values(["event_ts", "event_id"]).reset_index(drop=True)

    # ---------------- customer_accounts (CRM, current state at export) ---------------
    nps = cfg["nps"]
    IA = cfg["internal_accounts"]
    rows = []
    for acc in sim["ACC"]:
        if acc.get("test"):
            rows.append(dict(account_id=acc["aid"], signup_date=_iso(acc["start"]),
                             billing_email_domain=IA["email_domain"], country="US", industry="Software",
                             company_size="1-10", acquisition_channel="organic", onboarding_completed=True,
                             integrations_count=0, current_plan="Growth", current_seats=1, current_mrr=1.0,
                             account_status="active", cancellation_reason=None, last_cancellation_date=None,
                             last_login_date=_iso(EXPORT - 2), last_nps_score=None))
            continue
        i = acc["i"]
        if sim["deleted"][i]:
            continue
        score = acc["exit_nps"]
        if score is None and acc["status"] in ("active", "pending_cancellation", "free") \
                and rng.random() < nps["response_prob"]:
            score = int(np.clip(round(rng.normal(nps["mean"] + nps["engagement"] * sim["e_base"][i], nps["sd"])), 0, 10))
        rows.append(dict(
            account_id=f"A{i + 1:05d}", signup_date=_iso(sim["signup_days"][i]),
            billing_email_domain=f"cust{i + 1:05d}.example",
            country=sim["country"][i], industry=sim["industry"][i],
            company_size=A["company_sizes"][sim["size_i"][i]],
            acquisition_channel=A["channels"][sim["channel_i"][i]],
            onboarding_completed=bool(sim["onboarding"][i]), integrations_count=int(sim["integrations"][i]),
            current_plan=acc["plan"], current_seats=acc["seats"], current_mrr=acc["mrr"],
            account_status=acc["status"], cancellation_reason=acc["reason"],
            last_cancellation_date=_iso(acc.get("last_cancel")),
            last_login_date=_iso(acc["last_login"]), last_nps_score=score))
    accts = pd.DataFrame(rows)
    m1 = rng.random(len(accts)) < X["industry_missing_prob"]
    accts.loc[m1, "industry"] = None
    m2 = rng.random(len(accts)) < X["company_size_missing_prob"]
    accts.loc[m2, "company_size"] = None
    accts["last_nps_score"] = accts["last_nps_score"].astype("Int64")

    # ---------------- product_usage_monthly ------------------------------------------
    us = pd.DataFrame(sim["US"], columns=["account_id", "m", "active_users", "seats", "sessions",
                                          "reports_created", "api_calls", "avg_session_minutes"])
    us = us[us["m"] < N]
    us.insert(1, "month", us["m"].map(lambda m: dt.date.fromordinal(MS[m]).isoformat()))
    us = us.drop(columns="m").sort_values(["month", "account_id"]).reset_index(drop=True)

    # ---------------- support_tickets ------------------------------------------------
    tk = pd.DataFrame(sim["TK"], columns=["account_id", "day", "category", "priority", "channel",
                                          "first_response_hours", "csat_score", "_leaving"])
    tk = tk[tk["day"] <= EXPORT]
    tk["created_ts"] = (pd.to_datetime(tk["day"].map(lambda o: dt.date.fromordinal(int(o))))
                        + pd.to_timedelta(rng.integers(6 * 3600, 22 * 3600, len(tk)), unit="s")
                        ).dt.strftime("%Y-%m-%d %H:%M:%S")
    tk = tk.sort_values(["created_ts", "account_id"]).reset_index(drop=True)
    tk.insert(0, "ticket_id", [f"T{j + 1:06d}" for j in range(len(tk))])
    tk["csat_score"] = tk["csat_score"].astype("Int64")
    latent_tickets = tk[["ticket_id", "account_id", "_leaving"]].rename(
        columns={"_leaving": "filed_after_decision_to_leave"})
    tk = tk[["ticket_id", "account_id", "created_ts", "category", "priority", "channel",
             "first_response_hours", "csat_score"]]

    tables = dict(customer_accounts=accts, subscription_events=ev, support_tickets=tk,
                  product_usage_monthly=us)
    latent = dict(episodes=sim["EP"], tickets=latent_tickets, e_base=sim["e_base"], deleted=sim["deleted"],
                  test_ids=sim["test_ids"], n_acc=sim["n_acc"], duplicate_rows=int(len(dup)),
                  placeholder_event_ids=ev.loc[ev["event_ts"].str.startswith("1970"), "event_id"].tolist(),
                  cents_rows=int(cents.sum()))
    return tables, latent


def write_tables(tables: dict[str, pd.DataFrame], out_dir: str | Path) -> dict[str, str]:
    """Write CSVs + SQLite and return {file: sha256}."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    db = out / "saas_churn.sqlite"
    if db.exists():
        db.unlink()
    with sqlite3.connect(db) as con:
        for name in TABLES:
            tables[name].to_csv(out / f"{name}.csv", index=False, lineterminator="\n")  # same bytes on every OS
            tables[name].to_sql(name, con, index=False)
    con.close()
    sums = {f"{n}.csv": hashlib.sha256((out / f"{n}.csv").read_bytes()).hexdigest() for n in TABLES}
    (out / "CHECKSUMS.sha256").write_text("".join(f"{h}  {f}\n" for f, h in sums.items()))
    return sums
