"""Shared loading, churn definitions, feature building and result writing."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
RESULTS = ROOT / "results"
SQL = ROOT / "sql"

Q1_START, Q1_END = pd.Timestamp("2026-01-01"), pd.Timestamp("2026-04-01")
EXPORT = pd.Timestamp("2026-10-31")          # export date stated in the datasheet
INTERNAL_DOMAIN = "qa-internal.example"
SEED = 42


# --------------------------------------------------------------------------- #
# I/O
# --------------------------------------------------------------------------- #
def load(data_dir: Path | None = None):
    d = Path(data_dir or DATA)
    ev = pd.read_csv(d / "subscription_events.csv")
    acc = pd.read_csv(d / "customer_accounts.csv")
    tk = pd.read_csv(d / "support_tickets.csv")
    us = pd.read_csv(d / "product_usage_monthly.csv")
    ev["ts"] = pd.to_datetime(ev["event_ts"])
    tk["ts"] = pd.to_datetime(tk["created_ts"])
    us["month"] = pd.to_datetime(us["month"])
    return ev, acc, tk, us


def internal_ids(acc: pd.DataFrame) -> set:
    return set(acc.loc[acc["billing_email_domain"] == INTERNAL_DOMAIN, "account_id"])


def save(qid: str, title: str, question: str, lines: list[str], metrics: dict) -> None:
    """Write results/<qid>.md (human) and results/<qid>.json (machine)."""
    RESULTS.mkdir(exist_ok=True)
    md = [f"## {qid.upper()} · {title}", "", f"> **Question:** {question}", ""] + lines
    (RESULTS / f"{qid}.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    (RESULTS / f"{qid}.json").write_text(json.dumps(
        dict(id=qid, title=title, question=question, metrics=metrics), indent=2, default=_jsonable), encoding="utf-8")
    print("\n".join(md))


def _jsonable(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    return str(o)


# --------------------------------------------------------------------------- #
# Subscription state and churn definitions
# --------------------------------------------------------------------------- #
def paid_at(ev: pd.DataFrame, when: pd.Timestamp) -> set:
    """Accounts whose last subscription event before `when` left them with MRR > 0."""
    e = ev[ev["ts"] < when].sort_values(["ts", "event_id"])
    last = e.groupby("account_id").tail(1)
    return set(last.loc[last["mrr_after"] > 0, "account_id"])


def trials_at(ev: pd.DataFrame, when: pd.Timestamp) -> set:
    e = ev[ev["ts"] < when].sort_values(["ts", "event_id"])
    last = e.groupby("account_id").tail(1)
    return set(last.loc[last["event_type"] == "trial_start", "account_id"])


def churn_records(ev: pd.DataFrame) -> pd.DataFrame:
    """One row per cancellation (effective) or still-pending cancel request."""
    e = ev.drop_duplicates("event_id").sort_values(["account_id", "ts", "event_id"])
    rows = []
    for aid, g in e.groupby("account_id", sort=False):
        req = None
        types, tss = g["event_type"].values, g["ts"].values
        for j, (t, ts) in enumerate(zip(types, tss)):
            if t == "cancel_request":
                req = ts
            elif t == "cancellation":
                later = tss[j:][types[j:] == "reactivation"]
                refunded = bool(((types[j:j + 3] == "refund") &
                                 (tss[j:j + 3].astype("datetime64[D]") == ts.astype("datetime64[D]"))).any())
                react_days = (later[0].astype("datetime64[D]") - ts.astype("datetime64[D]")).astype(int) \
                    if len(later) else np.nan
                rows.append((aid, ts, req if req is not None else ts, refunded, react_days))
                req = None
        if req is not None:
            rows.append((aid, pd.NaT, req, False, np.nan))
    r = pd.DataFrame(rows, columns=["account_id", "effective_ts", "request_ts", "refunded", "react_days"])
    r["effective_ts"] = pd.to_datetime(r["effective_ts"])
    r["request_ts"] = pd.to_datetime(r["request_ts"])
    return r


def churn_rate(ev, recs, *, count_refunds=True, attribution="effective", same_day_counts=True,
               react_90d_counts=True, denominator="start", trials_and_downgrades=False) -> float:
    """Q1 2026 churn % under one combination of definition choices."""
    r = recs
    if not count_refunds:
        r = r[~r["refunded"]]
    if not same_day_counts:
        r = r[~(r["react_days"] == 0)]
    if not react_90d_counts:
        r = r[~r["react_days"].between(1, 90)]
    col = "effective_ts" if attribution == "effective" else "request_ts"
    churned = set(r.loc[(r[col] >= Q1_START) & (r[col] < Q1_END), "account_id"])
    start = paid_at(ev, Q1_START)
    if denominator == "start":
        denom = len(start)
    else:
        pts = [Q1_START, pd.Timestamp("2026-02-01"), pd.Timestamp("2026-03-01"), Q1_END]
        denom = np.mean([len(paid_at(ev, p)) for p in pts])
    if trials_and_downgrades:
        tr = trials_at(ev, Q1_START)
        q1 = ev[(ev["ts"] >= Q1_START) & (ev["ts"] < Q1_END)]
        expired = set(q1.loc[q1["event_type"] == "trial_expired", "account_id"]) & tr
        down = set(q1.loc[q1["event_type"] == "downgrade", "account_id"]) & start
        churned = churned | expired | down
        denom += len(tr)
    return 100 * len(churned) / denom


# --------------------------------------------------------------------------- #
# Analysis cohort and features (as of 2025-12-31)
# --------------------------------------------------------------------------- #
def build_cohort(ev, acc, tk, us) -> pd.DataFrame:
    """Paying accounts on 2026-01-01 (internal + CRM-deleted removed) with features and the Q1 label.

    Label `churned_q1` = a *voluntary* churn decision in Q1 2026: a cancel request or a downgrade to Free.
    """
    e = ev.drop_duplicates("event_id")
    e = e[e["ts"] > "2000-01-01"]
    cohort = (paid_at(e, Q1_START) - internal_ids(acc)) & set(acc["account_id"])
    q1 = e[(e["ts"] >= Q1_START) & (e["ts"] < Q1_END)]
    y_ids = set(q1.loc[q1["event_type"] == "cancel_request", "account_id"]) \
        | set(q1.loc[q1["event_type"] == "downgrade", "account_id"])
    df = acc[acc["account_id"].isin(cohort)].copy()
    df["churned_q1"] = df["account_id"].isin(y_ids).astype(int)

    pre = e[e["ts"] < Q1_START].sort_values("ts")
    last = pre.groupby("account_id").tail(1).set_index("account_id")
    df["plan"] = df["account_id"].map(last["plan"]).str.strip().str.title()
    df["billing_cycle"] = df["account_id"].map(last["billing_cycle"])
    df["seats"] = df["account_id"].map(last["seats"])
    df["mrr"] = df["account_id"].map(last["mrr_after"])
    first_paid = pre[pre["event_type"] == "subscription_start"].groupby("account_id")["ts"].min()
    # annual contracts can only be left at renewal: when is the next one?
    anchor = pre[pre["event_type"].isin(["subscription_start", "renewal", "reactivation", "upgrade"])] \
        .groupby("account_id")["ts"].max()
    next_renewal = df["account_id"].map(anchor) + pd.Timedelta(days=365)
    df["renewal_due_q1"] = np.where(df["billing_cycle"] == "monthly", 1,
                                    (next_renewal < Q1_END + pd.Timedelta(days=30)).astype(int))
    df["tenure_months"] = ((Q1_START - df["account_id"].map(first_paid)).dt.days / 30.4).round(1)

    u = us[us["month"] < Q1_START].copy()
    u["ratio"] = u["active_users"] / u["seats"].clip(lower=1)

    def agg(lo, hi, name):
        w = u[(u["month"] >= lo) & (u["month"] < hi)].groupby("account_id")
        return w.agg(**{f"au_{name}": ("active_users", "mean"), f"ratio_{name}": ("ratio", "mean"),
                        f"sess_{name}": ("sessions", "mean"), f"rep_{name}": ("reports_created", "mean"),
                        f"api_{name}": ("api_calls", "mean"), f"min_{name}": ("avg_session_minutes", "mean")})

    a1 = agg(pd.Timestamp("2025-12-01"), Q1_START, "last1")
    a3 = agg(pd.Timestamp("2025-10-01"), Q1_START, "last3")
    p3 = agg(pd.Timestamp("2025-07-01"), pd.Timestamp("2025-10-01"), "prev3")
    peak = u[u["month"] >= "2025-01-01"].groupby("account_id")["active_users"].max().rename("au_peak12")
    for t in (a1, a3, p3, peak):
        df = df.merge(t, left_on="account_id", right_index=True, how="left")
    df["ratio_trend"] = df["ratio_last3"] - df["ratio_prev3"]
    df["sess_trend"] = df["sess_last3"] / (df["sess_prev3"] + 1)
    df["au_drop_from_peak"] = 1 - df["au_last1"] / df["au_peak12"].clip(lower=1)
    t90 = tk[(tk["ts"] >= "2025-10-01") & (tk["ts"] < Q1_START)]
    df["tickets_90d"] = df["account_id"].map(t90.groupby("account_id").size()).fillna(0)
    df["low_csat_90d"] = df["account_id"].map(t90[t90["csat_score"] <= 2].groupby("account_id").size()).fillna(0)
    # CRM fields hold their value at EXPORT time (2026-10-31), i.e. after the label window
    df["days_since_last_login"] = (EXPORT - pd.to_datetime(df["last_login_date"])).dt.days
    df["days_since_last_cancellation"] = (EXPORT - pd.to_datetime(df["last_cancellation_date"])).dt.days
    df["sess_per_au_last3"] = df["sess_last3"] / df["au_last3"].clip(lower=0.5)

    df["is_starter_monthly"] = ((df.plan == "Starter") & (df.billing_cycle == "monthly")).astype(int)
    df["plan_starter"] = (df.plan == "Starter").astype(int)
    df["plan_enterprise"] = (df.plan == "Enterprise").astype(int)
    df["annual"] = (df.billing_cycle == "annual").astype(int)
    df["size_ord"] = df.company_size.map({"1-10": 0, "11-50": 1, "51-200": 2, "201-1000": 3, "1000+": 4})
    df["ch_paid_search"] = (df.acquisition_channel == "paid_search").astype(int)
    df["country_us"] = (df.country == "US").astype(int)
    df["onboarding"] = df.onboarding_completed.astype(int)
    return df


def cv_auc(model_fn, X, y, folds=5):
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import StratifiedKFold
    aucs = []
    for tr, te in StratifiedKFold(folds, shuffle=True, random_state=SEED).split(X, y):
        m = model_fn()
        m.fit(X.iloc[tr], y.iloc[tr])
        aucs.append(roc_auc_score(y.iloc[te], m.predict_proba(X.iloc[te])[:, 1]))
    return float(np.mean(aucs)), float(np.std(aucs))
