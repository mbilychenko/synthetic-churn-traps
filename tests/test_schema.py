"""Schema and integrity checks on the exported tables."""
import pandas as pd
import pytest

from churnlab.common import load

SCHEMA = {
    "customer_accounts": ["account_id", "signup_date", "billing_email_domain", "country", "industry",
                          "company_size", "acquisition_channel", "onboarding_completed", "integrations_count",
                          "current_plan", "current_seats", "current_mrr", "account_status", "cancellation_reason",
                          "last_cancellation_date", "last_login_date", "last_nps_score"],
    "subscription_events": ["event_id", "account_id", "event_ts", "event_type", "plan", "billing_cycle", "seats",
                            "mrr_before", "mrr_after", "reason_code", "source_system"],
    "support_tickets": ["ticket_id", "account_id", "created_ts", "category", "priority", "channel",
                        "first_response_hours", "csat_score"],
    "product_usage_monthly": ["account_id", "month", "active_users", "seats", "sessions", "reports_created",
                              "api_calls", "avg_session_minutes"],
}


@pytest.fixture(scope="module")
def tables():
    ev, acc, tk, us = load()
    return dict(subscription_events=ev, customer_accounts=acc, support_tickets=tk, product_usage_monthly=us)


@pytest.mark.parametrize("name", SCHEMA)
def test_columns(tables, name):
    assert [c for c in tables[name].columns if c not in ("ts",)] == SCHEMA[name]


def test_keys(tables):
    assert tables["customer_accounts"].account_id.is_unique
    assert tables["support_tickets"].ticket_id.is_unique
    ev = tables["subscription_events"]
    # duplicates are exact copies (re-run ingestion), never conflicting rows with the same id
    assert ev.drop(columns="ts").drop_duplicates().event_id.is_unique


def test_dates_within_export_window(tables):
    ev = tables["subscription_events"]
    real = ev[ev.ts > "2000-01-01"]
    assert real.ts.min() >= pd.Timestamp("2023-01-01")
    assert real.ts.max() <= pd.Timestamp("2026-11-01")
    assert tables["support_tickets"].ts.max() <= pd.Timestamp("2026-11-01")


def test_no_personal_data(tables):
    acc = tables["customer_accounts"]
    assert acc.billing_email_domain.str.endswith(".example").all()      # RFC 2606 reserved domain
    assert acc.account_id.str.fullmatch(r"A\d{5}").all()
