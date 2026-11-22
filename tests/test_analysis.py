"""Cross-checks between independent implementations of the same analysis."""
import importlib.util
import sqlite3
from pathlib import Path

from churnlab.common import DATA, SQL, churn_rate, churn_records, load

ROOT = Path(__file__).resolve().parents[1]


def test_sql_and_pandas_agree_on_q1_churn():
    ev, *_ = load()
    with sqlite3.connect(DATA / "saas_churn.sqlite") as con:
        _, _, sql_rate = con.execute((SQL / "q1_churn_rate.sql").read_text()).fetchone()
    assert abs(churn_rate(ev, churn_records(ev)) - sql_rate) < 0.01


def test_point_in_time_model_uses_no_export_time_columns():
    spec = importlib.util.spec_from_file_location("q05", ROOT / "analysis" / "q05_churn_prediction.py")
    q05 = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(q05)
    assert not set(q05.HUMAN_AI) & set(q05.EXPORT_TIME)
    assert not set(q05.MANUAL) & set(q05.EXPORT_TIME)
