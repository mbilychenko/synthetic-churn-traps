"""Q01 - What was the churn rate in Q1 2026? (the quick Text-to-SQL answer)"""
import sqlite3
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from churnlab.common import DATA, SQL, save  # noqa: E402

QUESTION = "Show me the churn rate for Q1 2026."


def main():
    sql = (SQL / "q1_churn_rate.sql").read_text()
    with sqlite3.connect(DATA / "saas_churn.sqlite") as con:
        n, c, rate = con.execute(sql).fetchone()
    L = [f"Running [`sql/q1_churn_rate.sql`](../sql/q1_churn_rate.sql) against `data/saas_churn.sqlite`:", "",
         f"**Q1 2026 churn rate = {rate:.1f}%** ({c:,} accounts with a cancellation / {n:,} paying accounts "
         f"on 2026-01-01).", "",
         "The query runs and returns one clean number. Every definition choice inside it is examined in Q02."]
    save("q01", "Q1 2026 churn rate", QUESTION, L,
         dict(churn_rate_pct=round(rate, 2), churned=c, paying_jan_1=n))


if __name__ == "__main__":
    main()
