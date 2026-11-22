"""Q02 - How much does the Q1 churn rate depend on what we call 'churn'?"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from churnlab.common import churn_rate, churn_records, internal_ids, load, save  # noqa: E402

QUESTION = "Which business decisions are hidden inside the Q1 churn number, and how much does each one move it?"

CHOICES = [
    ("Does a refund count as lost?", "Yes", "No", dict(count_refunds=False)),
    ("Which date do we attribute the cancellation to?", "End of term (effective)", "Request date",
     dict(attribution="request")),
    ("If they reactivate the same day, did they churn?", "Yes", "No", dict(same_day_counts=False)),
    ("If they reactivate within 90 days, did they churn?", "Yes", "No", dict(react_90d_counts=False)),
    ("Which denominator do we use?", "Start of period", "Average over period", dict(denominator="average")),
    ("Do expired trials or downgrades to Free count?", "Exclude", "Include", dict(trials_and_downgrades=True)),
]


def main():
    ev, acc, tk, us = load()
    recs = churn_records(ev)
    base = churn_rate(ev, recs)
    L = ["Each row changes **one** choice, starting from the straightforward query in Q01 "
         "(its choice is the first option in every row).", "",
         "| # | Question | Q01 choice | Alternative |", "|---|---|---|---|"]
    variants = {}
    for k, (q, a, b, kw) in enumerate(CHOICES, 1):
        v = churn_rate(ev, recs, **kw)
        variants[q] = round(v, 2)
        L.append(f"| {k} | {q} | {a} **{base:.1f}%** | {b} **{v:.1f}%** |")
    strict = churn_rate(ev, recs, attribution="request", same_day_counts=False, react_90d_counts=False,
                        denominator="average")
    ev_clean = ev[~ev["account_id"].isin(internal_ids(acc))]
    clean = churn_rate(ev_clean, churn_records(ev_clean))
    lo, hi = min([base, *variants.values()]), max([base, *variants.values()])
    L += ["", f"- Single-choice range: **{lo:.1f}% to {hi:.1f}%**. Six defensible answers, and the quick query picked one "
               f"without saying so.",
          f"- Combining request date, no same-day reactivations, no reactivations within 90 days and an average "
          f"denominator gives **{strict:.1f}%**.",
          f"- Removing the internal QA accounts (see Q00) moves the Q01 number to **{clean:.1f}%**."]
    save("q02", "Churn definition sensitivity", QUESTION, L,
         dict(base_pct=round(base, 2), variants_pct=variants, range_pct=[round(lo, 2), round(hi, 2)],
              combined_strict_pct=round(strict, 2), without_internal_pct=round(clean, 2)))


if __name__ == "__main__":
    main()
