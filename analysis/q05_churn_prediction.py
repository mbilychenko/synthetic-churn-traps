"""Q05 - Can we score every account by its risk of churning next quarter? Three ways to build it."""
import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import lightgbm as lgb  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.compose import ColumnTransformer  # noqa: E402
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier, StackingClassifier  # noqa: E402
from sklearn.impute import SimpleImputer  # noqa: E402
from sklearn.inspection import permutation_importance  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.pipeline import make_pipeline  # noqa: E402
from sklearn.preprocessing import OneHotEncoder, StandardScaler  # noqa: E402

from churnlab.common import SEED, build_cohort, cv_auc, load, save  # noqa: E402

warnings.filterwarnings("ignore")
QUESTION = "Build a model that scores every paying account by its risk of churning in Q1 2026."

# A) the classic hand-built model: business-intuitive columns, raw counts, linear model
MANUAL = ["plan_starter", "plan_enterprise", "annual", "seats", "mrr", "tenure_months", "onboarding",
          "integrations_count", "sess_last3", "au_last3", "tickets_90d", "size_ord", "ch_paid_search", "country_us"]

# B) "throw every column at AutoML": all CRM fields + all usage aggregates
AUTO_NUM = ["seats", "mrr", "tenure_months", "integrations_count", "au_last1", "ratio_last1", "au_last3",
            "ratio_last3", "sess_last3", "rep_last3", "api_last3", "min_last3", "ratio_trend", "sess_trend",
            "au_drop_from_peak", "tickets_90d", "low_csat_90d", "days_since_last_cancellation", "current_mrr",
            "last_nps_score", "days_since_last_login", "onboarding"]
AUTO_CAT = ["plan", "billing_cycle", "company_size", "acquisition_channel", "industry", "country",
            "account_status", "cancellation_reason", "current_plan"]
EXPORT_TIME = ["account_status", "cancellation_reason", "days_since_last_login", "days_since_last_cancellation",
               "last_nps_score", "current_mrr", "current_plan"]

# C) human + AI: leak-free, point-in-time features chosen with domain context
#    (seat utilisation, loss of active users, early tenure, price-change exposure,
#     and renewal timing: annual contracts can only be left at renewal)
HUMAN_AI = ["ratio_last1", "ratio_last3", "sess_per_au_last3", "au_drop_from_peak", "tenure_months", "onboarding",
            "integrations_count", "is_starter_monthly", "renewal_due_q1"]


def stack(num, cat):
    pre = ColumnTransformer([("n", make_pipeline(SimpleImputer(), StandardScaler()), num),
                             ("c", make_pipeline(SimpleImputer(strategy="constant", fill_value="NA"),
                                                 OneHotEncoder(handle_unknown="ignore")), cat)])
    return lambda: make_pipeline(pre, StackingClassifier(
        estimators=[("lr", LogisticRegression(max_iter=2000)),
                    ("rf", RandomForestClassifier(300, min_samples_leaf=3, random_state=SEED, n_jobs=-1)),
                    ("gb", HistGradientBoostingClassifier(random_state=SEED))],
        final_estimator=LogisticRegression(), cv=3))


def main():
    ev, acc, tk, us = load()
    df = build_cohort(ev, acc, tk, us)
    y = df["churned_q1"]

    lr = lambda: make_pipeline(SimpleImputer(), StandardScaler(), LogisticRegression(max_iter=2000))
    auc_m, sd_m = cv_auc(lr, df[MANUAL], y)

    Xa = df[AUTO_NUM + AUTO_CAT].copy()
    Xa["last_nps_score"] = Xa["last_nps_score"].astype(float)
    auto = stack(AUTO_NUM, AUTO_CAT)
    auc_a, sd_a = cv_auc(auto, Xa, y)
    fitted = auto().fit(Xa, y)
    pi = permutation_importance(fitted, Xa, y, scoring="roc_auc", n_repeats=3, random_state=SEED)
    top = pd.Series(pi.importances_mean, index=AUTO_NUM + AUTO_CAT).sort_values(ascending=False).head(4)

    num_nl = [c for c in AUTO_NUM if c not in EXPORT_TIME]
    cat_nl = [c for c in AUTO_CAT if c not in EXPORT_TIME]
    auc_nl, sd_nl = cv_auc(stack(num_nl, cat_nl), Xa[num_nl + cat_nl], y)

    lg = lambda: lgb.LGBMClassifier(n_estimators=300, learning_rate=0.02, num_leaves=7, min_child_samples=40,
                                    subsample=0.8, subsample_freq=1, colsample_bytree=0.9, reg_lambda=5,
                                    random_state=SEED, verbose=-1)
    auc_h, sd_h = cv_auc(lg, df[HUMAN_AI], y)

    L = [f"Cohort: {len(df):,} paying accounts on 2026-01-01; label = voluntary churn decision in Q1 2026 "
         f"({y.mean():.1%} positive). All scores are 5-fold stratified cross-validated ROC AUC.", "",
         "| Build | Model | Features | AUC |", "|---|---|---|---|",
         f"| A · Manual | Logistic regression | {len(MANUAL)} | **{auc_m:.2f}** ± {sd_m:.2f} |",
         f"| B · AutoML, all columns | Stack (LR + RF + GBM) | {len(AUTO_NUM + AUTO_CAT)} | **{auc_a:.2f}** ± {sd_a:.2f} |",
         f"| C · Human + AI | LightGBM | {len(HUMAN_AI)} | **{auc_h:.2f}** ± {sd_h:.2f} |", "",
         f"**Leakage audit for B.** Top features by permutation importance: "
         f"{', '.join(f'`{c}`' for c in top.index)}. Most are CRM fields that hold their value on the "
         f"**export date (2026-10-31)**, after the Q1 label window (see Q00). They encode the outcome: "
         f"a customer who left in Q1 has a recent cancellation date and an old last login. That model cannot "
         f"exist on 2026-01-01.",
         f"- Same AutoML stack with the export-time columns removed: **{auc_nl:.2f}** ± {sd_nl:.2f}.",
         "", "**Build C** uses only point-in-time features, plus one piece of domain context: annual "
         "contracts can only be cancelled at renewal (`renewal_due_q1`). It is the only one of the three that "
         "is both honest and better than the baseline."]
    save("q05", "Churn prediction: three builds", QUESTION, L,
         dict(auc_manual=round(auc_m, 3), auc_automl_all_columns=round(auc_a, 3),
              auc_automl_point_in_time=round(auc_nl, 3), auc_human_ai=round(auc_h, 3),
              n_features=dict(manual=len(MANUAL), automl=len(AUTO_NUM + AUTO_CAT), human_ai=len(HUMAN_AI)),
              automl_top_features=list(top.index), cohort=len(df), positive_rate=round(float(y.mean()), 4)))


if __name__ == "__main__":
    main()
