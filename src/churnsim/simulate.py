"""
Agent-style simulation of a fictional B2B SaaS company.

Each account is simulated month by month from trial signup to the export date.
Behaviour is driven entirely by `config/simulation.toml` and a seeded NumPy
generator; no external data is read.

The simulation keeps some *latent* state that a real company would never see
(e.g. the engagement trait, the day a customer privately decided to leave).
`simulate()` returns it alongside the observable records, but the public export
(`churnsim.export`) writes only what a real company's systems would record.
"""
from __future__ import annotations

import datetime as dt

import numpy as np

from .config import Calendar, date_ord


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def next_anniversary(day: int, dom: int) -> int:
    """First date strictly after `day` whose day-of-month equals `dom` (<= 28)."""
    d = dt.date.fromordinal(day)
    cand = dt.date(d.year, d.month, dom)
    if cand.toordinal() <= day:
        y, m = (d.year + 1, 1) if d.month == 12 else (d.year, d.month + 1)
        cand = dt.date(y, m, dom)
    return cand.toordinal()


def simulate(cfg: dict, seed: int | None = None) -> dict:
    seed = cfg["seed"] if seed is None else seed
    cal = Calendar.from_config(cfg)
    N = cal.n_months
    MS = [cal.month_start(m) for m in range(N + 1)]
    EXPORT = cal.export_ord

    rng = np.random.default_rng(seed)          # company, customers, billing
    rng_t = np.random.default_rng(seed + 1)    # support desk

    A, E, P = cfg["accounts"], cfg["engagement"], cfg["plans"]
    H, CP, INV = cfg["churn_hazard"], cfg["churn_process"], cfg["involuntary_churn"]
    BO, S, U, PC = cfg["billing_operations"], cfg["support"], cfg["usage"], cfg["price_change"]

    price_change_ord = date_ord(PC["effective_date"])
    pc_month = cal.month_of(price_change_ord)
    pc_months = tuple(range(pc_month, pc_month + PC["risk_months"]))
    outage_month = cal.month_index(cfg["export_artefacts"]["tracking_outage_month"])
    outage_capture = cfg["export_artefacts"]["tracking_outage_capture"]
    reasons = [r or None for r in BO["cancellation_reasons"]]
    product_cats = (S["product_categories"], S["product_category_probs"])
    leaving_cats = (S["leaving_categories"], S["leaving_category_probs"])

    # ---------------- signups ----------------------------------------------------
    sg = cfg["signups"]
    signups = []
    for m in range(N):
        n = int(rng.normal(sg["base_per_month"] + sg["growth_per_month"] * m, sg["noise_sd"]))
        signups.append(MS[m] + rng.integers(0, MS[m + 1] - MS[m], n))
    signup_days = np.sort(np.concatenate(signups))
    n_acc = len(signup_days)

    # ---------------- static traits ---------------------------------------------
    size_i = rng.choice(len(A["company_sizes"]), n_acc, p=A["company_size_probs"])
    ch_p = np.array(A["channel_probs_by_size"])
    channel_i = np.array([rng.choice(5, p=ch_p[s]) for s in size_i])
    ind_p = np.array(A["industry_weights"]); ind_p = ind_p / ind_p.sum()
    industry = rng.choice(np.array(A["industries"]), n_acc, p=ind_p)
    cty_p = np.array(A["country_weights"]); cty_p = cty_p / cty_p.sum()
    country = rng.choice(np.array(A["countries"]), n_acc, p=cty_p)
    e_base = rng.normal(0, 1, n_acc) + E["effect_referral"] * (channel_i == 2) \
        + E["effect_paid_search"] * (channel_i == 1)
    ob, ig, tc = cfg["onboarding"], cfg["integrations"], cfg["trial_conversion"]
    onboarding = rng.random(n_acc) < sigmoid(ob["intercept"] + ob["engagement"] * e_base)
    integrations = rng.poisson(np.exp(ig["intercept"] + ig["engagement"] * e_base + ig["size_51_plus"] * (size_i >= 2)))
    p_conv = sigmoid(tc["intercept"] + tc["engagement"] * e_base + tc["onboarding"] * onboarding
                     + tc["outbound_sales"] * (channel_i == 4))
    converts = rng.random(n_acc) < p_conv
    plan_p = np.array(P["probs_by_size"])
    plan_i = np.array([rng.choice(3, p=plan_p[s]) for s in size_i])
    plans = np.array(P["names"])
    annual = rng.random(n_acc) < np.array(P["annual_probs"])[plan_i]
    seat_rng = A["seat_range_by_size"]
    seats0 = np.array([rng.integers(seat_rng[s][0], seat_rng[s][1] + 1) for s in size_i])
    deleted = rng.random(n_acc) < A["deleted_account_rate"]

    EV, US, TK, EP, ACC = [], [], [], [], []
    k_counter = [0]

    def ev(aid, day, etype, plan, cycle, seats, mb, ma, reason=None):
        k_counter[0] += 1
        EV.append((aid, day, k_counter[0], etype, plan, cycle, seats, mb, ma, reason))

    def ticket(aid, day, pool, csat_mu, leaving):
        cats, p = pool
        cat = cats[rng_t.choice(len(cats), p=p)]
        frh = round(float(rng_t.lognormal(S["first_response_log_mean"], S["first_response_log_sd"])), 1)
        csat = int(np.clip(round(rng_t.normal(csat_mu, 1.0)), 1, 5)) if rng_t.random() < S["csat_response_prob"] else None
        TK.append((aid, day, cat, rng_t.choice(["low", "normal", "high"], p=[.3, .55, .15]),
                   rng_t.choice(["email", "chat", "phone"], p=[.5, .4, .1]), frh, csat, leaving))

    def mrr_of(plan, seats, cycle, day):
        if plan == "Free":
            return 0.0
        price = P["price_per_seat"][plan]
        if plan == PC["plan"] and cycle == PC["billing_cycle"] and day >= price_change_ord:
            price = PC["new_price_per_seat"]
        return round(seats * price * (P["annual_discount"] if cycle == "annual" else 1.0), 2)

    for i in range(n_acc):
        aid = f"A{i + 1:05d}"
        signup = int(signup_days[i])
        size = int(size_i[i])
        eb = float(e_base[i])
        integ = int(integrations[i])
        onb = bool(onboarding[i])
        plan = plans[plan_i[i]]
        cycle = "annual" if annual[i] else "monthly"
        seats = int(seats0[i])
        last_login = None
        last_reason = None
        exit_nps = None
        last_cancel = None

        ev(aid, signup, "trial_start", "Trial", None, seats, 0.0, 0.0)
        trial_end = signup + sg["trial_days"]
        if trial_end > EXPORT:
            ACC.append(dict(i=i, status="trial", plan="Trial", seats=seats, mrr=0.0,
                            last_login=signup + 1, reason=None, exit_nps=None))
            continue
        if not converts[i]:
            ev(aid, trial_end, "trial_expired", "Trial", None, seats, 0.0, 0.0)
            ACC.append(dict(i=i, status="trial_expired", plan="Trial", seats=seats, mrr=0.0,
                            last_login=signup + int(rng.integers(0, 10)), reason=None, exit_nps=None))
            continue

        conv = signup + int(rng.integers(tc["convert_day_min"], tc["convert_day_max"]))
        spell_start = conv
        dom = min(dt.date.fromordinal(conv).day, 28)
        term_end = conv + 365 if cycle == "annual" else None
        mrr = mrr_of(plan, seats, cycle, conv)
        ev(aid, conv, "subscription_start", plan, cycle, seats, 0.0, mrr)
        state = "paid"
        dev, champ_m, decided = 0.0, -99, False
        prev_ratio = None
        pending_eff = pending_kind = None
        react_day = None
        cur_episode = None

        for m in range(cal.month_of(conv), N):
            ms, me = MS[m], MS[m + 1] - 1
            # ---- latent engagement ------------------------------------------
            dev = E["ar_coefficient"] * dev + rng.normal(0, E["monthly_noise_sd"])
            if rng.random() < E["champion_left_monthly_prob"]:
                champ_m = m
                dev -= E["champion_left_drop"]
            e = eb + dev - (E["after_decision_drop"] if decided else 0.0) \
                - (E["free_plan_drop"] if state == "free" else 0.0)

            # ---- usage + product tickets -----------------------------------
            usage_ratio = None
            if state in ("paid", "pending", "free"):
                au = int(rng.binomial(seats, sigmoid(U["active_share_intercept"] + U["active_share_engagement"] * e)))
                usage_ratio = au / seats
                sess = int(rng.poisson(au * np.exp(U["sessions_log_base"] + U["sessions_log_engagement"] * e)))
                rep = int(rng.poisson(U["reports_per_session"] * sess))
                api = int(rng.poisson(integ * np.exp(U["api_log_base"] + U["api_log_engagement"] * e))) if integ > 0 else 0
                mins = round(max(1.0, float(rng.normal(U["minutes_base"] + U["minutes_engagement"] * e, U["minutes_sd"]))), 1)
                if m == outage_month:
                    sess, rep = int(sess * outage_capture), int(rep * outage_capture)
                US.append((aid, m, au, seats, sess, rep, api, mins))
                if au > 0:
                    last_login = min(me, EXPORT) - int(rng.integers(0, U["last_login_max_lag_days"]))
                for _ in range(int(rng_t.poisson(S["monthly_rate"] * (seats / 10) ** S["seat_exponent"]))):
                    ticket(aid, ms + int(rng_t.integers(0, me - ms + 1)), product_cats, S["product_csat_mean"], False)

            if state == "paid":
                # list-price change
                if m == pc_month and plan == PC["plan"] and cycle == PC["billing_cycle"]:
                    new = mrr_of(plan, seats, cycle, price_change_ord)
                    ev(aid, price_change_ord, "price_change", plan, cycle, seats, mrr, new)
                    mrr = new
                # annual renewal
                if cycle == "annual" and ms <= term_end <= me:
                    ev(aid, term_end, "renewal", plan, cycle, seats, mrr, mrr)
                    term_end += 365
                # seat changes
                if rng.random() < BO["seat_change_monthly_prob"]:
                    d = ms + int(rng.integers(0, me - ms + 1))
                    new_seats = max(1, int(round(seats * rng.uniform(*BO["seat_change_factor"]))))
                    if new_seats != seats:
                        new = mrr_of(plan, new_seats, cycle, d)
                        ev(aid, d, "seat_change", plan, cycle, new_seats, mrr, new)
                        seats, mrr = new_seats, new
                # billing-system operation: cancel + reactivate on the same day
                if rng.random() < BO["same_day_cancel_reactivate_prob"]:
                    d = ms + int(rng.integers(0, me - ms + 1))
                    reason = "billing_migration" if rng.random() < BO["migration_reason_prob"] else None
                    ev(aid, d, "cancellation", plan, cycle, seats, mrr, 0.0, reason)
                    last_cancel = d
                    ev(aid, d, "reactivation", plan, cycle, seats, 0.0, mrr)
                    EP.append(dict(account_id=aid, churn_type="technical_same_day", decision_day=None,
                                   request_day=d, effective_day=d, refunded=False, reactivation_day=d,
                                   contacted_support_after_decision=False))
                # failed payment
                elif rng.random() < INV["monthly_prob"]:
                    d = ms + int(rng.integers(0, me - ms + 1))
                    ev(aid, d, "payment_failed", plan, cycle, seats, mrr, mrr, "card_declined")
                    if rng.random() < INV["recover_prob"]:
                        ev(aid, d + int(rng.integers(*INV["recover_days"])), "payment_recovered",
                           plan, cycle, seats, mrr, mrr)
                    else:
                        state, pending_eff, pending_kind = "pending", d + INV["grace_days"], "involuntary"
                        cur_episode = dict(account_id=aid, churn_type="involuntary", decision_day=None,
                                           request_day=d, effective_day=d + INV["grace_days"], refunded=False,
                                           reactivation_day=None, contacted_support_after_decision=False)
                # voluntary decision to leave
                else:
                    in_window = True
                    x = H["intercept"]
                    if cycle == "annual":
                        in_window = ms < term_end <= ms + H["annual_window_days"]
                        x += H["annual_renewal_window"]
                    r_used = prev_ratio if prev_ratio is not None else usage_ratio
                    low = r_used is not None and r_used < H["low_usage_threshold"]
                    x += H["low_usage_previous_month"] * low
                    x += H["negative_engagement"] * max(0.0, -(eb + dev))
                    x += H["onboarding_completed"] * onb
                    x += H["per_integration"] * min(integ, 3)
                    x += H["plan_starter"] * (plan == "Starter")
                    x += H["plan_enterprise"] * (plan == "Enterprise")
                    x += H["price_change_exposure"] * (
                        plan == PC["plan"] and cycle == PC["billing_cycle"] and m in pc_months)
                    x += H["early_tenure"] * (m - cal.month_of(spell_start) <= 1)
                    x += H["champion_left_recently"] * (1 <= m - champ_m <= 4)
                    x += H["enterprise_low_usage"] * (plan == "Enterprise" and low)
                    x += H["company_size_1_10"] * (size == 0)
                    if in_window and rng.random() < sigmoid(x):
                        decided = True
                        dday = ms + int(rng.integers(0, me - ms + 1))
                        gap = int(rng.integers(*CP["request_gap_days"]))
                        req = dday + gap
                        contacts = rng_t.random() < S["leaving_contact_prob"]
                        if contacts:
                            ticket(aid, dday + int(rng_t.integers(0, gap + 1)), leaving_cats, S["leaving_csat_mean"], True)
                            if rng_t.random() < S["leaving_second_ticket_prob"]:
                                ticket(aid, dday + int(rng_t.integers(0, gap + 1)), leaving_cats,
                                       S["leaving_csat_mean"], True)
                        epi = dict(account_id=aid, churn_type="voluntary", decision_day=dday, request_day=req,
                                   effective_day=None, refunded=False, reactivation_day=None,
                                   contacted_support_after_decision=contacts)
                        if plan != "Enterprise" and rng.random() < CP["downgrade_to_free_prob"]:
                            ev(aid, req, "downgrade", "Free", None, seats, mrr, 0.0)
                            mrr, plan, cycle, state = 0.0, "Free", None, "free"
                            epi.update(churn_type="downgrade_to_free", effective_day=req)
                            EP.append(epi)
                            decided = False
                        else:
                            reason = reasons[rng.choice(len(reasons), p=BO["cancellation_reason_probs"])]
                            ev(aid, req, "cancel_request", plan, cycle, seats, mrr, mrr, reason)
                            last_reason = reason
                            if req - spell_start <= CP["money_back_days"] and rng.random() < CP["money_back_prob"]:
                                pending_eff, epi["refunded"] = req, True
                            elif rng.random() < (CP["goodwill_refund_prob_annual"] if cycle == "annual"
                                                 else CP["goodwill_refund_prob_monthly"]):
                                pending_eff, epi["refunded"] = req, True
                            elif cycle == "monthly":
                                pending_eff = next_anniversary(req, dom)
                            else:
                                pending_eff = max(term_end, req)
                            state, pending_kind = "pending", reason
                            epi["effective_day"] = pending_eff
                            cur_episode = epi

            prev_ratio = usage_ratio
            # ---- cancellation takes effect ----------------------------------
            if state == "pending" and pending_eff <= me and pending_eff <= EXPORT:
                reason = "payment_failed" if pending_kind == "involuntary" else pending_kind
                ev(aid, pending_eff, "cancellation", plan, cycle, seats, mrr, 0.0, reason)
                last_cancel = pending_eff
                if cur_episode and cur_episode["refunded"]:
                    ev(aid, pending_eff, "refund", plan, cycle, seats, 0.0, 0.0, "refund_issued")
                state, decided = "churned", False
                if pending_kind != "involuntary" and rng.random() < CP["exit_survey_prob"]:
                    exit_nps = int(rng.integers(0, 6))
                if pending_kind == "involuntary":
                    p90, plate = CP["involuntary_reactivate_within_90d_prob"], CP["involuntary_reactivate_later_prob"]
                else:
                    p90, plate = CP["reactivate_within_90d_prob"], CP["reactivate_later_prob"]
                r = rng.random()
                if r < p90:
                    react_day = pending_eff + int(rng.integers(1, 91))
                elif r < p90 + plate:
                    react_day = pending_eff + int(rng.integers(91, 301))
                else:
                    react_day = None
                if cur_episode is not None:
                    cur_episode["reactivation_day"] = react_day
                    EP.append(cur_episode)
                    cur_episode = None

            # ---- win-back -----------------------------------------------------
            if state == "churned" and react_day is not None and react_day <= min(me, EXPORT):
                spell_start, dom = react_day, min(dt.date.fromordinal(react_day).day, 28)
                if cycle == "annual":
                    term_end = react_day + 365
                mrr = mrr_of(plan, seats, cycle, react_day)
                ev(aid, react_day, "reactivation", plan, cycle, seats, 0.0, mrr)
                state, dev, react_day, champ_m = "paid", dev + E["reactivation_boost"], None, -99

            # ---- free accounts occasionally upgrade -------------------------
            if state == "free" and rng.random() < BO["free_upgrade_monthly_prob"]:
                d = ms + int(rng.integers(0, me - ms + 1))
                plan, cycle = "Starter", "monthly"
                mrr = mrr_of(plan, seats, cycle, d)
                ev(aid, d, "upgrade", plan, cycle, seats, 0.0, mrr)
                state, spell_start, dom = "paid", d, min(dt.date.fromordinal(d).day, 28)

        if cur_episode is not None:
            EP.append(cur_episode)
        status = {"paid": "active", "pending": "pending_cancellation", "free": "free",
                  "churned": "cancelled"}[state]
        ACC.append(dict(i=i, status=status, plan=plan, seats=seats,
                        mrr=mrr if state in ("paid", "pending") else 0.0,
                        last_login=last_login, reason=last_reason if status == "cancelled" else None,
                        exit_nps=exit_nps if status == "cancelled" else None,
                        last_cancel=last_cancel if last_cancel is None or last_cancel <= EXPORT else None))

    # ---------------- internal QA accounts on the production system ---------------
    IA = cfg["internal_accounts"]
    test_ids = []
    for t in range(IA["count"]):
        aid = f"A{n_acc + t + 1:05d}"
        test_ids.append(aid)
        start = int(rng.integers(MS[2], MS[33]))
        ev(aid, start, "subscription_start", "Growth", "monthly", 1, 0.0, 1.0)
        for m in range(cal.month_of(start) + 1, N):
            US.append((aid, m, 1, 1, int(rng.integers(0, 15)), 0, 0, 3.0))
            if rng.random() < IA["cancel_flow_test_monthly_prob"]:
                d = MS[m] + int(rng.integers(0, 25))
                ev(aid, d, "cancel_request", "Growth", "monthly", 1, 1.0, 1.0, "other")
                ev(aid, d, "cancellation", "Growth", "monthly", 1, 1.0, 0.0, "other")
                ev(aid, d + int(rng.integers(0, 4)), "reactivation", "Growth", "monthly", 1, 0.0, 1.0)
        ACC.append(dict(i=None, aid=aid, test=True, start=start))

    return dict(cfg=cfg, cal=cal, MS=MS, rng=rng, seed=seed, n_acc=n_acc, signup_days=signup_days,
                size_i=size_i, channel_i=channel_i, industry=industry, country=country, e_base=e_base,
                onboarding=onboarding, integrations=integrations, deleted=deleted,
                EV=EV, US=US, TK=TK, EP=EP, ACC=ACC, test_ids=test_ids)
