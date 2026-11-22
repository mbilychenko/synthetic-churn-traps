-- Q1 2026 churn rate: the straightforward query.
-- Churned  = accounts with a 'cancellation' event dated in Q1 2026
-- Base     = accounts whose latest subscription event before 2026-01-01 left MRR > 0
WITH last_state AS (
    SELECT account_id,
           mrr_after,
           ROW_NUMBER() OVER (PARTITION BY account_id ORDER BY event_ts DESC, event_id DESC) AS rn
    FROM subscription_events
    WHERE event_ts < '2026-01-01'
),
active_at_start AS (
    SELECT account_id FROM last_state WHERE rn = 1 AND mrr_after > 0
),
churned AS (
    SELECT DISTINCT account_id
    FROM subscription_events
    WHERE event_type = 'cancellation'
      AND event_ts >= '2026-01-01'
      AND event_ts <  '2026-04-01'
)
SELECT (SELECT COUNT(*) FROM active_at_start)                                   AS paying_accounts_jan_1,
       (SELECT COUNT(*) FROM churned)                                           AS churned_accounts,
       ROUND(100.0 * (SELECT COUNT(*) FROM churned)
                   / (SELECT COUNT(*) FROM active_at_start), 2)                 AS churn_rate_pct;
