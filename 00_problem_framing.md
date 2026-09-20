# Problem Framing — Repeat Purchase & Churn Intelligence

## Business question
Which users are about to stop ordering, and which of them can we actually change the outcome for?

This is deliberately two questions, not one. A model that predicts churn well but points us at users who were leaving no matter what is a model that helps no one. The second half — *can we change it* — is what turns this from a prediction exercise into a retention decision.

## Why this matters
Acquiring a new user costs far more than keeping one. In grocery/food-delivery-style marketplaces, the gap between a user's 1st and 2nd order is where most of the drop-off happens — get someone to a 3rd or 4th order and they're statistically far more likely to stick around. So the highest-leverage question a growth/retention team can answer isn't "how many users do we have" — it's "which of our existing users are drifting away, right now, while there's still time to act."

## North Star metric
**Repeat-order rate within a 30-day window** — of users who ordered in period *t*, what fraction order again within 30 days.

This is the metric a retention intervention should move. Everything downstream (churn label, model, tiers) is built to serve this number, not to optimize a leaderboard metric in isolation.

## Input metrics (the levers we can observe per user)
- Order frequency (orders per 30/60/90 days)
- Basket size (average, and trend — is it shrinking?)
- Category breadth (how many distinct categories/products)
- Days since last order (recency)
- Inter-order gap trend (widening = early warning)

## Guardrail metric
**Discount spend per retained user.**

Without a guardrail, "retention" is trivially gameable — blast everyone with a coupon and the repeat-order rate goes up while margin collapses. The guardrail forces the honest question: *did we retain the user, or did we just buy one order?* This is directly tied to the incrementality point in Phase 5 — some flagged users return regardless of intervention, and treating them anyway is pure margin loss.

## How churn will be defined (finalized, Phase 1)
**Threshold: 30 days since last order**, derived from the observed inter-order gap distribution (`sql/01_cohort_retention.sql`, item 4; n=3,214,874 gaps), not chosen arbitrarily:

- 75% of gaps resolve within 15 days; the distribution's 90th percentile lands at **30 days** almost exactly.
- Separately, Instacart censors `days_since_prior_order` at 30 — any true gap of 30+ days is recorded as exactly 30 (visible as a sharp spike: 11.5% of all gaps sit at `gap_days=30`). This means 30 days is also the finest threshold the data can actually support: the dataset cannot distinguish a 30-day gap from a 300-day gap, so setting the line any higher would be asserting precision the data doesn't have.

These two facts converge on the same number for different reasons — one behavioral (where the tail of "normal" reordering genuinely ends), one structural (where the data stops being able to tell churned from merely-slow apart) — which is why 30 days is used rather than an arbitrary round number.

**Operational tiers** (per user, based on `last_gap_days` — the gap immediately before their most recently observed order; see `sql/01_cohort_retention.sql` item 5 for why this is self-relative rather than measured against a shared "today"):
1. Active (normal cadence): `last_gap_days <= 15`
2. Gently slowing: `15 < last_gap_days < 30`
3. At/past churn threshold: `last_gap_days >= 30`

## Decision this project enables
Given a churn-risk score per user, decide:
1. Who enters a retention campaign
2. What it costs to reach them (discount, nudge, or nothing)
3. Who we deliberately leave alone, because intervening either won't work (likely-lost, low tenure) or isn't needed (still active, just naturally slower)

This is the frame that separates this project from a generic "predict churn" Kaggle notebook: the output isn't a probability column, it's a resourcing decision with an explicit cost trade-off (Phase 4) and a segmented action plan (Phase 5).

## Data constraint discovered in Phase 1: survivorship bias in the source data

Instacart's competition dataset deliberately includes only users with **between 4 and 100 historical orders** (verified: `MIN(lifetime_orders) = 4` across all 206,209 users). This means the dataset contains **zero** first-time or one-and-done users — everyone in it already made at least 4 purchases.

**Consequence:** this project cannot claim to predict whether a *new* user will make a second purchase (the headline framing above). What it actually measures is: among users who are *already* established, repeat customers, which of them show signs of drifting away (widening order gaps, declining recency) before going fully quiet. That is still a real, valuable retention question — it's what Phase 5's tiering (gently slowing / at risk / likely lost) is about — but it is narrower than "new user activation," and the write-up says so explicitly rather than overclaiming. If asked in an interview "so does this predict first-to-second-purchase churn?", the honest answer is no, and explaining *why* (the dataset's own sampling design) is itself the stronger answer.

## What this project is *not*
- Not a leaderboard exercise — accuracy/AUC alone will not be the headline number.
- Not proof that any intervention works — without a holdout/incrementality test (proposed, not run, given dataset constraints), this stops at "who to target," not "we proved retention lift."
- Not specific to one company's product — the same mechanics apply to food delivery, quick commerce, e-commerce, lending, and payments; only the label on the top bullet changes per audience.
