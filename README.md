# shakeup-radar

**How much will the private leaderboard reshuffle this competition?**
A calibrated answer from Kaggle competition history, instead of forum
folklore.

`shakeup = 1 − Spearman(public rank, private rank)` per competition,
computed with tie-averaged ranks over teams that finished on both boards.
Range is [0, 2]: 0 means the private board matched the public one, 1 means
uncorrelated, above 1 means anti-correlated. The tool reports the historical
shakeup distribution for competitions *like yours* (same metric family,
field size, public-LB fraction), fitted from the public
[Meta Kaggle](https://www.kaggle.com/datasets/kaggle/meta-kaggle) dataset.

## Quickstart

```bash
git clone <this repo> && cd shakeup-radar && pip install -e .
shakeup-radar predict --metric AUC --teams 3500 --lb-pct 20
```

Works immediately: a fitted artifact (Meta Kaggle snapshot 2026-07-12,
590 competitive competitions) ships with the package. To refit on fresh
data — recommended a couple of times a year, the regime moves:

```bash
# Competitions.csv + Teams.csv from kaggle.com/datasets/kaggle/meta-kaggle
shakeup-radar fit Competitions.csv Teams.csv   # prints the validation report
```

```json
{
 "expected_shakeup_p50": 0.031,
 "p75": 0.074, "p90": 0.19,
 "n_similar_comps": 87,
 "provenance": "stratum auc_rank|l|small",
 "risk_level": "MODERATE",
 "risk_level_basis": "stratum median vs global quantiles (p25=..., p50=..., p75=..., p90=...)"
}
```

Read it as: *half of historical competitions like this one shook up less
than 0.031; one in ten shook up more than 0.19.* `provenance` tells you
which calibration table answered (exact stratum → metric family → global
fallback); `n_similar_comps` says how much history backs it; `p90` is
suppressed rather than reported when fewer than 30 similar competitions
exist — a tail estimate from 15 samples would be noise in a lab coat.
`risk_level` is derived from the artifact's own quantiles, never from
hardcoded thresholds.

`--teams` means the expected **final leaderboard size**. Mid-competition
page counts differ from final ranked counts; the buckets are coarse, so
being off by ~30% rarely changes the answer — but know which number you're
giving it.

## What the data actually showed (2026-07-12 fit)

**The shakeup regime is non-stationary.** Competitions with a tiny public
leaderboard (<15% of test data) had median shakeup ~0.02–0.08 before 2019 —
and 0.4–1.5 every year since 2022. In the modern code-competition era, a
tiny public LB means the public and private boards are close to
uncorrelated (2023+ median: 0.96). This is why predictions are served from
a recent window (last 4 years of fitted data) and all-time numbers are
demoted to context: an all-history table under-predicts modern tiny-LB
shakeup by roughly 7×.

**Validation.** Temporal holdout, model vs predict-the-median baselines:
- Development estimate (split 2023-01-01, used during design iteration —
  treat as optimistic): +8.8% MAE vs recent-median baseline, Spearman 0.32,
  bootstrap 95% CI [1.6%, 14.2%].
- **Confirmatory (design frozen, splits untouched during development):**
  split 2024-01-01 → +6.2% MAE, Spearman 0.40 (n=99); split 2025-01-01 →
  +14.7% MAE, Spearman 0.46 (n=59).

The signal is modest for point prediction and strong for the thing the
tool actually answers: *which regime is your competition in, and what did
that regime's shakeup distribution look like.* Every artifact embeds its
own `holdout_report` with both baselines, so the evidence travels with the
calibration. **If a refit's improvement is near zero, believe it.**

## Scope and known limitations

- Population: Featured, Research, Recruitment, and Playground competitions
  with ≥30 ranked teams (override with `--hosts`). InClass and
  getting-started comps are excluded by default — they would dominate every
  stratum and answer a question nobody asks.
- Competitions with a 100% public leaderboard are excluded (public==private
  by construction).
- Meta Kaggle omits deleted competitions and deleted teams; that
  survivorship is inherited here and cannot be fixed from public data.
- Every exclusion is counted and printed in `population_stats` — nothing is
  dropped silently.
- 1−Spearman weights the whole leaderboard; a comp with a stable top-100
  and a noisy tail reads as higher shakeup than a medal-chaser might mean
  by the word. `top10_stay_rate` in the `targets` export is the
  medal-zone-specific view.
- No live scraping, no API keys, no telemetry. Inputs are numbers you read
  off the competition page.

## Design decisions (why a rewrite gets this wrong)

Benchmark entries excluded from ranks; teams missing either rank removed
(disqualifications, not shakeup); tie-averaged re-ranking of survivors
(raw ranks are gapped after filtering); temporal validation, never random
(era effects leak); strata under 15 comps fall back to coarser tables;
p90 suppressed under 30 comps; CSVs read as utf-8-sig (an Excel round-trip
BOM otherwise silently renames the first column and zeroes the dataset);
unknown metrics map to `other` loudly. The full numbered decision list is
in the source docstrings — it is the part of this tool that took the work.

## Provenance

Built as instrument T-01 / research card K-04 of the personal project:
"the public leaderboard reliably measures modeling skill" is a premise
under saturation-watch, and this is its measurement instrument. The golden
fixture plants three known noise regimes plus the adversarial cases found
in review (ties, missing ranks, BOM, population contamination); `pytest`
must pass before any release. v0.1 was adversarially reviewed before first
fit; the review and fixes are part of the repo history.
