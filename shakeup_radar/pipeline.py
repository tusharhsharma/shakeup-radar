"""shakeup-radar pipeline: Meta Kaggle CSVs -> shakeup targets -> features ->
calibrated risk artifact (JSON) + honest validation report.

Design rule inherited from EdgeWatch: the shipped artifact is a transparent
empirical lookup (auditable arithmetic), never a pickled model.

Decision cache (things a rewrite gets wrong; #s continue across modules):
  #5  IsBenchmark teams excluded (they hold ranks but are not competitors).
  #6  Teams need BOTH ranks present; a missing rank usually means a removed
      or disqualified team, not shakeup.
  #7  Competitions need >= MIN_TEAMS ranked teams.
  #8  Primary target = 1 - Spearman(public_rank, private_rank), computed
      with proper tie-averaged re-ranking of the SURVIVING teams (raw ranks
      are gapped once #5/#6 filters run). Range is [0,2]; >1 = anti-
      correlated boards. Top-k stay-rate is reported, not fitted.
  #9  Temporal validation only (train < split <= test); random splits leak
      era effects. The SHIPPED artifact is then refit on ALL rows — serving
      a 2022-frozen table in 2026 would discard the most relevant era.
  #10 Strata need >= MIN_STRATUM comps; p90 additionally needs >= P90_MIN
      (a p90 from 15 samples is a sample near-maximum wearing a lab coat).
  #16 Population filter: only competitive host segments (Featured/Research/
      Recruitment/Playground by default). InClass + Getting Started comps
      outnumber Featured comps and would dominate every stratum.
  #17 lb_pct >= 100 excluded: public==private by construction, shakeup=0
      rows that aren't competitions in the relevant sense.
  #18 CSVs read as utf-8-sig: an Excel round-trip BOM otherwise renames the
      first column and silently zeroes the dataset.
  #19 Dropped rows are COUNTED and reported (bad dates, filtered hosts),
      never silent. Survivorship in Meta Kaggle itself (deleted comps/teams
      absent at export) is a known, unfixable limitation — stated in README.
"""
import csv, json, math, datetime
from array import array
from collections import defaultdict

from .metrics_map import family_of

MIN_TEAMS = 30            # decision #7
MIN_STRATUM = 15          # decision #10
P90_MIN = 30              # decision #10
SPLIT_DATE = "2023-01-01"
TOPK = 10
COMPETITIVE_HOSTS = {"Featured", "Research", "Recruitment", "Playground"}  # #16

TEAM_BUCKETS = [(0, 100, "xs"), (100, 500, "s"), (500, 1500, "m"),
                (1500, 4000, "l"), (4000, 10**9, "xl")]

def team_bucket(n):
    for lo, hi, name in TEAM_BUCKETS:
        if lo <= n < hi:
            return name
    return "xl"

def lb_bucket(pct):
    if pct is None:
        return "unknown"
    if pct < 15:
        return "tiny"
    if pct < 35:
        return "small"
    return "large"

def rank_avg(vals):
    """Tie-averaged ranks (proper Spearman prep) — decisions #8, #13."""
    order = sorted(range(len(vals)), key=lambda i: vals[i])
    ranks = [0.0] * len(vals)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and vals[order[j + 1]] == vals[order[i]]:
            j += 1
        avg = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            ranks[order[k]] = avg
        i = j + 1
    return ranks

def spearman(xs, ys):
    xs, ys = rank_avg(xs), rank_avg(ys)
    n = len(xs)
    if n < 3:
        return None
    mx, my = sum(xs) / n, sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    syy = sum((y - my) ** 2 for y in ys)
    if sxx == 0 or syy == 0:
        return None
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    return sxy / math.sqrt(sxx * syy)

def quantile(sorted_vals, q):
    """Linear-interpolated quantile (numpy 'linear' method) — decision #10's
    fix for int-index bias (int(0.5n) was the UPPER median for even n)."""
    n = len(sorted_vals)
    if n == 0:
        return None
    if n == 1:
        return sorted_vals[0]
    pos = q * (n - 1)
    lo = int(pos)
    hi = min(lo + 1, n - 1)
    frac = pos - lo
    return round(sorted_vals[lo] * (1 - frac) + sorted_vals[hi] * frac, 6)

def parse_date(s):
    if not s:
        return None
    s = s.strip().split(".")[0]
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%d",
                "%m/%d/%Y %H:%M:%S", "%m/%d/%Y %I:%M:%S %p", "%m/%d/%Y"):
        try:
            return datetime.datetime.strptime(s, fmt)
        except ValueError:
            continue
    return None

def _int(v):
    try:
        return int(float(v))
    except (TypeError, ValueError):
        return None

def load_competitions(path, hosts=COMPETITIVE_HOSTS):
    """Returns (comps, stats). stats counts every exclusion — decision #19."""
    comps, stats = {}, defaultdict(int)
    with open(path, newline="", encoding="utf-8-sig") as f:   # decision #18
        for r in csv.DictReader(f):
            cid = r.get("Id")
            if not cid:
                stats["no_id"] += 1
                continue
            stats["total_rows"] += 1
            host = (r.get("HostSegmentTitle") or "").strip()
            if hosts is not None and host not in hosts:
                stats[f"excluded_host:{host or 'blank'}"] += 1
                continue
            try:
                lb_pct = float(r.get("LeaderboardPercentage") or "nan")
                if math.isnan(lb_pct):
                    lb_pct = None
            except ValueError:
                lb_pct = None
            if lb_pct is not None and lb_pct >= 100:          # decision #17
                stats["excluded_lb100"] += 1
                continue
            deadline = parse_date(r.get("DeadlineDate", ""))
            enabled = parse_date(r.get("EnabledDate", ""))
            if deadline is None:
                stats["excluded_bad_deadline"] += 1           # decision #19
                continue
            # decision #26 (review r3): an unfinished competition cannot have
            # a final private board; rows with future deadlines are rolling/
            # placeholder comps and would hijack the recency-window anchor.
            if deadline.date() > datetime.date.today():
                stats["excluded_future_deadline"] += 1
                continue
            # decision #23: recent Meta Kaggle rows store a numeric ID in the
            # abbreviation column; the readable metric lives in ...Name.
            metric = (r.get("EvaluationAlgorithmAbbreviation", "") or "").strip()
            if not metric or metric.isdigit():
                metric = (r.get("EvaluationAlgorithmName", "") or "").strip()
            comps[cid] = {
                "slug": r.get("Slug", ""),
                "host": host,
                "metric": metric,
                "family": family_of(metric),
                "lb_pct": lb_pct,
                "deadline": deadline.date().isoformat(),
                "duration_days": (deadline - enabled).days if enabled else None,
                "total_subs": _int(r.get("TotalSubmissions")),
            }
            stats["kept"] += 1
    return comps, dict(stats)

def compute_targets(teams_path, comps, stats=None):
    """Bounded memory: two int-arrays per competition (8 bytes/team), not
    Python tuples — Teams.csv has millions of rows (review finding #9).
    Decision #20 (review N1): BOTH ranks are parsed into locals inside one
    try before EITHER array is appended — a malformed second value must not
    desync the arrays (catches OverflowError too: rank 'inf' exists in the
    wild). Decision #21 (review N5): sub-MIN_TEAMS and degenerate-rho drops
    are counted, never silent."""
    stats = stats if stats is not None else {}
    pubs = defaultdict(lambda: array("l"))
    prvs = defaultdict(lambda: array("l"))
    with open(teams_path, newline="", encoding="utf-8-sig") as f:  # #18
        for r in csv.DictReader(f):
            cid = r.get("CompetitionId")
            if cid not in comps:
                continue
            if (r.get("IsBenchmark") or "").strip().lower() in ("true", "1"):
                continue                                       # decision #5
            pub, prv = r.get("PublicLeaderboardRank"), r.get("PrivateLeaderboardRank")
            if not pub or not prv:
                continue                                       # decision #6
            try:                                               # decision #20
                p_i, q_i = int(float(pub)), int(float(prv))
            except (ValueError, OverflowError):
                stats["malformed_rank_cells"] = stats.get("malformed_rank_cells", 0) + 1
                continue
            pubs[cid].append(p_i)
            prvs[cid].append(q_i)
    rows = []
    for cid in pubs:
        n = len(pubs[cid])
        if n < MIN_TEAMS:                                      # decision #7
            stats["excluded_small_comp"] = stats.get("excluded_small_comp", 0) + 1
            continue
        rho = spearman(list(pubs[cid]), list(prvs[cid]))
        if rho is None:
            stats["excluded_degenerate_rho"] = stats.get("excluded_degenerate_rho", 0) + 1
            continue
        top_idx = [i for i in range(n) if pubs[cid][i] <= TOPK]
        stayed = sum(1 for i in top_idx if prvs[cid][i] <= TOPK)
        moves = sorted(abs(pubs[cid][i] - prvs[cid][i]) for i in range(n))
        c = dict(comps[cid])
        c.update({
            "competition_id": cid,
            "n_ranked_teams": n,
            "shakeup": round(1.0 - rho, 6),                    # decision #8
            "top10_stay_rate": round(stayed / len(top_idx), 4) if top_idx else None,
            "median_abs_move_frac": round(moves[n // 2] / n, 6),
        })
        rows.append(c)
    return rows

def stratum_key(row):
    return (row["family"], team_bucket(row["n_ranked_teams"]), lb_bucket(row["lb_pct"]))

# decision #24 (real-data finding, 2026-07-12): the shakeup REGIME is
# non-stationary. Tiny-LB comps: median shakeup ~0.02-0.08 pre-2019 but
# 0.4-1.5 every year since 2022 (code-competition era). An all-history
# table under-predicts the modern regime by an order of magnitude, so
# predictions are served from a RECENT window (last RECENT_YEARS of the
# fitted data, self-adapting as data refreshes); all-time numbers are
# reported as context, never as the answer.
# decision #25 (same diagnostic): lb_bucket was the ONLY single feature
# with out-of-era holdout value (MAE 0.1572 vs baseline 0.1621; family and
# team-count ~nothing). The fallback chain therefore goes
# recent stratum -> recent LB-BUCKET table -> recent global.
RECENT_YEARS = 4
MIN_RECENT = 10

def _tables(rows, min_n=MIN_STRATUM):
    def table(keyfn, data):
        groups = defaultdict(list)
        for r in data:
            groups[keyfn(r)].append(r["shakeup"])
        out = {}
        for k, vals in groups.items():
            if len(vals) >= min_n:
                vals.sort()
                out[k] = {"n": len(vals),
                          "p25": quantile(vals, 0.25),
                          "p50": quantile(vals, 0.50),
                          "p75": quantile(vals, 0.75),
                          "p90": quantile(vals, 0.90) if len(vals) >= P90_MIN
                                 else None}                    # decision #10
        return out
    full = table(stratum_key, rows)
    lb_only = table(lambda r: (lb_bucket(r["lb_pct"]),), rows)
    all_vals = sorted(r["shakeup"] for r in rows)
    glob = None
    if all_vals:
        glob = {"n": len(all_vals),
                "p25": quantile(all_vals, 0.25), "p50": quantile(all_vals, 0.50),
                "p75": quantile(all_vals, 0.75),
                "p90": quantile(all_vals, 0.90) if len(all_vals) >= P90_MIN else None}
    return full, lb_only, glob

def _recent_start(rows):
    if not rows:
        return None
    latest = max(r["deadline"] for r in rows)
    return f"{int(latest[:4]) - RECENT_YEARS}{latest[4:10][:6]}"[:10]

def _tiers(rows):
    """(recent, alltime) table triples. Recent = last RECENT_YEARS of the
    data being fitted (decision #24), with the lower MIN_RECENT floor."""
    start = _recent_start(rows)
    recent_rows = [r for r in rows if r["deadline"] >= start]
    recent = _tables(recent_rows, min_n=MIN_RECENT)
    alltime = _tables(rows, min_n=MIN_STRATUM)
    return start, recent, alltime

def fit_artifact(rows, split_date=SPLIT_DATE, pop_stats=None):
    """Validate on temporal holdout (train tables predict post-split comps),
    then refit on ALL rows for the shipped tables (decision #9). The
    artifact carries its validation report and population stats."""
    train = [r for r in rows if r["deadline"] < split_date]
    test = [r for r in rows if r["deadline"] >= split_date]
    holdout = None
    if train and test:
        _, tr_recent, tr_alltime = _tiers(train)
        holdout = _evaluate(tr_recent, tr_alltime, test)
    if not rows:
        return None, train, test
    start, recent, alltime = _tiers(rows)
    if alltime[2] is None:
        return None, train, test
    def pack(triple):
        full, lb_only, glob = triple
        return {"strata": {"|".join(k): v for k, v in full.items()},
                "lb_bucket": {k[0]: v for k, v in lb_only.items()},
                "global": glob}
    artifact = {
        "version": "0.1.0",
        "target": "shakeup = 1 - spearman(public_rank, private_rank), range [0,2]",
        "validation_split_date": split_date,
        "n_train_comps": len(train), "n_test_comps": len(test),
        "n_total_comps_in_tables": len(rows),
        "recent_window_start": start,
        "holdout_report": holdout,
        "population_stats": pop_stats or {},
        "recent": pack(recent),
        "alltime": pack(alltime),
        "note": "predictions served from the recent tier (shakeup regime is "
                "non-stationary; see recent_window_start); alltime is context. "
                "holdout_report is the honest generalization estimate.",
    }
    return artifact, train, test

def _chain(tier, family, n_teams, lb_pct, label):
    key = "|".join((family, team_bucket(n_teams), lb_bucket(lb_pct)))
    if key in tier["strata"]:
        return tier["strata"][key], f"{label} stratum {key}"
    lb = lb_bucket(lb_pct)
    if lb in tier["lb_bucket"]:
        return tier["lb_bucket"][lb], f"{label} lb-bucket {lb}"
    if tier["global"]:
        return tier["global"], f"{label} global"
    return None, None

def _recent_trusted(artifact):
    """decision #27 (review r3): the recent tier is served only when its
    global holds >= MIN_RECENT comps — otherwise a near-empty window (sparse
    modern data) would answer with n=3 authority and the documented all-time
    fallback would be dead code."""
    g = artifact["recent"]["global"]
    return bool(g) and g["n"] >= MIN_RECENT

def predict(artifact, family, n_teams, lb_pct):
    q = prov = None
    if _recent_trusted(artifact):
        q, prov = _chain(artifact["recent"], family, n_teams, lb_pct, "recent")
    if q is None:
        q, prov = _chain(artifact["alltime"], family, n_teams, lb_pct, "all-time")
    if q is None:
        return None
    out = {"expected_shakeup_p50": q["p50"], "p75": q["p75"], "p90": q["p90"],
           "n_similar_comps": q["n"], "provenance": prov}
    if q["p90"] is None:
        out["p90_note"] = f"suppressed: needs >= {P90_MIN} similar comps"
    # all-time context (never the answer — decision #24)
    aq, aprov = _chain(artifact["alltime"], family, n_teams, lb_pct, "all-time")
    if aq:
        out["alltime_context_p50"] = aq["p50"]
    g = artifact["recent"]["global"] or artifact["alltime"]["global"]
    p50 = q["p50"]
    # note (review N6): when the global p90 is suppressed (tiny datasets),
    # SEVERE is unreachable by design — disclosed via risk_level_basis.
    out["risk_level"] = ("LOW" if p50 <= g["p25"] else
                         "MODERATE" if p50 <= g["p50"] else
                         "ELEVATED" if p50 <= g["p75"] else
                         "HIGH" if (g["p90"] is None or p50 <= g["p90"])
                         else "SEVERE")
    out["risk_level_basis"] = ("tier median vs recent global quantiles "
                               f"(p25={g['p25']}, p50={g['p50']}, p75={g['p75']}, p90={g['p90']})")
    return out

def _evaluate(recent, alltime, test_rows):
    glob = recent[2] or alltime[2]
    if not test_rows or glob is None or alltime[2] is None:
        return {"error": "insufficient data for holdout"}
    r_pack = {"strata": {"|".join(k): v for k, v in recent[0].items()},
              "lb_bucket": {k[0]: v for k, v in recent[1].items()},
              "global": recent[2]}
    a_pack = {"strata": {"|".join(k): v for k, v in alltime[0].items()},
              "lb_bucket": {k[0]: v for k, v in alltime[1].items()},
              "global": alltime[2]}
    art_like = {"recent": r_pack, "alltime": a_pack}
    pred, actual = [], []
    for r in test_rows:
        q = None
        if _recent_trusted(art_like):
            q, _ = _chain(r_pack, r["family"], r["n_ranked_teams"], r["lb_pct"], "recent")
        if q is None:
            q, _ = _chain(a_pack, r["family"], r["n_ranked_teams"], r["lb_pct"], "all-time")
        pred.append(q["p50"])
        actual.append(r["shakeup"])
    mae_model = sum(abs(p - a) for p, a in zip(pred, actual)) / len(actual)
    # decision #28 (review r3): BOTH baselines reported, so the recency
    # contribution and the stratification contribution are separable.
    out = {"n_test": len(actual), "mae_model": round(mae_model, 4)}
    for label, tier in (("recent", recent[2]), ("alltime", alltime[2])):
        if tier:
            gm = tier["p50"]
            mb = sum(abs(gm - a) for a in actual) / len(actual)
            out[f"mae_{label}_median_baseline"] = round(mb, 4)
            out[f"mae_improvement_vs_{label}_pct"] = \
                round(100 * (mb - mae_model) / mb, 1) if mb else None
    rho = spearman(pred, actual)   # tie-averaged — review finding #13
    out["spearman_pred_vs_actual"] = round(rho, 4) if rho is not None else None
    # back-compat aliases (older callers/tests read these names)
    out["mae_global_median_baseline"] = out.get("mae_recent_median_baseline",
                                                out.get("mae_alltime_median_baseline"))
    out["mae_improvement_pct"] = out.get("mae_improvement_vs_recent_pct",
                                         out.get("mae_improvement_vs_alltime_pct"))
    return out
