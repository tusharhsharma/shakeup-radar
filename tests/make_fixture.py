"""Golden fixture: synthetic Meta-Kaggle-shaped data with KNOWN structure,
including the adversarial cases from the 2026-07-12 review (findings 2,6,10).

Regimes (planted, recoverable):
  auc_rank / xl teams / large LB  -> LOW    (sigma=2 rank noise)      x30
  prob_loss / m teams / small LB  -> MODERATE (sigma=n/20)            x30
  accuracy_like / s teams / tiny  -> SEVERE (private ranks shuffled)  x30
Adversarial extras:
  - retrieval family: 16 comps split 8/8 across team buckets -> exercises
    the FAMILY-fallback tier (each stratum < MIN_STRATUM, family >= 15)
  - one comp with heavy rank TIES (quantized scores)
  - one comp where 30% of teams lack a private rank (must still pass, ranks
    filtered — decision #6)
  - one comp with blank LeaderboardPercentage (bucket 'unknown')
  - one comp with unknown metric 'WeirdMetric2026' (family 'other')
  - one InClass comp (MUST be excluded by host filter)
  - one lb_pct=100 comp (MUST be excluded, public==private by construction)
  - one comp with 10 teams (MUST be excluded, < MIN_TEAMS)
  - every comp carries one IsBenchmark row (must be excluded from ranks)
Expected surviving comps: 90 + 16 + 20 (code regime) + 4 = 130.
"""
import csv, os, random

HERE = os.path.dirname(__file__)
FIX = os.path.join(HERE, "fixtures")
random.seed(20260712)

def noisy_ranks(n, sigma):
    privs = list(range(1, n + 1))
    if sigma >= 9999:
        random.shuffle(privs)
        return privs
    noisy = sorted((p + random.gauss(0, sigma), p) for p in privs)
    out = [0] * n
    for new_rank, (_, p) in enumerate(noisy, 1):
        out[p - 1] = new_rank
    return out

def main():
    os.makedirs(FIX, exist_ok=True)
    comps, teams = [], []
    state = {"cid": 100, "tid": 0}

    def add_comp(metric, n_teams, lb_pct, sigma, year, host="Featured",
                 drop_private_frac=0.0, tie_quantize=0, code=False):
        state["cid"] += 1
        cid = state["cid"]
        comps.append({
            "Id": str(cid), "Slug": f"comp-{cid}", "Title": f"Comp {cid}",
            "HostSegmentTitle": host,
            "EnabledDate": f"{year}-01-01 00:00:00",
            "DeadlineDate": f"{year}-06-01 00:00:00",
            "EvaluationAlgorithmAbbreviation": metric,
            "OnlyAllowKernelSubmissions": str(code),
            "LeaderboardPercentage": "" if lb_pct is None else str(lb_pct),
            "TotalSubmissions": str(n_teams * 12),
            "TotalTeams": str(n_teams),
        })
        privs = noisy_ranks(n_teams, sigma)
        pubs = list(range(1, n_teams + 1))
        if tie_quantize:
            pubs = [((p - 1) // tie_quantize) * tie_quantize + 1 for p in pubs]
            privs = [((q - 1) // tie_quantize) * tie_quantize + 1 for q in privs]
        for i in range(n_teams):
            state["tid"] += 1
            row = {"Id": str(state["tid"]), "CompetitionId": str(cid),
                   "PublicLeaderboardRank": str(pubs[i]),
                   "PrivateLeaderboardRank": str(privs[i]),
                   "IsBenchmark": "False"}
            if drop_private_frac and random.random() < drop_private_frac:
                row["PrivateLeaderboardRank"] = ""
            teams.append(row)
        state["tid"] += 1
        teams.append({"Id": str(state["tid"]), "CompetitionId": str(cid),
                      "PublicLeaderboardRank": "1",
                      "PrivateLeaderboardRank": str(n_teams),
                      "IsBenchmark": "True"})

    # planted regimes (deadlines 2019-2024: both sides of the 2023 split)
    for i in range(30):
        add_comp("AUC", 5000, 50.0, 2.0, 2019 + i % 6)
    for i in range(30):
        add_comp("LogLoss", 600, 25.0, 30.0, 2019 + i % 6)
    for i in range(30):
        add_comp("Accuracy", 120, 10.0, 9999, 2019 + i % 6)
    # family-fallback tier: strata of 8 < MIN_STRATUM, family 16 >= 15
    for i in range(8):
        add_comp("MAP@{K}", 200, 20.0, 10.0, 2019 + i % 4)
    for i in range(8):
        add_comp("MAP@{K}", 800, 20.0, 40.0, 2019 + i % 4)
    # v0.2 planted regime: CODE comps, tiny LB, near-total shuffle — the
    # modern hidden-test pattern (exercises the (lb,code) tier)
    for i in range(20):
        add_comp("AUC", 400, 10.0, 9999, 2019 + i % 6, code=True)
    # adversarial extras that must SURVIVE
    add_comp("AUC", 400, 30.0, 5.0, 2020, tie_quantize=5)          # ties
    add_comp("AUC", 400, 30.0, 5.0, 2021, drop_private_frac=0.3)   # missing ranks
    add_comp("RMSE", 300, None, 8.0, 2020)                          # blank lb_pct
    add_comp("WeirdMetric2026", 300, 20.0, 8.0, 2021)               # unknown metric
    # adversarial extras that must be EXCLUDED
    add_comp("AUC", 500, 20.0, 5.0, 2020, host="InClass")           # host filter
    add_comp("AUC", 500, 100.0, 0.0, 2020)                          # lb=100
    add_comp("AUC", 10, 20.0, 2.0, 2020)                            # < MIN_TEAMS

    with open(os.path.join(FIX, "Competitions.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(comps[0].keys()))
        w.writeheader(); w.writerows(comps)
    with open(os.path.join(FIX, "Teams.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(teams[0].keys()))
        w.writeheader(); w.writerows(teams)
    print(f"fixture: {len(comps)} comps, {len(teams)} team rows")

if __name__ == "__main__":
    main()
