"""Golden tests incl. the adversarial cases from the 2026-07-12 review."""
import json, os, subprocess, sys

HERE = os.path.dirname(__file__)
FIX = os.path.join(HERE, "fixtures")
PKG = os.path.dirname(HERE)
sys.path.insert(0, PKG)

from shakeup_radar import pipeline  # noqa: E402

EXPECTED_SURVIVORS = 130  # 90 regimes + 16 retrieval + 20 code regime + 4 extras


def _load():
    comps, stats = pipeline.load_competitions(os.path.join(FIX, "Competitions.csv"))
    rows = pipeline.compute_targets(os.path.join(FIX, "Teams.csv"), comps)
    return rows, stats


def test_population_filters_and_counts():
    rows, stats = _load()
    assert len(rows) == EXPECTED_SURVIVORS, (len(rows), stats)
    assert stats.get("excluded_host:InClass") == 1
    assert stats.get("excluded_lb100") == 1
    slugs = {r["slug"] for r in rows}
    assert not any(r["host"] == "InClass" for r in rows)


def test_regimes_recovered_with_ties_and_gaps():
    rows, _ = _load()
    med = {}
    for fam in ("auc_rank", "prob_loss", "accuracy_like"):
        vals = sorted(r["shakeup"] for r in rows if r["family"] == fam)
        med[fam] = vals[len(vals) // 2]
    assert med["auc_rank"] < med["prob_loss"] < med["accuracy_like"], med
    assert med["accuracy_like"] > 0.8
    # tied-ranks comp and missing-rank comp must both survive with sane values
    tie_rows = [r for r in rows if r["family"] == "auc_rank"
                and r["n_ranked_teams"] not in (5000,)]
    assert tie_rows, "adversarial auc comps missing"
    assert all(0 <= r["shakeup"] <= 2 for r in rows)


def test_missing_rank_teams_filtered_not_fatal():
    rows, _ = _load()
    dropped = [r for r in rows if 250 < r["n_ranked_teams"] < 400
               and r["family"] == "auc_rank"]
    assert dropped, "the 30%-missing-private-rank comp should survive with ~280 teams"


def test_benchmark_rows_excluded():
    rows, _ = _load()
    xl = [r for r in rows if r["family"] == "auc_rank" and r["n_ranked_teams"] == 5000]
    assert len(xl) == 30
    assert all(r["shakeup"] < 0.01 for r in xl), "benchmark row leaked"


def test_artifact_fallback_chain_and_determinism():
    rows, stats = _load()
    art1, train, test = pipeline.fit_artifact(rows, pop_stats=stats)
    art2, _, _ = pipeline.fit_artifact(rows, pop_stats=stats)
    assert art1 == art2
    # exact recent-stratum hit (fixture max deadline 2024 -> recent = 2020+)
    p = pipeline.predict(art1, "auc_rank", 5000, 50.0, code=False)
    assert p["provenance"].startswith("recent stratum"), p["provenance"]
    assert p["expected_shakeup_p50"] < 0.01
    assert "alltime_context_p50" in p
    # v0.2: the code dimension separates the planted regimes sharing a bucket
    pc = pipeline.predict(art1, "auc_rank", 400, 10.0, code=True)
    assert pc["expected_shakeup_p50"] > 0.8, pc
    pn = pipeline.predict(art1, "accuracy_like", 120, 10.0, code=False)
    assert "hint" not in pc
    # code unknown -> falls to lb-bucket tier with a hint
    pu = pipeline.predict(art1, "auc_rank", 400, 10.0)
    assert "hint" in pu and "lb-bucket" in pu["provenance"], pu
    # LB-BUCKET fallback (retrieval strata are 8+8 in each era window)
    q = pipeline.predict(art1, "retrieval", 200, 20.0)
    assert q["provenance"] == "recent lb-bucket small", q["provenance"]
    # unseen family + unknown lb -> lb-bucket 'unknown' exists? no -> global
    g = pipeline.predict(art1, "text_sim", 50, None)
    assert "global" in g["provenance"] or "lb-bucket" in g["provenance"]
    # risk level comes from artifact quantiles, not folk constants
    assert "global quantiles" in p["risk_level_basis"]
    assert p["risk_level"] == "LOW"


def test_p90_suppressed_for_small_strata():
    rows, _ = _load()
    art, _, _ = pipeline.fit_artifact(rows)
    # tiny lb bucket in the alltime tier: 30 accuracy comps + extras -> p90
    # present; retrieval strata (8 each) are below even MIN_RECENT -> absent
    assert not any(k.startswith("retrieval|") for k in art["alltime"]["strata"]), \
        "sub-threshold strata must not be served"
    small = [v for k, v in art["recent"]["strata"].items() if v["n"] < 30]
    assert small and all(v["p90"] is None for v in small), \
        "p90 must be suppressed below 30 samples (review finding 4)"


def test_artifact_refit_on_all_after_validation():
    rows, _ = _load()
    art, train, test = pipeline.fit_artifact(rows)
    assert art["n_total_comps_in_tables"] == len(rows) > len(train)
    assert art["holdout_report"]["n_test"] == len(test)
    assert art["holdout_report"]["mae_model"] <= \
        art["holdout_report"]["mae_global_median_baseline"]


def test_bom_tolerated():
    src = os.path.join(FIX, "Competitions.csv")
    bom = os.path.join(FIX, "Competitions_bom.csv")
    with open(src, "rb") as f:
        data = f.read()
    with open(bom, "wb") as f:
        f.write(b"\xef\xbb\xbf" + data)
    try:
        comps, stats = pipeline.load_competitions(bom)
        assert stats["kept"] > 0, "BOM zeroed the dataset (review finding 6)"
    finally:
        os.remove(bom)


def test_cli_end_to_end_and_graceful_no_artifact(tmp_path):
    art = tmp_path / "artifact.json"
    r = subprocess.run(
        [sys.executable, "-m", "shakeup_radar.cli", "fit",
         os.path.join(FIX, "Competitions.csv"), os.path.join(FIX, "Teams.csv"),
         "--out", str(art)], capture_output=True, text=True, cwd=PKG)
    assert r.returncode == 0, r.stderr
    out = json.loads(r.stdout)
    assert out["n_competitions_used"] == EXPECTED_SURVIVORS
    r2 = subprocess.run(
        [sys.executable, "-m", "shakeup_radar.cli", "predict",
         "--metric", "AUC", "--teams", "5000", "--lb-pct", "50",
         "--artifact", str(art)], capture_output=True, text=True, cwd=PKG)
    assert r2.returncode == 0, r2.stderr
    assert json.loads(r2.stdout)["risk_level"] == "LOW"
    # from an empty cwd the BUNDLED artifact answers (new in v0.1 final)
    empty = tmp_path / "empty"
    empty.mkdir()
    packaged = os.path.join(PKG, "shakeup_radar", "artifact.json")
    if os.path.exists(packaged):
        r3 = subprocess.run(
            [sys.executable, "-m", "shakeup_radar.cli", "predict",
             "--metric", "AUC", "--teams", "100"],
            capture_output=True, text=True, cwd=str(empty),
            env={**os.environ, "PYTHONPATH": PKG})
        assert r3.returncode == 0, r3.stderr
        assert json.loads(r3.stdout)["artifact_used"].endswith(
            os.path.join("shakeup_radar", "artifact.json"))
    # graceful error when no artifact ANYWHERE (bundled one hidden)
    hidden = packaged + ".hidden"
    os.rename(packaged, hidden)
    try:
        r4 = subprocess.run(
            [sys.executable, "-m", "shakeup_radar.cli", "predict",
             "--metric", "AUC", "--teams", "100"],
            capture_output=True, text=True, cwd=str(empty),
            env={**os.environ, "PYTHONPATH": PKG})
        assert r4.returncode != 0
        assert "run `shakeup-radar fit" in r4.stderr, r4.stderr
    finally:
        os.rename(hidden, packaged)


def test_bad_split_date_rejected(tmp_path):
    r = subprocess.run(
        [sys.executable, "-m", "shakeup_radar.cli", "fit",
         os.path.join(FIX, "Competitions.csv"), os.path.join(FIX, "Teams.csv"),
         "--out", str(tmp_path / "a.json"), "--split-date", "01/06/2023"],
        capture_output=True, text=True, cwd=PKG)
    assert r.returncode != 0 and "ISO" in r.stderr


def test_malformed_rank_cells_do_not_desync(tmp_path):
    """Review N1: one dirty cell must not desync arrays or kill the fit."""
    import csv as _csv
    comps_src = os.path.join(FIX, "Competitions.csv")
    teams_src = os.path.join(FIX, "Teams.csv")
    dirty = tmp_path / "Teams_dirty.csv"
    with open(teams_src, newline="") as f:
        rows = list(_csv.DictReader(f))
    # corrupt three rows in the first surviving comp: N/A, inf, nan
    victims = [r for r in rows if r["CompetitionId"] == "101"][:3]
    victims[0]["PrivateLeaderboardRank"] = "N/A"
    victims[1]["PrivateLeaderboardRank"] = "inf"
    victims[2]["PublicLeaderboardRank"] = "nan"
    with open(dirty, "w", newline="") as f:
        w = _csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    comps, stats = pipeline.load_competitions(comps_src)
    out = pipeline.compute_targets(str(dirty), comps, stats)
    assert stats.get("malformed_rank_cells", 0) >= 2, stats
    assert len(out) == EXPECTED_SURVIVORS  # comp 101 survives, 3 teams lighter
    c101 = next(r for r in out if r["competition_id"] == "101")
    assert c101["n_ranked_teams"] == 4997
    assert c101["shakeup"] < 0.01, "desync would destroy the correlation"


def test_explicit_missing_artifact_refused(tmp_path):
    """Review N2: explicit --artifact must never silently fall back."""
    # create a valid ./artifact.json the buggy version would fall back to
    rows, _ = _load()
    art, _, _ = pipeline.fit_artifact(rows)
    with open(tmp_path / "artifact.json", "w") as f:
        json.dump(art, f)
    r = subprocess.run(
        [sys.executable, "-m", "shakeup_radar.cli", "predict",
         "--metric", "AUC", "--teams", "100", "--artifact", "/nonexistent/x.json"],
        capture_output=True, text=True, cwd=str(tmp_path),
        env={**os.environ, "PYTHONPATH": PKG})
    assert r.returncode != 0
    assert "refusing to fall back" in r.stderr, r.stderr


def test_compact_iso_split_date_normalized(tmp_path):
    """Review N3: '20230101' must behave exactly like '2023-01-01'."""
    a1, a2 = tmp_path / "a1.json", tmp_path / "a2.json"
    for out, sd in ((a1, "2023-01-01"), (a2, "20230101")):
        r = subprocess.run(
            [sys.executable, "-m", "shakeup_radar.cli", "fit",
             os.path.join(FIX, "Competitions.csv"), os.path.join(FIX, "Teams.csv"),
             "--out", str(out), "--split-date", sd],
            capture_output=True, text=True, cwd=PKG)
        assert r.returncode == 0, r.stderr
    j1, j2 = json.load(open(a1)), json.load(open(a2))
    assert j1["n_train_comps"] == j2["n_train_comps"] > 0
    assert j1["recent"] == j2["recent"] and j1["alltime"] == j2["alltime"]


def test_sparse_recent_window_falls_to_alltime():
    """Review r3: a near-empty recent window must not answer with n=3
    authority — the all-time tier takes over below MIN_RECENT."""
    rows, _ = _load()
    old = [dict(r) for r in rows if r["deadline"] < "2021-01-01"]
    # 3 lonely modern comps anchor the window; recent global n=3 < MIN_RECENT
    lonely = [dict(r) for r in rows if r["deadline"] >= "2024-01-01"][:3]
    for r in lonely:
        r["deadline"] = "2026-06-01"
    art, _, _ = pipeline.fit_artifact(old + lonely, split_date="2027-01-01")
    assert art["recent"]["global"]["n"] == 3
    p = pipeline.predict(art, "auc_rank", 5000, 50.0)
    assert p["provenance"].startswith("all-time"), p["provenance"]


def test_future_deadlines_excluded(tmp_path):
    """Review r3: a placeholder far-future deadline must not hijack the
    recency anchor — it is excluded (and counted) at load time."""
    import csv as _csv
    src = os.path.join(FIX, "Competitions.csv")
    rows = list(_csv.DictReader(open(src, newline="")))
    rows[0] = dict(rows[0])
    rows[0]["DeadlineDate"] = "2044-12-31 00:00:00"
    dst = tmp_path / "Competitions_future.csv"
    with open(dst, "w", newline="") as f:
        w = _csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    comps, stats = pipeline.load_competitions(str(dst))
    assert stats.get("excluded_future_deadline") == 1, stats
    assert all(c["deadline"] < "2044-01-01" for c in comps.values())
