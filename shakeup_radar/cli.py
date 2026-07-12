"""shakeup-radar CLI.

  shakeup-radar fit Competitions.csv Teams.csv [--out artifact.json]
      Fit the calibrated risk tables from Meta Kaggle. Validates on a
      temporal holdout (report printed — READ IT), then refits on all data.

  shakeup-radar predict --metric AUC --teams 3500 --lb-pct 20 [--artifact f]
      Deterministic risk estimate. --teams is the expected FINAL leaderboard
      size; mid-competition page counts include teams that never get a
      private rank, so if in doubt use the current count of SCORED teams
      (buckets are coarse, ~30% error rarely changes the answer).

  shakeup-radar targets Competitions.csv Teams.csv --out targets.csv
      Per-competition ground truth export (for your own analysis / paper).

Artifact resolution for predict: --artifact if given, else ./artifact.json,
else the packaged artifact if the install shipped one. Clear error otherwise.
"""
import argparse, csv, datetime, json, os, sys

from . import pipeline
from .metrics_map import family_of

PACKAGED = os.path.join(os.path.dirname(__file__), "artifact.json")

def _resolve_artifact(explicit):
    if explicit:  # review N2: an EXPLICIT path must never be silently swapped
        if os.path.exists(explicit):
            return explicit
        sys.exit(f"--artifact '{explicit}' does not exist "
                 "(refusing to fall back to a different calibration)")
    for cand in ("artifact.json", PACKAGED):
        if os.path.exists(cand):
            return cand
    sys.exit("no artifact found: pass --artifact, or run "
             "`shakeup-radar fit Competitions.csv Teams.csv` first "
             "(Meta Kaggle: kaggle.com/datasets/kaggle/meta-kaggle)")

def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    # adoption ergonomics: `shakeup-radar --metric AUC --teams 4000` works —
    # predict is the default subcommand (the phrase people type IS the tool)
    if argv and argv[0].startswith("--"):
        argv = ["predict"] + argv
    ap = argparse.ArgumentParser(prog="shakeup-radar")
    sub = ap.add_subparsers(dest="cmd", required=True)

    f = sub.add_parser("fit")
    f.add_argument("competitions"); f.add_argument("teams")
    f.add_argument("--out", default="artifact.json")
    f.add_argument("--split-date", default=pipeline.SPLIT_DATE)
    f.add_argument("--hosts", default=",".join(sorted(pipeline.COMPETITIVE_HOSTS)),
                   help="comma-separated HostSegmentTitle values to include; "
                        "'all' disables the population filter")

    p = sub.add_parser("predict")
    p.add_argument("--metric", required=True)
    p.add_argument("--teams", type=int, required=True,
                   help="expected FINAL number of ranked teams")
    p.add_argument("--lb-pct", type=float, default=None)
    p.add_argument("--artifact", default=None)

    t = sub.add_parser("targets")
    t.add_argument("competitions"); t.add_argument("teams")
    t.add_argument("--out", required=True)
    t.add_argument("--hosts", default=",".join(sorted(pipeline.COMPETITIVE_HOSTS)))

    a = ap.parse_args(argv)

    if a.cmd == "fit":  # review N3+N4: validate + NORMALIZE before touching CSVs
        try:
            a.split_date = datetime.date.fromisoformat(a.split_date).isoformat()
        except ValueError:
            sys.exit(f"--split-date must be ISO YYYY-MM-DD, got '{a.split_date}'")

    if a.cmd in ("fit", "targets"):
        hosts = None if a.hosts.strip().lower() == "all" else \
            {h.strip() for h in a.hosts.split(",") if h.strip()}
        comps, stats = pipeline.load_competitions(a.competitions, hosts)
        rows = pipeline.compute_targets(a.teams, comps, stats)
        if not rows:
            sys.exit(f"no competitions passed the filters — population stats: "
                     f"{json.dumps(stats)}")

    if a.cmd == "fit":
        artifact, train, test = pipeline.fit_artifact(rows, a.split_date, stats)
        if artifact is None:
            sys.exit("could not build tables (no data after filters)")
        unknown = sorted({r["metric"] for r in rows if r["family"] == "other"})
        with open(a.out, "w", encoding="utf-8") as fh:
            json.dump(artifact, fh, indent=1, sort_keys=True)
        print(json.dumps({"artifact": a.out,
                          "n_competitions_used": len(rows),
                          "population_stats": stats,
                          "holdout_report": artifact["holdout_report"],
                          "unmapped_metrics_sample": unknown[:40]}, indent=1))
        return

    if a.cmd == "predict":
        path = _resolve_artifact(a.artifact)
        with open(path, encoding="utf-8") as fh:
            artifact = json.load(fh)
        fam = family_of(a.metric)
        out = pipeline.predict(artifact, fam, a.teams, a.lb_pct)
        if out is None:
            sys.exit("artifact has no usable table — refit")
        out["metric_family"] = fam
        out["artifact_used"] = path
        print(json.dumps(out, indent=1))
        return

    if a.cmd == "targets":
        cols = ["competition_id", "slug", "deadline", "family", "metric",
                "n_ranked_teams", "lb_pct", "shakeup", "top10_stay_rate",
                "median_abs_move_frac", "host", "duration_days", "total_subs"]
        with open(a.out, "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore")
            w.writeheader(); w.writerows(rows)
        print(f"{len(rows)} competitions -> {a.out}")

if __name__ == "__main__":
    main()
