#!/usr/bin/env python3
import os
import json
import argparse
import glob
from typing import Any, Dict, List, Optional

import pandas as pd


def _safe_get(d: Dict[str, Any], path: List[str], default=None):
    cur: Any = d
    for k in path:
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur


def _read_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _infer_run_name(report_path: str, reports_root: Optional[str]) -> str:
    # Keep merge key stable as the concrete run directory name.
    # This avoids embedding experiment folder prefixes (e.g., exp_v3_en/run)
    # which would break downstream joins keyed on run_name/gen_run.
    if reports_root:
        try:
            rel = os.path.relpath(os.path.dirname(report_path), reports_root)
            return os.path.basename(rel.replace("\\", "/"))
        except Exception:
            pass
    return os.path.basename(os.path.dirname(report_path))


def _flatten_report(report_path: str, reports_root: Optional[str]) -> Dict[str, Any]:
    j = _read_json(report_path)

    run = _infer_run_name(report_path, reports_root)
    gen_csv = _safe_get(j, ["gen_csv"], "")

    integ = _safe_get(j, ["integrity"], {}) or {}
    dist_real = _safe_get(j, ["distribution", "real_test"], {}) or {}
    dist_gen = _safe_get(j, ["distribution", "gen"], {}) or {}
    dups = _safe_get(j, ["duplicates"], {}) or {}
    leak = _safe_get(j, ["leakage"], {}) or {}

    status_counts = integ.get("status_counts", {}) or {}
    rows = float(integ.get("rows", 0) or 0)

    ok = float(status_counts.get("OK", 0) or 0)
    ok_salv = float(status_counts.get("OK_SALVAGED", 0) or 0)
    usable = ok + ok_salv
    usable_rate = (usable / rows) if rows > 0 else 0.0

    return {
        "run": run,
        "report_json": report_path,
        "gen_csv": gen_csv,

        # integrity
        "rows": int(rows),
        "usable_rows": int(usable),
        "usable_rate": usable_rate,
        "empty_rate": integ.get("empty_rate", None),
        "blank_rate": integ.get("blank_rate", None),
        "score_intlike_rate": integ.get("score_intlike_rate", None),
        "score_nan_rate": integ.get("score_nan_rate", None),
        "score_min": integ.get("score_min", None),
        "score_max": integ.get("score_max", None),
        "status_OK": int(ok),
        "status_OK_SALVAGED": int(ok_salv),

        # real_test distribution (reference)
        "real_n": dist_real.get("n", None),
        "real_wc_mean": dist_real.get("wc_mean", None),
        "real_sc_mean": dist_real.get("sc_mean", None),
        "real_hindiish_row_rate": dist_real.get("hindiish_row_rate", None),
        "real_hindiish_token_fraction": dist_real.get("hindiish_token_fraction", None),

        # gen distribution
        "gen_n": dist_gen.get("n", None),
        "gen_wc_mean": dist_gen.get("wc_mean", None),
        "gen_wc_p50": dist_gen.get("wc_p50", None),
        "gen_wc_p90": dist_gen.get("wc_p90", None),
        "gen_sc_mean": dist_gen.get("sc_mean", None),
        "gen_devanagari_any_rate": dist_gen.get("devanagari_any_rate", None),
        "gen_latin_any_rate": dist_gen.get("latin_any_rate", None),
        "gen_hindiish_row_rate": dist_gen.get("hindiish_row_rate", None),
        "gen_hindiish_token_fraction": dist_gen.get("hindiish_token_fraction", None),
        "gen_hindiish_per_100w": dist_gen.get("hindiish_per_100w", None),
        "gen_hindiish_p95_per_review": dist_gen.get("hindiish_p95_per_review", None),
        "gen_hindiish_ge3_rate": dist_gen.get("hindiish_ge3_rate", None),

        # deltas vs real_test (if present in report)
        "delta_hindiish_row_rate_vs_real_test": _safe_get(j, ["distribution", "deltas_vs_real_test", "hindiish_row_rate_delta_vs_real_test"], None),
        "delta_hindiish_token_fraction_vs_real_test": _safe_get(j, ["distribution", "deltas_vs_real_test", "hindiish_token_fraction_delta_vs_real_test"], None),
        "delta_hindiish_per_100w_vs_real_test": _safe_get(j, ["distribution", "deltas_vs_real_test", "hindiish_per_100w_delta_vs_real_test"], None),

        # duplicates + leakage
        "dup_exact_rate": dups.get("gen_exact_duplicate_rate", None),
        "leak_exact_match_rate_vs_real_train": leak.get("gen_exact_match_rate_vs_real_train", None),
        "leak_exact_match_rate_vs_real_test": leak.get("gen_exact_match_rate_vs_real_test", None),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reports_root", required=True, help="Directory containing many run subdirs each with report.json")
    ap.add_argument("--glob", default="*/report.json", help="Glob under reports_root to find report.json files")
    ap.add_argument("--out_csv", required=True)
    ap.add_argument("--sort_by", default="run")
    args = ap.parse_args()

    pattern = os.path.join(args.reports_root, args.glob)
    report_paths = sorted(glob.glob(pattern, recursive=("**" in args.glob)))
    if not report_paths:
        raise SystemExit(f"No reports found with: {pattern}")

    rows: List[Dict[str, Any]] = []
    for rp in report_paths:
        try:
            rows.append(_flatten_report(rp, args.reports_root))
        except Exception as e:
            rows.append({
                "run": _infer_run_name(rp, args.reports_root),
                "report_json": rp,
                "ERROR": f"{type(e).__name__}: {e}",
            })

    df = pd.DataFrame(rows)

    # Add a couple useful computed deltas if real_* exists
    if "real_wc_mean" in df.columns and "gen_wc_mean" in df.columns:
        df["delta_wc_mean_vs_real"] = df["gen_wc_mean"] - df["real_wc_mean"]
    if "real_sc_mean" in df.columns and "gen_sc_mean" in df.columns:
        df["delta_sc_mean_vs_real"] = df["gen_sc_mean"] - df["real_sc_mean"]
    if "real_hindiish_row_rate" in df.columns and "gen_hindiish_row_rate" in df.columns:
        df["delta_hindiish_row_rate_vs_real"] = df["gen_hindiish_row_rate"] - df["real_hindiish_row_rate"]
    if "real_hindiish_token_fraction" in df.columns and "gen_hindiish_token_fraction" in df.columns:
        df["delta_hindiish_tokfrac_vs_real"] = df["gen_hindiish_token_fraction"] - df["real_hindiish_token_fraction"]

    if args.sort_by in df.columns:
        df = df.sort_values(args.sort_by)

    os.makedirs(os.path.dirname(args.out_csv), exist_ok=True)
    df.to_csv(args.out_csv, index=False)

    # Print compact view
    show_cols = [
        "run","rows","usable_rows","usable_rate",
        "gen_wc_mean","delta_wc_mean_vs_real",
        "gen_sc_mean","delta_sc_mean_vs_real",
        "gen_hindiish_row_rate","delta_hindiish_row_rate_vs_real",
        "dup_exact_rate",
        "status_OK","status_OK_SALVAGED",
    ]
    show_cols = [c for c in show_cols if c in df.columns]
    print(df[show_cols].to_string(index=False))
    print("\nWrote:", args.out_csv)


if __name__ == "__main__":
    main()
