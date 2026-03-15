#!/usr/bin/env python3
import argparse
import glob
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp, wasserstein_distance


def safe_read_csv(fp: str) -> pd.DataFrame:
    df = pd.read_csv(fp)

    # Drop duplicated header rows accidentally appended into data rows.
    if "target_id" in df.columns:
        df = df[df["target_id"].astype(str) != "target_id"]

    return df


def infer_gen_run(fp: str, df: pd.DataFrame) -> str:
    if "gen_run" in df.columns and len(df) > 0:
        return str(df["gen_run"].iloc[0])
    if "run_id" in df.columns and len(df) > 0:
        return str(df["run_id"].iloc[0])
    base = os.path.basename(fp)
    m = re.match(r"judged_single__(.+?)__.+\.csv$", base)
    return m.group(1) if m else base


def mean_or_none(s: pd.Series) -> Optional[float]:
    x = pd.to_numeric(s, errors="coerce").dropna()
    if x.empty:
        return None
    return float(x.mean())


def rate_at_threshold(s: pd.Series, thr: float = 0.5) -> Optional[float]:
    x = pd.to_numeric(s, errors="coerce").dropna()
    if x.empty:
        return None
    return float((x >= thr).mean())


def label_rate(labels: pd.Series, label: str) -> Optional[float]:
    y = labels.dropna().astype(str)
    if y.empty:
        return None
    return float((y == label).mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--judged_root", required=True, help="Directory containing judged_single__*.csv")
    ap.add_argument("--pattern", default="judged_single__*__qwen_single_v1.csv", help="Glob pattern under judged_root")
    ap.add_argument("--out_csv", required=True)
    ap.add_argument("--mu_real_drift_threshold", type=float, default=0.05)
    args = ap.parse_args()

    paths = sorted(glob.glob(os.path.join(args.judged_root, args.pattern), recursive=("**" in args.pattern)))
    if not paths:
        raise SystemExit(f"No files matched: {os.path.join(args.judged_root, args.pattern)}")

    rows: List[Dict[str, Any]] = []

    for fp in paths:
        df = safe_read_csv(fp)
        if "judge_status" not in df.columns:
            raise RuntimeError(f"{fp}: missing judge_status column. Columns: {list(df.columns)}")

        gen_run = infer_gen_run(fp, df)
        n = len(df)
        ok_mask = df["judge_status"].astype(str).str.upper() == "OK"
        ok = int(ok_mask.sum())
        ok_rate = float(ok / n) if n else 0.0

        probs_all = pd.to_numeric(df.loc[ok_mask, "judge_ai_probability_mean"], errors="coerce") if "judge_ai_probability_mean" in df.columns else pd.Series(dtype=float)
        conf_all = pd.to_numeric(df.loc[ok_mask, "judge_confidence_mean"], errors="coerce") if "judge_confidence_mean" in df.columns else pd.Series(dtype=float)

        has_source_type = "source_type" in df.columns
        if has_source_type:
            src = df["source_type"].fillna("").astype(str).str.lower()
            gen_mask = ok_mask & src.eq("gen")
            real_mask = ok_mask & src.eq("real")
        else:
            # Backward compatibility: legacy files are generated-only.
            gen_mask = ok_mask
            real_mask = pd.Series([False] * len(df), index=df.index)

        probs_gen = pd.to_numeric(df.loc[gen_mask, "judge_ai_probability_mean"], errors="coerce") if "judge_ai_probability_mean" in df.columns else pd.Series(dtype=float)
        probs_real = pd.to_numeric(df.loc[real_mask, "judge_ai_probability_mean"], errors="coerce") if "judge_ai_probability_mean" in df.columns else pd.Series(dtype=float)
        conf_gen = pd.to_numeric(df.loc[gen_mask, "judge_confidence_mean"], errors="coerce") if "judge_confidence_mean" in df.columns else pd.Series(dtype=float)
        conf_real = pd.to_numeric(df.loc[real_mask, "judge_confidence_mean"], errors="coerce") if "judge_confidence_mean" in df.columns else pd.Series(dtype=float)

        mu_gen = mean_or_none(probs_gen)
        mu_real = mean_or_none(probs_real)
        gap_G = (mu_gen - mu_real) if (mu_gen is not None and mu_real is not None) else None

        ks_stat = None
        w_dist = None
        if probs_gen.dropna().size > 0 and probs_real.dropna().size > 0:
            ks_stat = float(ks_2samp(probs_real.dropna(), probs_gen.dropna()).statistic)
            w_dist = float(wasserstein_distance(probs_real.dropna(), probs_gen.dropna()))

        labels_gen = df.loc[gen_mask, "judge_label"] if "judge_label" in df.columns else pd.Series(dtype=object)
        ai_rate_gen = label_rate(labels_gen, "AI")
        human_rate_gen = label_rate(labels_gen, "HUMAN")

        row: Dict[str, Any] = {
            "file": os.path.basename(fp),
            "gen_run": gen_run,
            "file_mtime": float(Path(fp).stat().st_mtime),
            "n": int(n),
            "ok": int(ok),
            "ok_rate": float(ok_rate),
            "source_type_present": bool(has_source_type),
            "n_gen": int(gen_mask.sum()),
            "n_real": int(real_mask.sum()),
            "mu_gen": mu_gen,
            "mu_real": mu_real,
            "gap_G": gap_G,
            "ks_stat": ks_stat,
            "wasserstein_dist": w_dist,
            "gen_ai_rate_at_0_5": rate_at_threshold(probs_gen, 0.5),
            "real_ai_rate_at_0_5": rate_at_threshold(probs_real, 0.5),
            "confidence_mean": mean_or_none(conf_all),
            "confidence_gen_mean": mean_or_none(conf_gen),
            "confidence_real_mean": mean_or_none(conf_real),
            # Backward-compatible fields
            "ai_prob_mean": mu_gen,
            "judge_label_AI_rate": ai_rate_gen,
            "judge_label_HUMAN_rate": human_rate_gen,
            # Explicit clarity aliases for gen-only legacy-style metrics.
            "ai_prob_mean_gen": mu_gen,
            "label_AI_rate_gen": ai_rate_gen,
            "label_HUMAN_rate_gen": human_rate_gen,
            # Optional overall mean over all OK rows.
            "ai_prob_mean_all_ok": mean_or_none(probs_all),
        }

        for c in ["dedupe_rows_before", "dedupe_rows_after", "dedupe_drop_n", "dedupe_drop_rate"]:
            if c in df.columns:
                v = pd.to_numeric(df[c], errors="coerce").dropna()
                row[c] = float(v.iloc[0]) if v.size else None

        rows.append(row)

    out = pd.DataFrame(rows)
    if out.empty:
        raise RuntimeError("No rows collected from judged files.")

    # Compute mu_real drift in chronological file write order.
    out = out.sort_values(["file_mtime", "gen_run"]).reset_index(drop=True)
    prev_mu_real: Optional[float] = None
    mu_prev_vals: List[Optional[float]] = []
    mu_drift_vals: List[Optional[float]] = []
    mu_alert_vals: List[bool] = []
    for _, row in out.iterrows():
        mu_real = row.get("mu_real", None)
        mu_prev_vals.append(prev_mu_real)
        if mu_real is None or (isinstance(mu_real, float) and not np.isfinite(mu_real)) or prev_mu_real is None:
            mu_drift_vals.append(None)
            mu_alert_vals.append(False)
        else:
            drift = float(mu_real - prev_mu_real)
            mu_drift_vals.append(drift)
            mu_alert_vals.append(abs(drift) > args.mu_real_drift_threshold)
        if mu_real is not None and (not isinstance(mu_real, float) or np.isfinite(mu_real)):
            prev_mu_real = float(mu_real)

    out["mu_real_prev"] = mu_prev_vals
    out["mu_real_drift_vs_prev"] = mu_drift_vals
    out["mu_real_drift_alert"] = mu_alert_vals

    out = out.sort_values(["gen_run", "file"]).reset_index(drop=True)
    out = out.drop(columns=["file_mtime"])

    Path(args.out_csv).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out_csv, index=False)

    print(out.to_string(index=False))
    print(f"\nWrote: {args.out_csv}")


if __name__ == "__main__":
    main()
