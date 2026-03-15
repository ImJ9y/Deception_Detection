#!/usr/bin/env python3
import argparse
from pathlib import Path
from typing import Optional

import pandas as pd


def read_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "gen_run" not in df.columns:
        for alt in ["run", "name", "model_run"]:
            if alt in df.columns:
                df = df.rename(columns={alt: "gen_run"})
                break
    if "gen_run" not in df.columns:
        raise RuntimeError(f"{path}: missing merge key 'gen_run'. Columns: {list(df.columns)}")

    df["gen_run"] = df["gen_run"].astype(str).str.strip()
    return df


def read_optional(path: Optional[str]) -> Optional[pd.DataFrame]:
    if not path:
        return None
    p = Path(path)
    if not p.exists():
        return None
    return read_csv(str(p))


def select_cols(df: pd.DataFrame, keep):
    return df[[c for c in keep if c in df.columns]]


def prefix_metric_cols(df: pd.DataFrame, prefix: str) -> pd.DataFrame:
    rename = {c: f"{prefix}_{c}" for c in df.columns if c != "gen_run"}
    return df.rename(columns=rename)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dist_csv", required=True)
    ap.add_argument("--single_csv", default="")
    ap.add_argument("--pairwise_csv", default="")
    ap.add_argument("--single_csv_qwen", default="")
    ap.add_argument("--pairwise_csv_qwen", default="")
    ap.add_argument("--single_csv_llama", default="")
    ap.add_argument("--pairwise_csv_llama", default="")
    ap.add_argument("--detector_csv", required=True)
    ap.add_argument("--out_csv", required=True)
    args = ap.parse_args()

    dist = read_csv(args.dist_csv)
    det = read_csv(args.detector_csv)

    single_qwen_path = args.single_csv_qwen or args.single_csv
    pair_qwen_path = args.pairwise_csv_qwen or args.pairwise_csv

    dist_keep = [
        "gen_run",
        "rows",
        "usable_rows",
        "usable_rate",
        "gen_wc_mean",
        "delta_wc_mean_vs_real",
        "gen_sc_mean",
        "delta_sc_mean_vs_real",
        "gen_hindiish_row_rate",
        "delta_hindiish_row_rate_vs_real",
        "dup_exact_rate",
        "leak_exact_match_rate_vs_real_test",
    ]

    single_keep = [
        "gen_run",
        "n",
        "ok",
        "ok_rate",
        "source_type_present",
        "n_gen",
        "n_real",
        "mu_gen",
        "mu_real",
        "gap_G",
        "ks_stat",
        "wasserstein_dist",
        "mu_real_prev",
        "mu_real_drift_vs_prev",
        "mu_real_drift_alert",
        "gen_ai_rate_at_0_5",
        "real_ai_rate_at_0_5",
        "confidence_mean",
        "confidence_gen_mean",
        "confidence_real_mean",
        "judge_label_AI_rate",
        "judge_label_HUMAN_rate",
        "ai_prob_mean",
        "ai_prob_mean_gen",
        "label_AI_rate_gen",
        "label_HUMAN_rate_gen",
        "ai_prob_mean_all_ok",
        "dedupe_rows_before",
        "dedupe_rows_after",
        "dedupe_drop_n",
        "dedupe_drop_rate",
    ]

    pair_keep = [
        "gen_run",
        "n_pairs_total",
        "n_pairs_main_raw",
        "n_pairs_main_raw_ok",
        "n_pairs_main_base",
        "n_pairs_main",
        "n_pairs_main_ok",
        "n_controls",
        "ok_rate_main",
        "main_aggregation_mode",
        "aggregate_main_by_base",
        "gen_win_rate",
        "tie_rate",
        "human_likeness_H",
        "real_win_rate",
        "net_advantage_delta",
        "gen_win_rate_ci_lo",
        "gen_win_rate_ci_hi",
        "tie_rate_ci_lo",
        "tie_rate_ci_hi",
        "human_likeness_H_ci_lo",
        "human_likeness_H_ci_hi",
        "net_advantage_delta_ci_lo",
        "net_advantage_delta_ci_hi",
        "mean_confidence",
        "mean_margin",
        "gen_left_rate_main",
        "H_gen_left",
        "H_gen_right",
        "position_bias_abs",
        "position_bias_abs_raw",
        "H_gen_left_raw",
        "H_gen_right_raw",
        "position_bias_pass",
        "swap_pairs_n",
        "swap_consistency",
        "swap_consistency_pass",
        "test_retest_pairs_n",
        "test_retest_agreement",
        "test_retest_soft_corr",
        "test_retest_pass",
        "control_rows_n",
        "control_pass_rate",
        "control_pass",
        "control_real_vs_template_pass_rate",
        "control_real_vs_template_real_win_rate",
        "control_real_vs_template_tie_rate",
        "control_real_vs_template_gen_win_rate",
        "control_real_vs_template_real_like_score",
        "control_real_vs_template_real_like_min",
        "control_real_vs_template_gen_win_max",
        "control_real_vs_real_side_gap",
        "control_real_vs_real_tie_rate",
        "control_gen_vs_gen_side_gap",
        "control_gen_vs_gen_tie_rate",
        "judge_reliable_flag",
        "judge_reliability_reasons",
        "more_human_A_rate",
        "more_human_B_rate",
        "ai_prob_A_mean",
        "ai_prob_B_mean",
        "ai_prob_B_minus_A",
        "confidence_mean",
        "dedupe_rows_before",
        "dedupe_rows_after",
        "dedupe_drop_n",
        "dedupe_drop_rate",
    ]

    det_keep = [
        "gen_run",
        "roc_auc",
        "acc",
        "f1",
        "precision",
        "recall",
        "balanced_acc",
        "test_real_n",
        "test_gen_n",
    ]

    dist = select_cols(dist, dist_keep)
    det = select_cols(det, det_keep)

    single_qwen = read_optional(single_qwen_path)
    pair_qwen = read_optional(pair_qwen_path)
    single_llama = read_optional(args.single_csv_llama)
    pair_llama = read_optional(args.pairwise_csv_llama)

    if single_qwen is not None:
        single_qwen = prefix_metric_cols(select_cols(single_qwen, single_keep), "qwen_single")
    if pair_qwen is not None:
        pair_qwen = prefix_metric_cols(select_cols(pair_qwen, pair_keep), "qwen_pair")
    if single_llama is not None:
        single_llama = prefix_metric_cols(select_cols(single_llama, single_keep), "llama_single")
    if pair_llama is not None:
        pair_llama = prefix_metric_cols(select_cols(pair_llama, pair_keep), "llama_pair")

    out = dist.copy()
    for df in [single_qwen, pair_qwen, single_llama, pair_llama]:
        if df is not None:
            out = out.merge(df, on="gen_run", how="outer")
    out = out.merge(det, on="gen_run", how="outer")

    order = ["llama31_fs5_nepali_v1", "llama31_zs_nepali_v1", "qwen3_fs5_nepali_v1", "qwen3_zs_nepali_v1"]
    out["__ord"] = out["gen_run"].apply(lambda x: order.index(x) if x in order else 999)
    out = out.sort_values(["__ord", "gen_run"]).drop(columns=["__ord"])

    Path(args.out_csv).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out_csv, index=False)

    print("\n--- MASTER SCOREBOARD (preview) ---")
    with pd.option_context("display.max_columns", 400, "display.width", 260):
        print(out.to_string(index=False))

    print("\nWrote:", args.out_csv)


if __name__ == "__main__":
    main()
