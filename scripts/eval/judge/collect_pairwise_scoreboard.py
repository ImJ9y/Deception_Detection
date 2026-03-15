#!/usr/bin/env python3
import argparse
import glob
import json
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


def parse_bool(x: Any) -> bool:
    if isinstance(x, bool):
        return x
    s = str(x).strip().lower()
    return s in {"1", "true", "t", "yes", "y"}


def safe_float_series(df: pd.DataFrame, col: str) -> pd.Series:
    if col not in df.columns:
        return pd.Series([np.nan] * len(df), index=df.index, dtype=float)
    return pd.to_numeric(df[col], errors="coerce")


def infer_gen_run(fp: str, df: pd.DataFrame) -> str:
    if "gen_run" in df.columns and len(df) > 0:
        return str(df["gen_run"].iloc[0])
    base = os.path.basename(fp)
    m = re.match(r"judged_pairs__(.+?)__.+\.csv$", base)
    return m.group(1) if m else base


def bootstrap_metrics(
    df_ok_main: pd.DataFrame,
    n_boot: int,
    seed: int,
    unit: str,
    cluster_col: str,
) -> Dict[str, Optional[Tuple[float, float]]]:
    if len(df_ok_main) == 0:
        return {
            "w": None,
            "t": None,
            "H": None,
            "delta": None,
            "w_pair": None,
            "t_pair": None,
            "H_pair": None,
            "delta_pair": None,
        }

    def calc(sub: pd.DataFrame) -> Tuple[float, float, float, float]:
        n = len(sub)
        if n == 0:
            return (np.nan, np.nan, np.nan, np.nan)
        winner = sub["normalized_winner"].astype(str)
        w = float((winner == "GEN").mean())
        t = float((winner == "TIE").mean())
        real_win = float((winner == "REAL").mean())
        H = w + 0.5 * t
        delta = w - real_win
        return (w, t, H, delta)

    rng = np.random.default_rng(seed)

    def percentile_ci(arr: np.ndarray) -> Optional[Tuple[float, float]]:
        arr = arr[np.isfinite(arr)]
        if arr.size == 0:
            return None
        return (float(np.quantile(arr, 0.025)), float(np.quantile(arr, 0.975)))

    def run_pair_bootstrap() -> Dict[str, Optional[Tuple[float, float]]]:
        n = len(df_ok_main)
        rows = np.arange(n)
        vals = np.zeros((n_boot, 4), dtype=float)
        for i in range(n_boot):
            take = rng.choice(rows, size=n, replace=True)
            sub = df_ok_main.iloc[take]
            vals[i, :] = calc(sub)
        return {
            "w": percentile_ci(vals[:, 0]),
            "t": percentile_ci(vals[:, 1]),
            "H": percentile_ci(vals[:, 2]),
            "delta": percentile_ci(vals[:, 3]),
        }

    def run_cluster_bootstrap() -> Dict[str, Optional[Tuple[float, float]]]:
        if cluster_col not in df_ok_main.columns:
            return run_pair_bootstrap()

        tmp = df_ok_main.copy()
        tmp[cluster_col] = tmp[cluster_col].fillna("__missing_cluster__").astype(str)
        keys = tmp[cluster_col].unique().tolist()
        if not keys:
            return run_pair_bootstrap()

        grouped = {k: g for k, g in tmp.groupby(cluster_col, sort=False)}
        vals = np.zeros((n_boot, 4), dtype=float)
        for i in range(n_boot):
            sample_keys = rng.choice(keys, size=len(keys), replace=True)
            frames = [grouped[k] for k in sample_keys]
            sub = pd.concat(frames, ignore_index=True)
            vals[i, :] = calc(sub)
        return {
            "w": percentile_ci(vals[:, 0]),
            "t": percentile_ci(vals[:, 1]),
            "H": percentile_ci(vals[:, 2]),
            "delta": percentile_ci(vals[:, 3]),
        }

    if unit == "pair":
        main_ci = run_pair_bootstrap()
        return {
            "w": main_ci["w"],
            "t": main_ci["t"],
            "H": main_ci["H"],
            "delta": main_ci["delta"],
            "w_pair": None,
            "t_pair": None,
            "H_pair": None,
            "delta_pair": None,
        }

    if unit == "both":
        cluster_ci = run_cluster_bootstrap()
        pair_ci = run_pair_bootstrap()
        return {
            "w": cluster_ci["w"],
            "t": cluster_ci["t"],
            "H": cluster_ci["H"],
            "delta": cluster_ci["delta"],
            "w_pair": pair_ci["w"],
            "t_pair": pair_ci["t"],
            "H_pair": pair_ci["H"],
            "delta_pair": pair_ci["delta"],
        }

    cluster_ci = run_cluster_bootstrap()
    return {
        "w": cluster_ci["w"],
        "t": cluster_ci["t"],
        "H": cluster_ci["H"],
        "delta": cluster_ci["delta"],
        "w_pair": None,
        "t_pair": None,
        "H_pair": None,
        "delta_pair": None,
    }


def normalize_pairwise_df(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    if "pair_role" not in out.columns:
        out["pair_role"] = "main"
    out["pair_role"] = out["pair_role"].fillna("main").astype(str).str.strip().str.lower()

    if "control_type" not in out.columns:
        out["control_type"] = ""
    out["control_type"] = out["control_type"].fillna("").astype(str).str.strip().str.lower()

    if "pair_order" not in out.columns:
        out["pair_order"] = ""

    if "is_gen_left" in out.columns:
        is_gen_left = out["is_gen_left"].map(parse_bool)
    else:
        is_gen_left = pd.Series([False] * len(out), index=out.index)
    out["is_gen_left"] = is_gen_left

    more = out.get("judge_more_human", pd.Series([""] * len(out))).fillna("").astype(str).str.strip().str.upper()
    out["judge_more_human"] = more

    ai_a = safe_float_series(out, "judge_ai_probability_A_mean")
    ai_b = safe_float_series(out, "judge_ai_probability_B_mean")
    out["judge_ai_probability_A_mean"] = ai_a
    out["judge_ai_probability_B_mean"] = ai_b

    ai_gen = np.where(out["is_gen_left"], ai_a, ai_b)
    ai_real = np.where(out["is_gen_left"], ai_b, ai_a)
    out["ai_prob_gen"] = ai_gen
    out["ai_prob_real"] = ai_real
    out["margin_gen_humanlike"] = out["ai_prob_real"] - out["ai_prob_gen"]

    winner: List[str] = []
    for m, gleft in zip(more.tolist(), out["is_gen_left"].tolist()):
        if m == "TIE":
            winner.append("TIE")
        elif m == "A":
            winner.append("GEN" if gleft else "REAL")
        elif m == "B":
            winner.append("REAL" if gleft else "GEN")
        else:
            winner.append("UNK")
    out["normalized_winner"] = winner

    status = out.get("judge_status", pd.Series([""] * len(out))).fillna("").astype(str).str.strip().str.upper()
    out["judge_status"] = status
    out["is_ok"] = status == "OK"

    if "pair_uid" not in out.columns:
        if "pair_id" in out.columns:
            out["pair_uid"] = out["pair_id"].astype(str)
        else:
            out["pair_uid"] = [f"pair_{i}" for i in range(len(out))]

    if "base_pair_uid" not in out.columns:
        out["base_pair_uid"] = out["pair_uid"]

    if "gen_id" not in out.columns:
        if "target_id" in out.columns:
            out["gen_id"] = out["target_id"].astype(str)
        else:
            out["gen_id"] = [f"gen_{i}" for i in range(len(out))]

    return out


def calc_H(sub: pd.DataFrame) -> Optional[float]:
    if len(sub) == 0:
        return None
    w = float((sub["normalized_winner"] == "GEN").mean())
    t = float((sub["normalized_winner"] == "TIE").mean())
    return w + 0.5 * t


def control_metrics(
    control_ok: pd.DataFrame,
    real_template_min: float,
    real_template_gen_max: float,
    balanced_max_gap: float,
    balanced_min_tie: float,
) -> Dict[str, Any]:
    out: Dict[str, Any] = {}

    def balanced_type(type_name: str) -> Dict[str, Any]:
        d = control_ok[control_ok["control_type"] == type_name]
        n = len(d)
        if n == 0:
            return {
                "n": 0,
                "gen_win": None,
                "real_win": None,
                "tie": None,
                "side_gap": None,
                "pass": False,
            }
        w_gen = float((d["normalized_winner"] == "GEN").mean())
        w_real = float((d["normalized_winner"] == "REAL").mean())
        tie = float((d["normalized_winner"] == "TIE").mean())
        side_gap = abs(w_gen - w_real)
        ok = (side_gap <= balanced_max_gap) and (tie >= balanced_min_tie)
        return {
            "n": int(n),
            "gen_win": w_gen,
            "real_win": w_real,
            "tie": tie,
            "side_gap": float(side_gap),
            "pass": bool(ok),
        }

    rt = control_ok[control_ok["control_type"] == "real_vs_template"]
    if len(rt) > 0:
        rt_real_win = float((rt["normalized_winner"] == "REAL").mean())
        rt_tie = float((rt["normalized_winner"] == "TIE").mean())
        rt_gen_win = float((rt["normalized_winner"] == "GEN").mean())
        rt_real_like = float(rt_real_win + 0.5 * rt_tie)
        rt_pass_rate = rt_real_win
        rt_pass = bool((rt_real_like >= real_template_min) and (rt_gen_win <= real_template_gen_max))
    else:
        rt_real_win = None
        rt_tie = None
        rt_gen_win = None
        rt_real_like = None
        rt_pass_rate = None
        rt_pass = False

    rr = balanced_type("real_vs_real")
    gg = balanced_type("gen_vs_gen")

    passes = [rt_pass, rr["pass"], gg["pass"]]
    out["control_types_present_n"] = int(sum([1 if len(rt) > 0 else 0, 1 if rr["n"] > 0 else 0, 1 if gg["n"] > 0 else 0]))
    out["control_pass_rate"] = float(np.mean(passes))
    out["control_pass"] = bool(all(passes))

    out["control_real_vs_template_n"] = int(len(rt))
    out["control_real_vs_template_pass_rate"] = rt_pass_rate
    out["control_real_vs_template_real_win_rate"] = rt_real_win
    out["control_real_vs_template_tie_rate"] = rt_tie
    out["control_real_vs_template_gen_win_rate"] = rt_gen_win
    out["control_real_vs_template_real_like_score"] = rt_real_like
    out["control_real_vs_template_real_like_min"] = float(real_template_min)
    out["control_real_vs_template_gen_win_max"] = float(real_template_gen_max)
    out["control_real_vs_template_pass"] = rt_pass

    out["control_real_vs_real_n"] = rr["n"]
    out["control_real_vs_real_side_gap"] = rr["side_gap"]
    out["control_real_vs_real_tie_rate"] = rr["tie"]
    out["control_real_vs_real_pass"] = rr["pass"]

    out["control_gen_vs_gen_n"] = gg["n"]
    out["control_gen_vs_gen_side_gap"] = gg["side_gap"]
    out["control_gen_vs_gen_tie_rate"] = gg["tie"]
    out["control_gen_vs_gen_pass"] = gg["pass"]
    return out


def sanity_random(n: int, seed: int) -> Dict[str, float]:
    rng = np.random.default_rng(seed)
    winners = rng.choice(["GEN", "REAL", "TIE"], size=n)
    is_left = rng.choice([True, False], size=n)
    df = pd.DataFrame({"normalized_winner": winners, "is_gen_left": is_left})
    H_left = calc_H(df[df["is_gen_left"]])
    H_right = calc_H(df[~df["is_gen_left"]])
    return {
        "sanity_n": float(n),
        "sanity_gen_left_rate": float(df["is_gen_left"].mean()),
        "sanity_H_left": float(H_left if H_left is not None else np.nan),
        "sanity_H_right": float(H_right if H_right is not None else np.nan),
        "sanity_position_bias_abs": float(abs(H_left - H_right) if H_left is not None and H_right is not None else np.nan),
    }


def aggregate_main_rows(df_main_all: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for base_uid, g_all in df_main_all.groupby("base_pair_uid", sort=False):
        g_all = g_all.copy()
        order_num = pd.to_numeric(g_all.get("pair_order", pd.Series([np.nan] * len(g_all))), errors="coerce")
        g_all["__pair_order_num"] = order_num
        g_all = g_all.sort_values(["__pair_order_num", "pair_uid"], na_position="last")

        if "__pair_order_num" in g_all.columns and np.isfinite(g_all["__pair_order_num"]).any():
            g_ref_df = g_all[np.isfinite(g_all["__pair_order_num"]) & (g_all["__pair_order_num"] == 1)]
            g_ref = g_ref_df.iloc[0] if len(g_ref_df) else g_all.iloc[0]
        else:
            g_ref = g_all.iloc[0]

        g_ok = g_all[g_all["is_ok"]].copy()
        winners_ok = g_ok["normalized_winner"].astype(str).tolist()

        # Debias rule: both orders must agree on GEN/REAL for a decisive win.
        # Any disagreement, missing order, unknown label, or explicit tie => TIE.
        agg_ok = len(g_ok) >= 1
        if len(g_ok) >= 2 and all(w == "GEN" for w in winners_ok):
            agg_winner = "GEN"
        elif len(g_ok) >= 2 and all(w == "REAL" for w in winners_ok):
            agg_winner = "REAL"
        else:
            agg_winner = "TIE"

        is_left = bool(g_ref["is_gen_left"])
        if agg_winner == "GEN":
            agg_more = "A" if is_left else "B"
        elif agg_winner == "REAL":
            agg_more = "B" if is_left else "A"
        elif agg_winner == "TIE":
            agg_more = "TIE"
        else:
            agg_more = "UNK"

        ai_gen_mean = float(pd.to_numeric(g_ok["ai_prob_gen"], errors="coerce").mean()) if len(g_ok) else np.nan
        ai_real_mean = float(pd.to_numeric(g_ok["ai_prob_real"], errors="coerce").mean()) if len(g_ok) else np.nan
        conf_mean = float(pd.to_numeric(g_ok["judge_confidence_mean"], errors="coerce").mean()) if len(g_ok) else np.nan

        rows.append(
            {
                "pair_uid": str(base_uid),
                "base_pair_uid": str(base_uid),
                "pair_role": "main",
                "control_type": "",
                "is_ok": bool(agg_ok),
                "judge_status": "OK" if agg_ok else "FAIL",
                "is_gen_left": is_left,
                "gen_id": str(g_ref.get("gen_id", "")),
                "real_id": str(g_ref.get("real_id", "")),
                "target_id": g_ref.get("target_id", None),
                "judge_more_human": agg_more,
                "normalized_winner": agg_winner,
                "ai_prob_gen": ai_gen_mean,
                "ai_prob_real": ai_real_mean,
                "margin_gen_humanlike": ai_real_mean - ai_gen_mean if np.isfinite(ai_real_mean) and np.isfinite(ai_gen_mean) else np.nan,
                "judge_ai_probability_A_mean": ai_gen_mean if is_left else ai_real_mean,
                "judge_ai_probability_B_mean": ai_real_mean if is_left else ai_gen_mean,
                "judge_confidence_mean": conf_mean,
                "n_order_rows_total": int(len(g_all)),
                "n_order_rows_ok": int(len(g_ok)),
            }
        )
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--judged_root", required=True, help="Directory containing judged_pairs__*.csv")
    ap.add_argument("--pattern", default="judged_pairs__*__qwen_pairwise_v1.csv", help="Glob pattern under judged_root")
    ap.add_argument("--out_csv", required=True)
    ap.add_argument("--raw_out_dir", default="")
    ap.add_argument("--bootstrap_iters", type=int, default=2000)
    ap.add_argument("--bootstrap_seed", type=int, default=12345)
    ap.add_argument("--bootstrap_unit", default="cluster", choices=["cluster", "pair", "both"])
    ap.add_argument("--position_bias_max_abs", type=float, default=0.05)
    ap.add_argument("--swap_consistency_min", type=float, default=0.80)
    ap.add_argument("--test_retest_agreement_min", type=float, default=0.75)
    ap.add_argument("--control_real_template_min", type=float, default=0.80)
    ap.add_argument("--control_real_template_gen_max", type=float, default=0.05)
    ap.add_argument("--control_balanced_max_gap", type=float, default=0.15)
    ap.add_argument("--control_balanced_min_tie", type=float, default=0.20)
    ap.add_argument("--aggregate_main_by_base", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--write_parquet", action="store_true")
    ap.add_argument("--sanity_random_n", type=int, default=0)
    args = ap.parse_args()

    files = sorted(glob.glob(os.path.join(args.judged_root, args.pattern), recursive=("**" in args.pattern)))
    if not files:
        raise SystemExit(f"No files matched: {os.path.join(args.judged_root, args.pattern)}")

    raw_out_dir = Path(args.raw_out_dir) if args.raw_out_dir else (Path(args.out_csv).parent / "pairwise_normalized_raw")
    raw_out_dir.mkdir(parents=True, exist_ok=True)

    rows: List[Dict[str, Any]] = []

    for fp in files:
        df_in = pd.read_csv(fp)
        df = normalize_pairwise_df(df_in)

        gen_run = infer_gen_run(fp, df)
        n_total = len(df)

        control_mask = (df["pair_role"] == "control") | (df["control_type"] != "")
        main_mask = (~control_mask) & df["pair_role"].isin(["", "main"])

        df_main = df[main_mask].copy()
        df_main_ok_raw = df_main[df_main["is_ok"]].copy()

        n_main_raw = len(df_main)
        n_main_raw_ok = len(df_main_ok_raw)

        n_main_base = int(df_main["base_pair_uid"].astype(str).nunique()) if len(df_main) else 0
        use_base_agg = bool(args.aggregate_main_by_base)
        main_aggregation_mode = "raw"

        if use_base_agg:
            base_counts = df_main["base_pair_uid"].astype(str).value_counts()
            if (not base_counts.empty) and int(base_counts.max()) >= 2:
                df_main_metric = aggregate_main_rows(df_main)
                df_main_metric_ok = df_main_metric[df_main_metric["is_ok"]].copy()
                main_aggregation_mode = "base_aggregated"
            else:
                df_main_metric = df_main.copy()
                df_main_metric_ok = df_main_metric[df_main_metric["is_ok"]].copy()
                main_aggregation_mode = "raw_fallback_no_two_orders"
        else:
            df_main_metric = df_main.copy()
            df_main_metric_ok = df_main_metric[df_main_metric["is_ok"]].copy()
            main_aggregation_mode = "raw_requested_off"

        n_main = len(df_main_metric)
        n_main_ok = len(df_main_metric_ok)
        ok_rate_main = float(n_main_ok / n_main) if n_main > 0 else 0.0

        winner_main = df_main_metric_ok["normalized_winner"].astype(str)
        gen_win = float((winner_main == "GEN").mean()) if n_main_ok else np.nan
        tie_rate = float((winner_main == "TIE").mean()) if n_main_ok else np.nan
        real_win = float((winner_main == "REAL").mean()) if n_main_ok else np.nan
        H = (gen_win + 0.5 * tie_rate) if n_main_ok else np.nan
        delta = (gen_win - real_win) if n_main_ok else np.nan

        conf = safe_float_series(df_main_metric_ok, "judge_confidence_mean")
        mean_conf = float(conf.mean()) if n_main_ok else np.nan
        mean_margin = float(pd.to_numeric(df_main_metric_ok["margin_gen_humanlike"], errors="coerce").mean()) if n_main_ok else np.nan

        more = df_main_metric_ok["judge_more_human"].astype(str)
        more_a_rate = float((more == "A").mean()) if n_main_ok else np.nan
        more_b_rate = float((more == "B").mean()) if n_main_ok else np.nan

        ai_prob_a_mean = float(safe_float_series(df_main_metric_ok, "judge_ai_probability_A_mean").mean()) if n_main_ok else np.nan
        ai_prob_b_mean = float(safe_float_series(df_main_metric_ok, "judge_ai_probability_B_mean").mean()) if n_main_ok else np.nan
        ai_prob_b_minus_a = float((safe_float_series(df_main_metric_ok, "judge_ai_probability_B_mean") - safe_float_series(df_main_metric_ok, "judge_ai_probability_A_mean")).mean()) if n_main_ok else np.nan

        if args.bootstrap_unit == "cluster":
            bootstrap_unit_used = "gen_item_cluster"
        elif args.bootstrap_unit == "pair":
            bootstrap_unit_used = "pair"
        else:
            bootstrap_unit_used = "both"

        cis = bootstrap_metrics(
            df_ok_main=df_main_metric_ok,
            n_boot=max(10, int(args.bootstrap_iters)),
            seed=int(args.bootstrap_seed),
            unit=args.bootstrap_unit,
            cluster_col="gen_id",
        )

        left = df_main_metric_ok[df_main_metric_ok["is_gen_left"]]
        right = df_main_metric_ok[~df_main_metric_ok["is_gen_left"]]
        H_left = calc_H(left)
        H_right = calc_H(right)
        gen_left_rate = float(df_main_metric_ok["is_gen_left"].mean()) if n_main_ok else np.nan
        if H_left is not None and H_right is not None:
            pos_bias_abs = abs(H_left - H_right)
            pos_bias_pass = bool(pos_bias_abs <= args.position_bias_max_abs)
        else:
            pos_bias_abs = np.nan
            pos_bias_pass = False

        left_raw = df_main_ok_raw[df_main_ok_raw["is_gen_left"]]
        right_raw = df_main_ok_raw[~df_main_ok_raw["is_gen_left"]]
        H_left_raw = calc_H(left_raw)
        H_right_raw = calc_H(right_raw)
        if H_left_raw is not None and H_right_raw is not None:
            pos_bias_abs_raw = abs(H_left_raw - H_right_raw)
        else:
            pos_bias_abs_raw = np.nan

        # Swap invariance gate.
        swap_ok = df[(df["pair_role"] == "swap") & (df["is_ok"])].copy()
        base_ok = df[(df["pair_role"] == "main") & (df["is_ok"])].copy()
        base_by_uid = {str(r["pair_uid"]): r for _, r in base_ok.iterrows()}
        swap_total = 0
        swap_good = 0
        for _, row in swap_ok.iterrows():
            b = base_by_uid.get(str(row["base_pair_uid"]))
            if b is None:
                continue
            base_more = str(b["judge_more_human"])
            swap_more = str(row["judge_more_human"])
            swap_total += 1
            if base_more == "A":
                ok = swap_more in {"B", "TIE"}
            elif base_more == "B":
                ok = swap_more in {"A", "TIE"}
            elif base_more == "TIE":
                ok = swap_more == "TIE"
            else:
                ok = False
            swap_good += int(ok)
        swap_consistency = float(swap_good / swap_total) if swap_total > 0 else np.nan
        swap_pass = bool((swap_total > 0) and (swap_consistency >= args.swap_consistency_min))

        # Test-retest gate.
        retest_ok = df[(df["pair_role"] == "retest") & (df["is_ok"])].copy()
        retest_total = 0
        retest_agree = 0
        base_margins: List[float] = []
        retest_margins: List[float] = []
        for _, row in retest_ok.iterrows():
            b = base_by_uid.get(str(row["base_pair_uid"]))
            if b is None:
                continue
            retest_total += 1
            base_w = str(b["normalized_winner"])
            retest_w = str(row["normalized_winner"])
            retest_agree += int(base_w == retest_w)

            bm = pd.to_numeric(pd.Series([b["margin_gen_humanlike"]]), errors="coerce").iloc[0]
            rm = pd.to_numeric(pd.Series([row["margin_gen_humanlike"]]), errors="coerce").iloc[0]
            if np.isfinite(bm) and np.isfinite(rm):
                base_margins.append(float(bm))
                retest_margins.append(float(rm))

        test_retest_agreement = float(retest_agree / retest_total) if retest_total > 0 else np.nan
        if len(base_margins) >= 2 and np.std(base_margins) > 0 and np.std(retest_margins) > 0:
            test_retest_soft_corr = float(np.corrcoef(base_margins, retest_margins)[0, 1])
        else:
            test_retest_soft_corr = np.nan
        test_retest_pass = bool((retest_total > 0) and (test_retest_agreement >= args.test_retest_agreement_min))

        control_ok = df[control_mask & df["is_ok"]].copy()
        controls = control_metrics(
            control_ok=control_ok,
            real_template_min=args.control_real_template_min,
            real_template_gen_max=args.control_real_template_gen_max,
            balanced_max_gap=args.control_balanced_max_gap,
            balanced_min_tie=args.control_balanced_min_tie,
        )

        fail_reasons: List[str] = []
        if not pos_bias_pass:
            fail_reasons.append("position_bias")
        if not swap_pass:
            fail_reasons.append("swap_invariance")
        if not test_retest_pass:
            fail_reasons.append("test_retest")
        if not controls["control_pass"]:
            fail_reasons.append("controls")
        judge_reliable = len(fail_reasons) == 0

        stem = Path(fp).stem
        norm_csv = raw_out_dir / f"{stem}__normalized.csv"
        norm_jsonl = raw_out_dir / f"{stem}__normalized.jsonl"
        df.to_csv(norm_csv, index=False)
        with open(norm_jsonl, "w", encoding="utf-8") as f:
            for rec in df.to_dict(orient="records"):
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")

        norm_parquet = None
        if args.write_parquet:
            try:
                norm_parquet = raw_out_dir / f"{stem}__normalized.parquet"
                df.to_parquet(norm_parquet, index=False)
            except Exception:
                norm_parquet = None

        row = {
            "file": os.path.basename(fp),
            "gen_run": gen_run,
            "n_pairs_total": int(n_total),
            "n_pairs_main_raw": int(n_main_raw),
            "n_pairs_main_raw_ok": int(n_main_raw_ok),
            "n_pairs_main_base": int(n_main_base),
            "n_pairs_main": int(n_main),
            "n_pairs_main_ok": int(n_main_ok),
            "n_controls": int(control_mask.sum()),
            "ok_rate_main": float(ok_rate_main),
            "main_aggregation_mode": main_aggregation_mode,
            "aggregate_main_by_base": bool(args.aggregate_main_by_base),
            "gen_win_rate": float(gen_win) if np.isfinite(gen_win) else None,
            "tie_rate": float(tie_rate) if np.isfinite(tie_rate) else None,
            "human_likeness_H": float(H) if np.isfinite(H) else None,
            "real_win_rate": float(real_win) if np.isfinite(real_win) else None,
            "net_advantage_delta": float(delta) if np.isfinite(delta) else None,
            "gen_win_rate_ci_lo": cis["w"][0] if cis["w"] else None,
            "gen_win_rate_ci_hi": cis["w"][1] if cis["w"] else None,
            "tie_rate_ci_lo": cis["t"][0] if cis["t"] else None,
            "tie_rate_ci_hi": cis["t"][1] if cis["t"] else None,
            "human_likeness_H_ci_lo": cis["H"][0] if cis["H"] else None,
            "human_likeness_H_ci_hi": cis["H"][1] if cis["H"] else None,
            "net_advantage_delta_ci_lo": cis["delta"][0] if cis["delta"] else None,
            "net_advantage_delta_ci_hi": cis["delta"][1] if cis["delta"] else None,
            "gen_win_rate_ci_pair_lo": cis["w_pair"][0] if cis["w_pair"] else None,
            "gen_win_rate_ci_pair_hi": cis["w_pair"][1] if cis["w_pair"] else None,
            "tie_rate_ci_pair_lo": cis["t_pair"][0] if cis["t_pair"] else None,
            "tie_rate_ci_pair_hi": cis["t_pair"][1] if cis["t_pair"] else None,
            "human_likeness_H_ci_pair_lo": cis["H_pair"][0] if cis["H_pair"] else None,
            "human_likeness_H_ci_pair_hi": cis["H_pair"][1] if cis["H_pair"] else None,
            "net_advantage_delta_ci_pair_lo": cis["delta_pair"][0] if cis["delta_pair"] else None,
            "net_advantage_delta_ci_pair_hi": cis["delta_pair"][1] if cis["delta_pair"] else None,
            "bootstrap_iters": int(args.bootstrap_iters),
            "bootstrap_unit": bootstrap_unit_used,
            "mean_confidence": float(mean_conf) if np.isfinite(mean_conf) else None,
            "mean_margin": float(mean_margin) if np.isfinite(mean_margin) else None,
            "gen_left_rate_main": float(gen_left_rate) if np.isfinite(gen_left_rate) else None,
            "H_gen_left": H_left,
            "H_gen_right": H_right,
            "position_bias_abs": float(pos_bias_abs) if np.isfinite(pos_bias_abs) else None,
            "position_bias_abs_raw": float(pos_bias_abs_raw) if np.isfinite(pos_bias_abs_raw) else None,
            "H_gen_left_raw": H_left_raw,
            "H_gen_right_raw": H_right_raw,
            "position_bias_pass": bool(pos_bias_pass),
            "swap_pairs_n": int(swap_total),
            "swap_consistency": float(swap_consistency) if np.isfinite(swap_consistency) else None,
            "swap_consistency_pass": bool(swap_pass),
            "test_retest_pairs_n": int(retest_total),
            "test_retest_agreement": float(test_retest_agreement) if np.isfinite(test_retest_agreement) else None,
            "test_retest_soft_corr": float(test_retest_soft_corr) if np.isfinite(test_retest_soft_corr) else None,
            "test_retest_pass": bool(test_retest_pass),
            "control_rows_n": int(len(control_ok)),
            **controls,
            "judge_reliable_flag": bool(judge_reliable),
            "judge_reliability_reasons": "|".join(fail_reasons),
            "n": int(n_main),
            "ok": int(n_main_ok),
            "ok_rate": float(ok_rate_main),
            "more_human_A_rate": float(more_a_rate) if np.isfinite(more_a_rate) else None,
            "more_human_B_rate": float(more_b_rate) if np.isfinite(more_b_rate) else None,
            "ai_prob_A_mean": float(ai_prob_a_mean) if np.isfinite(ai_prob_a_mean) else None,
            "ai_prob_B_mean": float(ai_prob_b_mean) if np.isfinite(ai_prob_b_mean) else None,
            "ai_prob_B_minus_A": float(ai_prob_b_minus_a) if np.isfinite(ai_prob_b_minus_a) else None,
            "confidence_mean": float(mean_conf) if np.isfinite(mean_conf) else None,
            "normalized_raw_csv": str(norm_csv),
            "normalized_raw_jsonl": str(norm_jsonl),
            "normalized_raw_parquet": str(norm_parquet) if norm_parquet else None,
        }

        if "dedupe_rows_before" in df.columns:
            v = pd.to_numeric(df["dedupe_rows_before"], errors="coerce").dropna()
            row["dedupe_rows_before"] = float(v.iloc[0]) if v.size else None
        if "dedupe_rows_after" in df.columns:
            v = pd.to_numeric(df["dedupe_rows_after"], errors="coerce").dropna()
            row["dedupe_rows_after"] = float(v.iloc[0]) if v.size else None
        if "dedupe_drop_n" in df.columns:
            v = pd.to_numeric(df["dedupe_drop_n"], errors="coerce").dropna()
            row["dedupe_drop_n"] = float(v.iloc[0]) if v.size else None
        if "dedupe_drop_rate" in df.columns:
            v = pd.to_numeric(df["dedupe_drop_rate"], errors="coerce").dropna()
            row["dedupe_drop_rate"] = float(v.iloc[0]) if v.size else None

        rows.append(row)

    out = pd.DataFrame(rows).sort_values(["gen_run", "file"]).reset_index(drop=True)

    if args.sanity_random_n > 0:
        sanity = sanity_random(args.sanity_random_n, args.bootstrap_seed)
        print("Sanity(random):", json.dumps(sanity, indent=2))

    Path(args.out_csv).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out_csv, index=False)

    print(out.to_string(index=False))
    print(f"\nWrote: {args.out_csv}")


if __name__ == "__main__":
    main()
