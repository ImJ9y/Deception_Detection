#!/usr/bin/env python3
import argparse
from pathlib import Path
from typing import Tuple

import pandas as pd


def baseline_names(model_family: str) -> Tuple[str, str]:
    mf = (model_family or "").lower()
    if mf == "qwen":
        return "qwen3_fs5_nepali_v1", "qwen3_zs_nepali_v1"
    if mf == "llama":
        return "llama31_fs5_nepali_v1", "llama31_zs_nepali_v1"
    return "", ""


def infer_model_family(run_name: str) -> str:
    rn = run_name.lower()
    if "qwen" in rn:
        return "qwen"
    if "llama" in rn:
        return "llama"
    return "unknown"


def main() -> None:
    ap = argparse.ArgumentParser(description="Write run-vs-baselines comparison from master scoreboard")
    ap.add_argument("--master_csv", required=True)
    ap.add_argument("--run_name", required=True)
    ap.add_argument("--model_family", default="")
    ap.add_argument("--out_csv", required=True)
    args = ap.parse_args()

    df = pd.read_csv(args.master_csv)
    if "gen_run" not in df.columns:
        raise RuntimeError(f"master_csv missing gen_run. Columns={list(df.columns)}")

    model_family = args.model_family.strip().lower() or infer_model_family(args.run_name)
    fs5_base, zs_base = baseline_names(model_family)

    keep = [args.run_name]
    if fs5_base:
        keep.append(fs5_base)
    if zs_base:
        keep.append(zs_base)

    out = df[df["gen_run"].astype(str).isin(keep)].copy()
    if out.empty:
        raise RuntimeError(f"No rows found for run/baselines in master_csv. keep={keep}")

    wanted_order = {n: i for i, n in enumerate(keep)}
    out["__ord"] = out["gen_run"].map(lambda x: wanted_order.get(str(x), 99))
    out = out.sort_values(["__ord", "gen_run"]).drop(columns=["__ord"])

    # Add delta columns vs FS5 and ZS for key numeric metrics.
    numeric_cols = [
        c
        for c in out.columns
        if c not in {"gen_run"}
        and pd.api.types.is_numeric_dtype(out[c])
    ]

    ref_map = {}
    for ref_name in [fs5_base, zs_base]:
        if ref_name and ref_name in set(out["gen_run"].astype(str)):
            ref_map[ref_name] = out[out["gen_run"].astype(str) == ref_name].iloc[0]

    missing_baselines = [r for r in [fs5_base, zs_base] if r and r not in ref_map]
    if missing_baselines:
        print(f"WARNING: missing baseline rows in master scoreboard: {missing_baselines}")

    run_mask = out["gen_run"].astype(str) == args.run_name
    if run_mask.any():
        ridx = out[run_mask].index[0]
        for ref_name, ref_row in ref_map.items():
            suffix = "fs5" if "fs5" in ref_name else "zs"
            for c in numeric_cols:
                if pd.isna(out.at[ridx, c]) or pd.isna(ref_row[c]):
                    out.at[ridx, f"delta_vs_{suffix}__{c}"] = None
                else:
                    out.at[ridx, f"delta_vs_{suffix}__{c}"] = float(out.at[ridx, c] - ref_row[c])

    Path(args.out_csv).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out_csv, index=False)

    print(out.to_string(index=False))
    print(f"\nWrote: {args.out_csv}")


if __name__ == "__main__":
    main()
