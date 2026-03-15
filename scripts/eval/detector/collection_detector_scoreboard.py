#!/usr/bin/env python3
import argparse
import json
from pathlib import Path
import pandas as pd


DEFAULT_COLS = [
    "gen_run",
    "exp_tag",
    "train_real_n",
    "train_gen_n",
    "test_real_n",
    "test_gen_n",
    "roc_auc",
    "acc",
    "f1",
    "precision",
    "recall",
    "balanced_acc",
    "prob_mean",
    "prob_std",
    "filter_mode_real_test",
    "report_json",
]


def infer_exp_tag(root: Path, report_json: Path) -> str:
    """
    Supports both:
    - <reports_root>/<run_name>/report.json
    - <reports_root>/<exp_tag>/<run_name>/report.json
    """
    rel_parent = report_json.parent.relative_to(root)
    parts = rel_parent.parts
    if len(parts) >= 2:
        return str(parts[0])
    return ""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reports_root", required=True, help="Directory containing per-run subdirs with report.json")
    ap.add_argument("--out_csv", required=True)
    args = ap.parse_args()

    root = Path(args.reports_root)
    if not root.exists():
        raise RuntimeError(f"--reports_root not found: {root}")

    rows = []
    report_paths = sorted(root.glob("**/report.json"))
    for rp in report_paths:
        obj = json.load(open(rp, "r", encoding="utf-8"))
        run = obj.get("run_name", rp.parent.name)
        exp_tag = infer_exp_tag(root, rp)

        sizes = obj.get("sizes", {})
        m = obj.get("metrics", {})

        rows.append({
            "gen_run": run,
            "exp_tag": exp_tag,
            "train_real_n": sizes.get("train_real_n"),
            "train_gen_n": sizes.get("train_gen_n"),
            "test_real_n": sizes.get("test_real_n"),
            "test_gen_n": sizes.get("test_gen_n"),
            "roc_auc": m.get("roc_auc"),
            "acc": m.get("acc"),
            "f1": m.get("f1"),
            "precision": m.get("precision"),
            "recall": m.get("recall"),
            "balanced_acc": m.get("balanced_acc"),
            "prob_mean": m.get("prob_mean"),
            "prob_std": m.get("prob_std"),
            "filter_mode_real_test": obj.get("filter_mode_real_test"),
            "report_json": str(rp),
        })

    if rows:
        df = pd.DataFrame(rows)
        sort_cols = [c for c in ["gen_run", "exp_tag"] if c in df.columns]
        if sort_cols:
            df = df.sort_values(sort_cols).reset_index(drop=True)
    else:
        df = pd.DataFrame(columns=DEFAULT_COLS)

    Path(args.out_csv).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out_csv, index=False)
    if len(df):
        print(df.to_string(index=False))
    else:
        print("No detector reports found.")
    print("Wrote:", args.out_csv)


if __name__ == "__main__":
    main()
