#!/usr/bin/env python3
import argparse
import json
import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd


FEATURE_SCORE_KEYS = [
    "human_naturalness",
    "human_specificity",
    "human_lived_experience",
    "human_coherence",
    "ai_templating",
    "ai_repetition",
    "ai_translationese",
    "ai_genericity",
]


def safe_str(x: Any) -> str:
    if x is None:
        return ""
    try:
        return str(x)
    except Exception:
        return repr(x)


def safe_float(x: Any) -> Optional[float]:
    try:
        if x is None:
            return None
        v = float(x)
        if pd.isna(v):
            return None
        return v
    except Exception:
        return None


def parse_json_obj(raw: Any) -> Dict[str, Any]:
    s = safe_str(raw).strip()
    if not s:
        return {}
    try:
        obj = json.loads(s)
        return obj if isinstance(obj, dict) else {}
    except Exception:
        return {}


def detect_key(df: pd.DataFrame) -> str:
    for c in ["single_item_id", "judge_row_id"]:
        if c in df.columns:
            return c
    raise RuntimeError(f"Could not infer join key for judged single CSV. Columns: {list(df.columns)}")


def load_backend(path: str, backend: str) -> Tuple[str, pd.DataFrame, pd.DataFrame]:
    df = pd.read_csv(path)
    key_col = detect_key(df)

    base_cols = [
        key_col,
        "single_item_id",
        "source_type",
        "run_id",
        "gen_run",
        "target_id",
        "gen_id",
        "real_id",
        "sentiment",
        "domain",
        "review_text",
        "Review",
    ]
    base_keep: List[str] = []
    for c in base_cols:
        if c in df.columns and c not in base_keep:
            base_keep.append(c)
    base = df[base_keep].copy()

    keep = [
        key_col,
        "judge_status",
        "judge_label",
        "judge_ai_probability_mean",
        "judge_confidence_mean",
        "judge_rationale",
        "judge_feature_scores_json",
        "judge_raw_json",
    ]
    for k in FEATURE_SCORE_KEYS:
        keep.append(f"judge_feat_{k}_mean")

    m = df[[c for c in keep if c in df.columns]].copy()
    rename = {}
    for c in list(m.columns):
        if c == key_col:
            continue
        rename[c] = f"{backend}_{c}"
    m = m.rename(columns=rename)
    return key_col, base, m


def merge_base_rows(left: pd.DataFrame, right: pd.DataFrame, key_col: str) -> pd.DataFrame:
    merged = left.merge(right, on=key_col, how="outer", suffixes=("", "__new"))
    for c in list(merged.columns):
        if not c.endswith("__new"):
            continue
        base = c[:-5]
        if base in merged.columns:
            merged[base] = merged[base].where(merged[base].notna() & (merged[base].astype(str) != ""), merged[c])
        else:
            merged[base] = merged[c]
        merged = merged.drop(columns=[c])
    return merged


def pick_examples(df: pd.DataFrame, n_per_source_type: int, seed: int) -> pd.DataFrame:
    if df.empty:
        return df
    rng = random.Random(seed)
    work = df.copy()

    label_cols = [c for c in work.columns if c.endswith("_judge_label")]
    conf_cols = [c for c in work.columns if c.endswith("_judge_confidence_mean")]
    status_cols = [c for c in work.columns if c.endswith("_judge_status")]

    if len(label_cols) >= 2:
        a, b = label_cols[0], label_cols[1]
        la = work[a].fillna("").astype(str).str.strip().str.upper()
        lb = work[b].fillna("").astype(str).str.strip().str.upper()
        work["__disagree"] = la.ne(lb) & la.ne("") & lb.ne("")
    else:
        work["__disagree"] = False

    if conf_cols:
        conf_num = work[conf_cols].apply(pd.to_numeric, errors="coerce")
        work["__max_conf"] = conf_num.max(axis=1)
    else:
        work["__max_conf"] = -1.0

    if status_cols:
        ok_any = work[status_cols].apply(
            lambda col: col.fillna("").astype(str).str.upper().eq("OK"),
            axis=0,
        ).any(axis=1)
        work = work[ok_any].copy()
        if work.empty:
            return work

    source_series = work.get("source_type", pd.Series(["unknown"] * len(work), index=work.index)).fillna("unknown")
    source_series = source_series.astype(str).str.lower()
    work["__source_norm"] = source_series
    work["__rand"] = [rng.random() for _ in range(len(work))]

    picked_idx: List[Any] = []
    for src in sorted(work["__source_norm"].dropna().unique().tolist()):
        grp = work[work["__source_norm"] == src].copy()
        if grp.empty:
            continue
        grp = grp.sort_values(["__disagree", "__max_conf", "__rand"], ascending=[False, False, True])
        picked_idx.extend(grp.head(max(0, int(n_per_source_type))).index.tolist())

    out = work.loc[picked_idx].copy()
    out = out.sort_values(["__source_norm", "__disagree", "__max_conf"], ascending=[True, False, False]).reset_index(drop=True)
    return out


def format_feature_line(row: pd.Series, backend: str) -> str:
    vals: List[str] = []
    for k in FEATURE_SCORE_KEYS:
        col = f"{backend}_judge_feat_{k}_mean"
        v = safe_float(row.get(col, None))
        if v is None:
            continue
        vals.append(f"{k}={v:.2f}")
    if not vals:
        payload = parse_json_obj(row.get(f"{backend}_judge_feature_scores_json", ""))
        for k in FEATURE_SCORE_KEYS:
            v = safe_float(payload.get(k))
            if v is None:
                continue
            vals.append(f"{k}={v:.2f}")
    return ", ".join(vals)


def maybe_trim(text: str, max_chars: int) -> str:
    t = safe_str(text).strip()
    if max_chars <= 0 or len(t) <= max_chars:
        return t
    return t[: max(0, max_chars - 3)].rstrip() + "..."


def render_md(df: pd.DataFrame, out_md: Path, language: str, run_name: str, max_review_chars: int) -> None:
    lines: List[str] = []
    lines.append(f"# Judge Interpretability Examples ({language})")
    lines.append("")
    lines.append(f"- Run: `{run_name}`")
    lines.append(f"- Examples: `{len(df)}`")
    lines.append("")

    if df.empty:
        lines.append("No examples found.")
    else:
        for i, row in df.iterrows():
            sid = safe_str(row.get("single_item_id", row.get("judge_row_id", f"row_{i+1}")))
            src = safe_str(row.get("source_type", ""))
            tid = safe_str(row.get("target_id", ""))
            review = maybe_trim(safe_str(row.get("review_text", row.get("Review", ""))), max_review_chars)
            lines.append(f"## Example {i+1}: `{sid}`")
            lines.append("")
            lines.append(f"- source_type: `{src}`")
            if tid:
                lines.append(f"- target_id: `{tid}`")
            lines.append("")
            lines.append("Review:")
            lines.append("```text")
            lines.append(review)
            lines.append("```")
            lines.append("")

            for backend in ["qwen", "llama"]:
                label = safe_str(row.get(f"{backend}_judge_label", "")).strip()
                ai_p = safe_float(row.get(f"{backend}_judge_ai_probability_mean", None))
                conf = safe_float(row.get(f"{backend}_judge_confidence_mean", None))
                rat = safe_str(row.get(f"{backend}_judge_rationale", "")).strip()
                feat_line = format_feature_line(row, backend)
                if not label and ai_p is None and not rat and not feat_line:
                    continue
                lines.append(f"### {backend.upper()} Judge")
                lines.append("")
                stat_parts = []
                if label:
                    stat_parts.append(f"label={label}")
                if ai_p is not None:
                    stat_parts.append(f"ai_probability={ai_p:.3f}")
                if conf is not None:
                    stat_parts.append(f"confidence={conf:.3f}")
                if stat_parts:
                    lines.append(f"- Result: `{', '.join(stat_parts)}`")
                if rat:
                    lines.append(f"- Explanation: {rat}")
                if feat_line:
                    lines.append(f"- Feature scores: {feat_line}")
                lines.append("")

    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser(description="Build interpretable judged examples from single-review judge outputs.")
    ap.add_argument("--single_csv_qwen", default="")
    ap.add_argument("--single_csv_llama", default="")
    ap.add_argument("--out_csv", required=True)
    ap.add_argument("--out_md", default="")
    ap.add_argument("--language", default="unknown")
    ap.add_argument("--run_name", default="")
    ap.add_argument("--examples_per_source_type", type=int, default=3)
    ap.add_argument("--max_review_chars", type=int, default=700)
    ap.add_argument("--seed", type=int, default=12345)
    args = ap.parse_args()

    sources: List[Tuple[str, str]] = []
    if args.single_csv_qwen and Path(args.single_csv_qwen).exists():
        sources.append(("qwen", args.single_csv_qwen))
    if args.single_csv_llama and Path(args.single_csv_llama).exists():
        sources.append(("llama", args.single_csv_llama))
    if not sources:
        raise RuntimeError("No single judged CSVs were found. Provide at least one existing --single_csv_qwen/--single_csv_llama.")

    key_col: Optional[str] = None
    base_df: Optional[pd.DataFrame] = None
    merged_df: Optional[pd.DataFrame] = None

    for backend, path in sources:
        k, b, m = load_backend(path, backend)
        if key_col is None:
            key_col = k
        elif key_col != k:
            raise RuntimeError(f"Mismatched merge keys across backends: {key_col} vs {k}")

        if base_df is None:
            base_df = b
        else:
            base_df = merge_base_rows(base_df, b, key_col)

        if merged_df is None:
            merged_df = m
        else:
            merged_df = merged_df.merge(m, on=key_col, how="outer")

    assert key_col is not None
    assert base_df is not None
    assert merged_df is not None

    df = base_df.merge(merged_df, on=key_col, how="outer")

    if "single_item_id" not in df.columns:
        df["single_item_id"] = df[key_col]
    if "review_text" not in df.columns and "Review" in df.columns:
        df["review_text"] = df["Review"]

    picked = pick_examples(df, n_per_source_type=args.examples_per_source_type, seed=args.seed).copy()
    picked.insert(0, "language", safe_str(args.language))
    picked.insert(1, "run_name", safe_str(args.run_name))

    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    picked.to_csv(out_csv, index=False)

    out_md = Path(args.out_md) if args.out_md else out_csv.with_suffix(".md")
    render_md(
        picked,
        out_md=out_md,
        language=safe_str(args.language),
        run_name=safe_str(args.run_name),
        max_review_chars=max(0, int(args.max_review_chars)),
    )

    print(f"Wrote examples CSV: {out_csv}")
    print(f"Wrote examples Markdown: {out_md}")


if __name__ == "__main__":
    main()
