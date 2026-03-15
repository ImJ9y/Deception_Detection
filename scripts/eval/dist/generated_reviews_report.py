#!/usr/bin/env python3
import os
import re
import json
import argparse
import pandas as pd
import numpy as np

DEVANAGARI_RE = re.compile(r"[\u0900-\u097F]")
LATIN_RE = re.compile(r"[A-Za-z]")

# Drift indicator only (not a true Hindi detector)
# NOTE: These are Devanagari tokens that often appear in Hindi / Hindi-influenced outputs.
HINDIISH_RE = re.compile(r"\b(बहुत|अच्छा|अभी|नहीं|है|था|कर|रहा|रही|रहे)\b")

def normalize_ws(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").strip())

def sent_count(s: str) -> int:
    s = normalize_ws(s)
    if not s:
        return 0
    parts = re.split(r"[।\.\?\!]+", s)
    return len([p for p in parts if p.strip()])

def word_count(s: str) -> int:
    s = normalize_ws(s)
    if not s:
        return 0
    return len(re.findall(r"\S+", s))

def has_latin(s: str) -> int:
    return 1 if LATIN_RE.search(s or "") else 0

def has_devanagari(s: str) -> int:
    return 1 if DEVANAGARI_RE.search(s or "") else 0

def hindiish_any(s: str) -> int:
    """Row-level: 1 if any hindi-ish token appears, else 0."""
    return 1 if HINDIISH_RE.search(s or "") else 0

def hindiish_count(s: str) -> int:
    """Count of hindi-ish token matches in the text."""
    return len(HINDIISH_RE.findall(s or ""))

def find_real_text_col(df: pd.DataFrame) -> str:
    for c in ["review_text", "text", "Review", "comment", "sentence"]:
        if c in df.columns:
            return c
    raise RuntimeError(f"Could not find a text column in real df. Columns={list(df.columns)}")

def safe_series(df: pd.DataFrame, col: str) -> pd.Series:
    if col not in df.columns:
        return pd.Series([""] * len(df))
    return df[col].fillna("").astype(str).map(normalize_ws)

def summarize_text(name: str, series: pd.Series) -> dict:
    """
    Summaries include:
      - hindiish_row_rate: fraction of rows containing >=1 hindi-ish token
      - hindiish_token_fraction: total hindi-ish matches / total word tokens
      - hindiish_per_100w: 100 * total hindi-ish matches / total word tokens
      - hindiish_p95_per_review: 95th percentile of hindi-ish match counts per review
      - hindiish_ge3_rate: fraction of rows with >=3 hindi-ish tokens (tail indicator)
    """
    wc = series.map(word_count)
    sc = series.map(sent_count)
    lat = series.map(has_latin)
    dev = series.map(has_devanagari)

    hin_any = series.map(hindiish_any)
    hin_cnt = series.map(hindiish_count)

    total_words = int(wc.sum())
    total_hin = int(hin_cnt.sum())

    # Avoid divide-by-zero
    token_frac = (total_hin / total_words) if total_words > 0 else 0.0
    per_100w = (100.0 * total_hin / total_words) if total_words > 0 else 0.0

    return {
        "name": name,
        "n": int(len(series)),
        "wc_mean": float(wc.mean()),
        "wc_p50": float(wc.median()),
        "wc_p90": float(wc.quantile(0.90)),
        "sc_mean": float(sc.mean()),
        "latin_any_rate": float(lat.mean()),
        "devanagari_any_rate": float(dev.mean()),

        # Hindi-ish drift (more informative bundle)
        "hindiish_row_rate": float(hin_any.mean()),
        "hindiish_token_fraction": float(token_frac),
        "hindiish_per_100w": float(per_100w),
        "hindiish_p95_per_review": float(hin_cnt.quantile(0.95)) if len(hin_cnt) else 0.0,
        "hindiish_ge3_rate": float((hin_cnt >= 3).mean()) if len(hin_cnt) else 0.0,

        # Helpful for debugging / reproducibility
        "total_words": int(total_words),
        "total_hindiish_matches": int(total_hin),
    }

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gen_csv", required=True, help="Generated CSV (e.g., qwen run output)")
    ap.add_argument("--real_train_csv", required=True)
    ap.add_argument("--real_test_csv", required=True)
    ap.add_argument("--out_dir", required=True, help="Directory to write report.json + helper CSVs")
    ap.add_argument("--gen_text_col", default="Review")
    ap.add_argument("--gen_score_col", default="Review_Score")
    ap.add_argument("--gen_sentiment_col", default="sentiment")
    ap.add_argument("--gen_status_col", default="status")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    gen = pd.read_csv(args.gen_csv)
    real_train = pd.read_csv(args.real_train_csv)
    real_test = pd.read_csv(args.real_test_csv)

    real_train_col = find_real_text_col(real_train)
    real_test_col = find_real_text_col(real_test)

    gen_text = safe_series(gen, args.gen_text_col)
    real_train_text = safe_series(real_train, real_train_col)
    real_test_text = safe_series(real_test, real_test_col)

    # --- Integrity ---
    # NOTE: gen_text already fillna("") so empty_rate is not meaningful; keep it but base on raw col.
    if args.gen_text_col in gen.columns:
        empty_rate = float(gen[args.gen_text_col].isna().mean())
    else:
        empty_rate = 1.0
    blank_rate = float((gen_text.str.strip() == "").mean())

    # score checks
    score = gen.get(args.gen_score_col, pd.Series([""] * len(gen)))
    score_is_intlike = score.apply(lambda x: str(x).strip().lstrip("-").isdigit())
    score_int_rate = float(score_is_intlike.mean())
    score_num = pd.to_numeric(score, errors="coerce")
    score_nan_rate = float(score_num.isna().mean())
    score_min = float(score_num.min()) if score_nan_rate < 1.0 else None
    score_max = float(score_num.max()) if score_nan_rate < 1.0 else None

    # optional columns
    status_counts = gen[args.gen_status_col].value_counts().to_dict() if args.gen_status_col in gen.columns else {}
    sentiment_counts = gen[args.gen_sentiment_col].value_counts().to_dict() if args.gen_sentiment_col in gen.columns else {}

    # --- Distribution summaries ---
    summary_real_test = summarize_text("REAL_test", real_test_text)
    summary_gen = summarize_text("GEN", gen_text)

    # Add deltas vs REAL_test for Hindi-ish drift metrics (so you can compare runs fairly)
    def add_deltas(gen_sum: dict, real_sum: dict) -> dict:
        out = {}
        for k in [
            "hindiish_row_rate",
            "hindiish_token_fraction",
            "hindiish_per_100w",
            "hindiish_p95_per_review",
            "hindiish_ge3_rate",
        ]:
            if k in gen_sum and k in real_sum:
                out[f"{k}_delta_vs_real_test"] = float(gen_sum[k] - real_sum[k])
        return out

    hindiish_deltas = add_deltas(summary_gen, summary_real_test)

    # --- Duplicates in GEN ---
    dup_mask = gen_text.duplicated(keep=False)
    dup_rate = float(gen_text.duplicated().mean())
    dups = gen.loc[dup_mask, :].copy()
    if len(dups) > 0:
        dups["_text_norm"] = gen_text[dup_mask].values
        top_dups = (
            dups.groupby("_text_norm")
            .size()
            .reset_index(name="count")
            .sort_values("count", ascending=False)
        )
        top_dups.to_csv(os.path.join(args.out_dir, "duplicates_groups.csv"), index=False)
        dups.to_csv(os.path.join(args.out_dir, "duplicates_rows.csv"), index=False)

    # --- Leakage exact matches ---
    train_set = set(real_train_text.tolist())
    test_set = set(real_test_text.tolist())

    leak_train_mask = gen_text.isin(train_set)
    leak_test_mask = gen_text.isin(test_set)

    leak_train_rate = float(leak_train_mask.mean())
    leak_test_rate = float(leak_test_mask.mean())

    if leak_train_rate > 0:
        gen.loc[leak_train_mask, :].to_csv(os.path.join(args.out_dir, "leakage_matches_train.csv"), index=False)
    if leak_test_rate > 0:
        gen.loc[leak_test_mask, :].to_csv(os.path.join(args.out_dir, "leakage_matches_test.csv"), index=False)

    # --- Row-level flags (useful for dashboards) ---
    flags = gen.copy()
    flags["_text_norm"] = gen_text
    flags["_wc"] = gen_text.map(word_count)
    flags["_sc"] = gen_text.map(sent_count)
    flags["_has_latin"] = gen_text.map(has_latin)
    flags["_has_devanagari"] = gen_text.map(has_devanagari)

    # Hindi-ish row + count (keep both)
    flags["_hindiish_any"] = gen_text.map(hindiish_any)
    flags["_hindiish_count"] = gen_text.map(hindiish_count)

    flags["_score_num"] = score_num
    flags["_score_intlike"] = score_is_intlike.astype(int)
    flags.to_csv(os.path.join(args.out_dir, "row_flags.csv"), index=False)

    report = {
        "gen_csv": args.gen_csv,
        "integrity": {
            "rows": int(len(gen)),
            "empty_rate": empty_rate,
            "blank_rate": blank_rate,
            "score_intlike_rate": score_int_rate,
            "score_nan_rate": score_nan_rate,
            "score_min": score_min,
            "score_max": score_max,
            "status_counts": status_counts,
            "sentiment_counts": sentiment_counts,
        },
        "distribution": {
            "real_test": summary_real_test,
            "gen": summary_gen,
            "deltas_vs_real_test": hindiish_deltas,
        },
        "duplicates": {
            "gen_exact_duplicate_rate": dup_rate,
        },
        "leakage": {
            "gen_exact_match_rate_vs_real_train": leak_train_rate,
            "gen_exact_match_rate_vs_real_test": leak_test_rate,
        },
    }

    out_json = os.path.join(args.out_dir, "report.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print("\n=== SANITY REPORT ===")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"\nWrote:")
    print(f"  - {out_json}")
    print(f"  - {os.path.join(args.out_dir,'row_flags.csv')}")
    if len(dups) > 0:
        print(f"  - {os.path.join(args.out_dir,'duplicates_groups.csv')}")
        print(f"  - {os.path.join(args.out_dir,'duplicates_rows.csv')}")
    if leak_train_rate > 0:
        print(f"  - {os.path.join(args.out_dir,'leakage_matches_train.csv')}")
    if leak_test_rate > 0:
        print(f"  - {os.path.join(args.out_dir,'leakage_matches_test.csv')}")

if __name__ == "__main__":
    main()
