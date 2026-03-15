#!/usr/bin/env python3
import argparse
import datetime as dt
import hashlib
import json
import random
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd


EVAL_STAGES_PER_RUN = ["register", "validate", "subset", "dist", "judge", "detector"]


@dataclass
class RunSpec:
    backend: str
    variant: str
    run_name: str
    gen_csv: Path
    prompt_config: Path
    prompt_lang: str


def run_cmd(cmd: List[str]) -> None:
    print("\n$", " ".join(cmd))
    subprocess.run(cmd, check=True)


def run_stage(args: List[str]) -> None:
    print("\n$", "stage", " ".join(args))
    old_argv = sys.argv[:]
    try:
        sys.argv = [__file__] + args
        _stage_main()
    finally:
        sys.argv = old_argv



# ---- Inlined stage execution library ----
STAGES = [
    "register",
    "validate",
    "subset",
    "dist",
    "judge",
    "detector",
    "master",
    "compare",
]

LANGUAGE_PRESETS: Dict[str, Dict[str, str]] = {
    # Keep only true path/data exceptions. Standard languages resolve dynamically.
    "nepali": {
        "dataset": "yt_nepali_movie_reviews",
        "dataset_root": "languages/nepali/datasets/yt_nepali_movie_reviews",
        "runs_root": "languages/nepali/runs/yt_nepali_movie_reviews",
    },
    "hausa": {
        # Preserve legacy shared runs root for Hausa.
        "runs_root": "languages/hausa/runs",
    },
    "hausa_hau_tts": {
        "dataset": "hau_tts",
        "dataset_root": "languages/hausa/datasets/hau_tts",
        "runs_root": "languages/hausa/runs/hau_tts",
    },
    "swahili": {
        # Preserve legacy misspelled on-disk directory.
        "dataset": "swahilli_reviews",
        "dataset_root": "languages/swahilli/datasets/swahilli_reviews",
        "runs_root": "languages/swahilli/runs",
    },
}

LANGUAGE_ALIASES: Dict[str, str] = {
    "swahilli": "swahili",
}


def utc_now_iso() -> str:
    return dt.datetime.utcnow().replace(microsecond=0).isoformat() + "Z"


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def read_json(path: Path, default: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    if not path.exists():
        return {} if default is None else dict(default)
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, obj: Dict[str, Any]) -> None:
    ensure_parent(path)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def sha256_file(path: Path) -> str:
    if not path.exists() or not path.is_file():
        return ""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(1024 * 1024)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def infer_model_family(run_name: str) -> str:
    rn = run_name.lower()
    if "qwen" in rn:
        return "qwen"
    if "llama" in rn:
        return "llama"
    return "unknown"


def infer_prompt_variant(run_name: str) -> str:
    rn = run_name.lower()
    if "fs5" in rn:
        return "fs5"
    if "zs" in rn:
        return "zs"
    return "custom"


def infer_language_label(dataset_root: Path, runs_root: Path) -> str:
    for p in [dataset_root, runs_root]:
        parts = [x.strip() for x in p.parts if x.strip()]
        if "languages" in parts:
            i = parts.index("languages")
            if i + 1 < len(parts):
                return parts[i + 1]
    return "unknown"


def infer_language_from_path_str(path_s: str) -> str:
    if not path_s:
        return ""
    parts = [x.strip() for x in Path(path_s).parts if x.strip()]
    if "languages" in parts:
        i = parts.index("languages")
        if i + 1 < len(parts):
            return parts[i + 1].lower()
    return ""


def language_token(language: str) -> str:
    raw = (language or "").strip().lower()
    token = re.sub(r"[^a-z0-9]+", "_", raw).strip("_")
    return token or "language"


def canonical_language(language: str) -> str:
    token = language_token(language)
    return LANGUAGE_ALIASES.get(token, token)


def resolve_language(language_arg: str, dataset_root_arg: str, runs_root_arg: str) -> str:
    raw = str(language_arg or "").strip().lower()
    if raw and raw != "auto":
        return canonical_language(raw)
    for cand in [dataset_root_arg, runs_root_arg]:
        g = infer_language_from_path_str(cand)
        if g:
            return canonical_language(g)
    return "hausa"


def preset_value(language: str, key: str, fallback: str = "") -> str:
    lang = canonical_language(language)
    values = default_preset_values(lang)
    values.update(LANGUAGE_PRESETS.get(lang, {}))
    return str(values.get(key, fallback))


def default_language_root(language: str) -> str:
    return f"languages/{canonical_language(language)}"


def default_dataset_name(language: str) -> str:
    return f"{canonical_language(language)}_reviews"


def default_dataset_root(language: str, dataset: str) -> str:
    return f"{default_language_root(language)}/datasets/{dataset}"


def default_runs_root(language: str, dataset: str) -> str:
    return f"{default_language_root(language)}/runs/{dataset}"


def default_preset_values(language: str) -> Dict[str, str]:
    dataset = default_dataset_name(language)
    return {
        "dataset": dataset,
        "dataset_root": default_dataset_root(language, dataset),
        "runs_root": default_runs_root(language, dataset),
        "judge_single_config": "configs/judge_single_config.json",
        "judge_pairwise_config": "configs/judge_pairwise_config.json",
    }


def default_control_template(language: str) -> str:
    lang = canonical_language(language)
    if lang == "hausa":
        return (
            "Wannan sharhi ne na samfurin rubutu. Ban kalli fim din ba amma ina maimaita jimloli iri daya. "
            "Fim din yana da kyau. Fim din yana da kyau. Fim din yana da kyau. "
            "Labarin yana da kyau. Labarin yana da kyau. Labarin yana da kyau. "
            "Aiki yana da kyau. Aiki yana da kyau. Aiki yana da kyau. "
            "Babu cikakkun bayanai na ainihin kwarewa ko scene. "
            "[TEMPLATE][GENERIC][NO_DETAIL]"
        )
    return (
        "This is a templated review. I did not directly experience the movie/product and I am repeating generic lines. "
        "It is good. It is good. It is good. "
        "The story is good. The story is good. The story is good. "
        "The acting is good. The acting is good. The acting is good. "
        "There are no concrete scenes, specific details, or lived-experience evidence. "
        "[TEMPLATE][GENERIC][NO_DETAIL]"
    )


def naming_warnings(run_name: str) -> List[str]:
    ws: List[str] = []
    if infer_model_family(run_name) == "unknown":
        ws.append("Could not infer model family from run_name; expected token like qwen/llama")
    if infer_prompt_variant(run_name) == "custom":
        ws.append("Could not infer prompt variant token (fs5/zs); treating as custom")
    return ws


def read_manifest(path: Path) -> pd.DataFrame:
    if not path.exists() or path.stat().st_size == 0:
        cols = [
            "run_name",
            "model_family",
            "prompt_variant",
            "dataset",
            "cohort",
            "exp_tag",
            "gen_csv",
            "subset_csv",
            "created_at",
            "updated_at",
            "status",
            "notes",
        ]
        return pd.DataFrame(columns=cols)
    return pd.read_csv(path)


def upsert_manifest_row(path: Path, row: Dict[str, Any]) -> None:
    df = read_manifest(path)
    if len(df):
        mask = (df["run_name"].astype(str) == str(row["run_name"])) & (df["cohort"].astype(str) == str(row["cohort"]))
    else:
        mask = pd.Series([], dtype=bool)

    if len(df) and mask.any():
        idx = df[mask].index[0]
        for k, v in row.items():
            df.at[idx, k] = v
    else:
        df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)

    ensure_parent(path)
    df.to_csv(path, index=False)


def expected_previous(stage: str) -> Optional[str]:
    i = STAGES.index(stage)
    if i == 0:
        return None
    return STAGES[i - 1]


def assert_stage_ready(state: Dict[str, Any], stage: str) -> None:
    prev = expected_previous(stage)
    if prev is None:
        return
    s = state.get("stage_status", {})
    if s.get(prev) != "done":
        raise RuntimeError(f"Stage '{stage}' requires previous stage '{prev}' to be done. Current '{prev}' status={s.get(prev)}")


def mark_stage(state: Dict[str, Any], stage: str, status: str, details: Dict[str, Any]) -> None:
    state.setdefault("stage_status", {})[stage] = status
    state.setdefault("stage_outputs", {})[stage] = details
    state["updated_at"] = utc_now_iso()


def parse_judge_backends(raw: str) -> List[str]:
    items = [x.strip().lower() for x in str(raw or "qwen").split(",") if x.strip()]
    if not items:
        items = ["qwen"]
    allowed = {"qwen", "llama"}
    bad = [x for x in items if x not in allowed]
    if bad:
        raise RuntimeError(f"Unsupported judge backend(s): {bad}. Allowed={sorted(allowed)}")
    out: List[str] = []
    seen = set()
    for x in items:
        if x not in seen:
            out.append(x)
            seen.add(x)
    return out


def replace_prompt_variant(run_name: str, variant: str) -> str:
    parts = str(run_name).split("_")
    for i, part in enumerate(parts):
        lower = part.lower()
        if lower == "zs" or lower.startswith("fs"):
            out = list(parts)
            out[i] = variant
            return "_".join(out)
    return ""


def baseline_run_names(model_family: str, language: str = "hausa", run_name: str = "") -> Tuple[str, str]:
    fs5_guess = replace_prompt_variant(run_name, "fs5")
    zs_guess = replace_prompt_variant(run_name, "zs")
    if fs5_guess or zs_guess:
        return fs5_guess or "", zs_guess or ""
    return ("", "")


def quality_flag_cols_for_language(language: str) -> List[str]:
    token = canonical_language(language)
    cols = [f"is_{token}_ok", "is_language_ok"]

    # Backward-compatibility aliases used in historical runs.
    if token == "swahili":
        cols = ["is_swahilli_ok", "is_swahili_ok"] + cols
    if token == "nepali":
        cols = ["is_devanagari_ok"] + cols

    out: List[str] = []
    seen = set()
    for col in cols:
        if col not in seen:
            out.append(col)
            seen.add(col)
    return out


def quality_rate_key_for_language(language: str) -> str:
    token = canonical_language(language)
    if token == "nepali":
        return "devanagari_ok_rate"
    return f"{token}_ok_rate"


def add_compat_quality_rate_aliases(stats: Dict[str, Any], language: str) -> Dict[str, Any]:
    out = dict(stats)
    lang = canonical_language(language)
    if lang == "swahili":
        if "swahili_ok_rate" in out:
            out["swahilli_ok_rate"] = out["swahili_ok_rate"]
        if "swahilli_ok_rate" in out:
            out["swahili_ok_rate"] = out["swahilli_ok_rate"]
    return out


def validate_gen_csv(gen_csv: Path, language: str) -> Dict[str, Any]:
    if not gen_csv.exists():
        raise RuntimeError(f"gen_csv does not exist: {gen_csv}")

    df = pd.read_csv(gen_csv)
    required = ["target_id", "Review", "sentiment", "status"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise RuntimeError(f"Missing required columns in gen_csv: {missing}")

    rows = len(df)
    target_id_nunique = int(pd.to_numeric(df["target_id"], errors="coerce").dropna().astype(int).nunique())
    ok_mask = df["status"].astype(str).isin(["OK", "OK_SALVAGED"])
    review = df["Review"].fillna("").astype(str)

    quality_cols = quality_flag_cols_for_language(language)
    quality_rate_key = quality_rate_key_for_language(language)
    quality_col = ""
    for c in quality_cols:
        if c in df.columns:
            quality_col = c
            break
    quality_series = pd.to_numeric(
        df.get(quality_col, pd.Series([None] * rows)) if quality_col else pd.Series([None] * rows),
        errors="coerce",
    ).fillna(0)
    quality_rate = float(quality_series.eq(1).mean()) if rows else 0.0
    blank_rate = float(review.str.strip().eq("").mean()) if rows else 0.0

    return {
        "rows": int(rows),
        "target_id_unique": target_id_nunique,
        "usable_rows": int(ok_mask.sum()),
        "usable_rate": float(ok_mask.mean()) if rows else 0.0,
        "blank_review_rate": blank_rate,
        quality_rate_key: quality_rate,
        "status_counts": df["status"].astype(str).value_counts().to_dict(),
        "sentiment_counts": df["sentiment"].astype(str).value_counts().to_dict(),
    }


def load_common_ids(common_ids_csv: Optional[Path]) -> Optional[set]:
    if common_ids_csv is None or not common_ids_csv.exists():
        return None
    cdf = pd.read_csv(common_ids_csv)
    id_col = "target_id" if "target_id" in cdf.columns else cdf.columns[0]
    ids = set(pd.to_numeric(cdf[id_col], errors="coerce").dropna().astype(int).tolist())
    return ids


def build_subset(gen_csv: Path, common_ids_csv: Path, out_csv: Path) -> Dict[str, Any]:
    gen = pd.read_csv(gen_csv)
    common_ids = load_common_ids(common_ids_csv)
    if common_ids is None:
        raise RuntimeError(f"common_ids_csv not found or empty: {common_ids_csv}")

    gen_tid = pd.to_numeric(gen["target_id"], errors="coerce").astype("Int64")
    sub = gen[gen_tid.isin(common_ids)].copy()

    ensure_parent(out_csv)
    sub.to_csv(out_csv, index=False)

    return {
        "subset_rows": int(len(sub)),
        "common_ids_n": int(len(common_ids)),
        "missing_common_ids_in_run": int(len(common_ids - set(pd.to_numeric(sub["target_id"], errors="coerce").dropna().astype(int).tolist()))),
        "subset_csv": str(out_csv),
    }


def resolve_real_test_df(real_test_csv: Path, domain_default: str = "movie") -> Tuple[pd.DataFrame, str]:
    rt = pd.read_csv(real_test_csv)

    if "target_id" in rt.columns:
        rt2 = rt.copy()
        rt2["target_id"] = pd.to_numeric(rt2["target_id"], errors="coerce").astype("Int64")
    elif "id" in rt.columns:
        rt2 = rt.copy()
        rt2["target_id"] = pd.to_numeric(rt2["id"], errors="coerce").astype("Int64")
    else:
        rt2 = rt.reset_index().rename(columns={"index": "target_id"})
        rt2["target_id"] = pd.to_numeric(rt2["target_id"], errors="coerce").astype("Int64")

    real_col = None
    for c in ["review_text", "text", "Review", "review"]:
        if c in rt2.columns:
            real_col = c
            break
    if real_col is None:
        raise RuntimeError(f"No real text column found in {real_test_csv}. Have {list(rt2.columns)}")

    if "sentiment" not in rt2.columns:
        if "label" in rt2.columns:
            rt2["sentiment"] = rt2["label"].map(lambda x: "POS" if str(x).strip() == "1" else "NEG")
        else:
            rt2["sentiment"] = ""

    if "domain" not in rt2.columns:
        rt2["domain"] = domain_default

    out = pd.DataFrame(
        {
            "target_id": pd.to_numeric(rt2["target_id"], errors="coerce").astype("Int64"),
            "real_id": pd.to_numeric(rt2["target_id"], errors="coerce").astype("Int64").astype(str),
            "review_text": rt2[real_col].fillna("").astype(str),
            "sentiment": rt2["sentiment"].fillna("").astype(str).str.upper(),
            "domain": rt2["domain"].fillna(domain_default).astype(str),
        }
    )
    out = out[out["target_id"].notna()].copy()
    out["target_id"] = out["target_id"].astype(int)
    out = out[out["review_text"].str.strip() != ""].reset_index(drop=True)
    return out, real_col


def load_real_reference_pool(real_test_csv: Path, common_ids_csv: Optional[Path], domain_default: str = "movie") -> Tuple[pd.DataFrame, str]:
    real_pool, real_text_col = resolve_real_test_df(real_test_csv, domain_default=domain_default)
    common_ids = load_common_ids(common_ids_csv)
    if common_ids is not None:
        real_pool = real_pool[real_pool["target_id"].isin(common_ids)].copy()
    real_pool = real_pool.reset_index(drop=True)
    if real_pool.empty:
        raise RuntimeError("Real reference pool is empty after filtering.")
    return real_pool, real_text_col


def dedupe_generated_for_judging(in_csv: Path, out_csv: Path, enabled: bool = True) -> Dict[str, Any]:
    df = pd.read_csv(in_csv)
    before = len(df)

    if enabled:
        if "Review" not in df.columns:
            raise RuntimeError("Cannot dedupe generated rows: missing Review column")
        out = df.drop_duplicates(subset=["Review"], keep="first").copy()
    else:
        out = df.copy()

    after = len(out)
    drop_n = int(before - after)
    drop_rate = float(drop_n / before) if before else 0.0

    out["dedupe_rows_before"] = before
    out["dedupe_rows_after"] = after
    out["dedupe_drop_n"] = drop_n
    out["dedupe_drop_rate"] = drop_rate

    ensure_parent(out_csv)
    out.to_csv(out_csv, index=False)

    return {
        "enabled": bool(enabled),
        "rows_before": int(before),
        "rows_after": int(after),
        "drop_n": drop_n,
        "drop_rate": drop_rate,
        "out_csv": str(out_csv),
    }


def _safe_int(x: Any) -> Optional[int]:
    try:
        if x is None or (isinstance(x, float) and pd.isna(x)):
            return None
        return int(float(x))
    except Exception:
        return None


def choose_control_n(preferred: Optional[int], fallback: int) -> int:
    p = _safe_int(preferred)
    fb = _safe_int(fallback) or 0
    if p is None or p <= 0:
        return max(0, fb)
    return max(0, p)


def build_balanced_left_flags(n: int, rng: random.Random, exact_balance: bool) -> List[bool]:
    if n <= 0:
        return []
    if not exact_balance:
        return [bool(rng.random() < 0.5) for _ in range(n)]
    n_left = n // 2
    n_right = n - n_left
    flags = [True] * n_left + [False] * n_right
    rng.shuffle(flags)
    return flags


def build_single_input(
    real_test_csv: Path,
    subset_gen_csv: Path,
    out_csv: Path,
    gen_run: str,
    common_ids_csv: Optional[Path],
    seed_base: int,
    domain_default: str,
    dedupe_meta: Dict[str, Any],
) -> Dict[str, Any]:
    gen = pd.read_csv(subset_gen_csv)
    real_pool, _ = load_real_reference_pool(real_test_csv, common_ids_csv, domain_default=domain_default)

    if "Review" not in gen.columns:
        raise RuntimeError(f"subset_gen_csv missing Review column: {subset_gen_csv}")

    source_hash_real = sha256_file(real_test_csv)
    source_hash_gen = sha256_file(subset_gen_csv)
    source_hash_common = sha256_file(common_ids_csv) if common_ids_csv else ""
    build_ts = utc_now_iso()

    rows: List[Dict[str, Any]] = []

    for i, row in gen.reset_index(drop=True).iterrows():
        gen_id = str(row.get("target_id", f"gen_{i}"))
        target_id = _safe_int(row.get("target_id"))
        sentiment = str(row.get("sentiment", "")).strip().upper()
        domain = str(row.get("domain", domain_default) or domain_default)
        review = str(row.get("Review", "") or "")
        if not review.strip():
            continue

        rows.append(
            {
                "single_item_id": f"{gen_run}__gen__{gen_id}",
                "source_type": "gen",
                "run_id": gen_run,
                "gen_run": gen_run,
                "target_id": target_id,
                "gen_id": gen_id,
                "real_id": "",
                "sentiment": sentiment,
                "domain": domain,
                "review_text": review,
                "Review": review,
                "judge_seed": int(seed_base + i * 1000),
                "build_timestamp_utc": build_ts,
                "source_hash_real_csv": source_hash_real,
                "source_hash_gen_csv": source_hash_gen,
                "source_hash_common_ids_csv": source_hash_common,
                "dedupe_rows_before": dedupe_meta.get("rows_before"),
                "dedupe_rows_after": dedupe_meta.get("rows_after"),
                "dedupe_drop_n": dedupe_meta.get("drop_n"),
                "dedupe_drop_rate": dedupe_meta.get("drop_rate"),
            }
        )

    offset = 1_000_000
    for i, row in real_pool.reset_index(drop=True).iterrows():
        target_id = _safe_int(row.get("target_id"))
        real_id = str(row.get("real_id", f"real_{i}"))
        review = str(row.get("review_text", "") or "")
        if not review.strip():
            continue

        rows.append(
            {
                "single_item_id": f"{gen_run}__real__{real_id}",
                "source_type": "real",
                "run_id": gen_run,
                "gen_run": gen_run,
                "target_id": target_id,
                "gen_id": "",
                "real_id": real_id,
                "sentiment": str(row.get("sentiment", "")).strip().upper(),
                "domain": str(row.get("domain", domain_default) or domain_default),
                "review_text": review,
                "Review": review,
                "judge_seed": int(seed_base + offset + i * 1000),
                "build_timestamp_utc": build_ts,
                "source_hash_real_csv": source_hash_real,
                "source_hash_gen_csv": source_hash_gen,
                "source_hash_common_ids_csv": source_hash_common,
                "dedupe_rows_before": dedupe_meta.get("rows_before"),
                "dedupe_rows_after": dedupe_meta.get("rows_after"),
                "dedupe_drop_n": dedupe_meta.get("drop_n"),
                "dedupe_drop_rate": dedupe_meta.get("drop_rate"),
            }
        )

    out = pd.DataFrame(rows)
    ensure_parent(out_csv)
    out.to_csv(out_csv, index=False)
    return {
        "rows": int(len(out)),
        "gen_rows": int((out["source_type"] == "gen").sum()),
        "real_rows": int((out["source_type"] == "real").sum()),
        "single_input_csv": str(out_csv),
    }


def build_pairwise_input(
    real_test_csv: Path,
    subset_gen_csv: Path,
    out_csv: Path,
    gen_run: str,
    common_ids_csv: Optional[Path],
    k_matches: int,
    match_seed: int,
    swap_fraction: float,
    retest_fraction: float,
    control_n_each: int,
    control_real_template_n: int,
    control_real_real_n: int,
    control_gen_gen_n: int,
    pairwise_debias_swap_all: bool,
    pair_exact_balance_lr: bool,
    judge_seed_base: int,
    domain_default: str,
    language: str,
    control_template_text: str,
    dedupe_meta: Dict[str, Any],
) -> Dict[str, Any]:
    gen = pd.read_csv(subset_gen_csv).reset_index(drop=True)
    real_pool, real_text_col = load_real_reference_pool(real_test_csv, common_ids_csv, domain_default=domain_default)

    if "Review" not in gen.columns:
        raise RuntimeError(f"subset_gen_csv missing Review column: {subset_gen_csv}")

    source_hash_real = sha256_file(real_test_csv)
    source_hash_gen = sha256_file(subset_gen_csv)
    source_hash_common = sha256_file(common_ids_csv) if common_ids_csv else ""
    build_ts = utc_now_iso()

    rng = random.Random(match_seed)
    rows: List[Dict[str, Any]] = []
    main_rows: List[Dict[str, Any]] = []

    def add_pair(
        *,
        pair_uid: str,
        base_pair_uid: str,
        pair_role: str,
        control_type: str,
        control_expected: str,
        sentiment: str,
        domain: str,
        gen_id: str,
        real_id: str,
        target_id: Optional[int],
        gen_review: str,
        real_review: str,
        is_gen_left: bool,
        shuffle_id: str,
        pair_order: Optional[int] = None,
    ) -> Dict[str, Any]:
        review_a = gen_review if is_gen_left else real_review
        review_b = real_review if is_gen_left else gen_review
        rec = {
            "pair_uid": pair_uid,
            "base_pair_uid": base_pair_uid,
            "pair_role": pair_role,
            "control_type": control_type,
            "control_expected": control_expected,
            "run_id": gen_run,
            "gen_run": gen_run,
            "gen_id": gen_id,
            "real_id": real_id,
            "target_id": target_id,
            "sentiment": sentiment,
            "domain": domain,
            "real_review": real_review,
            "gen_review": gen_review,
            "review_a": review_a,
            "review_b": review_b,
            "presented_left_text": review_a,
            "presented_right_text": review_b,
            "is_gen_left": bool(is_gen_left),
            "pair_order": pair_order if pair_order is not None else "",
            "judge_seed": int(judge_seed_base + len(rows) * 1000),
            "shuffle_id": shuffle_id,
            "build_timestamp_utc": build_ts,
            "source_hash_real_csv": source_hash_real,
            "source_hash_gen_csv": source_hash_gen,
            "source_hash_common_ids_csv": source_hash_common,
            "dedupe_rows_before": dedupe_meta.get("rows_before"),
            "dedupe_rows_after": dedupe_meta.get("rows_after"),
            "dedupe_drop_n": dedupe_meta.get("drop_n"),
            "dedupe_drop_rate": dedupe_meta.get("drop_rate"),
        }
        rows.append(rec)
        return rec

    # Main conceptual pairs: each generated row vs K real references in same sentiment bucket.
    k = max(1, int(k_matches))
    main_specs: List[Dict[str, Any]] = []
    for gi, g in gen.iterrows():
        g_review = str(g.get("Review", "") or "")
        if not g_review.strip():
            continue

        g_sent = str(g.get("sentiment", "")).strip().upper()
        g_domain = str(g.get("domain", domain_default) or domain_default)
        g_target_id = _safe_int(g.get("target_id"))
        g_id = str(g.get("target_id", f"gen_{gi}"))

        cand = real_pool
        if g_sent:
            cand2 = cand[cand["sentiment"].astype(str).str.upper() == g_sent]
            if len(cand2) > 0:
                cand = cand2
        cand = cand.reset_index(drop=True)

        if cand.empty:
            raise RuntimeError("No real references available for pairwise matching.")

        idxs = cand.index.tolist()
        if len(idxs) >= k:
            pick = rng.sample(idxs, k)
        else:
            pick = [rng.choice(idxs) for _ in range(k)]

        for kj, ridx in enumerate(pick):
            r = cand.loc[ridx]
            r_text = str(r.get("review_text", "") or "")
            if not r_text.strip():
                continue

            base_uid = f"{gen_run}__main__g{gi}__k{kj}"
            main_specs.append(
                {
                    "base_uid": base_uid,
                    "sentiment": g_sent,
                    "domain": g_domain,
                    "gen_id": g_id,
                    "real_id": str(r.get("real_id", "")),
                    "target_id": g_target_id,
                    "gen_review": g_review,
                    "real_review": r_text,
                }
            )

    if not main_specs:
        raise RuntimeError("Pairwise main pair construction produced zero rows.")

    main_flags = build_balanced_left_flags(len(main_specs), rng, pair_exact_balance_lr)
    main_rows: List[Dict[str, Any]] = []
    for i, spec in enumerate(main_specs):
        base_uid = str(spec["base_uid"])
        is_left = bool(main_flags[i])
        if pairwise_debias_swap_all:
            rec = add_pair(
                pair_uid=f"{base_uid}__o1",
                base_pair_uid=base_uid,
                pair_role="main",
                control_type="",
                control_expected="",
                sentiment=str(spec["sentiment"]),
                domain=str(spec["domain"]),
                gen_id=str(spec["gen_id"]),
                real_id=str(spec["real_id"]),
                target_id=_safe_int(spec.get("target_id")),
                gen_review=str(spec["gen_review"]),
                real_review=str(spec["real_review"]),
                is_gen_left=is_left,
                shuffle_id=f"main_seed_{match_seed}",
                pair_order=1,
            )
            add_pair(
                pair_uid=f"{base_uid}__o2",
                base_pair_uid=base_uid,
                pair_role="main",
                control_type="",
                control_expected="",
                sentiment=str(spec["sentiment"]),
                domain=str(spec["domain"]),
                gen_id=str(spec["gen_id"]),
                real_id=str(spec["real_id"]),
                target_id=_safe_int(spec.get("target_id")),
                gen_review=str(spec["gen_review"]),
                real_review=str(spec["real_review"]),
                is_gen_left=(not is_left),
                shuffle_id=f"main_seed_{match_seed}",
                pair_order=2,
            )
        else:
            rec = add_pair(
                pair_uid=base_uid,
                base_pair_uid=base_uid,
                pair_role="main",
                control_type="",
                control_expected="",
                sentiment=str(spec["sentiment"]),
                domain=str(spec["domain"]),
                gen_id=str(spec["gen_id"]),
                real_id=str(spec["real_id"]),
                target_id=_safe_int(spec.get("target_id")),
                gen_review=str(spec["gen_review"]),
                real_review=str(spec["real_review"]),
                is_gen_left=is_left,
                shuffle_id=f"main_seed_{match_seed}",
                pair_order=1,
            )
        # Keep order-1 main rows as anchors for swap/retest diagnostics.
        main_rows.append(rec)

    # Swap invariance subset.
    n_swap = max(0, int(round(float(swap_fraction) * len(main_rows))))
    if n_swap > 0:
        pick_swap = rng.sample(main_rows, min(n_swap, len(main_rows)))
        for base in pick_swap:
            add_pair(
                pair_uid=f"{base['pair_uid']}__swap",
                base_pair_uid=str(base["pair_uid"]),
                pair_role="swap",
                control_type="",
                control_expected="",
                sentiment=str(base["sentiment"]),
                domain=str(base["domain"]),
                gen_id=str(base["gen_id"]),
                real_id=str(base["real_id"]),
                target_id=_safe_int(base.get("target_id")),
                gen_review=str(base["gen_review"]),
                real_review=str(base["real_review"]),
                is_gen_left=(not bool(base["is_gen_left"])),
                shuffle_id=f"swap_seed_{match_seed}",
                pair_order="",
            )

    # Test-retest subset.
    n_retest = max(0, int(round(float(retest_fraction) * len(main_rows))))
    if n_retest > 0:
        pick_retest = rng.sample(main_rows, min(n_retest, len(main_rows)))
        for base in pick_retest:
            rec = add_pair(
                pair_uid=f"{base['pair_uid']}__retest",
                base_pair_uid=str(base["pair_uid"]),
                pair_role="retest",
                control_type="",
                control_expected="",
                sentiment=str(base["sentiment"]),
                domain=str(base["domain"]),
                gen_id=str(base["gen_id"]),
                real_id=str(base["real_id"]),
                target_id=_safe_int(base.get("target_id")),
                gen_review=str(base["gen_review"]),
                real_review=str(base["real_review"]),
                is_gen_left=bool(base["is_gen_left"]),
                shuffle_id=f"retest_seed_{match_seed}",
                pair_order="",
            )
            rec["judge_seed"] = int(base["judge_seed"]) + 17

    # Control injections. Template should be deliberately degenerate.
    template_ai = str(control_template_text or "").strip()
    if not template_ai:
        template_ai = default_control_template(language)

    def pick_real(sentiment: str = "") -> pd.Series:
        d = real_pool
        if sentiment:
            d2 = d[d["sentiment"].astype(str).str.upper() == sentiment.upper()]
            if len(d2) > 0:
                d = d2
        ridx = rng.choice(d.index.tolist())
        return d.loc[ridx]

    def pick_gen(sentiment: str = "") -> pd.Series:
        d = gen
        if sentiment and "sentiment" in d.columns:
            d2 = d[d["sentiment"].astype(str).str.upper() == sentiment.upper()]
            if len(d2) > 0:
                d = d2
        ridx = rng.choice(d.index.tolist())
        return d.loc[ridx]

    c_rt = choose_control_n(control_real_template_n, control_n_each)
    c_rr = choose_control_n(control_real_real_n, control_n_each)
    c_gg = choose_control_n(control_gen_gen_n, control_n_each)

    controls_rt: List[Dict[str, Any]] = []
    for i in range(c_rt):
        rr = pick_real()
        controls_rt.append(
            {
                "i": i,
                "sent": str(rr.get("sentiment", "")).strip().upper(),
                "dom": str(rr.get("domain", domain_default) or domain_default),
                "real_id": str(rr.get("real_id", "")),
                "target_id": _safe_int(rr.get("target_id")),
                "real_review": str(rr.get("review_text", "") or ""),
            }
        )
    for spec, is_left in zip(controls_rt, build_balanced_left_flags(len(controls_rt), rng, pair_exact_balance_lr)):
        i = int(spec["i"])
        add_pair(
            pair_uid=f"{gen_run}__control__real_vs_template__{i}",
            base_pair_uid=f"{gen_run}__control__real_vs_template__{i}",
            pair_role="control",
            control_type="real_vs_template",
            control_expected="real_win",
            sentiment=str(spec["sent"]),
            domain=str(spec["dom"]),
            gen_id=f"control_template_{i}",
            real_id=str(spec["real_id"]),
            target_id=_safe_int(spec.get("target_id")),
            gen_review=template_ai,
            real_review=str(spec["real_review"]),
            is_gen_left=bool(is_left),
            shuffle_id=f"control_seed_{match_seed}",
            pair_order="",
        )

    controls_rr: List[Dict[str, Any]] = []
    for i in range(c_rr):
        rr1 = pick_real()
        sent = str(rr1.get("sentiment", "")).strip().upper()
        rr2 = pick_real(sent)
        controls_rr.append(
            {
                "i": i,
                "sent": sent,
                "dom": str(rr1.get("domain", domain_default) or domain_default),
                "real_id": str(rr1.get("real_id", "")),
                "target_id": _safe_int(rr1.get("target_id")),
                "real_review": str(rr1.get("review_text", "") or ""),
                "gen_review": str(rr2.get("review_text", "") or ""),
            }
        )
    for spec, is_left in zip(controls_rr, build_balanced_left_flags(len(controls_rr), rng, pair_exact_balance_lr)):
        i = int(spec["i"])
        add_pair(
            pair_uid=f"{gen_run}__control__real_vs_real__{i}",
            base_pair_uid=f"{gen_run}__control__real_vs_real__{i}",
            pair_role="control",
            control_type="real_vs_real",
            control_expected="balanced",
            sentiment=str(spec["sent"]),
            domain=str(spec["dom"]),
            gen_id=f"control_real_b_{i}",
            real_id=str(spec["real_id"]),
            target_id=_safe_int(spec.get("target_id")),
            gen_review=str(spec["gen_review"]),
            real_review=str(spec["real_review"]),
            is_gen_left=bool(is_left),
            shuffle_id=f"control_seed_{match_seed}",
            pair_order="",
        )

    controls_gg: List[Dict[str, Any]] = []
    for i in range(c_gg):
        g1 = pick_gen()
        sent = str(g1.get("sentiment", "")).strip().upper()
        g2 = pick_gen(sent)
        controls_gg.append(
            {
                "i": i,
                "sent": sent,
                "dom": str(g1.get("domain", domain_default) or domain_default),
                "real_id": str(g1.get("target_id", f"control_gen_a_{i}")),
                "gen_id": str(g2.get("target_id", f"control_gen_b_{i}")),
                "target_id": _safe_int(g1.get("target_id")),
                "real_review": str(g1.get("Review", "") or ""),
                "gen_review": str(g2.get("Review", "") or ""),
            }
        )
    for spec, is_left in zip(controls_gg, build_balanced_left_flags(len(controls_gg), rng, pair_exact_balance_lr)):
        i = int(spec["i"])
        add_pair(
            pair_uid=f"{gen_run}__control__gen_vs_gen__{i}",
            base_pair_uid=f"{gen_run}__control__gen_vs_gen__{i}",
            pair_role="control",
            control_type="gen_vs_gen",
            control_expected="balanced",
            sentiment=str(spec["sent"]),
            domain=str(spec["dom"]),
            gen_id=str(spec["gen_id"]),
            real_id=str(spec["real_id"]),
            target_id=_safe_int(spec.get("target_id")),
            gen_review=str(spec["gen_review"]),
            real_review=str(spec["real_review"]),
            is_gen_left=bool(is_left),
            shuffle_id=f"control_seed_{match_seed}",
            pair_order="",
        )

    # Finalize pair_id ordering.
    for i, rec in enumerate(rows):
        rec["pair_id"] = i

    out = pd.DataFrame(rows)
    ensure_parent(out_csv)
    out.to_csv(out_csv, index=False)

    control_counts = out["control_type"].fillna("").astype(str).value_counts().to_dict()
    return {
        "rows": int(len(out)),
        "main_rows": int((out["pair_role"] == "main").sum()),
        "main_base_rows": int(len(main_specs)),
        "main_pair_orders": int(2 if pairwise_debias_swap_all else 1),
        "swap_rows": int((out["pair_role"] == "swap").sum()),
        "retest_rows": int((out["pair_role"] == "retest").sum()),
        "control_rows": int((out["pair_role"] == "control").sum()),
        "control_counts": control_counts,
        "is_gen_left_rate": float(out["is_gen_left"].mean()) if len(out) else 0.0,
        "pair_input_csv": str(out_csv),
        "real_text_col": real_text_col,
    }


def _stage_main() -> None:
    ap = argparse.ArgumentParser(description="Staged baseline-anchored eval pipeline")
    ap.add_argument("--stage", required=True, choices=STAGES)
    ap.add_argument("--run_name", required=True)
    ap.add_argument("--gen_csv", default="")

    ap.add_argument(
        "--language",
        default="auto",
        help="Language preset for defaults (auto|nepali|hausa|hausa_hau_tts|swahilli|swahili|korean|igbo)",
    )
    ap.add_argument("--dataset", default="")
    ap.add_argument("--cohort", default="common_subset_v1")
    ap.add_argument("--exp_tag", default="")

    ap.add_argument("--dataset_root", default="")
    ap.add_argument("--runs_root", default="")

    ap.add_argument("--model_family", default="")
    ap.add_argument("--prompt_variant", default="")

    ap.add_argument("--common_ids_csv", default="")
    ap.add_argument("--real_train_csv", default="")
    ap.add_argument("--real_test_csv", default="")

    ap.add_argument("--judge_single_config", default="")
    ap.add_argument("--judge_pairwise_config", default="")
    ap.add_argument("--judge_backends", default="qwen")
    ap.add_argument("--judge_n_samples", type=int, default=1)
    ap.add_argument("--judge_seed_base", type=int, default=12345)

    # Pair construction and reliability args.
    ap.add_argument("--pair_k_matches", type=int, default=3)
    ap.add_argument("--pair_match_seed", type=int, default=12345)
    ap.add_argument("--pair_swap_fraction", type=float, default=0.20)
    ap.add_argument("--pair_retest_fraction", type=float, default=0.20)
    ap.add_argument("--pair_control_n_each", type=int, default=12)
    ap.add_argument("--pair_control_real_template_n", type=int, default=60)
    ap.add_argument("--pair_control_real_real_n", type=int, default=60)
    ap.add_argument("--pair_control_gen_gen_n", type=int, default=60)
    ap.add_argument("--pairwise_debias_swap_all", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--pair_exact_balance_lr", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--pair_bootstrap_iters", type=int, default=2000)
    ap.add_argument("--pair_bootstrap_seed", type=int, default=12345)
    ap.add_argument("--pair_bootstrap_unit", default="cluster", choices=["cluster", "pair", "both"])

    ap.add_argument("--position_bias_max_abs", type=float, default=0.05)
    ap.add_argument("--swap_consistency_min", type=float, default=0.80)
    ap.add_argument("--test_retest_agreement_min", type=float, default=0.75)
    ap.add_argument("--control_real_template_min", type=float, default=0.80)
    ap.add_argument("--control_real_template_gen_max", type=float, default=0.05)
    ap.add_argument("--control_balanced_max_gap", type=float, default=0.15)
    ap.add_argument("--control_balanced_min_tie", type=float, default=0.20)

    ap.add_argument("--single_mu_real_drift_threshold", type=float, default=0.05)
    ap.add_argument("--judge_dedupe_exact", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--judge_sanity_random_n", type=int, default=0)
    ap.add_argument("--write_interpretability_examples", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--interpretability_examples_per_source_type", type=int, default=3)
    ap.add_argument("--interpretability_max_review_chars", type=int, default=700)
    ap.add_argument("--control_template_text", default="", help="Override template text for real-vs-template controls")

    ap.add_argument("--state_json", default="")
    ap.add_argument("--manifest_csv", default="")
    ap.add_argument("--comparison_csv", default="")
    args = ap.parse_args()

    stage = args.stage
    run_name = args.run_name
    stage_requires_gen_csv = {"register", "validate", "subset", "dist", "judge", "detector"}
    if stage in stage_requires_gen_csv and not str(args.gen_csv or "").strip():
        raise RuntimeError(f"Stage '{stage}' requires --gen_csv.")
    gen_csv = Path(args.gen_csv) if str(args.gen_csv or "").strip() else Path("")
    exp_tag = str(args.exp_tag or "").strip().strip("/")

    language = resolve_language(args.language, args.dataset_root, args.runs_root)
    dataset = str(args.dataset or preset_value(language, "dataset", default_dataset_name(language)))
    runs_root = Path(args.runs_root) if args.runs_root else Path(preset_value(language, "runs_root", default_runs_root(language, dataset)))
    dataset_root = Path(args.dataset_root) if args.dataset_root else Path(preset_value(language, "dataset_root", default_dataset_root(language, dataset)))
    judge_single_config = str(args.judge_single_config or preset_value(language, "judge_single_config", "configs/judge_single_config.json"))
    judge_pairwise_config = str(args.judge_pairwise_config or preset_value(language, "judge_pairwise_config", "configs/judge_pairwise_config.json"))
    control_template_text = str(args.control_template_text or default_control_template(language))
    domain_default = "movie" if "movie" in dataset.lower() else "review"

    cohort_base_dir = Path(args.cohort)
    cohort_exp_dir = cohort_base_dir / exp_tag if exp_tag else cohort_base_dir

    common_ids_csv = Path(args.common_ids_csv) if args.common_ids_csv else runs_root / "gen" / cohort_base_dir / "common_target_ids.csv"
    real_train_csv = Path(args.real_train_csv) if args.real_train_csv else dataset_root / "splits" / "train.csv"
    real_test_csv = Path(args.real_test_csv) if args.real_test_csv else dataset_root / "splits" / "test.csv"

    manifest_csv = Path(args.manifest_csv) if args.manifest_csv else runs_root / "eval" / "pipeline" / "pipeline_manifest.csv"
    state_json = Path(args.state_json) if args.state_json else runs_root / "eval" / "pipeline" / "states" / cohort_exp_dir / f"{run_name}.json"

    model_family = args.model_family.strip().lower() or infer_model_family(run_name)
    prompt_variant = args.prompt_variant.strip().lower() or infer_prompt_variant(run_name)

    subset_csv = runs_root / "gen" / cohort_exp_dir / f"{run_name}__common.csv"

    state = read_json(state_json, default={})
    if not state:
        state = {
            "run_name": run_name,
            "dataset": dataset,
            "cohort": args.cohort,
            "exp_tag": exp_tag,
            "created_at": utc_now_iso(),
            "stage_status": {},
            "stage_outputs": {},
        }

    assert_stage_ready(state, stage)

    if stage == "register":
        warnings = naming_warnings(run_name)

        info = {
            "run_name": run_name,
            "model_family": model_family,
            "prompt_variant": prompt_variant,
            "dataset": dataset,
            "cohort": args.cohort,
            "exp_tag": exp_tag,
            "gen_csv": str(gen_csv),
            "subset_csv": str(subset_csv),
            "common_ids_csv": str(common_ids_csv),
            "real_train_csv": str(real_train_csv),
            "real_test_csv": str(real_test_csv),
            "warnings": warnings,
        }

        mark_stage(state, stage, "done", info)
        write_json(state_json, state)

        now = utc_now_iso()
        upsert_manifest_row(
            manifest_csv,
            {
                "run_name": run_name,
                "model_family": model_family,
                "prompt_variant": prompt_variant,
                "dataset": dataset,
                "cohort": args.cohort,
                "exp_tag": exp_tag,
                "gen_csv": str(gen_csv),
                "subset_csv": str(subset_csv),
                "created_at": state.get("created_at", now),
                "updated_at": now,
                "status": "registered",
                "notes": " | ".join(warnings),
            },
        )

    elif stage == "validate":
        stats = validate_gen_csv(gen_csv, language=language)
        stats = add_compat_quality_rate_aliases(stats, language=language)
        out = {
            "validated_gen_csv": str(gen_csv),
            "stats": stats,
        }
        mark_stage(state, stage, "done", out)
        write_json(state_json, state)

        upsert_manifest_row(
            manifest_csv,
            {
                "run_name": run_name,
                "model_family": model_family,
                "prompt_variant": prompt_variant,
                "dataset": dataset,
                "cohort": args.cohort,
                "exp_tag": exp_tag,
                "gen_csv": str(gen_csv),
                "subset_csv": str(subset_csv),
                "created_at": state.get("created_at", utc_now_iso()),
                "updated_at": utc_now_iso(),
                "status": "validated",
                "notes": f"rows={stats['rows']} usable={stats['usable_rows']}",
            },
        )

    elif stage == "subset":
        if not common_ids_csv.exists():
            raise RuntimeError(f"common_ids_csv not found: {common_ids_csv}")
        out = build_subset(gen_csv, common_ids_csv, subset_csv)
        mark_stage(state, stage, "done", out)
        write_json(state_json, state)

        upsert_manifest_row(
            manifest_csv,
            {
                "run_name": run_name,
                "model_family": model_family,
                "prompt_variant": prompt_variant,
                "dataset": dataset,
                "cohort": args.cohort,
                "exp_tag": exp_tag,
                "gen_csv": str(gen_csv),
                "subset_csv": str(subset_csv),
                "created_at": state.get("created_at", utc_now_iso()),
                "updated_at": utc_now_iso(),
                "status": "subset_ready",
                "notes": f"subset_rows={out['subset_rows']}",
            },
        )

    elif stage == "dist":
        dist_reports_root = runs_root / "eval" / "dist" / args.cohort
        dist_dir = dist_reports_root / exp_tag / run_name if exp_tag else dist_reports_root / run_name
        dist_score_csv = dist_reports_root / f"scoreboard_dist_{args.cohort}.csv"

        run_cmd(
            [
                sys.executable,
                "scripts/eval/dist/generated_reviews_report.py",
                "--gen_csv",
                str(subset_csv),
                "--real_train_csv",
                str(real_train_csv),
                "--real_test_csv",
                str(real_test_csv),
                "--out_dir",
                str(dist_dir),
            ]
        )

        run_cmd(
            [
                sys.executable,
                "scripts/eval/dist/collection_dist_scoreboard.py",
                "--reports_root",
                str(dist_reports_root),
                "--glob",
                "**/report.json",
                "--out_csv",
                str(dist_score_csv),
                "--sort_by",
                "run",
            ]
        )

        out = {
            "dist_report": str(dist_dir / "report.json"),
            "dist_scoreboard": str(dist_score_csv),
        }
        mark_stage(state, stage, "done", out)
        write_json(state_json, state)

    elif stage == "judge":
        judge_inputs_root = runs_root / "eval" / "judge_inputs" / args.cohort / exp_tag if exp_tag else runs_root / "eval" / "judge_inputs" / args.cohort
        judge_outputs_root = runs_root / "eval" / "judge_outputs" / args.cohort
        judge_inputs_root.mkdir(parents=True, exist_ok=True)

        deduped_gen_csv = judge_inputs_root / f"judge_gen_deduped__{run_name}.csv"
        dedupe_meta = dedupe_generated_for_judging(
            in_csv=subset_csv,
            out_csv=deduped_gen_csv,
            enabled=bool(args.judge_dedupe_exact),
        )

        pair_in_csv = judge_inputs_root / f"judge_pairs__{run_name}.csv"
        pair_meta = build_pairwise_input(
            real_test_csv=real_test_csv,
            subset_gen_csv=deduped_gen_csv,
            out_csv=pair_in_csv,
            gen_run=run_name,
            common_ids_csv=common_ids_csv,
            k_matches=args.pair_k_matches,
            match_seed=args.pair_match_seed,
            swap_fraction=args.pair_swap_fraction,
            retest_fraction=args.pair_retest_fraction,
            control_n_each=args.pair_control_n_each,
            control_real_template_n=args.pair_control_real_template_n,
            control_real_real_n=args.pair_control_real_real_n,
            control_gen_gen_n=args.pair_control_gen_gen_n,
            pairwise_debias_swap_all=bool(args.pairwise_debias_swap_all),
            pair_exact_balance_lr=bool(args.pair_exact_balance_lr),
            judge_seed_base=args.judge_seed_base,
            domain_default=domain_default,
            language=language,
            control_template_text=control_template_text,
            dedupe_meta=dedupe_meta,
        )

        single_in_csv = judge_inputs_root / f"judge_single_input__{run_name}.csv"
        single_meta = build_single_input(
            real_test_csv=real_test_csv,
            subset_gen_csv=deduped_gen_csv,
            out_csv=single_in_csv,
            gen_run=run_name,
            common_ids_csv=common_ids_csv,
            seed_base=args.judge_seed_base,
            domain_default=domain_default,
            dedupe_meta=dedupe_meta,
        )

        backend_outputs: Dict[str, Any] = {}
        for backend in parse_judge_backends(args.judge_backends):
            backend_root = judge_outputs_root / backend
            backend_run_root = backend_root / exp_tag if exp_tag else backend_root
            backend_run_root.mkdir(parents=True, exist_ok=True)

            single_out_csv = backend_run_root / f"judged_single__{run_name}__{backend}_single_v1.csv"
            pair_out_csv = backend_run_root / f"judged_pairs__{run_name}__{backend}_pairwise_v1.csv"
            single_score_csv = backend_root / f"scoreboard_single__{backend}_single_v1.csv"
            pair_score_csv = backend_root / f"scoreboard_pairwise__{backend}_pairwise_v1.csv"

            run_cmd(
                [
                    sys.executable,
                    "scripts/eval/judge/llm_judge_single.py",
                    "--in_csv",
                    str(single_in_csv),
                    "--out_csv",
                    str(single_out_csv),
                    "--prompt_config",
                    judge_single_config,
                    "--judge_backend",
                    backend,
                    "--n_samples",
                    str(args.judge_n_samples),
                    "--seed_base",
                    str(args.judge_seed_base),
                    "--review_col",
                    "review_text",
                    "--seed_col",
                    "judge_seed",
                    "--id_cols",
                    "single_item_id",
                    "--resume",
                ]
            )

            run_cmd(
                [
                    sys.executable,
                    "scripts/eval/judge/llm_judge_pairwise.py",
                    "--in_csv",
                    str(pair_in_csv),
                    "--out_csv",
                    str(pair_out_csv),
                    "--prompt_config",
                    judge_pairwise_config,
                    "--judge_backend",
                    backend,
                    "--n_samples",
                    str(args.judge_n_samples),
                    "--seed_base",
                    str(args.judge_seed_base),
                    "--col_a",
                    "review_a",
                    "--col_b",
                    "review_b",
                    "--seed_col",
                    "judge_seed",
                    "--id_cols",
                    "pair_uid",
                    "--resume",
                ]
            )

            run_cmd(
                [
                    sys.executable,
                    "scripts/eval/judge/collect_single_scoreboard.py",
                    "--judged_root",
                    str(backend_root),
                    "--pattern",
                    f"**/judged_single__*__{backend}_single_v1.csv",
                    "--out_csv",
                    str(single_score_csv),
                    "--mu_real_drift_threshold",
                    str(args.single_mu_real_drift_threshold),
                ]
            )

            cmd = [
                sys.executable,
                "scripts/eval/judge/collect_pairwise_scoreboard.py",
                "--judged_root",
                str(backend_root),
                "--pattern",
                f"**/judged_pairs__*__{backend}_pairwise_v1.csv",
                "--out_csv",
                str(pair_score_csv),
                "--raw_out_dir",
                str(backend_root / "pairwise_normalized_raw"),
                "--bootstrap_iters",
                str(args.pair_bootstrap_iters),
                "--bootstrap_seed",
                str(args.pair_bootstrap_seed),
                "--bootstrap_unit",
                str(args.pair_bootstrap_unit),
                "--position_bias_max_abs",
                str(args.position_bias_max_abs),
                "--swap_consistency_min",
                str(args.swap_consistency_min),
                "--test_retest_agreement_min",
                str(args.test_retest_agreement_min),
                "--control_real_template_min",
                str(args.control_real_template_min),
                "--control_real_template_gen_max",
                str(args.control_real_template_gen_max),
                "--control_balanced_max_gap",
                str(args.control_balanced_max_gap),
                "--control_balanced_min_tie",
                str(args.control_balanced_min_tie),
                "--sanity_random_n",
                str(args.judge_sanity_random_n),
            ]
            if args.pairwise_debias_swap_all:
                cmd.extend(["--aggregate_main_by_base"])
            else:
                cmd.extend(["--no-aggregate_main_by_base"])
            run_cmd(cmd)

            backend_outputs[backend] = {
                "single_output_csv": str(single_out_csv),
                "pair_output_csv": str(pair_out_csv),
                "single_scoreboard": str(single_score_csv),
                "pair_scoreboard": str(pair_score_csv),
            }

        interpretability_outputs: Dict[str, Any] = {}
        if args.write_interpretability_examples:
            qwen_single_csv = str(backend_outputs.get("qwen", {}).get("single_output_csv", "") or "")
            llama_single_csv = str(backend_outputs.get("llama", {}).get("single_output_csv", "") or "")
            have_qwen = bool(qwen_single_csv and Path(qwen_single_csv).exists())
            have_llama = bool(llama_single_csv and Path(llama_single_csv).exists())

            if have_qwen or have_llama:
                interp_root = runs_root / "eval" / "master" / args.cohort / exp_tag if exp_tag else runs_root / "eval" / "master" / args.cohort
                interp_csv = interp_root / f"interpret_examples__{run_name}.csv"
                interp_md = interp_root / f"interpret_examples__{run_name}.md"

                cmd = [
                    sys.executable,
                    "scripts/eval/judge/build_interpretability_examples.py",
                    "--out_csv",
                    str(interp_csv),
                    "--out_md",
                    str(interp_md),
                    "--language",
                    infer_language_label(dataset_root=dataset_root, runs_root=runs_root),
                    "--run_name",
                    run_name,
                    "--examples_per_source_type",
                    str(args.interpretability_examples_per_source_type),
                    "--max_review_chars",
                    str(args.interpretability_max_review_chars),
                    "--seed",
                    str(args.judge_seed_base),
                ]
                if have_qwen:
                    cmd.extend(["--single_csv_qwen", qwen_single_csv])
                if have_llama:
                    cmd.extend(["--single_csv_llama", llama_single_csv])
                run_cmd(cmd)

                interpretability_outputs = {
                    "examples_csv": str(interp_csv),
                    "examples_md": str(interp_md),
                    "language": infer_language_label(dataset_root=dataset_root, runs_root=runs_root),
                    "examples_per_source_type": int(args.interpretability_examples_per_source_type),
                }

        out = {
            "dedupe": dedupe_meta,
            "pair_input": pair_meta,
            "single_input": single_meta,
            "backends": backend_outputs,
        }
        if interpretability_outputs:
            out["interpretability"] = interpretability_outputs
        mark_stage(state, stage, "done", out)
        write_json(state_json, state)

    elif stage == "detector":
        det_root = runs_root / "eval" / "det" / args.cohort
        det_run_dir = det_root / exp_tag / run_name if exp_tag else det_root / run_name
        det_score_csv = det_root / "scoreboard_detector__tfidf_lr_v1.csv"

        run_cmd(
            [
                sys.executable,
                "scripts/eval/detector/tfidf_lr.py",
                "--real_train_csv",
                str(real_train_csv),
                "--real_test_csv",
                str(real_test_csv),
                "--gen_csv",
                str(subset_csv),
                "--out_dir",
                str(det_run_dir),
                "--run_name",
                run_name,
                "--common_target_ids_csv",
                str(common_ids_csv),
            ]
        )

        run_cmd(
            [
                sys.executable,
                "scripts/eval/detector/collection_detector_scoreboard.py",
                "--reports_root",
                str(det_root),
                "--out_csv",
                str(det_score_csv),
            ]
        )

        out = {
            "detector_report": str(det_run_dir / "report.json"),
            "detector_scoreboard": str(det_score_csv),
        }
        mark_stage(state, stage, "done", out)
        write_json(state_json, state)

    elif stage == "master":
        dist_csv = runs_root / "eval" / "dist" / args.cohort / f"scoreboard_dist_{args.cohort}.csv"
        det_csv = runs_root / "eval" / "det" / args.cohort / "scoreboard_detector__tfidf_lr_v1.csv"
        out_csv = runs_root / "eval" / "master" / args.cohort / "master_scoreboard__v1.csv"

        qwen_single_csv = runs_root / "eval" / "judge_outputs" / args.cohort / "qwen" / "scoreboard_single__qwen_single_v1.csv"
        qwen_pair_csv = runs_root / "eval" / "judge_outputs" / args.cohort / "qwen" / "scoreboard_pairwise__qwen_pairwise_v1.csv"
        if not qwen_single_csv.exists():
            qwen_single_csv = runs_root / "eval" / "judge_outputs" / args.cohort / "scoreboard_single__qwen_single_v1.csv"
        if not qwen_pair_csv.exists():
            qwen_pair_csv = runs_root / "eval" / "judge_outputs" / args.cohort / "scoreboard_pairwise__qwen_pairwise_v1.csv"

        llama_single_csv = runs_root / "eval" / "judge_outputs" / args.cohort / "llama" / "scoreboard_single__llama_single_v1.csv"
        llama_pair_csv = runs_root / "eval" / "judge_outputs" / args.cohort / "llama" / "scoreboard_pairwise__llama_pairwise_v1.csv"

        cmd = [
            sys.executable,
            "scripts/eval/collect_master_scoreboard.py",
            "--dist_csv",
            str(dist_csv),
            "--detector_csv",
            str(det_csv),
            "--out_csv",
            str(out_csv),
        ]
        if qwen_single_csv.exists():
            cmd.extend(["--single_csv_qwen", str(qwen_single_csv)])
        if qwen_pair_csv.exists():
            cmd.extend(["--pairwise_csv_qwen", str(qwen_pair_csv)])
        if llama_single_csv.exists():
            cmd.extend(["--single_csv_llama", str(llama_single_csv)])
        if llama_pair_csv.exists():
            cmd.extend(["--pairwise_csv_llama", str(llama_pair_csv)])

        run_cmd(cmd)

        out = {
            "master_scoreboard": str(out_csv),
            "single_csv_qwen": str(qwen_single_csv) if qwen_single_csv.exists() else None,
            "pairwise_csv_qwen": str(qwen_pair_csv) if qwen_pair_csv.exists() else None,
            "single_csv_llama": str(llama_single_csv) if llama_single_csv.exists() else None,
            "pairwise_csv_llama": str(llama_pair_csv) if llama_pair_csv.exists() else None,
        }
        mark_stage(state, stage, "done", out)
        write_json(state_json, state)

    elif stage == "compare":
        master_csv = runs_root / "eval" / "master" / args.cohort / "master_scoreboard__v1.csv"
        if not master_csv.exists():
            raise RuntimeError(f"master scoreboard not found: {master_csv}")

        comparison_csv = (
            Path(args.comparison_csv)
            if args.comparison_csv
            else (
                runs_root / "eval" / "master" / args.cohort / exp_tag / f"comparison__{run_name}__vs_baselines.csv"
                if exp_tag
                else runs_root / "eval" / "master" / args.cohort / f"comparison__{run_name}__vs_baselines.csv"
            )
        )

        run_cmd(
            [
                sys.executable,
                "scripts/eval/write_baseline_comparison.py",
                "--master_csv",
                str(master_csv),
                "--run_name",
                run_name,
                "--model_family",
                model_family,
                "--out_csv",
                str(comparison_csv),
            ]
        )

        fs5, zs = baseline_run_names(model_family, language=language, run_name=run_name)
        out = {
            "comparison_csv": str(comparison_csv),
            "baseline_runs_expected": [fs5, zs],
        }
        mark_stage(state, stage, "done", out)
        write_json(state_json, state)

        upsert_manifest_row(
            manifest_csv,
            {
                "run_name": run_name,
                "model_family": model_family,
                "prompt_variant": prompt_variant,
                "dataset": dataset,
                "cohort": args.cohort,
                "exp_tag": exp_tag,
                "gen_csv": str(gen_csv),
                "subset_csv": str(subset_csv),
                "created_at": state.get("created_at", utc_now_iso()),
                "updated_at": utc_now_iso(),
                "status": "completed",
                "notes": f"comparison={comparison_csv}",
            },
        )

    print("\n=== Stage Complete ===")
    print(f"stage={stage}")
    print(f"state_json={state_json}")
    print(json.dumps(state.get("stage_outputs", {}).get(stage, {}), ensure_ascii=False, indent=2))


# ---- End inlined stage execution library ----

def parse_csv_list(raw: str) -> List[str]:
    return [x.strip() for x in str(raw or "").split(",") if x.strip()]


def sanitize_slug(s: str) -> str:
    out = re.sub(r"[^a-zA-Z0-9_]+", "_", str(s or "").strip().lower())
    out = re.sub(r"_+", "_", out).strip("_")
    return out or "dataset"


def parse_dataset_map(raw: str) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for item in parse_csv_list(raw):
        if ":" not in item:
            raise RuntimeError(f"Invalid --dataset_map entry (expected lang:dataset): {item}")
        lang, ds = item.split(":", 1)
        out[lang.strip().lower()] = ds.strip()
    return out


def discover_languages(languages_root: Path) -> List[str]:
    out = []
    if not languages_root.exists():
        return out
    for p in sorted(languages_root.iterdir()):
        if p.is_dir() and (p / "datasets").exists():
            out.append(p.name)
    return out


def discover_dataset(language: str, explicit_dataset: str, dataset_map: Dict[str, str], languages_root: Path) -> str:
    lang_key = language.lower()
    if lang_key in dataset_map:
        return dataset_map[lang_key]
    if explicit_dataset:
        return explicit_dataset

    droot = languages_root / language / "datasets"
    if not droot.exists():
        raise RuntimeError(f"No datasets directory found for language={language}: {droot}")

    candidates = [p.name for p in sorted(droot.iterdir()) if p.is_dir()]
    if not candidates:
        raise RuntimeError(f"No dataset subdirectories found in: {droot}")
    return candidates[0]


def choose_runs_root(language: str, dataset: str, languages_root: Path) -> Path:
    legacy = languages_root / language / "runs"
    dataset_specific = legacy / dataset
    if dataset_specific.exists():
        return dataset_specific
    if legacy.exists() and (legacy / "gen").exists():
        return legacy
    return dataset_specific


def candidate_language_dirs(language: str) -> List[str]:
    l = language.lower()
    if l == "swahili":
        return ["swahili", "swahilli"]
    if l == "swahilli":
        return ["swahilli", "swahili"]
    return [l]


def _score_config(path: Path, language: str, variant: str) -> Tuple[int, int, int]:
    name = path.name.lower()
    lang_score = 0 if any(x in name for x in candidate_language_dirs(language)) else 1
    var_score = 0 if variant in name else 1
    return (var_score, lang_score, len(name))


def find_generation_config(language: str, variant: str, repo_root: Path) -> Path:
    v = str(variant or "").strip().lower()
    if v not in {"fs5", "zs"}:
        raise RuntimeError(f"Unsupported prompt variant: {variant}")
    paths: List[Path] = []
    for lang_dir in candidate_language_dirs(language):
        gdir = repo_root / "configs" / lang_dir / "gen"
        if gdir.exists():
            paths.extend(sorted(gdir.glob("*.json")))
    if paths:
        ranked = sorted(paths, key=lambda p: _score_config(p, language, v))
        return ranked[0]

    cfg = repo_root / "configs" / f"gen_{v}_config.json"
    if cfg.exists():
        return cfg
    raise RuntimeError(
        f"No generation config found for language={language}, variant={variant}. "
        f"Tried configs/<language>/gen/*.json and {cfg}"
    )


def find_judge_configs(language: str, repo_root: Path, single_override: str, pair_override: str) -> Tuple[Path, Path]:
    if single_override and pair_override:
        single = Path(single_override)
        pair = Path(pair_override)
        if not single.exists():
            raise RuntimeError(f"Missing judge single config: {single}")
        if not pair.exists():
            raise RuntimeError(f"Missing judge pairwise config: {pair}")
        return single, pair

    candidates: List[Path] = []
    for lang_dir in candidate_language_dirs(language):
        jdir = repo_root / "configs" / lang_dir / "judge"
        if jdir.exists():
            candidates.extend(sorted(jdir.glob("*.json")))

    singles = [p for p in candidates if "judge_single" in p.name]
    pairs = [p for p in candidates if "judge_pairwise" in p.name]

    if singles:
        singles = sorted(singles, key=lambda p: (1 if p.name.endswith("_ha.json") else 0, len(p.name)))
        single = Path(single_override) if single_override else singles[0]
    else:
        single = Path(single_override) if single_override else (repo_root / "configs" / "judge_single_config.json")

    if pairs:
        pairs = sorted(pairs, key=lambda p: (1 if p.name.endswith("_ha.json") else 0, len(p.name)))
        pair = Path(pair_override) if pair_override else pairs[0]
    else:
        pair = Path(pair_override) if pair_override else (repo_root / "configs" / "judge_pairwise_config.json")

    if not single.exists():
        raise RuntimeError(
            f"Missing judge single config for language={language}. "
            f"Tried language-specific configs and fallback {single}"
        )
    if not pair.exists():
        raise RuntimeError(
            f"Missing judge pairwise config for language={language}. "
            f"Tried language-specific configs and fallback {pair}"
        )
    return single, pair


def infer_prompt_lang(prompt_config: Path, language: str) -> str:
    cfg = json.load(open(prompt_config, "r", encoding="utf-8"))
    keys = list((cfg.get("system_prompt") or {}).keys())
    if not keys:
        return "english"
    l = language.lower()
    if l in keys:
        return l
    if l == "swahili" and "swahilli" in keys:
        return "swahilli"
    if l == "swahilli" and "swahili" in keys:
        return "swahili"
    if "english" in keys:
        return "english"
    return keys[0]


def ensure_dataset_prepared(
    python_exe: str,
    repo_root: Path,
    language: str,
    dataset: str,
    dataset_root: Path,
    domain: str,
    prepare_data: bool,
) -> None:
    train_csv = dataset_root / "splits" / "train.csv"
    test_csv = dataset_root / "splits" / "test.csv"
    targets_csv = dataset_root / "targets" / "targets_for_generation.csv"

    if train_csv.exists() and test_csv.exists() and targets_csv.exists():
        return
    if not prepare_data:
        raise RuntimeError(
            "Required prepared files are missing while --no-prepare_data is set. "
            f"Expected: {train_csv}, {test_csv}, {targets_csv}"
        )

    cmd = [
        python_exe,
        str(repo_root / "scripts" / "utils" / "prepare_language_dataset.py"),
        "--language",
        language,
        "--dataset",
        dataset,
        "--out_root",
        str(dataset_root),
        "--domain",
        domain,
    ]
    run_cmd(cmd)


def run_generation(
    python_exe: str,
    repo_root: Path,
    targets_csv: Path,
    run_spec: RunSpec,
    domain: str,
    max_tries: int,
    resume: bool,
    language: str,
) -> None:
    if run_spec.backend == "qwen":
        script = repo_root / "scripts" / "gen" / "qwen" / "generate_qwen_reviews.py"
    elif run_spec.backend == "llama":
        script = repo_root / "scripts" / "gen" / "llama" / "generate_llama_reviews.py"
    else:
        raise RuntimeError(f"Unsupported generator backend: {run_spec.backend}")

    cmd = [
        python_exe,
        str(script),
        "--targets_csv",
        str(targets_csv),
        "--out_csv",
        str(run_spec.gen_csv),
        "--run_id",
        run_spec.run_name,
        "--lang",
        language,
        "--domain",
        domain,
        "--prompt_lang",
        run_spec.prompt_lang,
        "--prompt_config",
        str(run_spec.prompt_config),
        "--max_tries",
        str(max_tries),
    ]
    if resume:
        cmd.append("--resume")
    run_cmd(cmd)


def write_common_ids(run_specs: List[RunSpec], out_csv: Path, mode: str) -> int:
    sets: List[set] = []
    for spec in run_specs:
        if not spec.gen_csv.exists():
            raise RuntimeError(f"Generated CSV not found for common-id build: {spec.gen_csv}")
        df = pd.read_csv(spec.gen_csv)
        if "target_id" not in df.columns:
            raise RuntimeError(f"Generated CSV missing target_id: {spec.gen_csv}")
        ids = set(pd.to_numeric(df["target_id"], errors="coerce").dropna().astype(int).tolist())
        sets.append(ids)

    if not sets:
        raise RuntimeError("No runs available to build common target IDs")

    if mode == "first_run":
        common_ids = sorted(sets[0])
    else:
        common_ids = sorted(set.intersection(*sets))
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"target_id": common_ids}).to_csv(out_csv, index=False)
    return len(common_ids)


def run_eval_for_language(
    language: str,
    dataset: str,
    dataset_root: Path,
    runs_root: Path,
    run_specs: List[RunSpec],
    common_ids_csv: Path,
    cohort: str,
    exp_tag: str,
    judge_backends: str,
    judge_single_cfg: Path,
    judge_pair_cfg: Path,
) -> None:
    for spec in run_specs:
        for stage in EVAL_STAGES_PER_RUN:
            stage_args = [
                "--language",
                language,
                "--dataset",
                dataset,
                "--dataset_root",
                str(dataset_root),
                "--runs_root",
                str(runs_root),
                "--stage",
                stage,
                "--run_name",
                spec.run_name,
                "--gen_csv",
                str(spec.gen_csv),
                "--cohort",
                cohort,
                "--common_ids_csv",
                str(common_ids_csv),
                "--judge_backends",
                judge_backends,
                "--judge_single_config",
                str(judge_single_cfg),
                "--judge_pairwise_config",
                str(judge_pair_cfg),
            ]
            if exp_tag:
                stage_args.extend(["--exp_tag", exp_tag])
            run_stage(stage_args)

    # Build master scoreboard once after all runs complete.
    anchor = run_specs[0]
    stage_args = [
        "--language",
        language,
        "--dataset",
        dataset,
        "--dataset_root",
        str(dataset_root),
        "--runs_root",
        str(runs_root),
        "--stage",
        "master",
        "--run_name",
        anchor.run_name,
        "--gen_csv",
        str(anchor.gen_csv),
        "--cohort",
        cohort,
        "--common_ids_csv",
        str(common_ids_csv),
    ]
    if exp_tag:
        stage_args.extend(["--exp_tag", exp_tag])
    run_stage(stage_args)


def build_run_specs(
    language: str,
    dataset: str,
    runs_root: Path,
    cohort: str,
    exp_tag: str,
    backends: List[str],
    variants: List[str],
    repo_root: Path,
    run_suffix: str,
    reuse_existing: bool = False,
) -> List[RunSpec]:
    cohort_path = Path(cohort) / exp_tag if exp_tag else Path(cohort)
    gen_dir = runs_root / "gen" / cohort_path
    gen_dir.mkdir(parents=True, exist_ok=True)
    dataset_slug = sanitize_slug(dataset)
    suffix = f"_{sanitize_slug(run_suffix)}" if run_suffix else ""

    def parse_version(stem: str, prefix: str, suffix_part: str) -> Optional[int]:
        # Match names like "<prefix>_v12<suffix>".
        pat = re.compile(rf"^{re.escape(prefix)}_v(\d+){re.escape(suffix_part)}$")
        m = pat.match(stem)
        if not m:
            return None
        try:
            return int(m.group(1))
        except Exception:
            return None

    def choose_version(prefix: str, suffix_part: str) -> int:
        versions: List[int] = []
        for fp in gen_dir.glob(f"{prefix}_v*{suffix_part}.csv"):
            v = parse_version(fp.stem, prefix, suffix_part)
            if v is not None:
                versions.append(v)
        if reuse_existing:
            # For --skip_generate we want the latest existing run, not a new one.
            return max(versions) if versions else 1
        return (max(versions) + 1) if versions else 1

    specs: List[RunSpec] = []
    for backend in backends:
        for variant in variants:
            prefix = f"{backend}_{variant}_{sanitize_slug(language)}_{dataset_slug}"
            version = choose_version(prefix, suffix)
            run_name = f"{prefix}_v{version}{suffix}"
            prompt_cfg = find_generation_config(language=language, variant=variant, repo_root=repo_root)
            prompt_lang = infer_prompt_lang(prompt_cfg, language)
            specs.append(
                RunSpec(
                    backend=backend,
                    variant=variant,
                    run_name=run_name,
                    gen_csv=gen_dir / f"{run_name}.csv",
                    prompt_config=prompt_cfg,
                    prompt_lang=prompt_lang,
                )
            )
    return specs


def main() -> None:
    ap = argparse.ArgumentParser(description="Unified multi-language generation + evaluation runner")
    ap.add_argument("--languages", default="auto", help="Comma-separated language names or 'auto'")
    ap.add_argument("--dataset", default="", help="Single dataset name for all languages (optional)")
    ap.add_argument("--dataset_map", default="", help="Overrides like 'nepali:yt_nepali_movie_reviews,hausa:hausa_reviews'")
    ap.add_argument("--cohort", default="common_subset_auto_v1")
    ap.add_argument("--exp_tag", default="")
    ap.add_argument("--domain", default="review")
    ap.add_argument("--generator_backends", default="qwen,llama", help="Comma list: qwen,llama")
    ap.add_argument("--prompt_variants", default="zs,fs5", help="Comma list: zs,fs5")
    ap.add_argument("--judge_backends", default="qwen,llama")
    ap.add_argument("--judge_single_config", default="")
    ap.add_argument("--judge_pairwise_config", default="")
    ap.add_argument("--prepare_data", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--skip_generate", action="store_true")
    ap.add_argument("--skip_eval", action="store_true")
    ap.add_argument("--max_tries", type=int, default=3)
    ap.add_argument("--gen_resume", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--common_ids_mode", choices=["intersection", "first_run"], default="intersection")
    ap.add_argument("--run_suffix", default="")
    args = ap.parse_args()

    repo_root = Path(__file__).resolve().parents[3]
    languages_root = repo_root / "languages"
    dataset_map = parse_dataset_map(args.dataset_map)

    languages = parse_csv_list(args.languages)
    if not languages or (len(languages) == 1 and languages[0].lower() == "auto"):
        languages = discover_languages(languages_root)
    if not languages:
        raise RuntimeError("No languages found to run.")

    backends = [x.lower() for x in parse_csv_list(args.generator_backends)]
    variants = [x.lower() for x in parse_csv_list(args.prompt_variants)]
    if not backends or not variants:
        raise RuntimeError("generator_backends and prompt_variants must be non-empty")

    for b in backends:
        if b not in {"qwen", "llama"}:
            raise RuntimeError(f"Unsupported generator backend: {b}")
    for v in variants:
        if v not in {"zs", "fs5"}:
            raise RuntimeError(f"Unsupported prompt variant: {v}")

    for language in languages:
        dataset = discover_dataset(language, args.dataset, dataset_map, languages_root)
        dataset_root = languages_root / language / "datasets" / dataset
        runs_root = choose_runs_root(language, dataset, languages_root)
        runs_root.mkdir(parents=True, exist_ok=True)

        print("\n============================================================")
        print(f"LANGUAGE={language}")
        print(f"DATASET={dataset}")
        print(f"DATASET_ROOT={dataset_root}")
        print(f"RUNS_ROOT={runs_root}")
        print("============================================================")

        ensure_dataset_prepared(
            python_exe=sys.executable,
            repo_root=repo_root,
            language=language,
            dataset=dataset,
            dataset_root=dataset_root,
            domain=args.domain,
            prepare_data=bool(args.prepare_data),
        )

        targets_csv = dataset_root / "targets" / "targets_for_generation.csv"
        if not targets_csv.exists():
            raise RuntimeError(f"targets CSV not found after preparation: {targets_csv}")

        run_specs = build_run_specs(
            language=language,
            dataset=dataset,
            runs_root=runs_root,
            cohort=args.cohort,
            exp_tag=args.exp_tag,
            backends=backends,
            variants=variants,
            repo_root=repo_root,
            run_suffix=args.run_suffix,
            reuse_existing=bool(args.skip_generate),
        )
        print("Planned runs:")
        for spec in run_specs:
            print(
                f"  - {spec.run_name} | backend={spec.backend} variant={spec.variant} "
                f"prompt_lang={spec.prompt_lang} prompt_cfg={spec.prompt_config}"
            )

        if not args.skip_generate:
            for spec in run_specs:
                run_generation(
                    python_exe=sys.executable,
                    repo_root=repo_root,
                    targets_csv=targets_csv,
                    run_spec=spec,
                    domain=args.domain,
                    max_tries=int(args.max_tries),
                    resume=bool(args.gen_resume),
                    language=language,
                )

        if args.skip_generate and all(not spec.gen_csv.exists() for spec in run_specs):
            if args.skip_eval:
                print("No generated CSVs found and both --skip_generate and --skip_eval are set; skipping this language.")
                continue
            raise RuntimeError(
                "skip_generate was requested but generated CSVs were not found. "
                f"Expected files like: {run_specs[0].gen_csv}"
            )

        common_ids_csv = runs_root / "gen" / args.cohort / "common_target_ids.csv"
        n_common = write_common_ids(run_specs, common_ids_csv, mode=args.common_ids_mode)
        print(f"common_ids_csv={common_ids_csv} n={n_common}")

        if args.skip_eval:
            continue

        single_cfg, pair_cfg = find_judge_configs(
            language=language,
            repo_root=repo_root,
            single_override=args.judge_single_config,
            pair_override=args.judge_pairwise_config,
        )
        print(f"judge_single_config={single_cfg}")
        print(f"judge_pairwise_config={pair_cfg}")

        run_eval_for_language(
            language=language,
            dataset=dataset,
            dataset_root=dataset_root,
            runs_root=runs_root,
            run_specs=run_specs,
            common_ids_csv=common_ids_csv,
            cohort=args.cohort,
            exp_tag=args.exp_tag,
            judge_backends=args.judge_backends,
            judge_single_cfg=single_cfg,
            judge_pair_cfg=pair_cfg,
        )

    print("\nCompleted multi-language generation + evaluation pipeline.")


if __name__ == "__main__":
    main()
