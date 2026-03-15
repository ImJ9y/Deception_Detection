#!/usr/bin/env python3
import json
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    roc_auc_score, accuracy_score, f1_score,
    precision_score, recall_score, balanced_accuracy_score,
)


def detect_col(df: pd.DataFrame, candidates, what: str) -> str:
    for c in candidates:
        if c in df.columns:
            return c
    raise RuntimeError(
        f"Could not find {what} column. Tried {candidates}. Have: {list(df.columns)}"
    )


def read_common_ids(path: str):
    df = pd.read_csv(path)
    col = "target_id" if "target_id" in df.columns else df.columns[0]
    ids = pd.to_numeric(df[col], errors="coerce").dropna().astype(int).tolist()
    return set(ids)


def best_real_test_id_mode(real_test: pd.DataFrame, common_ids: set):
    """
    Decide whether common_ids align with real_test['target_id'], real_test['id'], or row index.
    Returns: mode, real_test_with_target_id
      mode in {"existing_target_id", "use_id", "use_row_index"}
    """
    rt = real_test.copy()

    overlap_target = -1
    if "target_id" in rt.columns:
        s = pd.to_numeric(rt["target_id"], errors="coerce").dropna().astype(int)
        overlap_target = len(set(s.tolist()) & common_ids)

    overlap_id = -1
    if "id" in rt.columns:
        s = pd.to_numeric(rt["id"], errors="coerce").dropna().astype(int)
        overlap_id = len(set(s.tolist()) & common_ids)

    rt2 = rt.reset_index().rename(columns={"index": "target_id"})
    s = rt2["target_id"].astype(int)
    overlap_index = len(set(s.tolist()) & common_ids)

    best = max(
        [
            (overlap_target, "existing_target_id"),
            (overlap_id, "use_id"),
            (overlap_index, "use_row_index"),
        ],
        key=lambda x: x[0],
    )
    mode = best[1]

    if mode == "use_row_index":
        return mode, rt2

    if mode == "existing_target_id":
        return mode, rt

    rt = rt.copy()
    rt["target_id"] = pd.to_numeric(rt["id"], errors="coerce")
    return mode, rt


def split_common_ids(common_ids: set, test_frac: float, seed: int):
    ids = sorted(list(common_ids))
    rng = np.random.RandomState(seed)
    rng.shuffle(ids)
    n = len(ids)
    n_test = int(np.ceil(test_frac * n))
    n_test = max(1, min(n - 1, n_test))  # ensure both sides non-empty
    test_ids = set(ids[:n_test])
    train_ids = set(ids[n_test:])
    return train_ids, test_ids


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--real_train_csv", required=True)
    ap.add_argument("--real_test_csv", required=True)
    ap.add_argument("--gen_csv", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--run_name", default=None)
    ap.add_argument("--common_target_ids_csv", default=None)

    ap.add_argument("--seed", type=int, default=12345)
    ap.add_argument("--ngram_min", type=int, default=3)
    ap.add_argument("--ngram_max", type=int, default=5)
    ap.add_argument("--min_df", type=int, default=2)
    ap.add_argument("--max_features", type=int, default=200000)

    ap.add_argument(
        "--shuffle_train_labels",
        action="store_true",
        help="Shuffle y_train (sanity check). Expect AUC ~ 0.5 when no leakage.",
    )

    # NEW: proper split when using common ids
    ap.add_argument("--common_test_frac", type=float, default=0.2)
    ap.add_argument("--common_split_seed", type=int, default=12345)

    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    run_name = args.run_name
    if run_name is None:
        run_name = Path(args.gen_csv).stem

    real_train = pd.read_csv(args.real_train_csv)
    real_test = pd.read_csv(args.real_test_csv)
    gen = pd.read_csv(args.gen_csv)

    real_train_text_col = detect_col(real_train, ["Review", "review_text", "text", "review"], "REAL_TRAIN text")
    real_test_text_col = detect_col(real_test, ["Review", "review_text", "text", "review"], "REAL_TEST text")
    gen_text_col = detect_col(gen, ["Review", "review_text", "text", "review"], "GEN text")

    common_ids = None
    filter_mode = None
    train_ids = None
    test_ids = None

    if args.common_target_ids_csv:
        common_ids = read_common_ids(args.common_target_ids_csv)
        train_ids, test_ids = split_common_ids(common_ids, args.common_test_frac, args.common_split_seed)

        # GEN must have target_id to split properly
        if "target_id" not in gen.columns:
            raise RuntimeError("GEN CSV is missing 'target_id' but common_target_ids_csv was provided.")

        gen_target = pd.to_numeric(gen["target_id"], errors="coerce").astype("Int64")
        gen_train = gen[gen_target.isin(train_ids)].copy()
        gen_test = gen[gen_target.isin(test_ids)].copy()

        # REAL_TEST split by best matching id mode
        filter_mode, real_test2 = best_real_test_id_mode(real_test, common_ids)
        rt_target = pd.to_numeric(real_test2["target_id"], errors="coerce").astype("Int64")
        real_test_split = real_test2[rt_target.isin(test_ids)].copy()

    else:
        # fallback (no common ids): keep previous behavior but NO split available
        gen_train = gen.copy()
        gen_test = gen.copy()
        real_test_split = real_test.copy()

    # Build train (real_train + gen_train)
    X_train_text = pd.concat(
        [real_train[real_train_text_col].astype(str), gen_train[gen_text_col].astype(str)],
        ignore_index=True,
    )
    y_train = np.array([0] * len(real_train) + [1] * len(gen_train), dtype=int)

    # Build test (real_test_split + gen_test)
    X_test_text = pd.concat(
        [real_test_split[real_test_text_col].astype(str), gen_test[gen_text_col].astype(str)],
        ignore_index=True,
    )
    y_test = np.array([0] * len(real_test_split) + [1] * len(gen_test), dtype=int)

    vec = TfidfVectorizer(
        analyzer="char",
        ngram_range=(args.ngram_min, args.ngram_max),
        min_df=args.min_df,
        max_features=args.max_features,
    )
    Xtr = vec.fit_transform(X_train_text)
    Xte = vec.transform(X_test_text)

    if args.shuffle_train_labels:
        rng = np.random.RandomState(args.seed)
        y_train = rng.permutation(y_train)

    clf = LogisticRegression(
        max_iter=2000,
        class_weight="balanced",
        random_state=args.seed,
        n_jobs=1,
    )
    clf.fit(Xtr, y_train)

    prob = clf.predict_proba(Xte)[:, 1]
    pred = (prob >= 0.5).astype(int)

    report = {
        "run_name": run_name,
        "inputs": {
            "real_train_csv": args.real_train_csv,
            "real_test_csv": args.real_test_csv,
            "gen_csv": args.gen_csv,
            "common_target_ids_csv": args.common_target_ids_csv,
            "shuffle_train_labels": bool(args.shuffle_train_labels),
            "common_test_frac": args.common_test_frac,
            "common_split_seed": args.common_split_seed,
        },
        "columns_used": {
            "real_train_text_col": real_train_text_col,
            "real_test_text_col": real_test_text_col,
            "gen_text_col": gen_text_col,
        },
        "filter_mode_real_test": filter_mode,
        "split_common_ids": {
            "train_ids_n": None if train_ids is None else int(len(train_ids)),
            "test_ids_n": None if test_ids is None else int(len(test_ids)),
        },
        "sizes": {
            "train_real_n": int(len(real_train)),
            "train_gen_n": int(len(gen_train)),
            "test_real_n": int(len(real_test_split)),
            "test_gen_n": int(len(gen_test)),
        },
        "metrics": {
            "roc_auc": float(roc_auc_score(y_test, prob)) if len(np.unique(y_test)) > 1 else None,
            "acc": float(accuracy_score(y_test, pred)),
            "f1": float(f1_score(y_test, pred)),
            "precision": float(precision_score(y_test, pred, zero_division=0)),
            "recall": float(recall_score(y_test, pred, zero_division=0)),
            "balanced_acc": float(balanced_accuracy_score(y_test, pred)),
            "prob_mean": float(np.mean(prob)),
            "prob_std": float(np.std(prob)),
        },
    }

    out_report = out_dir / "report.json"
    with open(out_report, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print("Wrote:", str(out_report))
    print(json.dumps(report["sizes"], indent=2))
    print(json.dumps(report["metrics"], indent=2))


if __name__ == "__main__":
    main()
