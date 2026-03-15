#!/usr/bin/env python3
"""
llm_judge_single.py

LLM-as-judge (single-review) using local backend (Qwen/Llama).
"""

import argparse
import csv
import datetime as dt
import hashlib
import json
import math
import os
import random
import re
import time
from typing import Any, Dict, List, Optional

import pandas as pd
try:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    _MODEL_IMPORT_ERROR = None
except Exception as _e:
    torch = None  # type: ignore
    AutoModelForCausalLM = None  # type: ignore
    AutoTokenizer = None  # type: ignore
    _MODEL_IMPORT_ERROR = _e


DEFAULT_MODEL_BY_BACKEND = {
    "qwen": os.environ.get("QWEN_MODEL_NAME", "/WAVE/datasets/oignat_lab/QWEN3"),
    "llama": os.environ.get("LLAMA_MODEL_NAME", "/WAVE/datasets/oignat_lab/Meta-Llama-3.1-8B-Instruct"),
}

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

_tokenizer = None
_model = None
_model_name = None


def utc_now_iso() -> str:
    return dt.datetime.utcnow().replace(microsecond=0).isoformat() + "Z"


def resolve_model_name(backend: str) -> str:
    b = (backend or "").strip().lower()
    if b not in DEFAULT_MODEL_BY_BACKEND:
        raise ValueError(f"Unsupported judge backend: {backend}. Expected one of {list(DEFAULT_MODEL_BY_BACKEND)}")
    return DEFAULT_MODEL_BY_BACKEND[b]


def _load_model():
    global _tokenizer, _model, _model_name
    if _MODEL_IMPORT_ERROR is not None:
        raise RuntimeError(f"Model dependencies unavailable: {_MODEL_IMPORT_ERROR}")
    if not _model_name:
        raise RuntimeError("Model name is not initialized. Set _model_name before generation.")
    if _tokenizer is None or _model is None:
        local_only = os.path.isdir(_model_name)
        _tokenizer = AutoTokenizer.from_pretrained(_model_name, local_files_only=local_only, trust_remote_code=True)
        _model = AutoModelForCausalLM.from_pretrained(
            _model_name,
            torch_dtype="auto",
            device_map="auto",
            local_files_only=local_only,
            trust_remote_code=True,
        )
        _model.eval()
    return _tokenizer, _model


def _seed_everything(seed: int):
    random.seed(seed)
    if torch is None:
        return
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_completion_from_messages(messages, seed: int, temperature: float, top_p: float, max_new_tokens: int) -> str:
    tokenizer, model = _load_model()
    _seed_everything(seed)

    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

    with torch.no_grad():
        generated_ids = model.generate(
            **model_inputs,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
            max_new_tokens=max_new_tokens,
            pad_token_id=tokenizer.eos_token_id,
        )

    output_ids = generated_ids[0][len(model_inputs.input_ids[0]) :]
    return tokenizer.decode(output_ids, skip_special_tokens=True).strip()


def safe_utf8(x: Any) -> str:
    try:
        s = str(x)
    except Exception:
        s = repr(x)
    return s.encode("utf-8", errors="replace").decode("utf-8", errors="replace")


def _extract_first_json(text: str) -> str:
    t = (text or "").strip()
    t = re.sub(r"^\s*```(?:json)?\s*", "", t)
    t = re.sub(r"\s*```\s*$", "", t)

    try:
        json.loads(t)
        return t
    except Exception:
        pass

    m = re.search(r"\{.*\}", t, flags=re.DOTALL)
    if not m:
        raise ValueError(f"No JSON object found in model output. Got: {t[:250]}")
    candidate = m.group(0)
    json.loads(candidate)
    return candidate


def stable_row_id(row: pd.Series, cols_for_id: List[str]) -> str:
    key_parts = [safe_utf8(row.get(c, "")) for c in cols_for_id]
    key = "||".join(key_parts)
    return hashlib.sha256(key.encode("utf-8")).hexdigest()[:16]


def detect_col(df: pd.DataFrame, candidates: List[str], purpose: str) -> str:
    for c in candidates:
        if c in df.columns:
            return c
    raise RuntimeError(
        f"Could not find {purpose} column. Tried: {candidates}\n"
        f"Available columns: {list(df.columns)}"
    )


def safe_format_braces(template: str, **kwargs) -> str:
    t = template if template is not None else ""
    tokens: Dict[str, str] = {}
    for k, v in kwargs.items():
        token = f"__FMT_{k.upper()}__"
        t = t.replace("{" + k + "}", token)
        tokens[token] = "" if v is None else str(v)

    t = t.replace("{", "{{").replace("}", "}}")

    for token, val in tokens.items():
        t = t.replace(token, val)

    return t


def append_row_csv(path: str, fieldnames: List[str], row: Dict[str, Any]):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    write_header = (not os.path.exists(path)) or (os.path.getsize(path) == 0)

    with open(path, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        if write_header:
            w.writeheader()
        w.writerow(row)


def load_prompt_config(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    for k in ["prompt_version", "system_prompt", "user_prompt_template"]:
        if k not in cfg:
            raise RuntimeError(f"Prompt config missing key: {k}")
    genp = cfg.get("generation_params", {}) or {}
    cfg["generation_params"] = {
        "temperature": float(genp.get("temperature", 0.2)),
        "top_p": float(genp.get("top_p", 1.0)),
        "max_new_tokens": int(genp.get("max_new_tokens", 220)),
    }
    return cfg


def parse_cols_arg(raw: str) -> List[str]:
    return [x.strip() for x in str(raw or "").split(",") if x.strip()]


def parse_seed_value(x: Any) -> Optional[int]:
    try:
        if x is None:
            return None
        return int(float(x))
    except Exception:
        return None


def parse_prob_01(x: Any, key: str) -> float:
    try:
        v = float(x)
    except Exception:
        raise ValueError(f"{key} is not numeric: {x}")
    if not (0.0 <= v <= 1.0):
        raise ValueError(f"{key} out of range [0,1]: {v}")
    return v


def parse_feature_scores(obj: Dict[str, Any]) -> Dict[str, Optional[float]]:
    raw = obj.get("feature_scores", {})
    if raw is None:
        raw = {}
    if not isinstance(raw, dict):
        raise ValueError("feature_scores must be an object")

    out: Dict[str, Optional[float]] = {}
    for k in FEATURE_SCORE_KEYS:
        v = raw.get(k, None)
        if v is None or (isinstance(v, str) and not v.strip()):
            out[k] = None
            continue
        out[k] = parse_prob_01(v, f"feature_scores.{k}")
    return out


def sample_mean(xs: List[float]) -> float:
    return sum(xs) / len(xs)


def sample_std(xs: List[float]) -> float:
    if len(xs) <= 1:
        return 0.0
    m = sample_mean(xs)
    return math.sqrt(sum((x - m) ** 2 for x in xs) / (len(xs) - 1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_csv", required=True)
    ap.add_argument("--out_csv", required=True)
    ap.add_argument("--prompt_config", required=True)
    ap.add_argument("--judge_backend", default="qwen", choices=["qwen", "llama"])
    ap.add_argument("--seed_base", type=int, default=12345)
    ap.add_argument("--n_samples", type=int, default=1, help="self-consistency samples per row")
    ap.add_argument("--max_rows", type=int, default=None)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--sleep", type=float, default=0.0)
    ap.add_argument("--review_col", default="")
    ap.add_argument("--seed_col", default="")
    ap.add_argument("--id_cols", default="")
    args = ap.parse_args()
    if args.n_samples > 0 and _MODEL_IMPORT_ERROR is not None:
        raise RuntimeError(
            f"Cannot run judging with n_samples={args.n_samples}: missing model dependencies ({_MODEL_IMPORT_ERROR})"
        )

    global _model_name
    _model_name = resolve_model_name(args.judge_backend)

    cfg = load_prompt_config(args.prompt_config)
    prompt_version = cfg["prompt_version"]
    system_prompt = cfg["system_prompt"]
    user_tmpl = cfg["user_prompt_template"]
    temperature = cfg["generation_params"]["temperature"]
    top_p = cfg["generation_params"]["top_p"]
    max_new_tokens = cfg["generation_params"]["max_new_tokens"]

    df = pd.read_csv(args.in_csv)
    if args.max_rows is not None:
        df = df.head(args.max_rows).copy()

    if args.review_col:
        if args.review_col not in df.columns:
            raise RuntimeError(f"--review_col not found in input CSV: {args.review_col}")
        col_review = args.review_col
    else:
        col_review = detect_col(df, ["review_text", "Review", "review", "text", "gen_review"], "review text")

    seed_col = args.seed_col.strip()
    if seed_col and seed_col not in df.columns:
        raise RuntimeError(f"--seed_col not found in input CSV: {seed_col}")

    already = set()
    if args.resume and os.path.exists(args.out_csv) and os.path.getsize(args.out_csv) > 0:
        prev = pd.read_csv(args.out_csv)
        if "judge_row_id" in prev.columns:
            already = set(prev["judge_row_id"].astype(str).tolist())

    cols_for_id = parse_cols_arg(args.id_cols)
    if not cols_for_id:
        for c in ["single_item_id", "run_id", "target_id", col_review]:
            if c in df.columns:
                cols_for_id.append(c)
    if not cols_for_id:
        cols_for_id = [col_review]

    base_fields = list(df.columns)
    judge_fields = [
        "judge_row_id",
        "judge_backend",
        "judge_model",
        "judge_prompt_version",
        "judge_prompt_config",
        "judge_n_samples",
        "judge_seed_used",
        "judge_timestamp_utc",
        "judge_ai_probability_mean",
        "judge_ai_probability_std",
        "judge_confidence_mean",
        "judge_confidence_std",
        "judge_label",
        "judge_label_raw",
        "judge_label_overridden",
        "judge_rationale",
        "judge_feature_scores_json",
        "judge_raw_json",
        "judge_status",
    ]
    for k in FEATURE_SCORE_KEYS:
        judge_fields.append(f"judge_feat_{k}_mean")
        judge_fields.append(f"judge_feat_{k}_std")
    fieldnames = base_fields + [c for c in judge_fields if c not in base_fields]

    new_rows = 0

    for i, row in df.iterrows():
        rid = stable_row_id(row, cols_for_id)
        if rid in already:
            continue

        review_text = "" if pd.isna(row.get(col_review, "")) else str(row.get(col_review, "")).strip()

        user_prompt = safe_format_braces(
            user_tmpl,
            review_text=review_text,
        )

        probs: List[float] = []
        confs: List[float] = []
        preds: List[str] = []
        feature_scores_samples: List[Dict[str, Optional[float]]] = []
        last_raw = ""
        last_err: Optional[str] = None
        last_obj: Optional[Dict[str, Any]] = None
        last_seed_used: Optional[int] = None

        base_seed = parse_seed_value(row.get(seed_col)) if seed_col else None
        if base_seed is None:
            base_seed = args.seed_base + i * 1000

        for s in range(args.n_samples):
            seed = int(base_seed) + s
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ]

            try:
                raw = get_completion_from_messages(
                    messages,
                    seed=seed,
                    temperature=temperature,
                    top_p=top_p,
                    max_new_tokens=max_new_tokens,
                )
                last_seed_used = seed
                last_raw = raw
                obj = json.loads(_extract_first_json(raw))

                for k in ["prediction", "ai_probability", "confidence", "rationale"]:
                    if k not in obj:
                        raise ValueError(f"Missing key: {k}")

                pred = str(obj.get("prediction", "")).strip().upper()
                if pred not in ("AI", "HUMAN"):
                    raise ValueError(f"Bad prediction: {pred}")

                ai_p = parse_prob_01(obj["ai_probability"], "ai_probability")
                conf = parse_prob_01(obj["confidence"], "confidence")
                rat = str(obj.get("rationale", "")).strip()
                feature_scores = parse_feature_scores(obj)

                probs.append(ai_p)
                confs.append(conf)
                preds.append(pred)
                feature_scores_samples.append(feature_scores)
                last_obj = {
                    "prediction": pred,
                    "ai_probability": ai_p,
                    "confidence": conf,
                    "rationale": rat,
                    "feature_scores": feature_scores,
                }
                last_err = None

            except Exception as e:
                last_err = safe_utf8(e)
                continue

        out = dict(row.to_dict())
        out["judge_row_id"] = rid
        out["judge_backend"] = args.judge_backend
        out["judge_model"] = _model_name
        out["judge_prompt_version"] = prompt_version
        out["judge_prompt_config"] = args.prompt_config
        out["judge_n_samples"] = args.n_samples
        out["judge_seed_used"] = last_seed_used
        out["judge_timestamp_utc"] = utc_now_iso()

        if len(probs) == 0 or last_obj is None:
            out["judge_ai_probability_mean"] = None
            out["judge_ai_probability_std"] = None
            out["judge_confidence_mean"] = None
            out["judge_confidence_std"] = None
            out["judge_label"] = None
            out["judge_label_raw"] = None
            out["judge_label_overridden"] = None
            out["judge_rationale"] = f"JUDGE_FAILED: {last_err}"
            out["judge_feature_scores_json"] = None
            for k in FEATURE_SCORE_KEYS:
                out[f"judge_feat_{k}_mean"] = None
                out[f"judge_feat_{k}_std"] = None
            out["judge_raw_json"] = last_raw
            out["judge_status"] = "JUDGE_FAILED"
        else:
            p_mean = sample_mean(probs)
            c_mean = sample_mean(confs)
            p_std = sample_std(probs)
            c_std = sample_std(confs)

            raw_label = max(set(preds), key=preds.count)
            label = "AI" if p_mean >= 0.5 else "HUMAN"

            feat_means: Dict[str, Optional[float]] = {}
            for k in FEATURE_SCORE_KEYS:
                vals = [d.get(k) for d in feature_scores_samples if d.get(k) is not None]
                if vals:
                    vals_f = [float(v) for v in vals if v is not None]
                    feat_means[k] = sample_mean(vals_f)
                    out[f"judge_feat_{k}_mean"] = feat_means[k]
                    out[f"judge_feat_{k}_std"] = sample_std(vals_f)
                else:
                    feat_means[k] = None
                    out[f"judge_feat_{k}_mean"] = None
                    out[f"judge_feat_{k}_std"] = None

            out["judge_ai_probability_mean"] = p_mean
            out["judge_ai_probability_std"] = p_std
            out["judge_confidence_mean"] = c_mean
            out["judge_confidence_std"] = c_std
            out["judge_label"] = label
            out["judge_label_raw"] = raw_label
            out["judge_label_overridden"] = int(raw_label != label)
            out["judge_rationale"] = last_obj.get("rationale", "")
            out["judge_feature_scores_json"] = json.dumps(feat_means, ensure_ascii=False)
            out["judge_raw_json"] = json.dumps(last_obj, ensure_ascii=False)
            out["judge_status"] = "OK"

        append_row_csv(args.out_csv, fieldnames, out)
        new_rows += 1

        if args.sleep > 0:
            time.sleep(args.sleep)

    print(f"Wrote: {args.out_csv} | new rows: {new_rows}")


if __name__ == "__main__":
    main()
