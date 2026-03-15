#!/usr/bin/env python3
import argparse
import csv
import datetime as dt
import hashlib
import json
import os
import random
import re
import time
from pathlib import Path
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
        _tokenizer = AutoTokenizer.from_pretrained(
            _model_name,
            local_files_only=local_only,
            trust_remote_code=True,
        )
        _model = AutoModelForCausalLM.from_pretrained(
            _model_name,
            torch_dtype="auto",
            device_map="auto",
            local_files_only=local_only,
            trust_remote_code=True,
        )
        _model.eval()
    return _tokenizer, _model


def _apply_chat_template(tokenizer, messages):
    try:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
    except Exception:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )


def _seed_everything(seed: int):
    random.seed(seed)
    if torch is None:
        return
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def safe_utf8(x: Any) -> str:
    if x is None:
        return ""
    if not isinstance(x, str):
        x = str(x)
    return x.encode("utf-8", errors="replace").decode("utf-8", errors="replace")


def _extract_first_json(text: str) -> Dict[str, Any]:
    t = safe_utf8(text).strip()
    if t.startswith("```"):
        t = re.sub(r"^\s*```[a-zA-Z]*\s*", "", t)
        t = re.sub(r"\s*```\s*$", "", t).strip()

    try:
        obj = json.loads(t)
        if isinstance(obj, dict):
            return obj
    except Exception:
        pass

    m = re.search(r"\{.*\}", t, flags=re.DOTALL)
    if not m:
        raise ValueError(f"No JSON object found. Got: {t[:250]}")
    cand = m.group(0).strip()
    obj = json.loads(cand)
    if not isinstance(obj, dict):
        raise ValueError("Parsed JSON is not an object")
    return obj


def get_completion(
    messages: List[Dict[str, str]],
    seed: int,
    temperature: float,
    top_p: float,
    max_new_tokens: int,
    deterministic: bool = False,
) -> str:
    tok, model = _load_model()
    _seed_everything(seed)

    chat_text = _apply_chat_template(tok, messages)
    inputs = tok([chat_text], return_tensors="pt").to(model.device)

    gen_kwargs: Dict[str, Any] = {
        **inputs,
        "max_new_tokens": int(max_new_tokens),
        "pad_token_id": tok.eos_token_id,
    }
    if deterministic:
        # Greedy decoding for deterministic judge behavior (no sampling).
        gen_kwargs["do_sample"] = False
    else:
        gen_kwargs["do_sample"] = True
        gen_kwargs["temperature"] = float(temperature)
        gen_kwargs["top_p"] = float(top_p)

    with torch.no_grad():
        out = model.generate(**gen_kwargs)

    gen_ids = out[0][inputs["input_ids"].shape[1] :]
    return safe_utf8(tok.decode(gen_ids, skip_special_tokens=True)).strip()


def load_prompt_config(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    for k in ["prompt_version", "system_prompt", "user_prompt_template"]:
        if k not in cfg:
            raise RuntimeError(f"prompt_config missing required key: {k}")
    return cfg


def detect_col(df: pd.DataFrame, candidates: List[str], name: str) -> str:
    for c in candidates:
        if c in df.columns:
            return c
    raise RuntimeError(
        f"Could not find {name} column. Tried: {candidates}\n"
        f"Available columns: {list(df.columns)}"
    )


def stable_row_id(row: pd.Series, cols: List[str]) -> str:
    key = "||".join([safe_utf8(row.get(c, "")) for c in cols])
    return hashlib.sha256(key.encode("utf-8")).hexdigest()[:16]


def parse_cols_arg(raw: str) -> List[str]:
    items = [x.strip() for x in str(raw or "").split(",") if x.strip()]
    return items


def parse_seed_value(x: Any) -> Optional[int]:
    try:
        if x is None:
            return None
        v = int(float(x))
        return v
    except Exception:
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_csv", required=True)
    ap.add_argument("--out_csv", required=True)
    ap.add_argument("--prompt_config", required=True)
    ap.add_argument("--judge_backend", default="qwen", choices=["qwen", "llama"])
    ap.add_argument("--seed_base", type=int, default=12345)
    ap.add_argument("--n_samples", type=int, default=1)
    ap.add_argument("--max_rows", type=int, default=None)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--sleep", type=float, default=0.0)
    ap.add_argument("--col_a", default="")
    ap.add_argument("--col_b", default="")
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
    prompt_version_l = str(prompt_version).strip().lower()
    system_prompt = cfg["system_prompt"]
    user_tmpl = cfg["user_prompt_template"]
    gen_params = cfg.get("generation_params", {}) or {}
    temperature = float(gen_params.get("temperature", 0.2))
    top_p = float(gen_params.get("top_p", 1.0))
    max_new_tokens = int(gen_params.get("max_new_tokens", 220))
    is_selfswap_diag = ("selfswap" in prompt_version_l) or ("selfswap" in Path(args.prompt_config).name.lower())
    deterministic_decode = bool(args.judge_backend == "llama" and temperature <= 0.0)
    do_sample = not deterministic_decode
    print(
        f"[DECODE CONFIG] backend={args.judge_backend} "
        f"temperature={temperature} top_p={top_p} "
        f"do_sample={do_sample} max_new_tokens={max_new_tokens}"
    )

    df = pd.read_csv(args.in_csv)
    if args.max_rows is not None:
        df = df.head(args.max_rows).copy()

    if args.col_a:
        if args.col_a not in df.columns:
            raise RuntimeError(f"--col_a not found in input CSV: {args.col_a}")
        col_a = args.col_a
    else:
        col_a = detect_col(
            df,
            ["review_a", "real_review", "Review_A", "text_a", "text_A", "A", "a"],
            "review A",
        )

    if args.col_b:
        if args.col_b not in df.columns:
            raise RuntimeError(f"--col_b not found in input CSV: {args.col_b}")
        col_b = args.col_b
    else:
        col_b = detect_col(
            df,
            ["review_b", "gen_review", "Review_B", "text_b", "text_B", "B", "b"],
            "review B",
        )

    id_cols = parse_cols_arg(args.id_cols)
    if not id_cols:
        for c in ["pair_uid", "pair_id", "base_pair_uid", "target_id"]:
            if c in df.columns:
                id_cols.append(c)
                break
    if not id_cols:
        id_cols = [col_a, col_b]

    seed_col = args.seed_col.strip()
    if seed_col and seed_col not in df.columns:
        raise RuntimeError(f"--seed_col not found in input CSV: {seed_col}")

    already = set()
    if args.resume and os.path.exists(args.out_csv):
        prev = pd.read_csv(args.out_csv)
        if "judge_row_id" in prev.columns:
            already = set(prev["judge_row_id"].astype(str).tolist())

    os.makedirs(os.path.dirname(args.out_csv), exist_ok=True)
    need_header = not os.path.exists(args.out_csv) or os.path.getsize(args.out_csv) == 0 or not args.resume

    judge_cols = [
        "judge_row_id",
        "judge_backend",
        "judge_model",
        "judge_prompt_version",
        "judge_prompt_config",
        "judge_n_samples",
        "judge_seed_used",
        "judge_timestamp_utc",
        "judge_pick_if_given",
        "judge_pick_if_swapped",
        "judge_more_human",
        "judge_ai_probability_A_mean",
        "judge_ai_probability_A_std",
        "judge_ai_probability_B_mean",
        "judge_ai_probability_B_std",
        "judge_confidence_mean",
        "judge_confidence_std",
        "judge_rationale",
        "judge_raw_json",
        "judge_status",
    ]
    out_fieldnames = list(df.columns) + [c for c in judge_cols if c not in df.columns]

    with open(args.out_csv, "a", newline="", encoding="utf-8", errors="replace") as f:
        w = csv.DictWriter(f, fieldnames=out_fieldnames)
        if need_header:
            w.writeheader()

        for i, row in df.iterrows():
            rid = stable_row_id(row, cols=id_cols)
            if args.resume and rid in already:
                continue

            review_a = safe_utf8(row.get(col_a, ""))
            review_b = safe_utf8(row.get(col_b, ""))

            probs_a: List[float] = []
            probs_b: List[float] = []
            confs: List[float] = []
            last_raw = ""
            last_err = None
            last_obj: Optional[Dict[str, Any]] = None
            last_seed_used: Optional[int] = None

            base_seed = parse_seed_value(row.get(seed_col)) if seed_col else None
            if base_seed is None:
                base_seed = args.seed_base + i * 1000

            for s in range(args.n_samples):
                seed = int(base_seed) + s
                user_prompt = user_tmpl.replace("{review_a}", review_a).replace("{review_b}", review_b)
                messages = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ]

                try:
                    raw = get_completion(
                        messages,
                        seed=seed,
                        temperature=temperature,
                        top_p=top_p,
                        max_new_tokens=max_new_tokens,
                        deterministic=deterministic_decode,
                    )
                    last_seed_used = seed
                    last_raw = raw
                    obj = _extract_first_json(raw)

                    required_keys = ["more_human", "ai_probability_A", "ai_probability_B", "confidence", "rationale"]
                    if is_selfswap_diag:
                        required_keys.extend(["pick_if_given", "pick_if_swapped"])
                    for k in required_keys:
                        if k not in obj:
                            raise ValueError(f"Missing key: {k}")

                    mh = str(obj["more_human"]).strip().upper()
                    if mh not in {"A", "B", "TIE"}:
                        raise ValueError(f"Invalid more_human: {obj['more_human']}")
                    obj["more_human"] = mh
                    if is_selfswap_diag:
                        pig = str(obj["pick_if_given"]).strip().upper()
                        pis = str(obj["pick_if_swapped"]).strip().upper()
                        if pig not in {"A", "B", "TIE"}:
                            raise ValueError(f"Invalid pick_if_given: {obj['pick_if_given']}")
                        if pis not in {"A", "B", "TIE"}:
                            raise ValueError(f"Invalid pick_if_swapped: {obj['pick_if_swapped']}")
                        obj["pick_if_given"] = pig
                        obj["pick_if_swapped"] = pis

                    probs_a.append(float(obj["ai_probability_A"]))
                    probs_b.append(float(obj["ai_probability_B"]))
                    confs.append(float(obj["confidence"]))
                    last_obj = obj
                    last_err = None
                except Exception as e:
                    last_err = str(e)
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

            if len(probs_a) == 0 or last_obj is None:
                out["judge_pick_if_given"] = None
                out["judge_pick_if_swapped"] = None
                out["judge_more_human"] = None
                out["judge_ai_probability_A_mean"] = None
                out["judge_ai_probability_A_std"] = None
                out["judge_ai_probability_B_mean"] = None
                out["judge_ai_probability_B_std"] = None
                out["judge_confidence_mean"] = None
                out["judge_confidence_std"] = None
                out["judge_rationale"] = f"JUDGE_FAILED: {last_err}"
                out["judge_raw_json"] = last_raw
                out["judge_status"] = "FAIL"
            else:
                import math

                def mean(xs):
                    return sum(xs) / len(xs)

                def std(xs):
                    if len(xs) <= 1:
                        return 0.0
                    m = mean(xs)
                    return math.sqrt(sum((x - m) ** 2 for x in xs) / (len(xs) - 1))

                out["judge_pick_if_given"] = safe_utf8(last_obj.get("pick_if_given", ""))
                out["judge_pick_if_swapped"] = safe_utf8(last_obj.get("pick_if_swapped", ""))
                out["judge_more_human"] = safe_utf8(last_obj.get("more_human", ""))
                out["judge_ai_probability_A_mean"] = mean(probs_a)
                out["judge_ai_probability_A_std"] = std(probs_a)
                out["judge_ai_probability_B_mean"] = mean(probs_b)
                out["judge_ai_probability_B_std"] = std(probs_b)
                out["judge_confidence_mean"] = mean(confs)
                out["judge_confidence_std"] = std(confs)
                out["judge_rationale"] = safe_utf8(last_obj.get("rationale", ""))
                out["judge_raw_json"] = json.dumps(last_obj, ensure_ascii=False)
                out["judge_status"] = "OK"

            w.writerow(out)
            f.flush()

            if args.sleep > 0:
                time.sleep(args.sleep)

    print(f"Wrote: {args.out_csv}")


if __name__ == "__main__":
    main()
