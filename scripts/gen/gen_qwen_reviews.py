#!/usr/bin/env python3
import argparse
import csv
import json
import os
import random
import re
import time
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


QWEN_MODEL_NAME = os.environ.get("QWEN_MODEL_NAME", "/WAVE/datasets/oignat_lab/QWEN3")
TEMPERATURE = float(os.environ.get("TEMPERATURE", "1.0"))
TOP_P = float(os.environ.get("TOP_P", "1.0"))
MAX_NEW_TOKENS = int(os.environ.get("MAX_NEW_TOKENS", "260"))

DEVANAGARI_RE = re.compile(r"[\u0900-\u097F]")
LATIN_RE = re.compile(r"[A-Za-z]")
HAUSA_EXT_LATIN_RE = re.compile(r"[A-Za-z\u0181\u018A\u0198\u0199\u0253\u0257\u01B3\u01B4]")

_tokenizer = None
_model = None


def safe_utf8(x: Any) -> str:
    if x is None:
        return ""
    if not isinstance(x, str):
        x = str(x)
    return x.encode("utf-8", errors="replace").decode("utf-8", errors="replace")


def normalize_ws(s: Any) -> str:
    return re.sub(r"\s+", " ", safe_utf8(s).strip())


def is_devanagari_ok(text: str) -> int:
    t = text or ""
    if LATIN_RE.search(t):
        return 0
    if not DEVANAGARI_RE.search(t):
        return 0
    return 1


def is_hausa_ok(text: str) -> int:
    t = text or ""
    if DEVANAGARI_RE.search(t):
        return 0
    if not HAUSA_EXT_LATIN_RE.search(t):
        return 0
    return 1


def is_any_nonempty(text: str) -> int:
    return 1 if normalize_ws(text) else 0


LANG_SPECS = {
    "hausa": {
        "prompt_lang_choices": [],
        "ok_col": "is_hausa_ok",
        "validator": is_hausa_ok,
        "retry_status": "RETRY_NON_HAUSA",
    },
    "nepali": {
        "prompt_lang_choices": [],
        "ok_col": "is_devanagari_ok",
        "validator": is_devanagari_ok,
        "retry_status": "RETRY_NON_DEVANAGARI",
    },
    "swahilli": {
        "prompt_lang_choices": [],
        "ok_col": "is_swahilli_ok",
        "validator": is_hausa_ok,
        "retry_status": "RETRY_NON_SWAHILLI",
    },
    "swahili": {
        "prompt_lang_choices": [],
        "ok_col": "is_swahilli_ok",
        "validator": is_hausa_ok,
        "retry_status": "RETRY_NON_SWAHILLI",
    },
}


def resolve_lang(lang: str) -> str:
    l = str(lang or "").strip().lower()
    if l in LANG_SPECS:
        return l
    return "generic"


def lang_spec(lang: str) -> Dict[str, Any]:
    l = resolve_lang(lang)
    if l in LANG_SPECS:
        return LANG_SPECS[l]
    return {
        "prompt_lang_choices": [],
        "ok_col": "is_language_ok",
        "validator": is_any_nonempty,
        "retry_status": "RETRY_EMPTY",
    }


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _load_model():
    global _tokenizer, _model
    if _tokenizer is None or _model is None:
        local_only = os.path.isdir(QWEN_MODEL_NAME)
        _tokenizer = AutoTokenizer.from_pretrained(
            QWEN_MODEL_NAME,
            local_files_only=local_only,
            trust_remote_code=True,
        )
        _model = AutoModelForCausalLM.from_pretrained(
            QWEN_MODEL_NAME,
            torch_dtype="auto",
            device_map="auto",
            local_files_only=local_only,
            trust_remote_code=True,
        )
        _model.eval()
    return _tokenizer, _model


def _apply_chat_template(tokenizer, messages: List[Dict[str, str]]) -> str:
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


def extract_first_json(text: str) -> Dict[str, Any]:
    t = safe_utf8(text).strip()
    if t.startswith("```"):
        t = re.sub(r"^\s*```[a-zA-Z]*\s*", "", t)
        t = re.sub(r"\s*```\s*$", "", t).strip()

    i = t.find("{")
    if i < 0:
        raise ValueError(f"No JSON object found in output: {t[:200]}")

    obj, _end = json.JSONDecoder().raw_decode(t[i:])
    if not isinstance(obj, dict):
        raise ValueError("First JSON value is not an object")
    return obj


def load_prompt_config(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    for k in ["prompt_version", "system_prompt", "user_prompt_templates"]:
        if k not in cfg:
            raise RuntimeError(f"prompt_config missing required key: {k}")
    return cfg


def load_fewshot_pool(cfg: Dict[str, Any], validator) -> Optional[pd.DataFrame]:
    fs = cfg.get("fewshot", {}) or {}
    if not fs.get("enabled", False):
        return None

    train_csv = fs.get("train_csv")
    if not train_csv:
        raise RuntimeError("fewshot.enabled=true but fewshot.train_csv missing in prompt_config")

    df = pd.read_csv(train_csv)
    text_col = fs.get("text_col", "review_text")
    label_col = fs.get("label_col", "label")
    if text_col not in df.columns or label_col not in df.columns:
        raise RuntimeError(f"fewshot train_csv missing required cols: {text_col}, {label_col}")

    out = df.copy()
    out[text_col] = out[text_col].fillna("").astype(str).map(normalize_ws)
    out = out[out[text_col].str.strip() != ""]
    out = out[out[text_col].map(lambda s: validator(s) == 1)]
    return out.reset_index(drop=True)


def pick_fewshot_examples(
    pool_df: Optional[pd.DataFrame],
    cfg: Dict[str, Any],
    sentiment: str,
    domain: str,
    prompt_lang: str,
    seed: int,
) -> Tuple[List[Dict[str, str]], str, int]:
    fs = cfg.get("fewshot", {}) or {}
    if not fs.get("enabled", False) or pool_df is None or len(pool_df) == 0:
        return [], "", 0

    k = int(fs.get("k", 0))
    if k <= 0:
        return [], "", 0

    text_col = fs.get("text_col", "review_text")
    label_col = fs.get("label_col", "label")
    match_sent = bool(fs.get("match_sentiment", True))
    rng = random.Random(int(seed) + int(fs.get("seed_offset", 0)))

    cand = pool_df
    if match_sent:
        want_label = 1 if sentiment == "POS" else 0
        labels = pd.to_numeric(pool_df[label_col], errors="coerce")
        cand = pool_df[labels == want_label]
    cand = cand.reset_index(drop=True)
    if cand.empty:
        return [], "", 0

    k_used = min(k, len(cand))
    idxs = rng.sample(range(len(cand)), k_used)
    user_prompt = cfg["user_prompt_templates"][sentiment][prompt_lang].format(domain=domain)
    if sentiment == "POS":
        smin, smax = int(fs.get("pos_score_min", 7)), int(fs.get("pos_score_max", 10))
    else:
        smin, smax = int(fs.get("neg_score_min", 1)), int(fs.get("neg_score_max", 6))

    pairs: List[Dict[str, str]] = []
    ids: List[str] = []
    for j in idxs:
        ex = cand.iloc[j]
        ex_text = normalize_ws(ex[text_col])
        ex_score = rng.randint(smin, smax)
        pairs.append({"role": "user", "content": user_prompt})
        pairs.append({"role": "assistant", "content": json.dumps({"Review": ex_text, "Review_Score": ex_score}, ensure_ascii=False)})
        ids.append(f"poolrow={int(ex.name)};label={int(ex[label_col])}")
    return pairs, "|".join(ids), k_used


def build_messages(cfg: Dict[str, Any], prompt_lang: str, sentiment: str, domain: str, fewshot_pairs: List[Dict[str, str]]) -> List[Dict[str, str]]:
    system_prompt = cfg["system_prompt"][prompt_lang]
    user_prompt = cfg["user_prompt_templates"][sentiment][prompt_lang].format(domain=domain)
    messages = [{"role": "system", "content": system_prompt}]
    if fewshot_pairs:
        messages.extend(fewshot_pairs)
    messages.append({"role": "user", "content": user_prompt})
    return messages


def generate_one(messages: List[Dict[str, str]], seed: int, temperature: float, top_p: float, max_new_tokens: int) -> str:
    tok, model = _load_model()
    _seed_everything(seed)
    text = _apply_chat_template(tok, messages)
    inputs = tok([text], return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(
            **inputs,
            do_sample=True,
            temperature=float(temperature),
            top_p=float(top_p),
            max_new_tokens=int(max_new_tokens),
        )
    gen_ids = out[0][inputs["input_ids"].shape[1] :]
    return safe_utf8(tok.decode(gen_ids, skip_special_tokens=True)).strip()


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Language-aware Qwen generator for review-style outputs")
    ap.add_argument("--targets_csv", required=True)
    ap.add_argument(
        "--out_csv",
        default="",
        help="Optional explicit output CSV path. Default: languages/{lang}/gen/{run_id}__qwen.csv",
    )
    ap.add_argument("--run_id", required=True)
    ap.add_argument("--lang", required=True)
    ap.add_argument("--domain", default="movie")
    ap.add_argument("--prompt_lang", default="")
    ap.add_argument("--prompt_config", required=True)
    ap.add_argument("--max_tries", type=int, default=3)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--sleep", type=float, default=0.0)
    args = ap.parse_args()

    return args


def default_out_csv(lang: str, run_id: str) -> str:
    lang_norm = (lang or "unknown").strip().lower()
    run_norm = (run_id or "run").strip()
    return os.path.join("languages", lang_norm, "gen", f"{run_norm}__qwen.csv")


def main() -> None:
    args = parse_args()
    if not str(args.out_csv or "").strip():
        args.out_csv = default_out_csv(args.lang, args.run_id)

    cfg = load_prompt_config(args.prompt_config)
    spec = lang_spec(args.lang)
    prompt_keys = list((cfg.get("system_prompt") or {}).keys())
    if not prompt_keys:
        raise RuntimeError("prompt_config has empty system_prompt map")

    if args.prompt_lang:
        prompt_lang = args.prompt_lang
    else:
        guess = args.lang.lower()
        prompt_lang = guess if guess in prompt_keys else ("english" if "english" in prompt_keys else prompt_keys[0])
    if prompt_lang not in prompt_keys:
        prompt_lang = "english" if "english" in prompt_keys else prompt_keys[0]

    gen_params = cfg.get("generation_params", {}) or {}
    temperature = float(gen_params.get("temperature", TEMPERATURE))
    top_p = float(gen_params.get("top_p", TOP_P))
    max_new_tokens = int(gen_params.get("max_new_tokens", MAX_NEW_TOKENS))

    tdf = pd.read_csv(args.targets_csv)
    required = ["target_id", "seed", "sentiment"]
    missing = [c for c in required if c not in tdf.columns]
    if missing:
        raise RuntimeError(f"targets_csv missing columns: {missing}")

    ok_col = spec["ok_col"]
    retry_status = spec["retry_status"]
    validator = spec["validator"]
    fewshot_pool = load_fewshot_pool(cfg, validator=validator)

    out_cols = [
        "run_id",
        "target_id",
        "seed",
        "label",
        "sentiment",
        "domain",
        "prompt_lang",
        "Review",
        "Review_Score",
        ok_col,
        "raw_model_output",
        "qwen_model_name",
        "prompt_version",
        "prompt_config",
        "fewshot_k",
        "fewshot_example_ids",
        "status",
    ]
    if ok_col == "is_swahilli_ok":
        out_cols.append("is_swahili_ok")

    already = set()
    if args.resume and os.path.exists(args.out_csv):
        prev = pd.read_csv(args.out_csv)
        if "target_id" in prev.columns:
            already = set(pd.to_numeric(prev["target_id"], errors="coerce").dropna().astype(int).tolist())
    if (not args.resume) and os.path.exists(args.out_csv):
        os.remove(args.out_csv)

    os.makedirs(os.path.dirname(args.out_csv) or ".", exist_ok=True)
    need_header = not os.path.exists(args.out_csv) or os.path.getsize(args.out_csv) == 0

    with open(args.out_csv, "a", newline="", encoding="utf-8", errors="replace") as f:
        w = csv.DictWriter(f, fieldnames=out_cols)
        if need_header:
            w.writeheader()

        for _, row in tdf.iterrows():
            tid = int(row["target_id"])
            if args.resume and tid in already:
                continue

            seed = int(row["seed"])
            sentiment = str(row["sentiment"]).strip().upper()
            label = row["label"] if "label" in row else ""

            fewshot_pairs, fewshot_ids, fewshot_k = pick_fewshot_examples(
                pool_df=fewshot_pool,
                cfg=cfg,
                sentiment=sentiment,
                domain=args.domain,
                prompt_lang=prompt_lang,
                seed=seed,
            )
            messages = build_messages(
                cfg=cfg,
                prompt_lang=prompt_lang,
                sentiment=sentiment,
                domain=args.domain,
                fewshot_pairs=fewshot_pairs,
            )

            review_text = ""
            review_score: Any = ""
            is_ok = 0
            raw_output = ""
            status = "OK"

            for attempt in range(1, int(args.max_tries) + 1):
                try:
                    raw_output = generate_one(
                        messages=messages,
                        seed=seed + (attempt - 1),
                        temperature=temperature,
                        top_p=top_p,
                        max_new_tokens=max_new_tokens,
                    )
                    obj = extract_first_json(raw_output)
                    review_text = normalize_ws(obj.get("Review", ""))
                    review_score = obj.get("Review_Score", "")
                    if isinstance(review_score, str) and review_score.strip().isdigit():
                        review_score = int(review_score.strip())

                    is_ok = int(validator(review_text))
                    if is_ok != 1:
                        status = retry_status
                        continue

                    if isinstance(review_score, int):
                        if sentiment == "POS" and not (7 <= review_score <= 10):
                            status = "WARN_SCORE_OUT_OF_RANGE_POS"
                        elif sentiment == "NEG" and not (1 <= review_score <= 6):
                            status = "WARN_SCORE_OUT_OF_RANGE_NEG"
                        else:
                            status = "OK"
                    else:
                        status = "WARN_SCORE_NOT_INT"
                    break
                except Exception as e:
                    status = f"FAIL_ATTEMPT_{attempt}: {type(e).__name__}: {safe_utf8(e)}"
                    raw_output = raw_output or status

            out_row = {
                "run_id": args.run_id,
                "target_id": tid,
                "seed": seed,
                "label": label,
                "sentiment": sentiment,
                "domain": args.domain,
                "prompt_lang": prompt_lang,
                "Review": review_text,
                "Review_Score": review_score,
                ok_col: int(is_ok),
                "raw_model_output": safe_utf8(raw_output),
                "qwen_model_name": QWEN_MODEL_NAME,
                "prompt_version": cfg.get("prompt_version", "unknown"),
                "prompt_config": args.prompt_config,
                "fewshot_k": int(fewshot_k),
                "fewshot_example_ids": fewshot_ids,
                "status": status,
            }
            if ok_col == "is_swahilli_ok":
                # Keep compatibility with both spellings.
                out_row["is_swahili_ok"] = int(is_ok)
            w.writerow(out_row)
            f.flush()
            if args.sleep > 0:
                time.sleep(args.sleep)

    print(f"Wrote: {args.out_csv}")


if __name__ == "__main__":
    main()
