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
import transformers
from transformers.utils import logging as hf_logging


hf_logging.set_verbosity_error()

LLAMA_MODEL_NAME = os.environ.get(
    "LLAMA_MODEL_NAME",
    os.environ.get("LLAMA_MODEL_ID", "/WAVE/datasets/oignat_lab/Meta-Llama-3.1-8B-Instruct"),
)
TEMPERATURE = float(os.environ.get("TEMPERATURE", "1.0"))
TOP_P = float(os.environ.get("TOP_P", "1.0"))
MAX_NEW_TOKENS = int(os.environ.get("MAX_NEW_TOKENS", "260"))

DEVANAGARI_RE = re.compile(r"[\u0900-\u097F]")
LATIN_RE = re.compile(r"[A-Za-z\u0181\u018A\u0198\u0199\u0253\u0257\u01B3\u01B4]")


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


def is_latin_ok(text: str) -> int:
    t = text or ""
    if DEVANAGARI_RE.search(t):
        return 0
    if not LATIN_RE.search(t):
        return 0
    return 1


def is_any_nonempty(text: str) -> int:
    return 1 if normalize_ws(text) else 0


LANG_SPECS: Dict[str, Dict[str, Any]] = {
    "nepali": {
        "ok_col": "is_devanagari_ok",
        "validator": is_devanagari_ok,
        "retry_status": "RETRY_NON_DEVANAGARI",
    },
    "hausa": {
        "ok_col": "is_hausa_ok",
        "validator": is_latin_ok,
        "retry_status": "RETRY_NON_HAUSA",
    },
    "swahilli": {
        "ok_col": "is_swahilli_ok",
        "validator": is_latin_ok,
        "retry_status": "RETRY_NON_SWAHILLI",
    },
    "swahili": {
        "ok_col": "is_swahilli_ok",
        "validator": is_latin_ok,
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
        "ok_col": "is_language_ok",
        "validator": is_any_nonempty,
        "retry_status": "RETRY_EMPTY",
    }


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_prompt_config(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    for k in ["prompt_version", "system_prompt", "user_prompt_templates"]:
        if k not in cfg:
            raise RuntimeError(f"prompt_config missing required key: {k}")
    return cfg


def _extract_first_json_obj(text: str) -> Optional[Dict[str, Any]]:
    t = safe_utf8(text).strip()
    if t.startswith("```"):
        t = re.sub(r"^\s*```[a-zA-Z]*\s*", "", t)
        t = re.sub(r"\s*```\s*$", "", t).strip()

    i = t.find("{")
    if i < 0:
        return None

    try:
        obj, _ = json.JSONDecoder().raw_decode(t[i:])
    except Exception:
        return None
    if isinstance(obj, dict):
        return obj
    return None


def _extract_score_any(text: str) -> Optional[int]:
    t = safe_utf8(text)
    pats = [
        r'(?i)"?Review_Score"?\s*:\s*([0-9]{1,2})',
        r'(?i)\breview[_\s]*score\b\s*[:=]\s*([0-9]{1,2})',
        r'([0-9]{1,2})\s*/\s*10',
    ]
    for p in pats:
        m = re.search(p, t)
        if m:
            n = int(m.group(1))
            if 1 <= n <= 10:
                return n
    return None


def _try_salvage(text: str) -> Optional[Dict[str, Any]]:
    score = _extract_score_any(text)
    if score is None:
        return None

    t = safe_utf8(text)
    m_review = re.search(r'(?i)"?Review"?\s*:\s*"([^"]+)"', t, flags=re.DOTALL)
    if not m_review:
        m_review = re.search(r'(?i)"?Review"?\s*:\s*([^\n\r]+)', t)

    if m_review:
        review = normalize_ws(m_review.group(1).strip().strip('"'))
    else:
        review = re.sub(r'(?i)"?Review_Score"?\s*:\s*[0-9]{1,2}', " ", t)
        review = re.sub(r'(?i)\breview[_\s]*score\b\s*[:=]\s*[0-9]{1,2}', " ", review)
        review = normalize_ws(review.strip().strip('"').strip("'"))

    if not review:
        return None
    return {"Review": review, "Review_Score": score}


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
        raise RuntimeError(f"fewshot train_csv missing cols: {text_col}, {label_col}")

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
        cand2 = pool_df[pd.to_numeric(pool_df[label_col], errors="coerce") == want_label]
        if len(cand2) > 0:
            cand = cand2
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
        ids.append(f"poolrow={int(ex.name)};label={safe_utf8(ex[label_col])}")
    return pairs, "|".join(ids), k_used


def build_messages(cfg: Dict[str, Any], prompt_lang: str, sentiment: str, domain: str, fewshot_pairs: List[Dict[str, str]]) -> List[Dict[str, str]]:
    system_prompt = cfg["system_prompt"][prompt_lang]
    user_prompt = cfg["user_prompt_templates"][sentiment][prompt_lang].format(domain=domain)
    messages = [{"role": "system", "content": system_prompt}]
    if fewshot_pairs:
        messages.extend(fewshot_pairs)
    messages.append({"role": "user", "content": user_prompt})
    return messages


class LlamaGenerator:
    def __init__(self, model_name: str):
        self.pipe = transformers.pipeline(
            "text-generation",
            model=model_name,
            device_map="auto",
            dtype=torch.bfloat16,
        )
        try:
            self.pipe.tokenizer.pad_token_id = self.pipe.tokenizer.eos_token_id
            self.pipe.model.config.pad_token_id = self.pipe.tokenizer.eos_token_id
        except Exception:
            pass

    def generate(self, messages: List[Dict[str, str]], seed: int, temperature: float, top_p: float, max_new_tokens: int) -> str:
        _seed_everything(seed)
        tok = self.pipe.tokenizer
        prompt_text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        outs = self.pipe(
            prompt_text,
            max_new_tokens=int(max_new_tokens),
            do_sample=True,
            temperature=float(temperature),
            top_p=float(top_p),
            return_full_text=False,
        )
        return safe_utf8(outs[0].get("generated_text", "")).strip()


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Language-aware Llama generator for review-style outputs")
    ap.add_argument("--targets_csv", required=True)
    ap.add_argument(
        "--out_csv",
        default="",
        help="Optional explicit output CSV path. Default: languages/{lang}/gen/{run_id}__llama.csv",
    )
    ap.add_argument("--run_id", required=True)
    ap.add_argument("--lang", required=True)
    ap.add_argument("--domain", default="review")
    ap.add_argument("--prompt_lang", default="")
    ap.add_argument("--prompt_config", required=True)
    ap.add_argument("--max_tries", type=int, default=3)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--sleep", type=float, default=0.0)
    return ap.parse_args()


def default_out_csv(lang: str, run_id: str) -> str:
    lang_norm = (lang or "unknown").strip().lower()
    run_norm = (run_id or "run").strip()
    return os.path.join("languages", lang_norm, "gen", f"{run_norm}__llama.csv")


def main() -> None:
    args = parse_args()
    if not str(args.out_csv or "").strip():
        args.out_csv = default_out_csv(args.lang, args.run_id)

    spec = lang_spec(args.lang)
    cfg = load_prompt_config(args.prompt_config)

    prompt_keys = list((cfg.get("system_prompt") or {}).keys())
    if not prompt_keys:
        raise RuntimeError("prompt_config has empty system_prompt map")
    if args.prompt_lang:
        prompt_lang = args.prompt_lang
    else:
        prompt_lang = args.lang.lower()
        if prompt_lang == "swahili":
            prompt_lang = "swahilli"
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
        spec["ok_col"],
        "raw_model_output",
        "llama_model_name",
        "prompt_version",
        "prompt_config",
        "fewshot_k",
        "fewshot_example_ids",
        "status",
    ]
    if spec["ok_col"] == "is_swahilli_ok":
        out_cols.append("is_swahili_ok")

    already = set()
    if args.resume and os.path.exists(args.out_csv):
        prev = pd.read_csv(args.out_csv)
        if "target_id" in prev.columns:
            already = set(pd.to_numeric(prev["target_id"], errors="coerce").dropna().astype(int).tolist())
    if (not args.resume) and os.path.exists(args.out_csv):
        os.remove(args.out_csv)

    fewshot_pool = load_fewshot_pool(cfg, validator=spec["validator"])
    gen = LlamaGenerator(LLAMA_MODEL_NAME)

    need_header = not os.path.exists(args.out_csv) or os.path.getsize(args.out_csv) == 0
    os.makedirs(os.path.dirname(args.out_csv) or ".", exist_ok=True)

    with open(args.out_csv, "a", newline="", encoding="utf-8", errors="replace") as f:
        w = csv.DictWriter(f, fieldnames=out_cols)
        if need_header:
            w.writeheader()

        for _, row in tdf.iterrows():
            tid = int(row["target_id"])
            if args.resume and tid in already:
                continue

            seed = int(row["seed"])
            sentiment = str(row.get("sentiment", "")).strip().upper()
            label = row.get("label", "")
            domain = str(row.get("domain", args.domain) or args.domain)

            fewshot_pairs, fewshot_ids, fewshot_k = pick_fewshot_examples(
                pool_df=fewshot_pool,
                cfg=cfg,
                sentiment=sentiment,
                domain=domain,
                prompt_lang=prompt_lang,
                seed=seed,
            )
            messages = build_messages(
                cfg=cfg,
                prompt_lang=prompt_lang,
                sentiment=sentiment,
                domain=domain,
                fewshot_pairs=fewshot_pairs,
            )

            review_text = ""
            review_score: Any = ""
            raw_output = ""
            quality_ok = 0
            status = "OK"

            for attempt in range(1, int(args.max_tries) + 1):
                try:
                    raw_output = gen.generate(
                        messages=messages,
                        seed=seed + (attempt - 1),
                        temperature=temperature,
                        top_p=top_p,
                        max_new_tokens=max_new_tokens,
                    )
                    obj = _extract_first_json_obj(raw_output)
                    salvaged = False
                    if obj is None:
                        obj = _try_salvage(raw_output)
                        salvaged = obj is not None
                    if obj is None:
                        raise ValueError("Model output did not contain parseable JSON")

                    review_text = normalize_ws(obj.get("Review", ""))
                    review_score = obj.get("Review_Score", "")
                    if isinstance(review_score, str) and review_score.strip().isdigit():
                        review_score = int(review_score.strip())

                    quality_ok = int(spec["validator"](review_text))
                    if quality_ok != 1:
                        status = str(spec["retry_status"])
                        continue

                    if isinstance(review_score, int):
                        if sentiment == "POS" and not (7 <= review_score <= 10):
                            status = "WARN_SCORE_OUT_OF_RANGE_POS"
                        elif sentiment == "NEG" and not (1 <= review_score <= 6):
                            status = "WARN_SCORE_OUT_OF_RANGE_NEG"
                        else:
                            status = "OK_SALVAGED" if salvaged else "OK"
                    else:
                        status = "WARN_SCORE_NOT_INT"
                    break
                except Exception as e:
                    status = f"FAIL_ATTEMPT_{attempt}: {type(e).__name__}: {safe_utf8(e)}"
                    raw_output = raw_output or status

            out_row: Dict[str, Any] = {
                "run_id": args.run_id,
                "target_id": tid,
                "seed": seed,
                "label": label,
                "sentiment": sentiment,
                "domain": domain,
                "prompt_lang": prompt_lang,
                "Review": review_text,
                "Review_Score": review_score,
                spec["ok_col"]: int(quality_ok),
                "raw_model_output": safe_utf8(raw_output),
                "llama_model_name": LLAMA_MODEL_NAME,
                "prompt_version": cfg.get("prompt_version", "unknown"),
                "prompt_config": args.prompt_config,
                "fewshot_k": int(fewshot_k),
                "fewshot_example_ids": fewshot_ids,
                "status": status,
            }
            if spec["ok_col"] == "is_swahilli_ok":
                out_row["is_swahili_ok"] = int(quality_ok)
            w.writerow(out_row)
            f.flush()
            if args.sleep > 0:
                time.sleep(args.sleep)

    print(f"Wrote: {args.out_csv}")


if __name__ == "__main__":
    main()
