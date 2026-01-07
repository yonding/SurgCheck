#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
LLM-as-a-Judge evaluator for Surgical VQA jsonl outputs.

- Traverse output_qwen/zeroshot/**/*.jsonl
- File-level parallel evaluation (ThreadPoolExecutor)
- Extract prediction text after '\nassistant\n'
- Judge with GPT (1–5 score)
- Save results in result_qwen/zeroshot_gpt/{same structure}.jsonl
- score >= 4 -> correct, else incorrect
- real-time disk flush
"""

import argparse
import json
import os
import random
import time
from pathlib import Path
from typing import Any, Dict, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed

from openai import OpenAI

ASSISTANT_MARK = "\nassistant\n"
# --------------------------------------------------
# Judge prompt
# --------------------------------------------------
SYSTEM_PROMPT = (
    "You are a strict, non-verbose judge. "
    "Follow the rubric exactly. Output valid JSON only."
)

RUBRIC_PROMPT = """[RUBRIC]
Task: Unified semantic and logical correctness scoring for surgical VQA (1–5).

Scoring:
1 = Completely different or opposite meaning
2 = Loosely related but logically incorrect
3 = Partially correct
4 = Clinically/logically interchangeable
5 = Perfectly equivalent

Rules:
- Yes/No answers must match exactly for 5
- Descriptive answers require both action and target for 4–5

Output JSON only:
{"score": 1|2|3|4|5, "reason": "..."}
"""

# --------------------------------------------------
# Utils
# --------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--in_dir", type=str, required=True)
    p.add_argument("--out_dir", type=str, required=True)
    p.add_argument("--model", type=str, default="gpt-5")
    p.add_argument("--resume", type=lambda x: str(x).lower() in {"1", "true", "yes"}, default=True)
    p.add_argument("--max_retries", type=int, default=5)
    p.add_argument("--num_workers", type=int, default=8)
    return p.parse_args()


def extract_pred(pred: Any) -> str:
    s = pred if isinstance(pred, str) else str(pred)
    if ASSISTANT_MARK in s:
        s = s.split(ASSISTANT_MARK)[-1]
    return s.strip()


def safe_json_loads(text: str) -> Optional[Dict]:
    if not text:
        return None
    text = text.strip()

    try:
        obj = json.loads(text)
        if isinstance(obj, dict):
            return obj
    except Exception:
        pass

    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        try:
            obj = json.loads(text[start:end + 1])
            if isinstance(obj, dict):
                return obj
        except Exception:
            pass

    return None


def count_lines(path: Path) -> int:
    if not path.exists():
        return 0
    with path.open("r", encoding="utf-8") as f:
        return sum(1 for _ in f)

# --------------------------------------------------
# Judge
# --------------------------------------------------
def judge_sample(
    client: OpenAI,
    model: str,
    question: str,
    pred: str,
    gt: str,
    max_retries: int,
) -> Dict[str, Any]:

    prompt = (
        RUBRIC_PROMPT
        + "\n[INPUTS]\n"
        + f"Question: {question}\n"
        + f"Prediction: {pred}\n"
        + f"References: {gt}\n"
    )

    last_err = None
    last_raw = ""

    for attempt in range(max_retries):
        try:
            resp = client.responses.create(
                model=model,
                input=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
            )

            raw = resp.output_text
            last_raw = raw

            parsed = safe_json_loads(raw)
            if parsed is None or "score" not in parsed:
                raise ValueError(f"Invalid judge output: {raw[:200]}")

            score = int(parsed["score"])
            reason = str(parsed.get("reason", "")).strip()

            if score < 1 or score > 5:
                raise ValueError(f"Score out of range: {score}")

            return {"score": score, "reason": reason, "raw": raw}

        except Exception as e:
            last_err = e
            time.sleep(0.5 * (2 ** attempt) + random.uniform(0, 0.2))

    raise RuntimeError(f"Judge failed: {last_err}\nLast output: {last_raw[:300]}")


# --------------------------------------------------
# Per-file processing (thread target)
# --------------------------------------------------
def process_one_file(in_file: Path, args) -> str:
    client = OpenAI()  # thread-local client

    rel_path = in_file.relative_to(Path(args.in_dir))
    out_file = Path(args.out_dir) / rel_path
    out_file.parent.mkdir(parents=True, exist_ok=True)

    skip = count_lines(out_file) if args.resume else 0
    print(f"[PROCESS] {rel_path} (resume skip={skip})")

    with in_file.open("r", encoding="utf-8") as fin, \
         out_file.open("a", encoding="utf-8") as fout:

        for idx, line in enumerate(fin):
            if args.resume and idx < skip:
                continue

            line = line.strip()
            if not line:
                continue

            data = json.loads(line)

            question = data.get("question", "")
            gt = data.get("gt", "")
            pred_extracted = extract_pred(data.get("pred", ""))

            judge = judge_sample(
                client,
                args.model,
                question,
                pred_extracted,
                gt,
                args.max_retries,
            )

            tag = "correct" if judge["score"] >= 4 else "incorrect"

            data["pred_extracted"] = pred_extracted
            data["judge"] = judge
            data["judge_tag"] = tag

            fout.write(json.dumps(data, ensure_ascii=False) + "\n")
            fout.flush()
            os.fsync(fout.fileno())

    return str(rel_path)


# --------------------------------------------------
# Main
# --------------------------------------------------
def main():
    args = parse_args()
    in_dir = Path(args.in_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    files = sorted(in_dir.rglob("*.jsonl"))
    if not files:
        print("No jsonl files found.")
        return

    num_workers = min(args.num_workers, len(files))
    print(f"[INFO] Using {num_workers} threads")

    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = [
            executor.submit(process_one_file, f, args)
            for f in files
        ]

        for fut in as_completed(futures):
            try:
                fname = fut.result()
                print(f"[DONE] {fname}")
            except Exception as e:
                print(f"[ERROR] {e}")

    print("✅ Evaluation finished.")


if __name__ == "__main__":
    main()
