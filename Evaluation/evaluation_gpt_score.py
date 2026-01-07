#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple


def safe_float(x) -> Optional[float]:
    try:
        return float(x)
    except Exception:
        return None


def norm_text(s: str) -> str:
    if s is None:
        return ""
    s = str(s).strip().lower()
    s = re.sub(r"\s+", " ", s)
    return s


def norm_label(s: str) -> str:
    # normalize labels like "blood_vessel" / "blood vessel" consistently
    s = norm_text(s)
    s = s.replace("-", " ").replace("_", " ")
    s = re.sub(r"\s+", " ", s).strip()
    return s


def read_jsonl(path: Path) -> List[Dict]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for ln, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as e:
                raise RuntimeError(f"[JSONDecodeError] {path} line {ln}: {e}") from e
    return rows


def build_label_vocab(rows: List[Dict]) -> List[str]:
    # vocab from gt labels in this file (normalized)
    vocab = sorted({norm_label(r.get("gt", "")) for r in rows if r.get("gt", "")})
    # longest-first for greedy matching
    vocab.sort(key=len, reverse=True)
    return vocab


def extract_pred_label(row: Dict, vocab: List[str]) -> str:
    """
    Map pred_extracted/pred text to one of the GT labels by substring match.
    - match longest vocab label appearing as a phrase in normalized pred text
    - if none matched: return normalized pred text (likely will count as wrong)
    """
    pred_text = row.get("pred_extracted", None)
    if pred_text is None:
        pred_text = row.get("pred", "")
    pred_norm = norm_label(pred_text)

    if not pred_norm:
        return ""

    # phrase match: word-boundary-ish using spaces
    # we search in a padded string to avoid partial word hits
    padded = f" {pred_norm} "
    for lab in vocab:
        if not lab:
            continue
        if f" {lab} " in padded:
            return lab

    # fallback: sometimes labels are single tokens but pred has punctuation
    # try regex word boundary on each token label (best effort)
    for lab in vocab:
        if not lab:
            continue
        pattern = r"\b" + re.escape(lab) + r"\b"
        if re.search(pattern, pred_norm):
            return lab

    return pred_norm


def macro_f1(y_true: List[str], y_pred: List[str]) -> float:
    if not y_true:
        return 0.0
    labels = sorted(set(y_true))  # macro over GT label set

    f1s: List[float] = []
    for c in labels:
        tp = sum(1 for yt, yp in zip(y_true, y_pred) if yt == c and yp == c)
        fp = sum(1 for yt, yp in zip(y_true, y_pred) if yt != c and yp == c)
        fn = sum(1 for yt, yp in zip(y_true, y_pred) if yt == c and yp != c)

        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0
        f1s.append(f1)

    return sum(f1s) / len(f1s) if f1s else 0.0

def eval_file(path: Path, positive_tag: str = "correct") -> Tuple[float, float, float, int]:
    rows = read_jsonl(path)
    if not rows:
        return 0.0, 0.0, 0.0, 0

    vocab = build_label_vocab(rows)

    scores: List[float] = []
    y_true: List[str] = []
    y_pred: List[str] = []

    total = 0
    correct_cnt = 0

    for r in rows:
        # score mean
        judge = r.get("judge", {}) or {}
        s = safe_float(judge.get("score", None))
        if s is not None:
            scores.append(s)

        # judge_tag-based accuracy
        tag = str(r.get("judge_tag", "")).strip().lower()
        total += 1
        if tag == positive_tag.lower():
            correct_cnt += 1

        # macro f1 still needs gt/pred-labels (best-effort extraction)
        gt = norm_label(r.get("gt", ""))
        pred = extract_pred_label(r, vocab)

        if gt == "":
            # gt 없는 row는 f1 계산에서 제외
            continue

        y_true.append(gt)
        y_pred.append(pred)

    score_mean = (sum(scores) / len(scores)) if scores else 0.0
    acc = (correct_cnt / total) if total > 0 else 0.0
    f1 = macro_f1(y_true, y_pred) if y_true else 0.0
    return score_mean, acc, f1, total


def main():
    ap = argparse.ArgumentParser(
        description="Per jsonl: mean(judge.score), accuracy, macro F1 (single-label multi-class) using gt as class label."
    )
    ap.add_argument(
        "--root_dir",
        type=str,
        default="/data/kayoung/repos/projects/Papers/IPCAI2026/Evaluation/result_endochat/zeroshot",
        help="Root directory containing jsonl files (recursive).",
    )
    ap.add_argument("--ext", type=str, default=".jsonl", help="Input extension (default: .jsonl)")
    ap.add_argument("--digits", type=int, default=4, help="Decimal digits for printing (default: 4)")
    args = ap.parse_args()

    root = Path(args.root_dir)
    if not root.exists():
        raise FileNotFoundError(f"root_dir not found: {root}")

    files = sorted(root.rglob(f"*{args.ext}"))
    if not files:
        print(f"(no files) under {root} with ext={args.ext}")
        return

    d = args.digits
    for fp in files:
        score_mean, acc, f1, n = eval_file(fp, positive_tag="correct")
        rel = fp.relative_to(root)
        print(f"{rel}/{score_mean:.{d}f}/{acc:.{d}f}/{f1:.{d}f}")


if __name__ == "__main__":
    main()
