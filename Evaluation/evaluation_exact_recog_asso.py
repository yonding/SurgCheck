#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Tuple, Optional, List
from collections import defaultdict

ASSIST_MARK = "\nassistant\n"


def extract_assistant_text(pred: str) -> str:
    if pred is None:
        return ""
    s = str(pred)
    if ASSIST_MARK in s:
        s = s.split(ASSIST_MARK, 1)[1]
    return s.strip()


def is_correct(pred_text: str, gt_text: str) -> bool:
    return pred_text.strip() == gt_text.strip()


def read_jsonl_build_index(path: Path) -> Dict[str, Tuple[str, str]]:
    idx: Dict[str, Tuple[str, str]] = {}
    with path.open("r", encoding="utf-8") as f:
        for ln, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except Exception as e:
                raise RuntimeError(f"[{path}] JSON parse error at line {ln}: {e}")

            image = str(row.get("image", "")).strip()
            if not image:
                continue

            pred_text = extract_assistant_text(row.get("pred", ""))
            gt_text = str(row.get("gt", "")).strip()
            idx[image] = (pred_text, gt_text)
    return idx


def build_prefix7_index(recog_idx: Dict[str, Tuple[str, str]]) -> Dict[str, List[str]]:
    pidx: Dict[str, List[str]] = {}
    for k in recog_idx.keys():
        p = k[:7]
        pidx.setdefault(p, []).append(k)
    return pidx


def lookup_recog_item(
    image: str,
    recog_idx: Dict[str, Tuple[str, str]],
    prefix7_idx: Dict[str, List[str]],
) -> Optional[Tuple[str, str]]:
    item = recog_idx.get(image)
    if item is not None:
        return item

    pref = image[:7]
    cands = prefix7_idx.get(pref, [])
    if len(cands) == 0:
        return None
    if len(cands) > 1:
        raise ValueError(
            f"Ambiguous prefix match for association image '{image}' (prefix='{pref}'): "
            f"{cands[:10]}{' ...' if len(cands) > 10 else ''}"
        )
    return recog_idx[cands[0]]


def blank_counts():
    return {
        (True, True): 0,     # A=맞, R=맞
        (True, False): 0,    # A=맞, R=틀
        (False, True): 0,    # A=틀, R=맞
        (False, False): 0,   # A=틀, R=틀
    }


def print_report(title: str, counts: Dict[Tuple[bool, bool], int], total: int, missing: int = 0):
    if total <= 0:
        print(f"\n[{title}] total=0 (skipped)")
        if missing:
            print(f"Unmatched/ignored samples: {missing}")
        return

    def pct(x: int) -> float:
        return 100.0 * x / total

    a_t_r_f = counts[(True, False)]
    a_t_r_t = counts[(True, True)]
    a_f_r_f = counts[(False, False)]
    a_f_r_t = counts[(False, True)]

    print(f"\n[{title}]")
    print(f"Total evaluated: {total}")
    if missing > 0:
        print(f"Unmatched/ignored samples: {missing}")

    print(f"A incorrect & R incorrect: {pct(a_f_r_f):.4f} ({a_f_r_f}/{total})")
    print(f"A correct & R incorrect  : {pct(a_t_r_f):.4f} ({a_t_r_f}/{total})")
    print(f"A incorrect & R correct  : {pct(a_f_r_t):.4f} ({a_f_r_t}/{total})")
    print(f"A correct & R correct    : {pct(a_t_r_t):.4f} ({a_t_r_t}/{total})")

    s = pct(a_t_r_f) + pct(a_t_r_t) + pct(a_f_r_f) + pct(a_f_r_t)
    print(f"Sum: {s:.6f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--association",
        type=str,
        default="output_llava_recog_asso/lora-infer/final_balanced_vqa_test_less_bias_association.jsonl",
    )
    ap.add_argument(
        "--recognition",
        type=str,
        default="output_llava_recog_asso/lora-infer/final_balanced_vqa_test_less_bias_recognition.jsonl",
    )
    ap.add_argument(
        "--ignore_missing",
        action="store_true",
        help="recognition에서 매칭(정확/7글자 prefix) 실패/모호 샘플은 제외하고 계산",
    )
    ap.add_argument(
        "--tag_field",
        type=str,
        default="sub_sub_tag",
        help="association jsonl에서 카테고리 필드명 (default: sub_sub_tag)",
    )
    args = ap.parse_args()

    assoc_path = Path(args.association)
    recog_path = Path(args.recognition)

    if not assoc_path.exists():
        print(f"ERROR: association file not found: {assoc_path}", file=sys.stderr)
        sys.exit(1)
    if not recog_path.exists():
        print(f"ERROR: recognition file not found: {recog_path}", file=sys.stderr)
        sys.exit(1)

    recog_idx = read_jsonl_build_index(recog_path)
    prefix7_idx = build_prefix7_index(recog_idx)

    # 전체 집계
    counts_all = blank_counts()
    total_all = 0
    missing_all = 0

    # sub_sub_tag별 집계
    counts_by_tag = defaultdict(blank_counts)  # tag -> counts dict
    total_by_tag = defaultdict(int)            # tag -> total
    missing_by_tag = defaultdict(int)          # tag -> missing

    with assoc_path.open("r", encoding="utf-8") as f:
        for ln, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except Exception as e:
                raise RuntimeError(f"[{assoc_path}] JSON parse error at line {ln}: {e}")

            image = str(row.get("image", "")).strip()
            if not image:
                continue

            tag = row.get(args.tag_field, None)
            tag = "UNKNOWN" if tag is None or str(tag).strip() == "" else str(tag).strip()

            assoc_pred = extract_assistant_text(row.get("pred", ""))
            assoc_gt = str(row.get("gt", "")).strip()
            assoc_correct = is_correct(assoc_pred, assoc_gt)

            try:
                recog_item = lookup_recog_item(image, recog_idx, prefix7_idx)
            except ValueError:
                # prefix 매칭 모호
                missing_all += 1
                missing_by_tag[tag] += 1
                if args.ignore_missing:
                    continue
                raise

            if recog_item is None:
                missing_all += 1
                missing_by_tag[tag] += 1
                if args.ignore_missing:
                    continue
                raise KeyError(
                    f"Recognition file has no matching image for '{image}' "
                    f"(exact or prefix7) (association line {ln}). "
                    f"원하면 --ignore_missing 옵션을 사용하세요."
                )

            recog_pred, recog_gt = recog_item
            recog_correct = is_correct(recog_pred, recog_gt)

            key = (assoc_correct, recog_correct)

            # 전체
            counts_all[key] += 1
            total_all += 1

            # 태그별
            counts_by_tag[tag][key] += 1
            total_by_tag[tag] += 1

    # 출력
    print_report("GLOBAL", counts_all, total_all, missing_all)

    # tag별 출력 (샘플 많은 순으로)
    for tag in sorted(total_by_tag.keys(), key=lambda t: total_by_tag[t], reverse=True):
        print_report(f"{args.tag_field}={tag}", counts_by_tag[tag], total_by_tag[tag], missing_by_tag[tag])


if __name__ == "__main__":
    main()
