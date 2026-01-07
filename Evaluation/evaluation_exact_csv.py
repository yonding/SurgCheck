#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Evaluate CSV outputs containing `pred` and `gt` columns (plus optional tag columns)
and save metrics/reports in the same format as evaluation_exact.py.
"""

import argparse
import csv
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

ASSISTANT_MARK = "assistant\n"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Evaluate CSV outputs per-file + global: accuracy + F1 (macro/micro/weighted)."
    )
    p.add_argument("--in_dir", type=str, default="output_endochat/lora-infer", help="Root dir containing csv files")
    p.add_argument("--out_dir", type=str, default="result_endochat/lora-infer", help="Root dir to save json results with same structure")
    p.add_argument("--ext", type=str, default=".csv", help="Input extension to search (default: .csv)")
    p.add_argument("--pred_col", type=str, default="Pred", help="CSV column name for prediction text")
    p.add_argument("--gt_col", type=str, default="GT", help="CSV column name for ground-truth text")
    p.add_argument("--casefold",  default=True, help="Lowercase normalization")
    p.add_argument("--strip_punct",  default=False, help="Strip punctuation before compare")
    p.add_argument("--print_topk", type=int, default=15, help="Top-k groups to print per breakdown")
    p.add_argument("--print_per_file",  default=True, help="Print per-file report")
    p.add_argument("--include_confusion",  default=False, help="Save confusion counts per file (can be big)")
    return p.parse_args()


def _normalize_text(s: str, casefold: bool, strip_punct: bool) -> str:
    if s is None:
        return ""
    s = str(s).strip()
    s = re.sub(r"\s+", " ", s)

    if casefold:
        s = s.casefold()

    if strip_punct:
        s = re.sub(r"[^0-9a-zA-Z가-힣\s]+", "", s)
        s = re.sub(r"\s+", " ", s).strip()

    return s


def extract_answer_from_pred(pred: Any) -> str:
    if pred is None:
        return ""
    text = str(pred)

    if ASSISTANT_MARK in text:
        ans = text.split(ASSISTANT_MARK)[-1]
    else:
        ans = text

    ans = ans.strip()

    # cut off if model continues another role
    for cut in ["\nuser\n", "\nsystem\n", "\nassistant\n"]:
        if cut in ans:
            ans = ans.split(cut)[0].strip()

    return ans


def safe_get(obj: Dict[str, Any], key: str) -> str:
    v = obj.get(key, None)
    return "" if v is None else str(v)


class AccCounter:
    __slots__ = ("n", "correct")

    def __init__(self) -> None:
        self.n = 0
        self.correct = 0

    def add(self, ok: bool) -> None:
        self.n += 1
        if ok:
            self.correct += 1

    @property
    def acc(self) -> float:
        return (self.correct / self.n) if self.n > 0 else float("nan")


def fmt_pct(x: float) -> str:
    if x != x:
        return "nan"
    return f"{100.0 * x:.2f}%"


class ConfusionCounter:
    """
    Stores counts for (gt, pred).
    Also keeps supports for gt classes.
    """

    def __init__(self) -> None:
        self.pairs = Counter()  # (gt, pred) -> count
        self.support = Counter()  # gt -> count
        self.total = 0

    def add(self, gt: str, pred: str) -> None:
        self.pairs[(gt, pred)] += 1
        self.support[gt] += 1
        self.total += 1

    def merge_(self, other: "ConfusionCounter") -> None:
        self.pairs.update(other.pairs)
        self.support.update(other.support)
        self.total += other.total


def compute_f1_from_confusion(conf: ConfusionCounter) -> Dict[str, Any]:
    """
    Multiclass F1:
      - micro-F1 = accuracy for single-label multiclass, but we compute properly from sums.
      - macro-F1 = average F1 over classes (classes from gt-support union pred labels seen)
      - weighted-F1 = support-weighted average F1
    """
    classes = set(conf.support.keys())
    for (_gt, _pred), _c in conf.pairs.items():
        classes.add(_pred)

    tp = Counter()
    fp = Counter()
    fn = Counter()

    for (g, p), c in conf.pairs.items():
        if g == p:
            tp[g] += c
        else:
            fn[g] += c
            fp[p] += c

    def safe_div(a: float, b: float) -> float:
        return a / b if b > 0 else 0.0

    per_class = []
    for cls in sorted(classes):
        TP = tp[cls]
        FP = fp[cls]
        FN = fn[cls]
        prec = safe_div(TP, TP + FP)
        rec = safe_div(TP, TP + FN)
        f1 = safe_div(2 * prec * rec, prec + rec) if (prec + rec) > 0 else 0.0
        sup = conf.support.get(cls, 0)
        per_class.append(
            {
                "class": cls,
                "support": int(sup),
                "tp": int(TP),
                "fp": int(FP),
                "fn": int(FN),
                "precision": prec,
                "recall": rec,
                "f1": f1,
            }
        )

    if len(per_class) == 0:
        macro_f1 = micro_f1 = weighted_f1 = float("nan")
    else:
        macro_f1 = sum(r["f1"] for r in per_class) / len(per_class)
        total_sup = sum(r["support"] for r in per_class)
        weighted_f1 = (sum(r["f1"] * r["support"] for r in per_class) / total_sup) if total_sup > 0 else 0.0

        TP_sum = sum(r["tp"] for r in per_class)
        FP_sum = sum(r["fp"] for r in per_class)
        FN_sum = sum(r["fn"] for r in per_class)
        micro_prec = safe_div(TP_sum, TP_sum + FP_sum)
        micro_rec = safe_div(TP_sum, TP_sum + FN_sum)
        micro_f1 = safe_div(2 * micro_prec * micro_rec, micro_prec + micro_rec) if (micro_prec + micro_rec) > 0 else 0.0

    return {
        "macro_f1": macro_f1,
        "micro_f1": micro_f1,
        "weighted_f1": weighted_f1,
        "num_classes": len(per_class),
        "per_class": per_class,
    }


def counter_dict_to_list(d: Dict[Any, AccCounter], key_names: Optional[List[str]] = None) -> List[Dict[str, Any]]:
    items = []
    for k, c in d.items():
        row: Dict[str, Any] = {"n": c.n, "correct": c.correct, "acc": c.acc, "acc_percent": fmt_pct(c.acc)}
        if isinstance(k, tuple) and key_names:
            for name, val in zip(key_names, k):
                row[name] = val
        else:
            row["key"] = k
        items.append(row)

    items.sort(key=lambda r: (-(r["acc"] if r["acc"] == r["acc"] else -1), -r["n"]))
    return items


def print_topk(title: str, rows: List[Dict[str, Any]], topk: int, key_fields: Optional[List[str]] = None) -> None:
    print(f"\n  - {title} (top {topk})")
    for r in rows[:topk]:
        if key_fields:
            key_str = " | ".join(str(r.get(k, "")) for k in key_fields)
        else:
            key_str = str(r.get("key", ""))
        print(f"    {key_str:60s}  n={r['n']:6d}  acc={r['acc_percent']}")


def main() -> None:
    args = parse_args()

    in_root = Path(args.in_dir)
    out_root = Path(args.out_dir)
    if not in_root.exists():
        raise FileNotFoundError(f"--in_dir not found: {in_root}")

    csv_paths = sorted(in_root.rglob(f"*{args.ext}"))
    if not csv_paths:
        print(f"[WARN] No '{args.ext}' files found under {in_root}")
        return

    global_acc = AccCounter()
    global_conf = ConfusionCounter()

    global_by_main: Dict[str, AccCounter] = defaultdict(AccCounter)
    global_by_sub: Dict[str, AccCounter] = defaultdict(AccCounter)
    global_by_subsub: Dict[str, AccCounter] = defaultdict(AccCounter)
    global_by_combo: Dict[Tuple[str, str, str], AccCounter] = defaultdict(AccCounter)

    global_conf_by_main: Dict[str, ConfusionCounter] = defaultdict(ConfusionCounter)
    global_conf_by_sub: Dict[str, ConfusionCounter] = defaultdict(ConfusionCounter)
    global_conf_by_subsub: Dict[str, ConfusionCounter] = defaultdict(ConfusionCounter)
    global_conf_by_combo: Dict[Tuple[str, str, str], ConfusionCounter] = defaultdict(ConfusionCounter)

    per_file_summaries: List[Dict[str, Any]] = []

    for cp in csv_paths:
        rel = cp.relative_to(in_root)
        out_path = (out_root / rel).with_suffix(".json")
        out_path.parent.mkdir(parents=True, exist_ok=True)

        file_acc = AccCounter()
        file_conf = ConfusionCounter()

        file_by_main: Dict[str, AccCounter] = defaultdict(AccCounter)
        file_by_sub: Dict[str, AccCounter] = defaultdict(AccCounter)
        file_by_subsub: Dict[str, AccCounter] = defaultdict(AccCounter)
        file_by_combo: Dict[Tuple[str, str, str], AccCounter] = defaultdict(AccCounter)

        file_conf_by_main: Dict[str, ConfusionCounter] = defaultdict(ConfusionCounter)
        file_conf_by_sub: Dict[str, ConfusionCounter] = defaultdict(ConfusionCounter)
        file_conf_by_subsub: Dict[str, ConfusionCounter] = defaultdict(ConfusionCounter)
        file_conf_by_combo: Dict[Tuple[str, str, str], ConfusionCounter] = defaultdict(ConfusionCounter)

        records_out: List[Dict[str, Any]] = []

        with cp.open("r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            # line number: header is line 1, so start rows at 2
            for line_idx, row in enumerate(reader, start=2):
                if row is None:
                    continue

                pred_raw = row.get(args.pred_col, "")
                gt_raw = row.get(args.gt_col, "")

                main_tag = safe_get(row, "main_tag")
                sub_tag = safe_get(row, "sub_tag")
                sub_sub_tag = safe_get(row, "sub_sub_tag")

                pred_ans = extract_answer_from_pred(pred_raw)
                pred_norm = _normalize_text(pred_ans, args.casefold, args.strip_punct)
                gt_norm = _normalize_text(gt_raw, args.casefold, args.strip_punct)

                ok = pred_norm == gt_norm

                file_acc.add(ok)
                file_conf.add(gt_norm, pred_norm)

                file_by_main[main_tag].add(ok)
                file_by_sub[sub_tag].add(ok)
                file_by_subsub[sub_sub_tag].add(ok)
                file_by_combo[(main_tag, sub_tag, sub_sub_tag)].add(ok)

                file_conf_by_main[main_tag].add(gt_norm, pred_norm)
                file_conf_by_sub[sub_tag].add(gt_norm, pred_norm)
                file_conf_by_subsub[sub_sub_tag].add(gt_norm, pred_norm)
                file_conf_by_combo[(main_tag, sub_tag, sub_sub_tag)].add(gt_norm, pred_norm)

                global_acc.add(ok)
                global_conf.add(gt_norm, pred_norm)

                global_by_main[main_tag].add(ok)
                global_by_sub[sub_tag].add(ok)
                global_by_subsub[sub_sub_tag].add(ok)
                global_by_combo[(main_tag, sub_tag, sub_sub_tag)].add(ok)

                global_conf_by_main[main_tag].add(gt_norm, pred_norm)
                global_conf_by_sub[sub_tag].add(gt_norm, pred_norm)
                global_conf_by_subsub[sub_sub_tag].add(gt_norm, pred_norm)
                global_conf_by_combo[(main_tag, sub_tag, sub_sub_tag)].add(gt_norm, pred_norm)

                records_out.append(
                    {
                        "line": line_idx,
                        "pred_raw": pred_raw,
                        "pred_extracted": pred_ans,
                        "gt": gt_raw,
                        "pred_norm": pred_norm,
                        "gt_norm": gt_norm,
                        "correct": ok,
                        "main_tag": main_tag,
                        "sub_tag": sub_tag,
                        "sub_sub_tag": sub_sub_tag,
                        "ori_img_tag": row.get("ori_img_tag", None),
                        "image": row.get("image", None),
                        "question": row.get("question", None),
                    }
                )

        file_f1 = compute_f1_from_confusion(file_conf)

        file_by_main_rows = counter_dict_to_list(file_by_main)
        for r in file_by_main_rows:
            k = r["key"]
            r["f1"] = compute_f1_from_confusion(file_conf_by_main[k])["macro_f1"]

        file_by_sub_rows = counter_dict_to_list(file_by_sub)
        for r in file_by_sub_rows:
            k = r["key"]
            r["f1"] = compute_f1_from_confusion(file_conf_by_sub[k])["macro_f1"]

        file_by_subsub_rows = counter_dict_to_list(file_by_subsub)
        for r in file_by_subsub_rows:
            k = r["key"]
            r["f1"] = compute_f1_from_confusion(file_conf_by_subsub[k])["macro_f1"]

        file_by_combo_rows = counter_dict_to_list(file_by_combo, key_names=["main_tag", "sub_tag", "sub_sub_tag"])
        for r in file_by_combo_rows:
            key = (r["main_tag"], r["sub_tag"], r["sub_sub_tag"])
            f1 = compute_f1_from_confusion(file_conf_by_combo[key])
            r["macro_f1"] = f1["macro_f1"]
            r["micro_f1"] = f1["micro_f1"]
            r["weighted_f1"] = f1["weighted_f1"]

        file_summary = {
            "input_file": str(rel),
            "num_samples": file_acc.n,
            "correct": file_acc.correct,
            "accuracy": file_acc.acc,
            "accuracy_percent": fmt_pct(file_acc.acc),
            "macro_f1": file_f1["macro_f1"],
            "micro_f1": file_f1["micro_f1"],
            "weighted_f1": file_f1["weighted_f1"],
            "num_classes": file_f1["num_classes"],
            "by_main_tag": file_by_main_rows,
            "by_sub_tag": file_by_sub_rows,
            "by_sub_sub_tag": file_by_subsub_rows,
            "by_combo_main_sub_subsub": file_by_combo_rows,
        }

        payload = {"file_summary": file_summary, "records": records_out}

        if args.include_confusion:
            payload["confusion_pairs"] = [
                {"gt": g, "pred": p, "count": c} for (g, p), c in file_conf.pairs.most_common()
            ]

        with out_path.open("w", encoding="utf-8") as wf:
            json.dump(payload, wf, ensure_ascii=False, indent=2)

        per_file_summaries.append(file_summary)

        if args.print_per_file:
            print("\n" + "-" * 90)
            print(f"📄 FILE: {rel}")
            print(f"  ✅ file_accuracy: {file_summary['accuracy_percent']}  ({file_summary['accuracy']:.6f})")
            print(
                f"  ✅ file_F1: macro={file_summary['macro_f1']:.6f} | micro={file_summary['micro_f1']:.6f} | weighted={file_summary['weighted_f1']:.6f}"
            )
            print(
                f"  - samples: {file_summary['num_samples']}  correct: {file_summary['correct']}  classes(seen): {file_summary['num_classes']}"
            )

            print_topk("By main_tag", file_summary["by_main_tag"], args.print_topk)
            print_topk("By sub_tag", file_summary["by_sub_tag"], args.print_topk)
            print_topk("By sub_sub_tag", file_summary["by_sub_sub_tag"], args.print_topk)

            print(f"\n  - By (main_tag | sub_tag | sub_sub_tag) (top {args.print_topk})")
            for r in file_summary["by_combo_main_sub_subsub"][: args.print_topk]:
                key_str = f"{r['main_tag']} | {r['sub_tag']} | {r['sub_sub_tag']}"
                print(f"    {key_str:60s}  n={r['n']:6d}  acc={r['acc_percent']}  macroF1={r['macro_f1']:.4f}")

    global_f1 = compute_f1_from_confusion(global_conf)

    print("\n" + "=" * 90)
    print("✅ OVERALL (MAIN RESULT)")
    print(f"  - Files scanned   : {len(csv_paths)}")
    print(f"  - Total samples   : {global_acc.n}")
    print(f"  - Correct         : {global_acc.correct}")
    print(f"  - Accuracy        : {fmt_pct(global_acc.acc)}  ({global_acc.acc:.6f})")
    print(
        f"  - F1              : macro={global_f1['macro_f1']:.6f} | micro={global_f1['micro_f1']:.6f} | weighted={global_f1['weighted_f1']:.6f}"
    )
    print(f"  - Classes(seen)   : {global_f1['num_classes']}")
    print("=" * 90)

    global_by_main_rows = counter_dict_to_list(global_by_main)
    for r in global_by_main_rows:
        k = r["key"]
        r["macro_f1"] = compute_f1_from_confusion(global_conf_by_main[k])["macro_f1"]

    global_by_sub_rows = counter_dict_to_list(global_by_sub)
    for r in global_by_sub_rows:
        k = r["key"]
        r["macro_f1"] = compute_f1_from_confusion(global_conf_by_sub[k])["macro_f1"]

    global_by_subsub_rows = counter_dict_to_list(global_by_subsub)
    for r in global_by_subsub_rows:
        k = r["key"]
        r["macro_f1"] = compute_f1_from_confusion(global_conf_by_subsub[k])["macro_f1"]

    global_by_combo_rows = counter_dict_to_list(global_by_combo, key_names=["main_tag", "sub_tag", "sub_sub_tag"])
    for r in global_by_combo_rows:
        key = (r["main_tag"], r["sub_tag"], r["sub_sub_tag"])
        f1 = compute_f1_from_confusion(global_conf_by_combo[key])
        r["macro_f1"] = f1["macro_f1"]
        r["micro_f1"] = f1["micro_f1"]
        r["weighted_f1"] = f1["weighted_f1"]

    out_root.mkdir(parents=True, exist_ok=True)
    summary_path = out_root / "_summary.json"

    summary = {
        "in_dir": str(in_root),
        "out_dir": str(out_root),
        "num_files": len(csv_paths),
        "overall": {
            "total_samples": global_acc.n,
            "total_correct": global_acc.correct,
            "accuracy": global_acc.acc,
            "accuracy_percent": fmt_pct(global_acc.acc),
            "macro_f1": global_f1["macro_f1"],
            "micro_f1": global_f1["micro_f1"],
            "weighted_f1": global_f1["weighted_f1"],
            "num_classes": global_f1["num_classes"],
        },
        "by_main_tag": global_by_main_rows,
        "by_sub_tag": global_by_sub_rows,
        "by_sub_sub_tag": global_by_subsub_rows,
        "by_combo_main_sub_subsub": global_by_combo_rows,
        "per_file": sorted(
            per_file_summaries,
            key=lambda r: (-(r["accuracy"] if r["accuracy"] == r["accuracy"] else -1), -r["num_samples"]),
        ),
    }

    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"\n📌 Saved global summary: {summary_path}")
    print("Done.\n")


if __name__ == "__main__":
    main()
