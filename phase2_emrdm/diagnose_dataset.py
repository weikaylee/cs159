#!/usr/bin/env python3
"""Diagnose the triplet-matching gap between Phase 1 (110K) and Phase 2 (12K).

Phase 1 uses a simple glob that finds all .tif files.
Phase 2 uses build_triplets() which requires strict regex matching on filenames.
This script shows exactly how many files each method finds per season/modality,
and prints sample filenames so you can see if the regex is missing files.

Usage:
    python phase2_emrdm/diagnose_dataset.py \
        --data_root /resnick/groups/perona/oywang/cs159/data
"""

import argparse
import glob
import os
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
try:
    from eval_emrdm import build_triplets, _index_dir
    HAS_BUILD_TRIPLETS = True
except ImportError:
    HAS_BUILD_TRIPLETS = False

MODALITIES = ("s2", "s2_cloudy", "s1")


def count_glob(data_root: str) -> dict:
    """Count files using Phase 1's simple glob — no filename validation."""
    counts = {}
    for d in sorted(os.listdir(data_root)):
        for mod in MODALITIES:
            if d.endswith(f"_{mod}") and os.path.isdir(os.path.join(data_root, d)):
                n = len(glob.glob(os.path.join(data_root, d, "**", "*.tif"), recursive=True))
                counts[d] = n
    return counts


def count_regex(data_root: str) -> dict:
    """Count files that match the build_triplets() regex per season/modality."""
    roi_re = re.compile(r'^ROIs(\d+)_(\w+)_s2$')
    results = {}
    for d in sorted(os.listdir(data_root)):
        m = roi_re.match(d)
        if not m:
            continue
        roi_num, season = m.group(1), m.group(2)
        for mod in MODALITIES:
            root = os.path.join(data_root, f"ROIs{roi_num}_{season}_{mod}")
            if not os.path.isdir(root):
                results[f"ROIs{roi_num}_{season}_{mod}"] = 0
                continue
            # Same regex as _index_dir
            pat = re.compile(
                rf"ROIs{roi_num}_{re.escape(season)}_{re.escape(mod)}_(\d+)_p(\d+)\.tif$"
            )
            matched = []
            all_tifs = glob.glob(os.path.join(root, "**", "*.tif"), recursive=True)
            for path in all_tifs:
                if pat.search(os.path.basename(path)):
                    matched.append(path)
            results[f"ROIs{roi_num}_{season}_{mod}"] = (len(matched), len(all_tifs))
    return results


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data_root", required=True)
    p.add_argument("--sample_files", type=int, default=5,
                   help="Number of sample filenames to show per directory")
    args = p.parse_args()

    print(f"Data root: {args.data_root}\n")

    # ── Phase 1 style: simple glob ────────────────────────────────────────────
    print("=" * 60)
    print("Phase 1 glob (no filename validation) — what train_mirror_map sees:")
    print("=" * 60)
    glob_counts = count_glob(args.data_root)
    total_glob = 0
    for dirname, count in glob_counts.items():
        print(f"  {count:7d}  {dirname}")
        total_glob += count
    print(f"  {'TOTAL':>7}  {total_glob}")

    # ── Phase 2 style: regex matching ─────────────────────────────────────────
    print("\n" + "=" * 60)
    print("Phase 2 regex (build_triplets) — matched vs total .tif per dir:")
    print("=" * 60)
    regex_results = count_regex(args.data_root)
    for dirname, val in regex_results.items():
        if isinstance(val, tuple):
            matched, total = val
            pct = 100 * matched / max(total, 1)
            flag = "  ← MISMATCH" if matched < total else ""
            print(f"  {matched:7d} / {total:7d}  ({pct:5.1f}%)  {dirname}{flag}")
        else:
            print(f"  {'MISSING':>7}               {dirname}")

    # ── Sample filenames from a representative directory ───────────────────────
    print("\n" + "=" * 60)
    print(f"Sample filenames (first {args.sample_files} .tif files per modality):")
    print("=" * 60)
    roi_re = re.compile(r'^ROIs(\d+)_(\w+)_s2$')
    printed_seasons = set()
    for d in sorted(os.listdir(args.data_root)):
        m = roi_re.match(d)
        if not m:
            continue
        roi_num, season = m.group(1), m.group(2)
        if season in printed_seasons:
            continue
        printed_seasons.add(season)
        print(f"\n  Season: {season}  (ROIs{roi_num})")
        for mod in MODALITIES:
            root = os.path.join(args.data_root, f"ROIs{roi_num}_{season}_{mod}")
            if not os.path.isdir(root):
                print(f"    {mod}: directory not found")
                continue
            tifs = sorted(glob.glob(os.path.join(root, "**", "*.tif"), recursive=True))
            print(f"    {mod} ({len(tifs)} total):")
            for t in tifs[: args.sample_files]:
                print(f"      {os.path.basename(t)}")

    # ── build_triplets result ─────────────────────────────────────────────────
    if HAS_BUILD_TRIPLETS:
        print("\n" + "=" * 60)
        print("build_triplets() total matched triplets:")
        print("=" * 60)
        triplets = build_triplets(args.data_root)
        print(f"  {len(triplets)} matched triplets")
        if triplets:
            print("  Example triplet paths:")
            for k, v in triplets[0].items():
                print(f"    {k}: {v}")


if __name__ == "__main__":
    main()
