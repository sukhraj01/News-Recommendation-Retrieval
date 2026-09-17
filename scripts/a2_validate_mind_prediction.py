#!/usr/bin/env python3
"""Validate a MIND Codabench prediction.txt: line count, format, permutation
correctness, no duplicate impression ids. Same discipline as every prior
submission's ad-hoc validation in this project (see ADR-012's "Validation"
section) -- now a real script instead of an inline heredoc.

Usage: python3 a2_validate_mind_prediction.py <prediction.txt>
"""
import sys

path = sys.argv[1]
seen_ids = set()
bad = 0
dup = 0
n = 0
with open(path) as f:
    for n, line in enumerate(f, 1):
        line = line.rstrip("\n")
        parts = line.split(" ", 1)
        if len(parts) != 2:
            bad += 1
            if bad <= 3:
                print("MALFORMED:", repr(line[:200]))
            continue
        imp_id, ranks_str = parts
        if not ranks_str.startswith("[") or not ranks_str.endswith("]") or " " in ranks_str:
            bad += 1
            if bad <= 3:
                print("MALFORMED RANKS:", repr(line[:200]))
            continue
        if imp_id in seen_ids:
            dup += 1
        seen_ids.add(imp_id)
        ranks = [int(x) for x in ranks_str[1:-1].split(",")]
        k = len(ranks)
        if sorted(ranks) != list(range(1, k + 1)):
            bad += 1
            if bad <= 3:
                print("NOT A PERMUTATION:", imp_id, "n=", k)

print("total lines checked:", n)
print("unique impression ids:", len(seen_ids))
print("duplicates:", dup)
print("malformed/non-permutation:", bad)
