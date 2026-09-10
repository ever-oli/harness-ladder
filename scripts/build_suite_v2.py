#!/usr/bin/env python3
"""Build tasks/suite_v2.json from GSM8K + BBH + MBPP (fixed seed).

Papers:
  Cobbe et al. 2021 GSM8K (arXiv:2110.14168)
  Suzgun et al. 2022 BIG-Bench Hard (arXiv:2210.09261)
  Austin et al. 2021 MBPP (arXiv:2108.07732)
"""
from __future__ import annotations

import argparse
import json
import random
import re
from pathlib import Path

from datasets import load_dataset

ROOT = Path(__file__).resolve().parents[1]
SEED = 42
BBH_CONFIGS = [
    "boolean_expressions",
    "multistep_arithmetic_two",
    "navigate",
    "object_counting",
    "tracking_shuffled_objects_three_objects",
    "logical_deduction_three_objects",
    "temporal_sequences",
    "web_of_lies",
]


def gsm8k_answer(ans: str) -> str:
    m = re.search(r"####\s*(-?[\d,]+(?:\.\d+)?)", ans)
    if not m:
        nums = re.findall(r"-?\d+(?:\.\d+)?", ans.replace(",", ""))
        return nums[-1] if nums else ans.strip()
    return m.group(1).replace(",", "")


def build_gsm8k(rng: random.Random, n: int = 24) -> list[dict]:
    ds = load_dataset("openai/gsm8k", "main", split="test")
    idxs = list(range(len(ds)))
    rng.shuffle(idxs)
    out: list[dict] = []
    for i in idxs:
        if len(out) >= n:
            break
        row = ds[i]
        exp = gsm8k_answer(row["answer"])
        if not re.fullmatch(r"-?\d+(?:\.\d+)?", exp):
            continue
        out.append(
            {
                "id": f"v2_gsm8k_{len(out)+1:02d}",
                "prompt": row["question"].strip() + "\n\nFinal Answer: bare number only.",
                "expected": exp,
                "category": "math",
                "match": "exact",
                "source": "openai/gsm8k",
                "paper": "Cobbe et al. 2021 (arXiv:2110.14168)",
            }
        )
    return out


def build_bbh(rng: random.Random, n: int = 24) -> list[dict]:
    per = max(1, n // len(BBH_CONFIGS))
    out: list[dict] = []
    for cfg in BBH_CONFIGS:
        ds = load_dataset("lukaemon/bbh", cfg, split="test")
        idxs = list(range(len(ds)))
        rng.shuffle(idxs)
        taken = 0
        for i in idxs:
            if taken >= per or len(out) >= n:
                break
            row = ds[i]
            inp = row.get("input") or ""
            tgt = str(row.get("target") or "").strip()
            if not inp or not tgt or len(inp) > 1200:
                continue
            cat = (
                "long_horizon"
                if any(k in cfg for k in ("arithmetic", "tracking", "navigate", "temporal"))
                else "math"
            )
            out.append(
                {
                    "id": f"v2_bbh_{len(out)+1:02d}",
                    "prompt": inp.strip() + "\n\nFinal Answer: bare answer only (exact match).",
                    "expected": tgt,
                    "category": cat,
                    "match": "exact",
                    "source": f"lukaemon/bbh:{cfg}",
                    "paper": "Suzgun et al. 2022 BBH (arXiv:2210.09261)",
                }
            )
            taken += 1
        if len(out) >= n:
            break
    return out[:n]


def build_mbpp(rng: random.Random, n: int = 16) -> list[dict]:
    ds = load_dataset("google-research-datasets/mbpp", "sanitized", split="test")
    idxs = list(range(len(ds)))
    rng.shuffle(idxs)
    out: list[dict] = []
    for i in idxs:
        if len(out) >= n:
            break
        row = ds[i]
        text = row.get("text") or ""
        tests = row.get("test_list") or []
        if not text or not tests:
            continue
        m = None
        for t in tests:
            m = re.search(r"assert\s+(.+?)\s*==\s*(.+)$", t.strip())
            if m:
                break
        if not m:
            continue
        call, val = m.group(1).strip(), m.group(2).strip()
        expected = val
        if (expected.startswith("'") and expected.endswith("'")) or (
            expected.startswith('"') and expected.endswith('"')
        ):
            expected = expected[1:-1]
        out.append(
            {
                "id": f"v2_mbpp_{len(out)+1:02d}",
                "prompt": (
                    f"Implement and run this Python task.\n{text.strip()}\n\n"
                    f"Then evaluate: {call}\n"
                    "Final Answer: the bare result only."
                ),
                "expected": expected,
                "category": "code",
                "match": "exact",
                "source": "mbpp",
                "paper": "Austin et al. 2021 MBPP (arXiv:2108.07732)",
            }
        )
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()
    rng = random.Random(args.seed)
    gsm = build_gsm8k(rng, 24)
    bbh = build_bbh(rng, 24)
    mbpp = build_mbpp(rng, 16)
    suite = gsm + bbh + mbpp
    out = ROOT / "tasks" / "suite_v2.json"
    meta = ROOT / "tasks" / "suite_v2_meta.json"
    out.write_text(json.dumps(suite, indent=2) + "\n", encoding="utf-8")
    meta.write_text(
        json.dumps(
            {
                "name": "suite_v2",
                "seed": args.seed,
                "n_tasks": len(suite),
                "composition": {"gsm8k": len(gsm), "bbh": len(bbh), "mbpp": len(mbpp)},
                "papers": [
                    "Cobbe et al. 2021 — GSM8K (arXiv:2110.14168)",
                    "Suzgun et al. 2022 — BIG-Bench Hard (arXiv:2210.09261)",
                    "Austin et al. 2021 — MBPP (arXiv:2108.07732)",
                ],
                "notes": "Harder ruler after suite_v1 saturation. Fixed seed sampling.",
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"wrote {out} n={len(suite)}")


if __name__ == "__main__":
    main()
