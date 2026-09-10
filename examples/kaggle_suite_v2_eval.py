#!/usr/bin/env python3
"""Kaggle entry: suite_v2 P8 then P2/P4/P6 on MiniCPM5-2B."""
from __future__ import annotations
import json, os, subprocess, sys
from pathlib import Path

ROOT = Path("/kaggle/working/harness-ladder")
WORK = Path("/kaggle/working")


def main() -> None:
    import torch
    assert torch.cuda.is_available(), "Enable GPU accelerator"
    print("GPU", torch.cuda.get_device_name(0))
    if not ROOT.exists():
        subprocess.check_call(["git", "clone", "--depth", "1", "https://github.com/ever-oli/harness-ladder.git", str(ROOT)])
    else:
        subprocess.check_call(["git", "-C", str(ROOT), "fetch", "origin"])
        subprocess.check_call(["git", "-C", str(ROOT), "reset", "--hard", "origin/main"])
    print(subprocess.check_output(["git", "-C", str(ROOT), "log", "-1", "--oneline"], text=True))
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "transformers>=4.44.0", "accelerate", "sentencepiece", "protobuf", "einops"])
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "-r", str(ROOT / "requirements.txt")])

    def run(rung: int, name: str, tokens: int = 160):
        out = WORK / name
        env = os.environ.copy()
        env["PYTHONPATH"] = str(ROOT / "src")
        subprocess.check_call(
            [
                sys.executable,
                str(ROOT / "examples/run_p4_transformers.py"),
                "--rung", str(rung),
                "--suite", str(ROOT / "tasks/suite_v2.json"),
                "--output", str(out),
                "--ledger", str(WORK / "ledger_suite_v2.csv"),
                "--max-new-tokens", str(tokens),
            ],
            cwd=str(ROOT),
            env=env,
        )
        data = json.loads(out.read_text())
        print(f"P{rung} {data['n_success']}/{data['n_tasks']} {data['success_rate']:.2%}", flush=True)
        return data

    results = {
        "P8": run(8, "p8_real_minicpm5_suite_v2_fix.json", 192),
        "P2": run(2, "p2_real_minicpm5_suite_v2.json"),
        "P4": run(4, "p4_real_minicpm5_suite_v2.json"),
        "P6": run(6, "p6_real_minicpm5_suite_v2.json"),
    }
    summary = {k: {"passed": f"{v['n_success']}/{v['n_tasks']}", "rate": v["success_rate"]} for k, v in results.items()}
    (WORK / "suite_v2_kaggle_summary.json").write_text(json.dumps(summary, indent=2))
    print(summary)


if __name__ == "__main__":
    main()
