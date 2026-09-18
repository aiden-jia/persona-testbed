#!/usr/bin/env python3
"""
Run a cold call eval and save results + analysis to a timestamped file.

Usage (from promptfoo_sim/):
  python run_eval.py                        # eval_2026-05-30_1200.json
  python run_eval.py backchannel_fix        # eval_2026-05-30_1200_backchannel_fix.json
  python run_eval.py --no-analysis          # skip the show_results step
  python run_eval.py my_label --no-analysis
"""
import datetime
import subprocess
import sys
from pathlib import Path

THIS_DIR = Path(__file__).parent
RESULTS_DIR = THIS_DIR / "results"
CONFIG = THIS_DIR / "promptfooconfig.coldcall.yaml"


def main():
    args = sys.argv[1:]
    run_analysis = "--no-analysis" not in args
    labels = [a for a in args if a != "--no-analysis"]
    label = "_".join(labels) if labels else ""

    ts = datetime.datetime.now().strftime("%Y-%m-%d_%H%M")
    stem = f"eval_{ts}" + (f"_{label}" if label else "")
    out_path = RESULTS_DIR / f"{stem}.json"
    RESULTS_DIR.mkdir(exist_ok=True)

    print(f"\nRunning eval  →  {out_path.name}\n")
    result = subprocess.run(
        ["promptfoo", "eval", "--config", str(CONFIG), "--output", str(out_path)],
        cwd=THIS_DIR,
    )
    if result.returncode != 0:
        print("\nEval failed — aborting.")
        sys.exit(result.returncode)

    if run_analysis:
        print(f"\nGenerating analysis for {out_path.name} ...\n")
        subprocess.run(
            [sys.executable, str(THIS_DIR / "show_results.py"), str(out_path)],
            cwd=THIS_DIR,
        )


if __name__ == "__main__":
    main()
