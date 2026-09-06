#!/usr/bin/env python3
"""Regenerate ALL paper figures (main + supplementary) from the committed data.

Thin wrapper that runs both figure-generation scripts in sequence. Invoked by
the repo-root ``make figures`` target.
"""
from pathlib import Path
import subprocess, sys

HERE = Path(__file__).resolve().parent


SCRIPTS = (
    HERE / "regenerate_figures.py",
    HERE / "regenerate_extra_figures.py",
    HERE / "regenerate_main_boxplot.py",
    HERE / "regenerate_finetuning_figure.py",
    HERE / "supplementary" / "scripts" / "gen_sim_validation_figures.py",
    HERE / "supplementary" / "scripts" / "gen_wall_override_dependence.py",
    HERE / ".." / "data" / "rl_ablation" / "gen_ppo_ablation_fig.py",
    HERE / ".." / "data" / "rl_ablation" / "gen_ablation_summary_fig.py",
)


def main() -> int:
    rc = 0
    for script in SCRIPTS:
        print(f"\n=== running {script.name} ===")
        r = subprocess.run([sys.executable, str(script)], cwd=str(HERE))
        rc = rc or r.returncode
    if rc:
        print("\nOne or more figure scripts failed.", file=sys.stderr)
    else:
        print("\nAll figures regenerated.")
    return rc


if __name__ == "__main__":
    sys.exit(main())
