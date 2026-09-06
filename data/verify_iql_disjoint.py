#!/usr/bin/env python3
"""Verify that IQL's training corpus is disjoint from its benchmark evaluation.

The deployed IQL checkpoint is ``model_280000`` (evaluated on 2026-05-28
15:50:23). Its training corpus (arcball_post_recalib_flat.h5) is rebuilt from
the 222 source logs in evaluation/results/real/ over the window
2026-05-20 00:00:00 .. 2026-05-28 15:03:00 -- a cutoff 47 minutes before that
benchmark run. This script confirms, from the released data alone, that:

  1. the corpus contains exactly 1,415 trials / 333,497 transitions, and
  2. the deployed checkpoint ``model_280000`` is NOT among the source
     controllers (it could not be -- it did not exist until training finished),

so no IQL benchmark trial can have leaked into IQL's training data.

Usage:  python data/verify_iql_disjoint.py
"""
import sys
from collections import Counter
from pathlib import Path

# Reuse the exact corpus reconstruction used to build the training file.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_flat_dataset import collect_jsons, load_trials  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
EVAL_DIR = ROOT / "evaluation" / "results" / "real"
AFTER, BEFORE = "20260520000000", "20260528150300"   # training-window cutoff
DEPLOYED = "model_280000"


def main():
    _, jsons = collect_jsons(EVAL_DIR, None, AFTER, BEFORE)
    ctrl = Counter()
    n_trials = n_trans = 0
    for jf in jsons:
        for obs, acts, name, success in load_trials(jf):
            ctrl[name] += 1
            n_trials += 1
            n_trans += len(obs) - 1

    leaked = [c for c in ctrl if DEPLOYED in c]
    print(f"source logs         : {len(jsons)}")
    print(f"trials / transitions: {n_trials} / {n_trans:,}")
    print(f"source controllers  : {len(ctrl)}")
    print(f"deployed checkpoint '{DEPLOYED}' in training sources: "
          f"{leaked if leaked else 'NO'}")
    ok = (not leaked) and n_trials == 1415 and n_trans == 333497
    print("\nDISJOINT + reproduced" if ok else "\nMISMATCH -- check inputs")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
