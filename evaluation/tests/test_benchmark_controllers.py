"""Integration smoke test for the simulation benchmark.

Asserts that the canonical continuous-action benchmark builds exactly the
13 deployed controllers - in particular that PPO-WM (ppo_wm_v1.zip), which
shares the PPO algorithm class, is discovered as its own controller rather
than being masked by ppo.zip.

Requires the model checkpoints (Git LFS) to be materialized. Run via:
    make smoke
    cd evaluation && PYTHONPATH=. python -m pytest tests/ -q
"""
import os
import sys

EVAL_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, EVAL_DIR)

from utils.load_controllers import build_benchmark_controllers  # noqa: E402


def test_benchmark_builds_all_13_controllers():
    cwd = os.getcwd()
    os.chdir(EVAL_DIR)  # build_benchmark_controllers uses paths relative to evaluation/
    try:
        ctrls = build_benchmark_controllers(
            action_type="cont",
            model_dir="models",
            world_model_dir="models/world_model",
        )
    finally:
        os.chdir(cwd)

    labels = [name for name, _ in ctrls]
    assert len(ctrls) == 13, f"expected 13 controllers, got {len(ctrls)}: {labels}"
    assert any("PPO_WM" in lbl.upper() for lbl in labels), \
        f"PPO-WM (world-model policy) missing from the benchmark: {labels}"
