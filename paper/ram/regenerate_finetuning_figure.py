#!/usr/bin/env python3
"""Regenerate supplementary/figures/finetuning_results.png.

Left panel: PPO (WM) settling-time distribution before vs after hardware
fine-tuning, from the 50-trial per-trial logs.
Right panel: success-rate comparison (pre- vs post-FT) for the three
fine-tuned models, annotated with mean settling time.

All right-panel summary values are taken to match Table `tab:ft_results`
in F_finetuning.tex (pre-FT rows are the 10-trial baselines for the two
simulation-trained models, 50-trial for PPO (WM)). This closes the prior
mismatch where the right panel's pre-FT bars showed the 50-trial classical
baselines (PPO(BZ+DR) 98%/5.1s, TD3(DR) 91%/11.4s) instead of the table's
pre-FT rows (both 100%; 4.98s and 8.16s).
"""
import json
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Anchor all paths to this file so the script runs from any cwd (e.g. the
# make-figures pipeline runs it with cwd=paper/ram).
HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "..", "paper", "data", "exp1_hardware")


def settling(path, key):
    d = json.load(open(path))
    c = d.get("controllers", d)
    v = c[key]
    return [t["settling_time"] for t in v
            if isinstance(t, dict) and t.get("success")
            and t.get("settling_time") is not None]


pre = settling(f"{DATA}/ppo_wm_v1.json", "RLController_ppo_wm_v1")
post = settling(f"{DATA}/ppo_wm_v1_ft.json", "RLController_ppo_wm_v1_ft")

# Right-panel summary values, matching tab:ft_results in F_finetuning.tex.
models = ["PPO(WM)\nFT 50t", "PPO(BZ+DR)\nFT 10t", "TD3(DR)\nFT 10t"]
sr_pre = [100, 100, 100]
sr_post = [100, 90, 80]
mean_pre = [6.84, 4.98, 8.16]
mean_post = [7.03, 6.50, 8.60]

C_PRE_BOX, C_POST_BOX = "#e2837c", "#78b869"
C_PRE_BAR, C_POST_BAR = "#4c85c4", "#5aa84f"

fig, (axL, axR) = plt.subplots(1, 2, figsize=(14.5, 5.8))

# ---- Left: PPO (WM) before/after box plot ----
bp = axL.boxplot([pre, post], labels=["PPO (WM)\npre-FT", "PPO (WM)\npost-FT"],
                 widths=0.5, patch_artist=True,
                 medianprops=dict(color="orange", linewidth=1.6))
for patch, col in zip(bp["boxes"], [C_PRE_BOX, C_POST_BOX]):
    patch.set_facecolor(col)
    patch.set_alpha(0.9)
axL.set_ylabel("Settling Time (s)", fontsize=12)
axL.set_title(f"PPO (WM) Fine-tuning\n(Pre: {len(pre)*2}%/{np.mean(pre):.2f}s, "
              f"Post: {len(post)*2}%/{np.mean(post):.2f}s)", fontsize=13)
axL.grid(True, axis="y", alpha=0.3)
axL.tick_params(labelsize=11)

# ---- Right: SR pre vs post, annotated with mean settling ----
x = np.arange(len(models))
w = 0.38
bars_pre = axR.bar(x - w / 2, sr_pre, w, label="Pre-FT", color=C_PRE_BAR)
bars_post = axR.bar(x + w / 2, sr_post, w, label="Post-FT", color=C_POST_BAR)
for b, m in zip(bars_pre, mean_pre):
    axR.text(b.get_x() + b.get_width() / 2, b.get_height() + 1.2,
             f"{m:.1f}s", ha="center", va="bottom", fontsize=11, color=C_PRE_BAR)
for b, m in zip(bars_post, mean_post):
    axR.text(b.get_x() + b.get_width() / 2, b.get_height() + 1.2,
             f"{m:.1f}s", ha="center", va="bottom", fontsize=11, color="#3f7a37")
axR.set_ylabel("Success Rate (%)", fontsize=12)
axR.set_title("Fine-tuning: Before vs After", fontsize=13)
axR.set_xticks(x)
axR.set_xticklabels(models, fontsize=11)
axR.set_ylim(0, 112)
axR.legend(fontsize=11, loc="upper right")
axR.grid(True, axis="y", alpha=0.3)

plt.tight_layout()
for out in [os.path.join(HERE, "supplementary", "figures", "finetuning_results.png")]:
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print("wrote", out)
