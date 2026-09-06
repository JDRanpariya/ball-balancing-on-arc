#!/usr/bin/env python3

import argparse
import json
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import glob
import os
from scipy.stats import gaussian_kde

# ------------ Helper Functions ------------

def load_json(filename):
    with open(filename, "r") as f:
        return json.load(f)

def save_json(data, filename):
    with open(filename, "w") as f:
        json.dump(data, f, indent=2)

def analyze_experiment(exp_results):
    rows = []
    ctrl_settling_times = {}
    for ctrl_name, trials in exp_results["controllers"].items():
        if not trials:
            continue
        settling_times = [t["settling_time"] if t.get("settling_time") is not None else 60 for t in trials]
        settling_times_without_failed = [t["settling_time"] for t in trials if t.get("settling_time") is not None]
        overshoots = [t["overshoot"] if t.get("overshoot") is not None else 0 for t in trials]
        violations = [t["violations"] if t.get("violations") is not None else 0 for t in trials]
        rows.append({
            "Controller": ctrl_name,
            "Avg Settling Time": round(float(np.mean(settling_times)), 2),
            "Avg Settling (Success Only)": round(float(np.mean(settling_times_without_failed)), 2) if settling_times_without_failed else None,
            "Avg Overshoot": round(float(np.mean(overshoots)), 2),
            "Avg Violations": round(float(np.mean(violations)), 2),
            "Success Rate (%)": round(100 * (len(settling_times_without_failed) / len(trials)), 2),
        })
        ctrl_settling_times[ctrl_name] = settling_times_without_failed
    return pd.DataFrame(rows), ctrl_settling_times

name_map = {
    "RLController_10M_noH_noDR_2g_dip_reward": "RL-NoH-NoDR-2gDip",
    "RLController_10M_withH_noDR": "RL-H-NoDR",
    "RLController_3M_noH_noDR": "RL-3M-NoH-NoDR",
    "RLController_3M_noH_noDR_finetuned_step_97k": "RL-ft-97k",
    "RLController_3M_noH_noDR_finetuned_step_147k": "RL-ft-147k",
    "RLController_3M_noH_noDR_finetuned_step_197k": "RL-ft-197k",
    "RLController_10M_noH_DR_2g_no_param_in_state": "RL-NoH-DR-2g",
    "PIDController": "PID",
    "LQRController_LQR_dip": "LQR_dip",
    "SMCController_SMC_dip": "SMC_dip",
    "MPCController_MPC_dip": "MPC_dip",
    "NMPCController_NMPC_dip": "NMPC_dip",
    "LQRController_LQR_no_dip": "LQR_no_dip",
    "SMCController_SMC_no_dip": "SMC_no_dip",
    "MPCController_MPC_no_dip": "MPC_no_dip",
    "NMPCController_NMPC_no_dip": "NMPC_no_dip",
    "RLController_3M_eq3_dipR_12N": "RL_dipR_12N",
    "RLController_3M_eq3_dipR_18N": "RL_dipR_18N",
    "RLController_5M_eq3_dipR_18N": "RL_dipR_18N_5M",
    "RLController_3M_eq3_dipR_18N_n": "RL_dipR_18N_n",
    "RLController_3M_eq3_dipR_20N": "RL_dipR_20N",
    "RLController_3M_eq3_gaussR_18N": "RL_gaussR_18N",
}

def clean_experiment_name(filename: str) -> str:
    name = os.path.splitext(filename)[0]
    name = name.replace("experiment_", "")
    parts = name.split("_")
    ball_type = " ".join(parts[0:2])
    if "magnet" in name:
        return f"{ball_type} + magnet"
    else:
        return ball_type

# ------------ Main Functionalities ------------

def remove_trials_below_settling_time(filename, threshold=1):
    data = load_json(filename)
    controllers = data.get("controllers", {})
    for controller_name, trials in controllers.items():
        filtered_trials = [t for t in trials if t.get("settling_time") is None or t["settling_time"] >= threshold]
        removed_count = len(trials) - len(filtered_trials)
        if removed_count > 0:
            print(f"Removed {removed_count} trial(s) from {controller_name}")
        data["controllers"][controller_name] = filtered_trials
    save_json(data, filename)
    print("Checkpoint updated.")

def remove_trials_by_index(filename, controller, start, end):
    data = load_json(filename)
    indices_to_remove = list(range(start, end+1))
    controllers = data.get("controllers", {})
    if controller in controllers:
        trials = controllers[controller]
        filtered_trials = [t for i, t in enumerate(trials) if i not in indices_to_remove]
        removed_count = len(trials) - len(filtered_trials)
        if removed_count > 0:
            print(f"Removed {removed_count} trial(s) from {controller} at indices {indices_to_remove}")
        data["controllers"][controller] = filtered_trials
        save_json(data, filename)
    else:
        print(f"Controller {controller} not found.")

def clear_controller_trials(filename, controller):
    data = load_json(filename)
    if controller in data.get("controllers", {}):
        data["controllers"][controller] = []
        print(f"Cleared all trials for {controller}.")
        save_json(data, filename)
    else:
        print(f"Controller {controller} not found in checkpoint.")

def summarize_and_plot(filename):
    data = load_json(filename)
    df, ctrl_settling_times = analyze_experiment(data)
    df["Controller"] = df["Controller"].replace(name_map)
    # Table plot
    fig, ax = plt.subplots(figsize=(12, 3))
    ax.axis('tight')
    ax.axis('off')
    table = ax.table(cellText=df.values, colLabels=df.columns,
                     loc='center', cellLoc='center', colLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1.2, 1.2)
    for (row, col), cell in table.get_celld().items():
        if row == 0:
            cell.set_text_props(weight='bold')
            cell.set_facecolor("#f0f0f0")
    plt.tight_layout()
    out_file = f"{os.path.splitext(filename)[0]}_table.png"
    fig.savefig(out_file, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved table for {filename} -> {out_file}")

    # Boxplot
    controllers = []
    data_to_plot = []
    for c in ctrl_settling_times.keys():
        if ctrl_settling_times[c]:
            controllers.append(name_map.get(c, c))
            data_to_plot.append(ctrl_settling_times[c])
    if data_to_plot:
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.boxplot(data_to_plot, tick_labels=controllers, showmeans=True)
        ax.set_title(f"Boxplot of Settling Times\n({filename})")
        ax.set_ylabel("Settling Time (s)")
        ax.set_xlabel("Controller")
        plt.xticks(rotation=30, ha="right")
        out_file = f"{os.path.splitext(filename)[0]}_box.png"
        fig.savefig(out_file, dpi=300, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved boxplot for {filename} -> {out_file}")

        fig, ax = plt.subplots(figsize=(10, 5))
        x_min = min(min(vals) for vals in data_to_plot)
        x_max = max(max(vals) for vals in data_to_plot)
        x = np.linspace(x_min, x_max, 200)
        for i, vals in enumerate(data_to_plot):
            if len(vals) > 1:
                kde = gaussian_kde(vals)
                ax.plot(x, kde(x), label=controllers[i])
        ax.set_title(f"Gaussian KDE of Settling Times\n({filename})")
        ax.set_xlabel("Settling Time (s)")
        ax.set_ylabel("Density")
        ax.legend()
        out_file = f"{os.path.splitext(filename)[0]}_gaussian.png"
        fig.savefig(out_file, dpi=300, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved Gaussian KDE for {filename} -> {out_file}")

def compare_across_files(json_files):
    comparison = {}
    for file in json_files:
        if not os.path.exists(file):
            print(f"File not found: {file}")
            continue
        try:
            with open(file, "r") as f:
                data = json.load(f)
        except json.JSONDecodeError as e:
            print(f"[FAIL] JSON error in {file}: {e}")
            continue
        exp_name = clean_experiment_name(file)
        df, _ = analyze_experiment(data)
        df["Controller"] = df["Controller"].replace(name_map)
        for _, row in df.iterrows():
            ctrl = row["Controller"]
            val = row["Avg Settling Time"]
            if ctrl not in comparison:
                comparison[ctrl] = {}
            comparison[ctrl][exp_name] = val
    comparison_df = pd.DataFrame(comparison).T.reset_index().rename(columns={"index": "Controller"})
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.axis('tight')
    ax.axis('off')
    table = ax.table(cellText=comparison_df.values, colLabels=comparison_df.columns,
                     loc='center', cellLoc='center', colLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.2)
    for (row, col), cell in table.get_celld().items():
        if row == 0:
            cell.set_text_props(weight='bold')
            cell.set_facecolor("#f0f0f0")
    plt.tight_layout()
    out_file = "comparison_table.png"
    fig.savefig(out_file, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] Saved cross-experiment comparison table -> {out_file}")

def compare_noise_experiments():
    noise_files = sorted(glob.glob("*noise_*.json"))
    if not noise_files:
        print("No noise experiments found!")
        return
    comparison_rows = {}
    for file in noise_files:
        with open(file, "r") as f:
            data = json.load(f)
        df, _ = analyze_experiment(data)
        df["Controller"] = df["Controller"].replace(name_map)
        noise_level = os.path.splitext(file)[0].split("_")[-1]
        for _, row in df.iterrows():
            ctrl = row["Controller"]
            if ctrl not in comparison_rows:
                comparison_rows[ctrl] = {}
            comparison_rows[ctrl][f"Noise {noise_level} (Settling)"] = row["Avg Settling Time"]
            comparison_rows[ctrl][f"Noise {noise_level} (Success %)"] = row["Success Rate (%)"]
    comp_df = pd.DataFrame(comparison_rows).T.reset_index()
    comp_df.rename(columns={"index": "Controller"}, inplace=True)
    fig, ax = plt.subplots(figsize=(16, 5))
    ax.axis('tight')
    ax.axis('off')
    table = ax.table(cellText=comp_df.values, colLabels=comp_df.columns,
                     loc='center', cellLoc='center', colLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.2)
    for (row, col), cell in table.get_celld().items():
        if row == 0:
            cell.set_text_props(weight='bold')
            cell.set_facecolor("#f0f0f0")
    plt.tight_layout()
    fig.savefig("comparison_table_with_noise.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("[OK] Saved overall noise comparison table -> comparison_table_with_noise.png")

# ------------ CLI ------------

def main():
    parser = argparse.ArgumentParser(
        description="Experiment JSON utilities: clean, summarize, and plot experiment results.",
        epilog="""
Example usage:
  python utils.py remove_below test.json --threshold 1
  python utils.py remove_by_index test.json NMPCController_NMPC_dip 75 100
  python utils.py clear_controller test.json SMCController_SMC_dip
  python utils.py summarize experiment1.json
  python utils.py compare
  python utils.py compare_noise
        """,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    subparsers = parser.add_subparsers(dest='command', required=True, help="Available commands")

    p_remove = subparsers.add_parser(
        'remove_below',
        help="Remove trials with settling_time below threshold (default 1s)."
    )
    p_remove.add_argument('filename', help="Path to experiment JSON file.")
    p_remove.add_argument('--threshold', type=float, default=1.0, help="Minimum settling_time to keep (default: 1.0s).")

    p_index = subparsers.add_parser(
        'remove_by_index',
        help="Remove trials by index range for a specific controller."
    )
    p_index.add_argument('filename', help="Path to experiment JSON file.")
    p_index.add_argument('controller', help="Controller name.")
    p_index.add_argument('start', type=int, help="Start index (inclusive).")
    p_index.add_argument('end', type=int, help="End index (inclusive).")

    p_clear = subparsers.add_parser(
        'clear_controller',
        help="Remove all trials for a specific controller."
    )
    p_clear.add_argument('filename', help="Path to experiment JSON file.")
    p_clear.add_argument('controller', help="Controller name.")

    p_summarize = subparsers.add_parser(
        'summarize',
        help="Summarize and plot results for a single experiment file."
    )
    p_summarize.add_argument('filename', help="Path to experiment JSON file.")

    p_compare = subparsers.add_parser(
        'compare',
        help="Create a comparison table of average settling times across specified experiment JSON files."
    )

    p_compare.add_argument(
        'json_files',
        nargs='+',
        help="List of experiment JSON files to compare."
    )

    p_noise = subparsers.add_parser(
        'compare_noise',
        help="Create a comparison table for noise experiments (files matching '*noise_*.json')."
    )

    args = parser.parse_args()

    if args.command == 'remove_below':
        remove_trials_below_settling_time(args.filename, args.threshold)
    elif args.command == 'remove_by_index':
        remove_trials_by_index(args.filename, args.controller, args.start, args.end)
    elif args.command == 'clear_controller':
        clear_controller_trials(args.filename, args.controller)
    elif args.command == 'summarize':
        summarize_and_plot(args.filename)
    elif args.command == 'compare':
        compare_across_files(args.json_files)
    elif args.command == 'compare_noise':
        compare_noise_experiments()


if __name__ == '__main__':
    main()
