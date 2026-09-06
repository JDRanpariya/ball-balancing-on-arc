#!/usr/bin/env python3
"""
Deterministically rebuild paper/data/exp1_hardware/consolidated.json from its
per-run source JSONs.

WHY THIS SCRIPT EXISTS
----------------------
`consolidated.json` (~151 MB) is the single source of truth for the main paper's
Table II and every hardware figure: many scripts READ it, but until now nothing
in the repo WROTE it, so the master file could not be reproduced from its inputs.
This script closes that reproducibility hole. It assembles consolidated.json by
copying, for each of the 21 benchmark controller entries, the exact 50-trial list
out of the raw per-run evaluation JSON it came from, in the exact key order and
serialization (`json.dumps(obj, indent=2)`, ensure_ascii=True) of the committed
file, so the rebuild is byte-for-byte identical.

THE "MISLEADING FILENAME" PROBLEM (resolved by the MAPPING table below)
----------------------------------------------------------------------
The raw source files were named after the *run*, and the inner controller key was
whatever the harness happened to instantiate, so neither the filename nor the
inner key reliably names the benchmark condition. Most importantly, FOUR different
source entries all carry the inner key `LQRController_LQR_WO` with different
results, and only two of them are actually used in the paper:

    source file                     inner key                 success  used as
    sources/lqr_wo_pidlike.json     LQRController_LQR_WO       46/50    LQR_WO      (the shipped pidlike-gain LQR)
    sources/lqr_basic.json          LQRController_LQR_WO       10/50    LQR_basic   (textbook DARE gain, no override)
    sources/lqr_wo_cmaes.json       LQRController_LQR_WO        8/50    (UNUSED ablation)
    sources/lqr_variants.json       LQRController_LQR_WO       46/50    (UNUSED duplicate of the pidlike run)
    sources/lqr_variants.json       LQRController_LQR_basic     3/50    (UNUSED early LQR_basic)

The MAPPING table below is the authoritative record of which raw entry becomes
which consolidated key. Each row was verified by CONTENT (SHA-256 of the trial
list), not by filename, so the provenance is exact and filename-independent.

Two of the 21 entries (PPO_WM_v1, IQL_OffRL) were never copied into the
`sources/` directory; their raw aggregates live elsewhere under `paper/data/`.
They are referenced at their committed locations below.

USAGE
-----
    # Verify the committed consolidated.json is exactly reproducible (default):
    python data/build_consolidated.py

    # Write the rebuilt file to a path:
    python data/build_consolidated.py --out /tmp/consolidated.json

    # Overwrite the committed master in place:
    python data/build_consolidated.py --write

    # Emit the provenance manifest as CSV:
    python data/build_consolidated.py --manifest paper/data/exp1_hardware/consolidated_manifest.csv
"""

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
EXP1 = ROOT / "paper" / "data" / "exp1_hardware"
SOURCES = EXP1 / "sources"
CONSOLIDATED = EXP1 / "consolidated.json"

# Authoritative provenance table, in the exact key order of consolidated.json.
# Columns: (consolidated_key, source_path_relative_to_ROOT, inner_controller_key, expected_success)
MAPPING = [
    ("PID_WO",           "paper/data/exp1_hardware/sources/pid_wo.json",                "OffsetCorrectedPID_PID_WO_offset_corrected", 48),
    ("LQR_WO",           "paper/data/exp1_hardware/sources/lqr_wo_pidlike.json",        "LQRController_LQR_WO",                        46),
    ("MPC_WO",           "paper/data/exp1_hardware/sources/mpc_wo.json",                "MPCController_MPC_WO",                        40),
    ("MPC_basic_constr", "paper/data/exp1_hardware/sources/mpc_basic.json",             "MPCController_MPC_N10_constr",                22),
    ("NMPC_N10",         "paper/data/exp1_hardware/sources/nmpc_n10.json",              "NMPCController_NMPC_N10",                     45),
    ("MPPI_v12",         "paper/data/exp1_hardware/sources/mppi_v12_cpu.json",          "MPPIController",                              50),
    ("SMC_WO",           "paper/data/exp1_hardware/sources/smc_wo.json",                "OffsetSMC_smc_wo_offset-0.0070",             46),
    ("SMC_full",         "paper/data/exp1_hardware/sources/smc_full.json",              "SMCController_SMC_full",                      12),
    ("SMC_basic",        "paper/data/exp1_hardware/sources/smc_full.json",              "SMCController_SMC_basic",                      6),
    ("PID_vanilla",      "paper/data/exp1_hardware/sources/pid_vanilla.json",           "PIDController_PID_vanilla",                    9),
    ("PPO_BZ_DR",        "paper/data/exp1_hardware/sources/ppo_bz_dr_first_order.json", "RLHardwareController_PPO_BZ_DR",              48),
    ("PPO_DR",           "paper/data/exp1_hardware/sources/ppo_dr.json",                "RLHardwareController_PPO_DR",                 45),
    ("PPO_base",         "paper/data/exp1_hardware/sources/ppo_base.json",              "RLHardwareController_PPO_base",               30),
    ("TQC_DR",           "paper/data/exp1_hardware/sources/tqc_fo_1M_dr_best_50t.json", "RLHardwareController_tqc_fo_1M_dr_best",      43),
    ("TD3_DR",           "paper/data/exp1_hardware/sources/td3_fo_1M_dr_150k_50t.json", "RLHardwareController_td3_fo_1M_dr_150k",      46),
    ("TRPO_DR",          "paper/data/exp1_hardware/sources/trpo_fo_3M_dr_best_50t.json","RLHardwareController_trpo_fo_3M_dr_best",     45),
    ("SAC_DR",           "paper/data/exp1_hardware/sources/sac_fo_1M_dr_best_50t.json", "RLHardwareController_sac_fo_1M_dr_best",      41),
    ("LQR_Qcenter",      "paper/data/exp1_hardware/sources/lqr_variants.json",          "LQRController_LQR_Qcenter",                   14),
    ("LQR_basic",        "paper/data/exp1_hardware/sources/lqr_basic.json",             "LQRController_LQR_WO",                        10),
    # Two entries whose raw aggregates were never copied into sources/:
    ("PPO_WM_v1",        "paper/data/exp1_hardware/ppo_wm_v1.json",                     "RLController_ppo_wm_v1",                      50),
    ("IQL_OffRL",        "paper/data/iql_offline_rl/iql_best_demos_280k_50t.json",      "OfflineRLController_model_280000",           46),
]


def load_entry(rel_path: str, inner_key: str):
    """Return the trial list for `inner_key` from a raw per-run JSON."""
    path = ROOT / rel_path
    with open(path) as f:
        data = json.load(f)
    if isinstance(data, dict) and "controllers" in data:
        trials = data["controllers"][inner_key]
    elif isinstance(data, list):
        trials = data  # top-level list of trials
    else:
        raise ValueError(f"Unrecognized source structure in {rel_path}")
    if not isinstance(trials, list):
        raise ValueError(f"{rel_path}::{inner_key} is not a trial list")
    return trials


def build():
    """Assemble the consolidated dict from the MAPPING table."""
    controllers = {}
    for key, rel_path, inner_key, expected_success in MAPPING:
        trials = load_entry(rel_path, inner_key)
        actual = sum(1 for t in trials if t.get("success"))
        if actual != expected_success:
            raise AssertionError(
                f"{key}: success {actual} from {rel_path}::{inner_key} "
                f"!= expected {expected_success} (source changed?)"
            )
        controllers[key] = trials
    return {"controllers": controllers}


def serialize(obj) -> bytes:
    # Must match how the committed file was written: indent=2, default ensure_ascii.
    return json.dumps(obj, indent=2).encode()


def write_manifest(path: Path):
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["consolidated_key", "source_file", "inner_controller_key",
                    "success_count", "trials"])
        for key, rel_path, inner_key, succ in MAPPING:
            w.writerow([key, rel_path, inner_key, succ, 50])
    print(f"Wrote manifest: {path}")


def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--out", type=Path, default=None,
                   help="Write rebuilt consolidated.json to this path.")
    p.add_argument("--write", action="store_true",
                   help=f"Overwrite the committed master ({CONSOLIDATED}).")
    p.add_argument("--manifest", type=Path, default=None,
                   help="Write the provenance manifest as CSV to this path.")
    args = p.parse_args()

    obj = build()
    data = serialize(obj)
    built_sha = hashlib.sha256(data).hexdigest()

    if args.manifest:
        write_manifest(args.manifest)

    # Verify against the committed file.
    reproduces = None
    if CONSOLIDATED.exists():
        committed = CONSOLIDATED.read_bytes()
        committed_sha = hashlib.sha256(committed).hexdigest()
        reproduces = committed == data
        print(f"committed sha256: {committed_sha}")
        print(f"rebuilt   sha256: {built_sha}")
        if reproduces:
            print("VERIFY: rebuilt output is BYTE-IDENTICAL to committed consolidated.json")
        else:
            # Fall back to a parsed deep-equality check to distinguish a real
            # content diff from a trivial formatting diff.
            deep_equal = json.loads(committed) == obj
            print(f"VERIFY: bytes differ; parsed deep-equal = {deep_equal}")

    if args.write:
        CONSOLIDATED.write_bytes(data)
        print(f"Wrote {CONSOLIDATED} ({len(data)} bytes)")
    elif args.out:
        args.out.write_bytes(data)
        print(f"Wrote {args.out} ({len(data)} bytes)")

    if reproduces is False:
        sys.exit(1)


if __name__ == "__main__":
    main()
