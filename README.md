# Ball Balancing on Arc

**A Ball-on-Arc Benchmark: Classical and Learning-Based Control on Realistic Deployment Hardware**

A reproducible benchmark of 13 controllers spanning classical feedback, predictive control, simulation-trained RL, and data-driven methods on a real ball-on-arc platform - a nonlinear, underactuated system with hard workspace constraints.

This repository accompanies the **arXiv preprint** and contains the code,
trained models, checkpoints, hardware design, datasets, experimental records,
and figure-generation scripts needed to reproduce the reported results. See
[`reproducibility_guide.md`](reproducibility_guide.md) for the end-to-end
pipeline.

<p align="center">
  <img src="paper/ram/figures/experimental_setup.jpeg" width="500" alt="Experimental Setup">
</p>

______________________________________________________________________

## Quick Start

```bash
git clone https://github.com/JDRanpariya/ball-balancing-on-arc.git
cd ball-balancing-on-arc
git lfs pull
make install-full
```

See [`reproducibility_guide.md`](reproducibility_guide.md) for the complete reproduction pipeline.

### Train PPO with blind-zone + domain randomization

```bash
python training/scripts/train_all_cont.py --algos ppo --rewards balanced --dr --blind-zone --timesteps 3000000 --tag ppo_bz_dr
```

### Evaluate all controllers in simulation

```bash
make eval-sim
```

### Generate paper figures

```bash
make figures
```

### Build the paper PDFs

The arXiv preprint and supplementary material build from the LaTeX source
(see [`paper/ram/Makefile`](paper/ram/Makefile)):

```bash
cd paper/ram
make all             # build main.pdf + supplementary/supplementary.pdf
```

______________________________________________________________________

## Repository Structure

```
.
+-- balancer/           # Core Python package (pip install -e balancer/[all])
|   +-- controllers/    #   PD, LQR, MPC, NMPC, SMC, RL, Offline RL, MPPI
|   +-- core/           #   Dynamics, linearization, parameters, state
|   +-- envs/           #   Gymnasium environments (sim + real)
|   +-- hardware/       #   Serial comms, sensor processing, calibration
|   +-- world_model/    #   LSTM dynamics model for MPPI/PPO-WM
+-- training/           # RL training pipeline (SB3 + Hydra)
|   +-- configs/        #   Algorithm, environment, wrapper configs
|   +-- scripts/        #   train.py, train_all_cont.py
|   +-- offline_rl/     #   IQL training via d3rlpy
+-- evaluation/         # Experiments, tuning, analysis
|   +-- scripts/        #   eval.py, eval_sim.py, figure generation
|   +-- tuning/         #   CMA-ES automated controller tuning
|   +-- models/         #   Trained model weights (.zip, .pth)
|   +-- experiment/     #   Experiment runner infrastructure
|   +-- utils/          #   Controller loading, analysis helpers
+-- data/               # Dataset collection scripts
|   +-- dataset/        #   Three 1M exploration datasets (two calibration eras + pre-negative-reward) + demonstrations
+-- hardware/           # Twincat Setup + Arduino sensor firmware (VL53L0X_Setup)
+-- paper/              # arXiv preprint source + experimental data + figures
|   +-- ram/            #   LaTeX source, sections, supplementary
|   +-- data/           #   Hardware trial data, ablation results, tuned params
```

**State vector:** `[cart_pos, cart_vel, ball_angle, ball_angular_vel]`

______________________________________________________________________

## Reproducibility

Designed for reproducible control research. All code, data, and trained models needed to reproduce every table and every scripted figure are included. Three supplement figures (`dataset_sample.png`, `training_curves_all.png`, `ppo_fo_5M_reward_curves.png`) are static training/dataset artifacts with no regeneration script.

| Resource                         | Location                                                                                                                                              |
| -------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------- |
| Full reproducibility guide       | [`reproducibility_guide.md`](reproducibility_guide.md)                                                                                                |
| Paper source + supplementary     | [`paper/ram/`](paper/ram/)                                                                                                                            |
| Hardware trial data (650 trials) | [`paper/data/exp1_hardware/`](paper/data/exp1_hardware/)                                                                                              |
| Trained RL models                | [`evaluation/models/cont/`](evaluation/models/cont/)                                                                                                  |
| World model checkpoints          | [`evaluation/models/world_model/`](evaluation/models/world_model/)                                                                                    |
| Datasets                         | [`data/dataset/`](data/dataset/)                                                                                                                      |
| World model training             | [`training/configs/env/world_model.yaml`](training/configs/env/world_model.yaml) + [`balancer/balancer/world_model/`](balancer/balancer/world_model/) |
| Offline RL training              | [`training/offline_rl/`](training/offline_rl/)                                                                                                        |

```bash
make install-full    # Install all dependencies
make figures         # Generate all paper figures from raw data
make eval-sim        # Run all controllers in simulation
make check           # Run reproducibility checklist
```

______________________________________________________________________

## System

- **Plant:** Ball (24 g) on convex arc (R = 2.101 m) with Gaussian dip, on motorized cart
- **Actuator:** SMC LEFB32 linear actuator, Beckhoff EtherCAT servo (velocity interface)
- **Sensors:** Dual VL53L0X ToF at 12 Hz (sequential mode), motor encoder, IR beam break
- **Control rate:** 20 Hz
- **State:** 4D continuous `[cart_pos, cart_vel, ball_angle, ball_ang_vel]`
- **Action:** Scalar velocity command ∈ [-0.9, 0.9] m/s

______________________________________________________________________

## Acknowledgments

This work is built on the open-source scientific-Python and robotics ecosystem,
including NumPy, SciPy, PyTorch, Gymnasium, Stable-Baselines3, d3rlpy, CasADi,
OSQP, and pycma. See [THIRD_PARTY.md](THIRD_PARTY.md) for the full list of
third-party software we depend on and the works credited for the algorithms we
build on.

## License

The code is released under the MIT License. The datasets, evaluation results,
and trained models are released under CC BY 4.0. See [`LICENSE`](LICENSE).
