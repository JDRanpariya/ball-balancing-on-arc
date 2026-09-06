# Third-Party Software and Acknowledgments

This project is built on the open-source scientific-Python, deep-learning, and
robotics ecosystem. We gratefully acknowledge the authors and maintainers of
the libraries below. Each remains under its own license; this file records the
dependencies our code builds on and the works to credit when an algorithm is
implemented through one of them.

Versions are declared in [`balancer/pyproject.toml`](balancer/pyproject.toml)
and the single conda environment file [`env.yaml`](env.yaml) (with a
pip-installable mirror in [`requirements.lock.txt`](requirements.lock.txt)).

## Scientific computing and data

- **NumPy**: array programming (Harris et al., *Nature*, 2020).
- **SciPy**: optimization, linear algebra, signal processing (Virtanen et al.,
  *Nature Methods*, 2020). Used for LQR (`solve_discrete_are`) and matrix
  exponential discretization.
- **pandas**, **Matplotlib**, **seaborn**: data handling and plotting.
- **h5py**: HDF5 dataset storage. **scikit-learn**: metrics and data
  splitting. **tqdm**, **rich**, **joblib**, **PyYAML**: utilities.

## Deep learning

- **PyTorch**: neural-network training and inference, incl. the LSTM world
  model (Paszke et al., NeurIPS, 2019).
- **ONNX Runtime**: deployment of exported policies.

## Reinforcement learning

- **Gymnasium**: RL environment interface.
- **Stable-Baselines3** and **SB3-Contrib**: implementations of the
  simulation-trained policies (PPO, SAC, TD3, TRPO, TQC, and others)
  (Raffin et al., *JMLR*, 2021).
- **d3rlpy**: offline-RL algorithms used for the hardware-data-driven policies
  (IQL, CQL, TD3+BC, BCQ) (Seno & Imai, *JMLR*, 2022).

## Control and optimization

- **OSQP**: quadratic-program solver for linear MPC (Stellato et al., *Math.
  Program. Comput.*, 2020).
- **CasADi**: nonlinear optimization and automatic differentiation for NMPC,
  solved with IPOPT (Andersson et al., *Math. Program. Comput.*, 2019).
- **pycma**: CMA-ES evolution strategy used to tune the classical and
  predictive controllers (Hansen et al.).

## Experiment infrastructure

- **Hydra** and **OmegaConf**: configuration management for training.
- **Weights & Biases**, **TensorBoard**: experiment tracking.

## Hardware and media

- **pyserial**: serial communication with the sensor microcontroller.
- **MoviePy**, **pygame** (optional) - video export and rendering.

## Algorithm provenance

Several controllers implement algorithms from the literature; these are also
cited in the paper and in the relevant source files:

- **MPPI**: information-theoretic Model Predictive Path Integral control
  (Williams et al., ICRA 2017), in
  [`balancer/balancer/controllers/mppi.py`](balancer/balancer/controllers/mppi.py).
- **IQL**: Implicit Q-Learning (Kostrikov et al., ICLR 2022), via d3rlpy.
- **World-model planning / policy training**: learned-dynamics control in the
  spirit of Dreamer/PlaNet (Hafner et al.).
