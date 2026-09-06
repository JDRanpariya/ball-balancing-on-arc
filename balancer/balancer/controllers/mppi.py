"""MPPI controller with learned world-model dynamics.

Implements information-theoretic Model Predictive Path Integral control
(Williams et al., "Information Theoretic MPC for Model-Based Reinforcement
Learning," ICRA 2017), rolling out a learned LSTM world model to score
sampled action sequences and executing their cost-weighted average.

Beyond the base algorithm this adds model-error correction (prediction-error
bias), context warm-up of the LSTM from recent observations, an LQR terminal
cost-to-go, adaptive Gaussian action sampling, and warm-start trajectory
shifting.
"""

import numpy as np
import torch

from balancer.core.device import resolve_device
from collections import deque
from balancer.controllers.base import BaseController


class MPPIController(BaseController):
    """Model Predictive Path Integral (MPPI) controller with neural world model.

    Uses importance-weighted sampling over action sequences rolled out through
    a learned LSTM dynamics model. Includes model error correction and context
    warm-up for robust real-time performance.

    Args:
        predictor: Trained WorldModelPredictor instance.
        horizon (int): Planning horizon length. Default 15.
        n_samples (int): Number of action sequences to sample. Default 800.
        lambda_ (float): Temperature parameter. Default 0.4.
        device (str): 'cpu' or 'cuda'. Default 'cpu'.
        Q_diag (list): State cost weights [cart_pos, cart_v, ball_pos, ball_v].
        R (float): Control effort cost weight.
        cart_limit_cost (float): Penalty for cart limit violation.
        stateful (bool): Maintain LSTM hidden state across steps.
        context_window (int): Number of past observations for LSTM warm-up.
        error_correction (bool): Apply model prediction error as bias.
        error_alpha (float): Error correction EMA factor (0=ignore, 1=full).
        terminal_cost (bool): Add LQR terminal cost-to-go.
        terminal_P (list): 4x4 terminal cost matrix (flattened). If None, uses default.
        noise_type (str): 'gaussian' or 'uniform'. Default 'gaussian'.
        noise_sigma (float): Std dev for gaussian sampling. Default 0.5.
        sigma_decay (float): Decay sigma when costs are good. Default 0.95.
        sigma_min (float): Minimum sigma. Default 0.1.
        beta (float): Temporal smoothing factor [0,1]. Default 0.9.
    """

    def __init__(
        self,
        predictor,
        horizon: int = 15,
        n_samples: int = 800,
        lambda_: float = 0.4,
        device: str = "cpu",
        Q_diag: list = None,
        R: float = 0.01,
        cart_limit_cost: float = 100.0,
        stateful: bool = False,
        # -- Context warm-up --
        context_window: int = 10,
        # -- Model error correction --
        error_correction: bool = False,
        error_alpha: float = 0.3,
        # -- Terminal cost --
        terminal_cost: bool = False,
        terminal_P: list = None,
        # -- Sampling --
        noise_type: str = "uniform",
        noise_sigma: float = 0.5,
        sigma_decay: float = 0.95,
        sigma_min: float = 0.1,
        beta: float = 0.9,
        # -- Cart wall handling --
        cart_clamp: bool = False,
        wall_margin: float = 0.0,
        override_gain: float = 0.5,
    ):
        self.predictor = predictor
        self.H = horizon
        self.n_samples = n_samples
        self.lambda_ = lambda_
        self.device = resolve_device(device)

        if Q_diag is None:
            Q_diag = [0.0, 1.0, 80.0, 1.0]
        self.Q = torch.tensor(Q_diag, device=self.device, dtype=torch.float32)
        self.R = torch.tensor([R], device=self.device, dtype=torch.float32)
        self.cart_limit_cost = cart_limit_cost

        self.action_dim = 1
        self.beta = beta
        self.stateful = stateful

        # -- Cart wall handling --
        self.cart_clamp = cart_clamp
        self.wall_margin = wall_margin
        self.override_gain = override_gain

        # -- Context warm-up --
        self.context_window = context_window
        self._obs_buffer = deque(maxlen=context_window)  # stores (state, action) tuples

        # -- Model error correction --
        self.error_correction = error_correction
        self.error_alpha = error_alpha
        self._error_bias = torch.zeros(4, device=self.device, dtype=torch.float32)
        self._last_predicted_next = None

        # -- Terminal cost (LQR cost-to-go) --
        self.terminal_cost = terminal_cost
        if terminal_P is not None:
            self._P = torch.tensor(
                terminal_P, device=self.device, dtype=torch.float32
            ).reshape(4, 4)
        else:
            # Default: approximate LQR Riccati solution for the arc-ball system
            # P = diag-dominant approximation matching Q weights with cross-terms
            self._P = torch.diag(torch.tensor(
                [1.0, 2.0, 100.0, 5.0], device=self.device, dtype=torch.float32
            ))

        # -- Sampling --
        self.noise_type = noise_type
        self.noise_sigma = noise_sigma
        self.sigma_decay = sigma_decay
        self.sigma_min = sigma_min

        # Trajectory mean (GPU tensor) for warm-starting
        self.prev_sol = torch.zeros(
            (self.H, self.action_dim),
            device=self.device,
            dtype=torch.float32,
        )

    def sample_action_sequences(self) -> torch.Tensor:
        """Sample action sequences around previous solution.

        Uses Gaussian noise centred on the shifted previous solution,
        with temporal smoothing for coherent trajectories.

        Returns:
            (n_samples, H, 1) action tensor, clamped to [-0.9, 0.9].
        """
        if self.noise_type == "gaussian":
            noise = torch.randn(
                (self.n_samples, self.H, self.action_dim),
                device=self.device,
                dtype=torch.float32,
            ) * self.noise_sigma
        else:
            noise = (
                torch.rand(
                    (self.n_samples, self.H, self.action_dim),
                    device=self.device,
                    dtype=torch.float32,
                ) * 2 - 1
            )

        # Centre on previous solution with temporal smoothing
        actions = torch.zeros_like(noise)
        actions[:, 0, :] = self.prev_sol[0] + noise[:, 0, :]
        for t in range(1, self.H):
            actions[:, t, :] = (
                self.beta * (self.prev_sol[t] + noise[:, t, :])
                + (1.0 - self.beta) * actions[:, t - 1, :]
            )

        return actions.clamp(-0.9, 0.9)

    def compute_cost(
        self,
        traj_states: torch.Tensor,
        traj_actions: torch.Tensor,
        cart_limit: float,
    ) -> torch.Tensor:
        """Compute trajectory cost with terminal value.

        Args:
            traj_states: (n_samples, H+1, 4) state trajectories.
            traj_actions: (n_samples, H, 1) action sequences.
            cart_limit: Maximum allowed cart position (m).

        Returns:
            (n_samples,) total cost per trajectory.
        """
        # Running state cost
        state_cost = (self.Q * traj_states[:, 1:, :] ** 2).sum(dim=-1)  # (N, H)

        # Control effort
        action_cost = self.R * (traj_actions.squeeze(-1) ** 2)  # (N, H)

        # Cart constraint
        cart_pos = traj_states[:, 1:, 0]
        violation = torch.relu(cart_pos.abs() - cart_limit)
        constraint_cost = self.cart_limit_cost * violation  # (N, H)

        running_cost = (state_cost + action_cost + constraint_cost).sum(dim=-1)

        # Terminal cost: x_H^T P x_H
        if self.terminal_cost:
            x_H = traj_states[:, -1, :]  # (N, 4)
            term = (x_H @ self._P * x_H).sum(dim=-1)  # (N,)
            running_cost = running_cost + term

        return running_cost

    def _warmup_context(self):
        """Feed observation buffer through LSTM to build context hidden state.

        Returns:
            Hidden state tuple (h, c) or None if buffer is empty.
        """
        if len(self._obs_buffer) == 0:
            return None

        hc = None
        for state, action in self._obs_buffer:
            s = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
            a = torch.tensor([[action]], dtype=torch.float32, device=self.device)
            s_norm = self.predictor.normalise_state(s)
            a_norm = self.predictor.normalise_action(a)
            x = torch.cat([s_norm, a_norm], dim=-1).unsqueeze(1)  # (1, 1, 5)
            _, hc = self.predictor.model(x, hc)

        return hc

    def _expand_hc_for_samples(self, hc):
        """Clone hidden state for N parallel rollout samples."""
        if hc is None:
            return None
        h, c = hc
        h_exp = h.expand(-1, self.n_samples, -1).contiguous()
        c_exp = c.expand(-1, self.n_samples, -1).contiguous()
        return (h_exp, c_exp)

    def _update_error_bias(self, actual_state: np.ndarray):
        """Update model error correction bias using EMA.

        Compares what the model predicted for this step vs what actually happened.
        The accumulated bias is added to all future predictions.
        """
        if not self.error_correction or self._last_predicted_next is None:
            return

        actual = torch.tensor(
            actual_state, dtype=torch.float32, device=self.device
        )
        error = actual - self._last_predicted_next  # (4,)

        # Exponential moving average of error
        self._error_bias = (
            (1.0 - self.error_alpha) * self._error_bias
            + self.error_alpha * error
        )

    def reset(self):
        """Reset controller state."""
        self.prev_sol.zero_()
        self._obs_buffer.clear()
        self._error_bias.zero_()
        self._last_predicted_next = None
        self.noise_sigma = 0.5  # reset adaptive sigma

    @torch.no_grad()
    def step(self, state: np.ndarray, cart_limit: float) -> float:
        """Compute control action using MPPI.

        Args:
            state: (4,) numpy array [cart_pos, cart_vel, ball_pos, ball_vel].
            cart_limit: Maximum allowed cart position (m).

        Returns:
            Continuous action in [-0.9, 0.9].
        """
        # -- 1. Update error bias with actual observation --
        self._update_error_bias(state)

        # -- 2. Build context hidden state --
        if self.stateful and len(self._obs_buffer) > 0:
            hc_context = self._warmup_context()
        else:
            hc_context = None

        # -- 3. Convert state to tensor --
        state_t = torch.as_tensor(
            state, device=self.device, dtype=torch.float32,
        ).unsqueeze(0)  # (1, 4)

        # -- 4. Sample action sequences --
        action_seqs = self.sample_action_sequences()

        # -- 5. Rollout world model --
        traj_states = torch.zeros(
            (self.n_samples, self.H + 1, 4),
            device=self.device,
            dtype=torch.float32,
        )
        traj_states[:, 0, :] = state_t

        hc = self._expand_hc_for_samples(hc_context) if self.stateful else None
        current_states = state_t.expand(self.n_samples, -1)

        for t in range(self.H):
            next_states, hc = self.predictor.predict_batch(
                current_states, action_seqs[:, t, :], hc
            )
            # Apply error correction bias
            if self.error_correction:
                next_states = next_states + self._error_bias

            # -- Cart wall clamping (optional) --
            # Simulate physical wall: cart cannot exceed limits.
            if self.cart_clamp:
                cart_pos = next_states[:, 0]
                hit_right = cart_pos > cart_limit
                hit_left = cart_pos < -cart_limit
                hit_wall = hit_right | hit_left
                if hit_wall.any():
                    next_states = next_states.clone()
                    next_states[:, 0] = cart_pos.clamp(-cart_limit, cart_limit)
                    next_states[:, 1] = torch.where(
                        hit_wall, torch.zeros_like(next_states[:, 1]), next_states[:, 1]
                    )

            traj_states[:, t + 1, :] = next_states
            current_states = next_states

        # -- 6. Compute costs and weights --
        traj_costs = self.compute_cost(traj_states, action_seqs, cart_limit)
        min_cost = traj_costs.min()

        exp_costs = torch.exp(-(traj_costs - min_cost) / self.lambda_)
        weights = exp_costs / (exp_costs.sum() + 1e-8)

        # -- 7. Compute weighted mean action --
        weighted_actions = weights[:, None, None] * action_seqs  # (N, H, 1)
        new_mean_sol = weighted_actions.sum(dim=0)  # (H, 1)

        u_next = new_mean_sol[0, 0].item()

        # -- 8. Update trajectory mean (shift + append) --
        self.prev_sol[:-1] = new_mean_sol[1:]
        self.prev_sol[-1] = new_mean_sol[-1]

        # -- 9. Adapt noise sigma --
        # If best cost is low, reduce exploration; if high, increase
        effective_ratio = (weights.max() / (1.0 / self.n_samples)).item()
        if effective_ratio > 5.0:  # good solutions found
            self.noise_sigma = max(
                self.sigma_min, self.noise_sigma * self.sigma_decay
            )
        else:  # struggling, explore more
            self.noise_sigma = min(0.9, self.noise_sigma / self.sigma_decay)

        # -- 10. Wall-proximity override --
        # Like PID/LQR: when cart is near wall and MPPI pushes toward it,
        # hard-override with motion away from wall.
        u_clipped = float(np.clip(u_next, -0.9, 0.9))

        if self.wall_margin > 0:
            cart_pos = state[0]
            near_right = cart_pos > (cart_limit - self.wall_margin)
            near_left = cart_pos < -(cart_limit - self.wall_margin)
            if near_right and u_clipped > 0:
                u_clipped = -self.override_gain
            elif near_left and u_clipped < 0:
                u_clipped = self.override_gain

        # -- 11. Store for error correction and context --
        # Predict what we expect next state to be (for error correction)
        pred_next, _ = self.predictor.predict_batch(
            state_t,
            torch.tensor([[u_clipped]], device=self.device, dtype=torch.float32),
            hc_context if self.stateful else None,
        )
        self._last_predicted_next = pred_next.squeeze(0)  # (4,)

        # Add to observation buffer for context warm-up
        self._obs_buffer.append((state.copy(), u_clipped))

        return u_clipped
