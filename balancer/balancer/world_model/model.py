"""
model.py  -  LSTM world model for ArcBall (ball-and-cart)
==========================================================
Input  : [s_t (4) | a (1)]  ->  5 dims (continuous action scalar)
Output : Δstate (4)          (next_state = s_t + Δstate)

Loss   : weighted MSE on Δstate
         weights = [1, 5, 1, 5]  (velocity dims penalised more)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# -- Config --------------------------------------------------------------------

@dataclass
class WorldModelConfig:
    """Configuration for LSTMWorldModel.

    Attributes:
        input_dim: Dimensionality of input (state_dim + action_dim).
        state_dim: Number of state variables [cart_pos, cart_vel, ball_pos, ball_vel].
        hidden_dim: LSTM hidden layer size.
        num_layers: Number of stacked LSTM layers.
        dropout: Dropout probability between LSTM layers and before output.
        state_weights: Per-dimension loss weights for weighted MSE.
    """
    input_dim:     int   = 5      # state_dim(4) + action_dim(1)
    state_dim:     int   = 4      # [cart_pos, cart_vel, ball_pos, ball_vel]
    hidden_dim:    int   = 64
    num_layers:    int   = 2
    dropout:       float = 0.2
    state_weights: list  = field(default_factory=lambda: [1.0, 5.0, 1.0, 5.0])


# -- Model ---------------------------------------------------------------------

class LSTMWorldModel(nn.Module):
    """
    Sequence-to-sequence LSTM world model.

    Forward
    -------
    x  : (B, T, input_dim)   - concatenated [state | action] at each timestep
    hc : optional (h, c) tuple for stateful / autoregressive use

    Returns
    -------
    delta_pred : (B, T, state_dim)   - predicted Δstate
    (h, c)     : updated LSTM hidden state
    """

    def __init__(self, cfg: WorldModelConfig):
        super().__init__()
        self.cfg = cfg

        self.lstm = nn.LSTM(
            input_size  = cfg.input_dim,
            hidden_size = cfg.hidden_dim,
            num_layers  = cfg.num_layers,
            batch_first = True,
            dropout     = cfg.dropout if cfg.num_layers > 1 else 0.0,
        )
        self.dropout = nn.Dropout(cfg.dropout)
        self.fc_out  = nn.Linear(cfg.hidden_dim, cfg.state_dim)

        self._init_weights()

    # -- weight initialisation -------------------------------------------------

    def _init_weights(self):
        for name, param in self.lstm.named_parameters():
            if "weight_ih" in name:
                nn.init.xavier_uniform_(param)
            elif "weight_hh" in name:
                nn.init.orthogonal_(param)
            elif "bias" in name:
                nn.init.zeros_(param)
                # set forget-gate bias = 1 for better gradient flow
                n = param.size(0)
                param.data[n // 4 : n // 2].fill_(1.0)
        nn.init.xavier_uniform_(self.fc_out.weight)
        nn.init.zeros_(self.fc_out.bias)

    # -- forward ---------------------------------------------------------------

    def forward(
        self,
        x:  torch.Tensor,
        hc: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:

        out, (h, c) = self.lstm(x, hc)         # (B, T, hidden)
        out = self.dropout(out)
        delta_pred = self.fc_out(out)           # (B, T, state_dim)
        return delta_pred, (h, c)

    # -- loss ------------------------------------------------------------------

    def loss(
        self,
        pred:   torch.Tensor,   # (B, T, state_dim)
        target: torch.Tensor,   # (B, T, state_dim)
    ) -> Tuple[torch.Tensor, dict]:
        """
        Weighted MSE.  Returns (total_loss, metrics_dict).
        metrics_dict is ready to pass directly to wandb.log().
        """
        weights     = torch.tensor(self.cfg.state_weights, device=pred.device)
        mse_per_dim = F.mse_loss(pred, target, reduction="none")    # (B, T, 4)
        total_loss  = (mse_per_dim * weights).mean()

        with torch.no_grad():
            per_dim = mse_per_dim.mean(dim=(0, 1))                  # (4,)
            metrics = {
                "loss":          total_loss.item(),
                "loss_cart_pos": per_dim[0].item(),
                "loss_cart_vel": per_dim[1].item(),
                "loss_ball_pos": per_dim[2].item(),
                "loss_ball_vel": per_dim[3].item(),
            }

        return total_loss, metrics

    # -- autoregressive rollout ------------------------------------------------

    @torch.no_grad()
    def rollout(
        self,
        seed:    torch.Tensor,   # (B, seed_len, input_dim)  - warm-up context
        actions: torch.Tensor,   # (B, rollout_len, 1)       - continuous actions
        hc:      Optional[Tuple] = None,
    ) -> torch.Tensor:
        """
        Warm up on `seed`, then autoregressively unroll for len(actions) steps.
        Returns predicted states: (B, rollout_len, state_dim).
        """
        # warm up hidden state
        _, hc = self.forward(seed, hc)

        # initial state = last state in seed window
        state = seed[:, -1, :self.cfg.state_dim]   # (B, 4)

        trajectory = []
        for t in range(actions.size(1)):
            x_t = torch.cat([state, actions[:, t, :]], dim=-1).unsqueeze(1)  # (B,1,5)
            delta, hc = self.forward(x_t, hc)
            state = state + delta[:, 0, :]          # integrate Δstate
            trajectory.append(state.unsqueeze(1))

        return torch.cat(trajectory, dim=1)         # (B, rollout_len, 4)
