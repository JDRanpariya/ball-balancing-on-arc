"""
dataset.py  -  ArcBall (ball-and-cart) HDF5 dataset
=====================================================
HDF5 column layout (from data config):
    [0]  prev_cart_pos
    [1]  prev_cart_vel
    [2]  prev_ball_pos  - offset corrected before use (see below)
    [3]  prev_ball_vel
    [4]  action          - continuous cart velocity in [-0.9, 0.9]
    [5]  cur_cart_pos
    [6]  cur_cart_vel
    [7]  cur_ball_pos   - offset corrected before use (see below)
    [8]  cur_ball_vel
    [9]  reward          - ignored
    [10] dip_reward      - ignored

ball_pos offset correction(only for dataset collected at calibration value 175,170 otherwise use calibrated values):
    if ball_pos > 0 : corrected = ball_pos - 0.00476
    if ball_pos <= 0: corrected = ball_pos + 0.00857
Applied to both prev and cur before computing delta, so delta is unaffected
but s_t reflects corrected absolute ball_pos for the state input.

__getitem__ returns a sliding window of length seq_len:
    s_t   : (seq_len, 4)  normalised state  [cart_pos, cart_vel, ball_pos, ball_vel]
    a_t   : (seq_len, 1)  continuous action scalar
    delta : (seq_len, 4)  normalised Δstate = cur - prev
"""

from __future__ import annotations

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset


class ArcBallDataset(Dataset):

    STATE_NAMES = ["cart_pos", "cart_vel", "ball_pos", "ball_vel"]

    # column indices - 15-col calib layout (action at col 6)
    _S_COLS = slice(0, 4)    # prev state
    _A_COL = slice(6, 7)    # action
    _SNEXT_COLS = slice(7, 11)   # cur state

    # column indices - 11-col pre-negation layout (action at col 4).
    # Used by the v1 world-model training data (arcball_cont_1M.h5,
    # collected at 50 Hz before the serial-reader velocity negation).
    _S_COLS_11 = slice(0, 4)
    _A_COL_11 = slice(4, 5)
    _SNEXT_COLS_11 = slice(5, 9)

    # ball_pos offset constants (column index 2 within state)
    _BALL_POS_COL = 2
    _BALL_POS_POS_OFFSET = 0.0   # -0.00476  when ball_pos > 0
    _BALL_POS_NEG_OFFSET = 0.0   # +0.00857  when ball_pos <= 0

    def __init__(
        self,
        path:            str,
        mode:            str = "train",
        split:           tuple = (0.8, 0.1, 0.1),
        seq_len:         int = 10,
        normalise_state: bool = True,
        normalise_delta: bool = True,
        max_rows:        int = None,   # use first N rows
        # use last N rows (mutually exclusive with max_rows)
        tail_rows:       int = None,
    ):
        assert mode in ("train", "val", "test")
        assert abs(sum(split) - 1.0) < 1e-6

        # -- load ------------------------------------------------------
        with h5py.File(path, "r") as f:
            data = f["dataset"][:]

        if max_rows is not None:
            data = data[:max_rows]
        elif tail_rows is not None:
            data = data[-tail_rows:]

        # Auto-detect column layout from width.
        # 15-col: [state(4) raw(2) action(1) next_state(4) raw(2) reward(2)]
        # 11-col: [state(4) action(1) next_state(4) reward(2)]           (v1 data)
        if data.shape[1] >= 13:
            s_cols, a_col, snext_cols = self._S_COLS, self._A_COL, self._SNEXT_COLS
        else:
            s_cols, a_col, snext_cols = self._S_COLS_11, self._A_COL_11, self._SNEXT_COLS_11

        s_t = data[:, s_cols].astype(np.float32)
        a_t = data[:, a_col].astype(np.float32)    # (N, 1)
        s_next = data[:, snext_cols].astype(np.float32)

        # -- ball_pos offset correction ---------------------------------
        s_t = self._correct_ball_pos(s_t)
        s_next = self._correct_ball_pos(s_next)

        delta = s_next - s_t                                # (N, 4)

        # -- split -----------------------------------------------------
        N = len(s_t)
        i1 = int(N * split[0])
        i2 = int(N * (split[0] + split[1]))
        sl = {"train": slice(0, i1), "val": slice(
            i1, i2), "test": slice(i2, N)}[mode]

        s_t, a_t, delta = s_t[sl], a_t[sl], delta[sl]

        # -- normalise (stats computed on this split only) --------------
        def _normalise(x: np.ndarray, enabled: bool):
            mean = x.mean(0, keepdims=True)
            std = x.std(0,  keepdims=True) + np.finfo(np.float32).eps
            return ((x - mean) / std) if enabled else (x - mean), mean, std

        s_norm,     self.s_mean,     self.s_std = _normalise(
            s_t,   normalise_state)
        delta_norm, self.delta_mean, self.delta_std = _normalise(
            delta, normalise_delta)

        # action is already bounded [-0.9, 0.9] - store as-is
        self.s_t = torch.from_numpy(s_norm)
        self.a = torch.from_numpy(a_t)
        self.delta = torch.from_numpy(delta_norm)
        self.seq_len = seq_len

    @classmethod
    def _correct_ball_pos(cls, s: np.ndarray) -> np.ndarray:
        """Apply signed offset to ball_pos column (col 2)."""
        out = s.copy()
        pos = s[:, cls._BALL_POS_COL]
        out[:, cls._BALL_POS_COL] = np.where(
            pos > 0,
            pos + cls._BALL_POS_POS_OFFSET,
            pos + cls._BALL_POS_NEG_OFFSET,
        )
        return out

    def __len__(self) -> int:
        return len(self.s_t) - self.seq_len

    def __getitem__(self, idx: int):
        sl = slice(idx, idx + self.seq_len)
        return self.s_t[sl], self.a[sl], self.delta[sl]
