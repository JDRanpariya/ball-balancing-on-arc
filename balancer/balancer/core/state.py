from dataclasses import dataclass

# Index constants (used everywhere as array indices)

CART_X   = 0
CART_DOT = 1
BALL_X   = 2
BALL_DOT = 3

@dataclass
class State:
    cart_position: float = 0.0
    cart_velocity: float = 0.0
    ball_position: float = 0.0
    ball_velocity: float = 0.0
    reward:        float = 0.0
    action:        float = 0.0

    def to_array(self):
        import numpy as np
        return np.array(
            [self.cart_position, self.cart_velocity,
             self.ball_position, self.ball_velocity],
            dtype=np.float32,
        )