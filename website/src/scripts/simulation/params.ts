/**
 * Physical parameters: single source of truth for the website simulation.
 * Ported from: balancer/core/params.py, balancer/hardware/constants.py
 * Tuned params from: evaluation/tuning/tuning/cont/{ctrl}/best_params.json
 */

export const PARAMS = {
  // Physical constants (from DEFAULT_PARAMS)
  g: 9.81,
  M: 0.351,       // Cart mass [kg]
  m: 0.024,       // Ball mass [kg]
  R: 2.101,       // Arc radius [m]
  r: 0.009,       // Ball radius [m]
  d: 0.002,       // Gaussian dip depth [m]
  sigma: 0.020 / 2.354820045,  // Gaussian dip width (FWHM=20mm)
  tau_v: 0.15,    // Velocity lag time constant [s]
  mu_ball_rolling: 0.025,
  mu_ball_viscous: 0.009,
  max_cart_velocity: 0.9,  // [m/s]
  ball_restitution: 0.3,   // Steel on PLA

  // System limits (from SYSTEM constants)
  cart_limit: 0.7765,      // [m] operational cart limit
  ball_limit: 0.0810,      // [rad] ball angle limit

  // Settling criteria
  settling_band: 0.01,     // [rad]
  settling_duration: 1.0,  // [s]
  fail_time: 30.0,         // [s]

  // Simulation
  Ts: 0.05,               // Control period [s] (20 Hz)
  command_delay_steps: 1,  // 1-step command delay
} as const;

// Derived constant
export const I_BALL = (2 / 5) * PARAMS.m * PARAMS.r * PARAMS.r;

// Gaussian width parameter k = R²/(2σ²)
export const K_GAUSS = PARAMS.R * PARAMS.R / (2.0 * PARAMS.sigma * PARAMS.sigma);

// State indices
export const CART_X = 0;
export const CART_DOT = 1;
export const BALL_X = 2;
export const BALL_DOT = 3;

// Linearized discrete-time matrices (velocity-input plant, Ts=0.05, tau_v=0.15)
// From: get_discrete_system_velocity(DEFAULT_PARAMS, 0.05, tau_v=0.15)
export const Ad_vel: number[][] = [
  [1.0, 0.042520303413931614, 0.0, 0.0],
  [0.0, 0.7165313105737892, 0.0, 0.0],
  [0.0, 0.0024390138616396558, 0.770664769632359, 0.04611587278218193],
  [0.0, 0.08826127560471216, -8.805554104234606, 0.770664769632359],
];

export const Bd_vel: number[][] = [
  [0.007479696586068389],
  [0.2834686894262108],
  [-0.0024390138616396553],
  [-0.08826127560471214],
];
