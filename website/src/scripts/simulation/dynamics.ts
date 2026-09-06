/**
 * NonLinearDynamics: exact port of balancer/core/dynamics.py
 * Motor model: first_order (velocity-input plant)
 * Ball integration: RK4
 * Matches eval_sim.py configuration exactly.
 */

import { PARAMS, I_BALL, K_GAUSS, CART_X, CART_DOT, BALL_X, BALL_DOT } from './params';

export type State = [number, number, number, number];

// ===========================================================================
// GEOMETRY HELPERS (Gaussian dip on arc surface)
// ===========================================================================

function alpha(th: number): number {
  return 2.0 * PARAMS.d * K_GAUSS * th * Math.exp(-K_GAUSS * th * th);
}

function alphaPrime(th: number): number {
  return 2.0 * PARAMS.d * K_GAUSS * Math.exp(-K_GAUSS * th * th) * (1.0 - 2.0 * K_GAUSS * th * th);
}

function cEff(th: number): number {
  const a = alpha(th);
  const C = PARAMS.R * PARAMS.R - 2.0 * PARAMS.R * a * Math.sin(th) + a * a;
  return C + (I_BALL / PARAMS.m) * (PARAMS.R / PARAMS.r) * (PARAMS.R / PARAMS.r);
}

function cEffPrime(th: number): number {
  const a = alpha(th);
  const ap = alphaPrime(th);
  return -2.0 * PARAMS.R * (ap * Math.sin(th) + a * Math.cos(th)) + 2.0 * a * ap;
}

// ===========================================================================
// BALL DYNAMICS (reduced equation: cart accel is known input)
// ===========================================================================

function ballDerivatives(th: number, thdot: number, xddot: number): number {
  const { R, g, mu_ball_rolling, mu_ball_viscous } = PARAMS;
  const a = alpha(th);
  const ce = cEff(th);
  const cep = cEffPrime(th);

  const b2 = -0.5 * cep * thdot * thdot - g * (-R * Math.sin(th) + a);

  // Friction
  const eps_roll = 0.005;
  const Q_rr = mu_ball_rolling * g * R * Math.cos(th) * Math.tanh(thdot / eps_roll);
  const Q_visc = mu_ball_viscous * thdot;

  return (-R * Math.cos(th) * xddot + b2 - (Q_rr + Q_visc)) / ce;
}

function ballStepRK4(th: number, thdot: number, xddot: number, dt: number): [number, number] {
  const k1_th = thdot;
  const k1_thdot = ballDerivatives(th, thdot, xddot);

  const k2_th = thdot + 0.5 * dt * k1_thdot;
  const k2_thdot = ballDerivatives(th + 0.5 * dt * k1_th, k2_th, xddot);

  const k3_th = thdot + 0.5 * dt * k2_thdot;
  const k3_thdot = ballDerivatives(th + 0.5 * dt * k2_th, k3_th, xddot);

  const k4_th = thdot + dt * k3_thdot;
  const k4_thdot = ballDerivatives(th + dt * k3_th, k4_th, xddot);

  const th_new = th + (dt / 6) * (k1_th + 2 * k2_th + 2 * k3_th + k4_th);
  const thdot_new = thdot + (dt / 6) * (k1_thdot + 2 * k2_thdot + 2 * k3_thdot + k4_thdot);

  return [th_new, thdot_new];
}

// ===========================================================================
// CART DYNAMICS (first-order lag, exact integration)
// ===========================================================================

function cartSubstepFirstOrder(
  x: number, xdot: number, v_cmd: number, dt: number
): [number, number, number] {
  const tau = PARAMS.tau_v;
  const decay = Math.exp(-dt / tau);

  // No cart friction (enable_cart_friction=False in eval_sim)
  const v_eff = v_cmd;

  // Exact integration of first-order ODE
  const xdot_new = v_eff + (xdot - v_eff) * decay;
  const x_new = x + v_eff * dt + (xdot - v_eff) * tau * (1.0 - decay);
  const xddot = dt > 0 ? (xdot_new - xdot) / dt : 0;

  return [x_new, xdot_new, xddot];
}

// ===========================================================================
// NonLinearDynamics CLASS
// ===========================================================================

export class NonLinearDynamics {
  private _n_sub: number;
  private _actual_dt: number;
  private _cmd_buffer: number[];
  private _cart_accel: number = 0;
  // Observation filter (matches TwinCAT NC PT1 on encoder velocity)
  private _filtered_xdot: number | null = null;
  private _velocity_filter_tau = 0.01;

  constructor() {
    // first_order: physics_dt = min(tau, 0.01) = 0.01
    // n_sub = round(tau / physics_dt) = round(0.05 / 0.01) = 5
    const physics_dt = Math.min(PARAMS.Ts, 0.01);
    this._n_sub = Math.max(1, Math.round(PARAMS.Ts / physics_dt));
    this._actual_dt = PARAMS.Ts / this._n_sub;
    this._cmd_buffer = new Array(PARAMS.command_delay_steps).fill(0.0);
  }

  reset(): void {
    this._cmd_buffer = new Array(PARAMS.command_delay_steps).fill(0.0);
    this._cart_accel = 0;
    this._filtered_xdot = null;
  }

  /**
   * Advance state by one control step (Ts=0.05s).
   * action: continuous velocity command normalised to [-1, 1]
   * Returns new state.
   */
  step(state: State, action: number): State {
    // Command delay: use buffered action, store new one
    let effective_action: number;
    if (PARAMS.command_delay_steps > 0) {
      effective_action = this._cmd_buffer.shift()!;
      this._cmd_buffer.push(action);
    } else {
      effective_action = action;
    }

    // Resolve velocity command: for cont action_type, action IS the velocity
    // in m/s, clipped to [-max_cart_velocity, +max_cart_velocity]
    // (matches Python _resolve_vcmd: clip(action, -0.9, 0.9))
    const v_cmd = Math.max(-PARAMS.max_cart_velocity,
      Math.min(PARAMS.max_cart_velocity, effective_action));

    let [x, xdot, th, thdot] = state;
    const dt = this._actual_dt;

    for (let i = 0; i < this._n_sub; i++) {
      // Cart soft limits (TwinCAT command saturation)
      let effective_v = v_cmd;
      if (x >= PARAMS.cart_limit && v_cmd > 0) effective_v = 0;
      else if (x <= -PARAMS.cart_limit && v_cmd < 0) effective_v = 0;

      // Integrate cart
      let xddot: number;
      [x, xdot, xddot] = cartSubstepFirstOrder(x, xdot, effective_v, dt);

      // Hard clamp cart position
      if (Math.abs(x) > PARAMS.cart_limit) {
        x = Math.sign(x) * PARAMS.cart_limit;
        if ((x >= PARAMS.cart_limit && xdot > 0) || (x <= -PARAMS.cart_limit && xdot < 0)) {
          xdot = 0;
          xddot = 0;
          this._cart_accel = 0;
        }
      }

      // Integrate ball (RK4)
      [th, thdot] = ballStepRK4(th, thdot, xddot, dt);

      // Ball wall collision (partially inelastic)
      if (Math.abs(th) > PARAMS.ball_limit) {
        th = Math.sign(th) * PARAMS.ball_limit;
        if (Math.sign(th) * thdot > 0) {
          thdot = -PARAMS.ball_restitution * thdot;
        }
      }
    }

    // Apply velocity observation filter (matches NC PT1 in TwinCAT)
    if (this._filtered_xdot === null) {
      this._filtered_xdot = xdot;
    }
    const filterAlpha = PARAMS.Ts / (this._velocity_filter_tau + PARAMS.Ts);
    this._filtered_xdot = (1.0 - filterAlpha) * this._filtered_xdot + filterAlpha * xdot;

    return [x, this._filtered_xdot, th, thdot];
  }
}

// ===========================================================================
// Exported geometry helpers (used by SMC and NMPC controllers)
// ===========================================================================

export { alpha, alphaPrime, cEff, cEffPrime, ballDerivatives, ballStepRK4, cartSubstepFirstOrder };
