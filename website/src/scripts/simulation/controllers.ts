/**
 * Controllers: exact port of balancer/controllers/*.py with CMA-ES tuned parameters.
 * Matches evaluation/tuning/tuning/cont/ best_params.json for each controller.
 */

import {
  PARAMS, I_BALL, K_GAUSS,
  CART_X, CART_DOT, BALL_X, BALL_DOT,
  Ad_vel, Bd_vel,
} from './params';
import {
  type State,
  alpha, cEff, cEffPrime, ballDerivatives, ballStepRK4, cartSubstepFirstOrder,
} from './dynamics';

// ===========================================================================
// Controller interface
// ===========================================================================

export interface Controller {
  name: string;
  step(state: State, cart_limit: number): number;
  reset(): void;
}

// ===========================================================================
// PID Controller
// Tuned: Kp=19.782, Ki=0.0, Kd=1.367, wall_margin=0.12, override_gain=0.5
// ===========================================================================

export class PIDController implements Controller {
  name = 'PID';
  private Kp = 19.782;
  private Ki = 0.0;
  private Kd = 1.367;
  private Ts = 0.05;
  private integral_limit = 5.0;
  private wall_margin = 0.12;
  private override_gain = 0.5;
  private _integral = 0;

  step(state: State, cart_limit: number): number {
    const theta = state[BALL_X];
    const theta_dot = state[BALL_DOT];

    // Integrate
    this._integral += theta * this.Ts;
    this._integral = clamp(this._integral, -this.integral_limit, this.integral_limit);

    // PI-D control law
    let u = this.Kp * theta + this.Ki * this._integral + this.Kd * theta_dot;

    // Wall-proximity override
    const cart_pos = state[CART_X];
    if (cart_pos > (cart_limit - this.wall_margin) && u > 0) {
      u = -this.override_gain;
    } else if (cart_pos < -(cart_limit - this.wall_margin) && u < 0) {
      u = this.override_gain;
    }

    // Saturate
    const u_unsat = u;
    u = clamp(u, -1, 1);

    // Anti-windup
    if (Math.abs(this.Ki) > 1e-12 && Math.abs(u_unsat) > 1) {
      this._integral -= theta * this.Ts;
    }

    return u;
  }

  reset(): void {
    this._integral = 0;
  }
}

// ===========================================================================
// LQR Controller
// Tuned: K = [3.2215, 1.9804, -36.9167, -2.0228]
// ===========================================================================

export class LQRController implements Controller {
  name = 'LQR';
  // Paper Exp-1 params (arxiv Table I footnote): K=[0.175,-0.242,-20.746,-6.054], WO gain=0.4
  private K = [0.17524246, -0.24188056, -20.74557764, -6.05351731];
  private wall_margin = 0.12;
  private override_gain = 0.4;

  step(state: State, cart_limit: number): number {
    // u = -K @ x
    let u = -(this.K[0] * state[0] + this.K[1] * state[1] +
              this.K[2] * state[2] + this.K[3] * state[3]);

    // Wall-proximity override
    const cart_pos = state[CART_X];
    if (cart_pos > (cart_limit - this.wall_margin) && u > 0) {
      u = -this.override_gain;
    } else if (cart_pos < -(cart_limit - this.wall_margin) && u < 0) {
      u = this.override_gain;
    }

    return clamp(u, -1, 1);
  }

  reset(): void {}
}

// ===========================================================================
// SMC Controller (velocity-level sliding mode)
// Tuned: lam=2.0, k=3.0, phi=0.85, lam_x=0.0
// ===========================================================================

export class SMCController implements Controller {
  name = 'SMC';
  // Deployed SMC+WO params (Supplementary A3, sec:smc_impl):
  //   "We find k_d=0.5 sufficient on hardware"
  //   "CMA-ES selects phi=0.806 (SMC+WO)"
  //   "retained in the deployed SMC+WO variant (k_d=0.5)"
  //   wall-override: d_w=0.12m, u_ov=0.5
  private lam = 3.0;      // Supplementary A3 CMA-ES table: lambda=3.0 (SMC+WO)
  private k = 6.0;        // Supplementary A3 CMA-ES table: k=6.0 (SMC+WO)
  private phi = 0.806;    // SMC+WO boundary layer (CMA-ES, Supplementary A3)
  private k_d = 0.5;      // cart-velocity damping (Supplementary A3: "k_d=0.5 sufficient on hardware")
  private wall_margin = 0.12;
  private override_gain = 0.5;
  private _prev_u = 0;

  step(state: State, cart_limit: number): number {
    const cart_pos = state[CART_X];
    const xdot = state[CART_DOT];
    const th = state[BALL_X];  // sim: no sensor offset (hardware uses -0.007 for rig calibration)
    const thdot = state[BALL_DOT];

    // 1. Sliding surface: sigma = thdot + lam*th
    const sigma = thdot + this.lam * th;

    // 2. Dynamics decomposition (f_vel, g_vel)
    const [f_vel, g_vel] = this._ballDynamicsSplit(th, thdot, xdot);

    // 3. Equivalent + switching control (drives sigma -> 0)
    const sat_s = Math.tanh(sigma / this.phi);
    const F = f_vel + this.lam * thdot;
    // 4. v_cmd = -(F + k*sat(sigma)) / G  -  k_d*xdot
    //    First term: SMC drives sigma->0 (ball stabilisation).
    //    k_d*xdot: explicit cart-velocity damping - without it the equivalent
    //    control's residual bias (model error, friction) integrates into
    //    unbounded cart drift after the ball is balanced.
    const v_cmd = -(F + this.k * sat_s) / g_vel - this.k_d * xdot;

    // 5. Normalise to [-1, 1]
    let u = clamp(v_cmd / PARAMS.max_cart_velocity, -1, 1);

    // 6. Wall-proximity override (safety)
    if (cart_pos > (cart_limit - this.wall_margin) && u > 0) {
      u = -this.override_gain;
    } else if (cart_pos < -(cart_limit - this.wall_margin) && u < 0) {
      u = this.override_gain;
    }

    u = clamp(u, -1, 1);
    this._prev_u = u;
    return u;
  }

  private _ballDynamicsSplit(th: number, thdot: number, xdot: number): [number, number] {
    const { R, g, mu_ball_rolling, mu_ball_viscous, tau_v } = PARAMS;
    const I = I_BALL;

    const a = alpha(th);
    const ce = cEff(th);
    const cep = cEffPrime(th);

    const b2 = -0.5 * cep * thdot * thdot - g * (-R * Math.sin(th) + a);

    // Ball friction
    const eps = 0.005;
    const friction = mu_ball_rolling * g * R * Math.cos(th) * Math.tanh(thdot / eps)
                   + mu_ball_viscous * thdot;

    const f_vel = (b2 - friction + R * Math.cos(th) * xdot / tau_v) / ce;
    let g_vel_val = -R * Math.cos(th) / (ce * tau_v);

    if (Math.abs(g_vel_val) < 1e-8) {
      g_vel_val = Math.sign(g_vel_val) * 1e-8 || 1e-8;
    }

    return [f_vel, g_vel_val];
  }

  reset(): void {
    this._prev_u = 0;
  }
}

// ===========================================================================
// MPC+WO Controller (Linear QP via ADMM + wall override)
// Tuned: Q_diag=[0.001, 0, 40, 0], R=0.017, N=10, tau_v=0.15
// Wall override: margin=0.12, gain=0.5
// Uses linearized velocity-input discrete model (Ad_vel, Bd_vel)
// ===========================================================================

export class MPCController implements Controller {
  name = 'MPC+WO';
  // tuning/tuning/cont/mpc_wo/best_params.json: Q=[0.01,0.05,30,0.05], R=0.001, N=10
  // wall_margin=0.10, override_gain=0.4, cart_constraint=false
  private N = 10;
  private Q_diag = [0.01, 0.05, 30.0, 0.05];
  private R_val = 0.001;
  private wall_margin = 0.10;
  private override_gain = 0.4;
  private A_bar: number[][];   // (N*4, 4)
  private B_bar: number[][];   // (N*4, N)
  private H: number[][];       // (N, N) Hessian
  private H_inv: number[][];   // (N, N) inverse of H (precomputed for unconstrained solve)
  private A_cart: number[][];  // (N, 4) - cart position rows of A_bar
  private B_cart: number[][];  // (N, N) - cart position rows of B_bar
  private last_u: number[];

  constructor() {
    const n = 4, N = this.N;
    // Build A_bar, B_bar
    this.A_bar = zeros2d(N * n, n);
    this.B_bar = zeros2d(N * n, N);

    for (let i = 0; i < N; i++) {
      const Ai = matPow(Ad_vel, i + 1);
      for (let r = 0; r < n; r++) {
        for (let c = 0; c < n; c++) {
          this.A_bar[i * n + r][c] = Ai[r][c];
        }
      }
      for (let j = 0; j <= i; j++) {
        const Aij = matPow(Ad_vel, i - j);
        const AijB = matMul(Aij, Bd_vel);
        for (let r = 0; r < n; r++) {
          this.B_bar[i * n + r][j] = AijB[r][0];
        }
      }
    }

    // Hessian: H = B_bar^T @ Q_block @ B_bar + R*I
    this.H = zeros2d(N, N);
    for (let i = 0; i < N; i++) {
      for (let j = 0; j < N; j++) {
        let sum = 0;
        for (let k = 0; k < N; k++) {
          const qMult = (k === N - 1) ? 2.0 : 1.0;
          for (let s = 0; s < n; s++) {
            sum += this.B_bar[k * n + s][i] * this.Q_diag[s] * qMult * this.B_bar[k * n + s][j];
          }
        }
        this.H[i][j] = sum + (i === j ? this.R_val : 0);
      }
    }

    // Precompute H inverse for unconstrained solve (used in ADMM)
    this.H_inv = invertMatrix(this.H);

    // Extract cart position rows (state index 0)
    this.A_cart = zeros2d(N, n);
    this.B_cart = zeros2d(N, N);
    for (let k = 0; k < N; k++) {
      for (let c = 0; c < n; c++) {
        this.A_cart[k][c] = this.A_bar[k * n][c];
      }
      for (let c = 0; c < N; c++) {
        this.B_cart[k][c] = this.B_bar[k * n][c];
      }
    }

    this.last_u = new Array(N).fill(0);
  }

  step(state: State, cart_limit: number): number {
    const n = 4, N = this.N;

    // Linear cost: f = B_bar^T @ Q_block @ (A_bar @ x0)
    const Ax = new Array(N * n).fill(0);
    for (let i = 0; i < N * n; i++) {
      for (let j = 0; j < n; j++) {
        Ax[i] += this.A_bar[i][j] * state[j];
      }
    }

    const f = new Array(N).fill(0);
    for (let i = 0; i < N; i++) {
      for (let k = 0; k < N; k++) {
        const qMult = (k === N - 1) ? 2.0 : 1.0;
        for (let s = 0; s < n; s++) {
          f[i] += this.B_bar[k * n + s][i] * this.Q_diag[s] * qMult * Ax[k * n + s];
        }
      }
    }

    // Cart constraint: free_cart + B_cart @ u in [-cart_limit, cart_limit]
    const free_cart = new Array(N).fill(0);
    for (let k = 0; k < N; k++) {
      for (let j = 0; j < n; j++) {
        free_cart[k] += this.A_cart[k][j] * state[j];
      }
    }

    // Solve QP via ADMM (matches OSQP behavior)
    // min 0.5 u^T H u + f^T u
    // s.t. -1 <= u_i <= 1  (input)
    //      -cart_limit <= free_cart[k] + sum_j B_cart[k][j]*u[j] <= cart_limit
    const u = this.solveQP_ADMM(f, free_cart, cart_limit, N);

    // Warm start for next step
    for (let i = 0; i < N - 1; i++) this.last_u[i] = u[i + 1];
    this.last_u[N - 1] = u[N - 1];

    let action = clamp(u[0], -1, 1);

    // Wall-proximity override (matches Python MPC+WO)
    const cart_pos = state[CART_X];
    if (cart_pos > (cart_limit - this.wall_margin) && action > 0) {
      action = -this.override_gain;
    } else if (cart_pos < -(cart_limit - this.wall_margin) && action < 0) {
      action = this.override_gain;
    }

    return action;
  }

  private solveQP_ADMM(f: number[], free_cart: number[], cart_limit: number, N: number): number[] {
    // ADMM for QP: min 0.5 u^T H u + f^T u, s.t. constraints
    // Penalty parameter
    const rho = 1.0;
    const maxIter = 200;

    // Augmented system: (H + rho*I)^{-1}; precompute
    const H_aug_inv = zeros2d(N, N);
    for (let i = 0; i < N; i++) {
      for (let j = 0; j < N; j++) {
        H_aug_inv[i][j] = this.H[i][j] + (i === j ? rho : 0);
      }
    }
    const H_rho_inv = invertMatrix(H_aug_inv);

    // Variables
    const u = this.last_u.slice(); // warm start
    const z = u.slice(); // auxiliary
    const y = new Array(N).fill(0); // dual

    for (let iter = 0; iter < maxIter; iter++) {
      // u-step: u = (H + rho*I)^{-1} * (rho*z - y - f)
      const rhs = new Array(N).fill(0);
      for (let i = 0; i < N; i++) {
        rhs[i] = rho * z[i] - y[i] - f[i];
      }
      for (let i = 0; i < N; i++) {
        u[i] = 0;
        for (let j = 0; j < N; j++) {
          u[i] += H_rho_inv[i][j] * rhs[j];
        }
      }

      // z-step: project u + y/rho onto constraints
      for (let i = 0; i < N; i++) {
        z[i] = clamp(u[i] + y[i] / rho, -1, 1);
      }

      // Also enforce cart constraints via projection
      // B_cart @ z must satisfy -cart_limit - free_cart <= B_cart@z <= cart_limit - free_cart
      for (let k = 0; k < N; k++) {
        let cart_pred = free_cart[k];
        for (let j = 0; j < N; j++) {
          cart_pred += this.B_cart[k][j] * z[j];
        }
        if (cart_pred > cart_limit) {
          // Scale down inputs proportionally to fix violation
          const excess = cart_pred - cart_limit;
          // Find the most responsible input at this horizon step
          let maxContrib = 0, maxIdx = 0;
          for (let j = 0; j <= Math.min(k, N - 1); j++) {
            const contrib = Math.abs(this.B_cart[k][j] * z[j]);
            if (contrib > maxContrib) { maxContrib = contrib; maxIdx = j; }
          }
          if (Math.abs(this.B_cart[k][maxIdx]) > 1e-10) {
            z[maxIdx] -= excess / this.B_cart[k][maxIdx];
            z[maxIdx] = clamp(z[maxIdx], -1, 1);
          }
        } else if (cart_pred < -cart_limit) {
          const deficit = -cart_limit - cart_pred;
          let maxContrib = 0, maxIdx = 0;
          for (let j = 0; j <= Math.min(k, N - 1); j++) {
            const contrib = Math.abs(this.B_cart[k][j] * z[j]);
            if (contrib > maxContrib) { maxContrib = contrib; maxIdx = j; }
          }
          if (Math.abs(this.B_cart[k][maxIdx]) > 1e-10) {
            z[maxIdx] -= deficit / this.B_cart[k][maxIdx];
            z[maxIdx] = clamp(z[maxIdx], -1, 1);
          }
        }
      }

      // y-step (dual update)
      for (let i = 0; i < N; i++) {
        y[i] += rho * (u[i] - z[i]);
      }

      // Convergence check (primal residual)
      let primal_res = 0;
      for (let i = 0; i < N; i++) primal_res += (u[i] - z[i]) * (u[i] - z[i]);
      if (primal_res < 1e-8) break;
    }

    return z;
  }

  reset(): void {
    this.last_u = new Array(this.N).fill(0);
  }
}

// ===========================================================================
// NMPC Controller (Nonlinear MPC via sequence shooting + coordinate descent)
// Tuned: Q_diag=[0, 0, 9, 0], R=0.01, N=10
// Optimizes full N-step action sequence (not just constant input)
// ===========================================================================

export class NMPCController implements Controller {
  name = 'NMPC';
  // tuning/tuning/cont/nmpc/best_params.json: 100% sim SR
  private N = 10;
  private Q_theta = 9.0;
  private R_val = 0.006;
  private dt = 0.05;
  private sequence: number[];
  // Perturbation deltas for coordinate descent
  private deltas = [-0.4, -0.2, -0.1, -0.05, 0.05, 0.1, 0.2, 0.4];

  constructor() {
    this.sequence = new Array(this.N).fill(0);
  }

  step(state: State, cart_limit: number): number {
    const N = this.N;

    // Warm start: shift previous solution left
    for (let i = 0; i < N - 1; i++) this.sequence[i] = this.sequence[i + 1];
    this.sequence[N - 1] = this.sequence[N - 2] || 0;

    // Coordinate descent: 3 rounds, optimize each position
    for (let round = 0; round < 3; round++) {
      for (let k = 0; k < N; k++) {
        let bestCost = this._rolloutSequenceCost(state, cart_limit);
        let bestVal = this.sequence[k];

        for (const delta of this.deltas) {
          const trial = clamp(this.sequence[k] + delta, -1, 1);
          if (trial === bestVal) continue;
          const old = this.sequence[k];
          this.sequence[k] = trial;
          const cost = this._rolloutSequenceCost(state, cart_limit);
          if (cost < bestCost) {
            bestCost = cost;
            bestVal = trial;
          }
          this.sequence[k] = old;
        }
        this.sequence[k] = bestVal;
      }
    }

    return clamp(this.sequence[0], -1, 1);
  }

  private _rolloutSequenceCost(state: State, cart_limit: number): number {
    let px = state[CART_X], pxd = state[CART_DOT];
    let pth = state[BALL_X], pthd = state[BALL_DOT];
    let cost = 0;
    const dt = this.dt;
    const decayFactor = Math.exp(-dt / PARAMS.tau_v);

    for (let step = 0; step < this.N; step++) {
      const u = this.sequence[step];
      let eff_v = u * PARAMS.max_cart_velocity;

      // Cart soft limits
      if (px >= cart_limit && eff_v > 0) eff_v = 0;
      else if (px <= -cart_limit && eff_v < 0) eff_v = 0;

      // Cart dynamics (first-order lag)
      const pxd_new = eff_v + (pxd - eff_v) * decayFactor;
      const px_new = px + eff_v * dt + (pxd - eff_v) * PARAMS.tau_v * (1 - decayFactor);
      const xddot = (pxd_new - pxd) / dt;
      px = px_new;
      pxd = pxd_new;

      // Hard clamp cart
      if (Math.abs(px) > cart_limit) {
        px = Math.sign(px) * cart_limit;
        if (Math.sign(px) * pxd > 0) { pxd = 0; }
      }

      // Ball dynamics (full nonlinear RK4)
      [pth, pthd] = ballStepRK4(pth, pthd, xddot, dt);

      // Ball wall
      if (Math.abs(pth) > PARAMS.ball_limit) {
        pth = Math.sign(pth) * PARAMS.ball_limit;
        if (Math.sign(pth) * pthd > 0) pthd = -PARAMS.ball_restitution * pthd;
      }

      // Stage cost
      cost += this.Q_theta * pth * pth + this.R_val * u * u;

      // Cart constraint penalty (soft)
      const proximity = Math.abs(px) / cart_limit;
      if (proximity > 0.9) cost += 50 * (proximity - 0.9) * (proximity - 0.9);

      // Ball limit penalty
      const ballProx = Math.abs(pth) / PARAMS.ball_limit;
      if (ballProx > 0.8) cost += 200 * (ballProx - 0.8) * (ballProx - 0.8);
    }

    // Terminal cost (2x weight on final state)
    cost += 2.0 * this.Q_theta * pth * pth;

    return cost;
  }

  reset(): void {
    this.sequence = new Array(this.N).fill(0);
  }
}

// ===========================================================================
// Factory
// ===========================================================================

export type ControllerName = 'pid' | 'lqr' | 'smc' | 'mpc' | 'nmpc';

export function createController(name: ControllerName): Controller {
  switch (name) {
    case 'pid': return new PIDController();
    case 'lqr': return new LQRController();
    case 'smc': return new SMCController();
    case 'mpc': return new MPCController();
    case 'nmpc': return new NMPCController();
    default: throw new Error(                             );
  }
}

// ===========================================================================
// Utilities
// ===========================================================================

function clamp(v: number, lo: number, hi: number): number {
  return v < lo ? lo : v > hi ? hi : v;
}

function zeros2d(rows: number, cols: number): number[][] {
  return Array.from({ length: rows }, () => new Array(cols).fill(0));
}

function matMul(A: number[][], B: number[][]): number[][] {
  const m = A.length, n = B[0].length, p = B.length;
  const C = zeros2d(m, n);
  for (let i = 0; i < m; i++)
    for (let j = 0; j < n; j++)
      for (let k = 0; k < p; k++)
        C[i][j] += A[i][k] * B[k][j];
  return C;
}

function matPow(A: number[][], power: number): number[][] {
  const n = A.length;
  let result = zeros2d(n, n);
  for (let i = 0; i < n; i++) result[i][i] = 1; // identity
  let base = A.map(row => row.slice());
  let p = power;
  while (p > 0) {
    if (p & 1) result = matMul(result, base);
    base = matMul(base, base);
    p >>= 1;
  }
  return result;
}

/** Gauss-Jordan matrix inversion for small (N×N) matrices. */
function invertMatrix(M: number[][]): number[][] {
  const n = M.length;
  // Augment [M | I]
  const aug = M.map((row, i) => {
    const ext = new Array(n).fill(0);
    ext[i] = 1;
    return [...row, ...ext];
  });

  for (let col = 0; col < n; col++) {
    // Partial pivot
    let maxRow = col;
    for (let row = col + 1; row < n; row++) {
      if (Math.abs(aug[row][col]) > Math.abs(aug[maxRow][col])) maxRow = row;
    }
    [aug[col], aug[maxRow]] = [aug[maxRow], aug[col]];

    const pivot = aug[col][col];
    if (Math.abs(pivot) < 1e-15) {
      // Singular: return identity as fallback
      const I = zeros2d(n, n);
      for (let i = 0; i < n; i++) I[i][i] = 1;
      return I;
    }

    // Scale pivot row
    for (let j = col; j < 2 * n; j++) aug[col][j] /= pivot;

    // Eliminate column
    for (let row = 0; row < n; row++) {
      if (row === col) continue;
      const factor = aug[row][col];
      for (let j = col; j < 2 * n; j++) {
        aug[row][j] -= factor * aug[col][j];
      }
    }
  }

  // Extract right half
  return aug.map(row => row.slice(n));
}
