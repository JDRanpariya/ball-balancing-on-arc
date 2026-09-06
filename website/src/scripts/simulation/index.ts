/**
 * Simulation engine: replicates eval_sim.py protocol exactly.
 * Exports everything needed by the Hero component.
 */

export { PARAMS, CART_X, CART_DOT, BALL_X, BALL_DOT } from './params';
export { NonLinearDynamics, type State } from './dynamics';
export {
  type Controller, type ControllerName,
  createController,
  PIDController, LQRController, SMCController, MPCController, NMPCController,
} from './controllers';

import { PARAMS } from './params';
import { NonLinearDynamics, type State } from './dynamics';
import { type Controller, createController, type ControllerName } from './controllers';

/**
 * Run a full episode and return settling time (or null if failed).
 * Matches runner_sim.py::run_trial + _compute_settling_time.
 */
export function runEpisode(
  controller: Controller,
  initialState: State,
  maxTime: number = PARAMS.fail_time,
): { settlingTime: number | null; states: State[]; actions: number[] } {
  const dyn = new NonLinearDynamics();
  dyn.reset();
  controller.reset();

  const steps = Math.round(maxTime / PARAMS.Ts);
  let state: State = [...initialState];
  const states: State[] = [];
  const actions: number[] = [];

  for (let k = 0; k < steps; k++) {
    const u = controller.step(state, PARAMS.cart_limit);
    const clipped = Math.max(-1, Math.min(1, u));
    actions.push(clipped);

    state = dyn.step(state, clipped);
    states.push([...state]);

    // Check for NaN/divergence
    if (state.some(v => !isFinite(v))) break;
  }

  // Compute settling time (same as _compute_settling_time in runner_sim.py)
  const settlingTime = computeSettlingTime(
    states.map(s => s[2]),  // theta history
    states.map((_, i) => (i + 1) * PARAMS.Ts),  // time history
  );

  return { settlingTime, states, actions };
}

/**
 * Compute settling time: first time after which |theta| < band
 * for at least            seconds continuously.
 */
export function computeSettlingTime(
  thetaHist: number[],
  timeHist: number[],
  band: number = PARAMS.settling_band,
  duration: number = PARAMS.settling_duration,
): number | null {
  let enterTime: number | null = null;

  for (let i = 0; i < thetaHist.length; i++) {
    if (Math.abs(thetaHist[i]) < band) {
      if (enterTime === null) enterTime = timeHist[i];
      if (timeHist[i] - enterTime >= duration) {
        return enterTime;
      }
    } else {
      enterTime = null;
    }
  }

  return null;
}
