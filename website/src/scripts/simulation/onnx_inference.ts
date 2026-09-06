/**
 * ONNX Runtime Web inference for RL policy models.
 * Loads exported .onnx files from public/models/ and runs deterministic inference.
 * Fully lazy: nothing runs until an RL controller is selected.
 *
 * Follows: https://onnxruntime.ai/docs/build/web.html
 */

const BASE_URL = (import.meta.env?.BASE_URL || '/').replace(/\/$/, '');

export type RLModelName = 'ppo' | 'td3' | 'sac' | 'tqc' | 'trpo';
export const RL_MODELS: RLModelName[] = ['ppo', 'td3', 'sac', 'tqc', 'trpo'];

// All state is lazily initialized
let ort: any = null;
const sessionCache: Map<string, any> = new Map();
const loadingPromises: Map<string, Promise<any>> = new Map();

async function initOrt(): Promise<any> {
  if (ort) return ort;
  // Dynamic import from CDN: the ESM bundle + WASM both come from jsDelivr.
  // (onnxruntime-web is NOT bundled: its WASM exceeds Cloudflare Pages'
  // 25 MiB per-file limit, and a bare external import resolves to an
  // empty stub in the browser - InferenceSession would be undefined.)
  const mod = await import(
    /* @vite-ignore */
    'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.26.0/dist/ort.min.mjs'
  );
  ort = mod;
  // Load WASM files from jsDelivr CDN (avoids Vite/Astro public/ import issues)
  ort.env.wasm.wasmPaths = 'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.26.0/dist/';
  // Single-threaded to avoid SharedArrayBuffer/COOP/COEP requirements
  ort.env.wasm.numThreads = 1;
  return ort;
}

async function loadModel(name: RLModelName): Promise<any> {
  if (sessionCache.has(name)) return sessionCache.get(name);
  if (loadingPromises.has(name)) return loadingPromises.get(name);

  const promise = (async () => {
    const runtime = await initOrt();
    const url = BASE_URL + '/models/' + name + '_policy.onnx';
    const session = await runtime.InferenceSession.create(url, {
      executionProviders: ['wasm'],
    });
    sessionCache.set(name, session);
    loadingPromises.delete(name);
    return session;
  })();

  loadingPromises.set(name, promise);
  return promise;
}

/**
 * Run deterministic inference.
 * @param name - RL model name
 * @param state - [cart_x, cart_vel, ball_x, ball_vel]
 * @returns action clipped to [-1, 1], or 0 on failure
 */
export async function predict(
  name: RLModelName,
  state: [number, number, number, number],
): Promise<number> {
  try {
    const runtime = await initOrt();
    const session = await loadModel(name);
    const input = new runtime.Tensor('float32', Float32Array.from(state), [1, 4]);
    const results = await session.run({ obs: input });
    const action = (results.action.data as Float32Array)[0];
    return Math.max(-1, Math.min(1, action));
  } catch (e) {
    console.warn('[onnx_inference] predict failed:', e);
    return 0;
  }
}

/**
 * Preload a model so first inference is fast.
 */
export function preloadModel(name: RLModelName): Promise<any> {
  return loadModel(name).catch(() => null);
}
