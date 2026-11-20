import { mad } from '../film/util';

/**
 * Impulse (click/crackle) removal for the recovered optical waveform.
 *
 * Detection uses the residual against a median of NEIGHBORING samples (the
 * center sample is excluded, so a multi-row click cannot hide in its own
 * baseline). On real track data the separation is wide: a dust/scratch drives
 * the normalized waveform to an extreme several times the normal residual
 * spread, while the 99th-percentile ordinary residual stays far below the
 * threshold. The robust scale is a 68th-percentile residual magnitude with a
 * quantization floor, avoiding the MAD->0 collapse of a smooth quantized track.
 *
 * A modest dilation covers the flat interior of a multi-row click (its edge
 * rows fire, its middle rows do not); flagged runs are then filled by monotone
 * cubic interpolation between the bracketing clean samples.
 */

const NEIGHBOR_HALF = 3;
export const DEFAULT_CLICK_K = 10;
const SCALE_FLOOR = 0.01;
/** Longest click run the bridge fills between two detected edge seeds. */
const MAX_CLICK_SPAN = 5;
const MAX_GAP = 16;

export function declick(input: Float32Array, kThreshold: number = DEFAULT_CLICK_K): { output: Float32Array; clicks: number } {
  const x = input;
  const n = x.length;

  const prediction = new Float32Array(n);
  const residual = new Float32Array(n);
  const window: number[] = [];
  for (let i = 0; i < n; i++) {
    window.length = 0;
    for (let j = Math.max(0, i - NEIGHBOR_HALF); j <= Math.min(n - 1, i + NEIGHBOR_HALF); j++) {
      if (j !== i) window.push(x[j]);
    }
    window.sort((a, b) => a - b);
    prediction[i] = window[window.length >> 1];
    residual[i] = x[i] - prediction[i];
  }

  const { mad: m } = mad(residual);
  const sortedAbs = Float32Array.from(residual, Math.abs).sort();
  const q68 = sortedAbs[Math.floor(sortedAbs.length * 0.68)] ?? 0;
  const scale = Math.max(SCALE_FLOOR, 1.4826 * m, q68);
  const threshold = kThreshold * scale;

  // Strong-impulse seeds mark certain click rows. A multi-row click fires on
  // its edge rows but may not on its flat interior, so bridge two seeds that
  // lie within one click span; the bounded span keeps this from running across
  // clean program samples.
  const defective = new Uint8Array(n);
  const seeds: number[] = [];
  for (let i = 0; i < n; i++) {
    if (Math.abs(residual[i]) > threshold) {
      seeds.push(i);
      defective[i] = 1;
    }
  }
  for (let s = 0; s < seeds.length - 1; s++) {
    const a = seeds[s];
    const b = seeds[s + 1];
    if (b - a <= MAX_CLICK_SPAN) {
      for (let k = a + 1; k < b; k++) defective[k] = 1;
    }
  }

  const out = x.slice();
  let i = 0;
  while (i < n) {
    if (!defective[i]) {
      i++;
      continue;
    }
    const start = i;
    while (i < n && defective[i]) i++;
    const end = i;
    if (end - start <= MAX_GAP) {
      const l = start - 1;
      const r = end;
      if (l >= 0 && r < n) {
        for (let k = start; k < end; k++) {
          const t = (k - l) / (r - l);
          out[k] = monotoneInterp(x[l], x[r], prediction[l], prediction[r], t);
        }
      } else {
        for (let k = start; k < end; k++) out[k] = prediction[k];
      }
    } else {
      for (let k = start; k < end; k++) out[k] = prediction[k];
    }
  }

  return { output: out, clicks: seeds.length };
}

/** Cubic Hermite blend with clamped slopes (no overshoot ring around a click). */
function monotoneInterp(y0: number, y1: number, m0: number, m1: number, t: number): number {
  const t2 = t * t;
  const t3 = t2 * t;
  const h00 = 2 * t3 - 3 * t2 + 1;
  const h10 = t3 - 2 * t2 + t;
  const h01 = -2 * t3 + 3 * t2;
  const h11 = t3 - t2;
  const s0 = clampSlope(y0, y1, m0);
  const s1 = clampSlope(y0, y1, m1);
  return h00 * y0 + h10 * s0 + h01 * y1 + h11 * s1;
}

function clampSlope(y0: number, y1: number, m: number): number {
  const secant = y1 - y0;
  if (secant === 0) return 0;
  if (m * secant < 0) return 0;
  return Math.abs(m) > 3 * Math.abs(secant) ? 3 * secant : m;
}
