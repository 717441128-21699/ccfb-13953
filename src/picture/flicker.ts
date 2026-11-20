import { PlanarImage } from '../film/image';

/**
 * Flicker flattening that preserves genuine lighting events.
 *
 * Within a shot the clean scene's global luma level is constant, so every
 * deviation of the observed per-frame mean from the shot reference level is
 * flicker and is inverted by a multiplicative gain. The one exception is a
 * sustained, strongly dimmed run (a genuine light-off): such frames fall well
 * below the reference and are left untouched, because flattening them would
 * erase real cinematography rather than remove a telecine brightness jump.
 *
 * The shot reference is the median of frame means EXCLUDING the dimmed run, so
 * the light event cannot drag the target brightness down.
 */

/** A frame below this fraction of the shot reference is treated as a real light event. */
const LIGHT_EVENT_RATIO = 0.5;

function meanLuma(img: PlanarImage): number {
  const n = img.width * img.height;
  let s = 0;
  for (let p = 0; p < n; p++) {
    s += 0.299 * img.r[p] + 0.587 * img.g[p] + 0.114 * img.b[p];
  }
  return s / n;
}

function median(values: number[]): number {
  const s = values.slice().sort((a, b) => a - b);
  return s[s.length >> 1];
}

export interface FlickerResult {
  frames: PlanarImage[];
  /** Per-frame gain actually applied. */
  appliedGain: Float32Array;
  observedLevel: Float32Array;
  sustainedLevel: Float32Array;
}

/** Compute the per-frame flicker correction gains without touching pixels. */
export function computeFlickerGains(
  frames: PlanarImage[],
  shots: { start: number; end: number }[],
): { gain: Float32Array; observedLevel: Float32Array; sustainedLevel: Float32Array } {
  const observed = new Float32Array(frames.length);
  const sustained = new Float32Array(frames.length);
  const gain = new Float32Array(frames.length);

  for (let i = 0; i < frames.length; i++) observed[i] = meanLuma(frames[i]);

  for (const shot of shots) {
    const levels: number[] = [];
    for (let i = shot.start; i < shot.end; i++) levels.push(observed[i]);
    const reference = median(levels);
    const lightFloor = reference * LIGHT_EVENT_RATIO;

    // Refine the reference excluding genuine light-off frames.
    const normalLevels = levels.filter((v) => v >= lightFloor);
    const target = normalLevels.length ? median(normalLevels) : reference;

    for (let i = shot.start; i < shot.end; i++) {
      sustained[i] = target;
      if (observed[i] < lightFloor) {
        // Sustained real lighting event: preserve it (no corrective gain).
        gain[i] = 1;
      } else {
        const g = target / Math.max(1e-6, observed[i]);
        gain[i] = Math.min(1.6, Math.max(0.625, g));
      }
    }
  }
  return { gain, observedLevel: observed, sustainedLevel: sustained };
}

export function removeFlicker(frames: PlanarImage[], shots: { start: number; end: number }[]): FlickerResult {
  const { gain, observedLevel, sustainedLevel } = computeFlickerGains(frames, shots);
  const out = frames.map((img) => img.clone());
  for (let f = 0; f < frames.length; f++) {
    const dst = out[f];
    for (let p = 0; p < dst.width * dst.height; p++) {
      dst.r[p] = dst.r[p] * gain[f];
      dst.g[p] = dst.g[p] * gain[f];
      dst.b[p] = dst.b[p] * gain[f];
    }
  }

  return { frames: out, appliedGain: gain, observedLevel, sustainedLevel };
}

export function applyFlickerGains(frames: PlanarImage[], gain: Float32Array): PlanarImage[] {
  const out = frames.map((img) => img.clone());
  for (let f = 0; f < frames.length; f++) {
    const dst = out[f];
    for (let p = 0; p < dst.width * dst.height; p++) {
      dst.r[p] = dst.r[p] * gain[f];
      dst.g[p] = dst.g[p] * gain[f];
      dst.b[p] = dst.b[p] * gain[f];
    }
  }
  return out;
}
