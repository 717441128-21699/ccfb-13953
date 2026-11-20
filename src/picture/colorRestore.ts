import { PlanarImage } from '../film/image';

/**
 * Per-shot color restoration for faded dye stock.
 *
 * Old 16mm color film loses blue and gains red (the pink cast). Estimation is
 * blind: on flicker-stabilized frames the per-channel robust means should
 * share a common gray-world level, so the required reciprocal multipliers are
 * derived from shot statistics, never from the synthesizer's known gains.
 */

function channelRobustMean(plane: Uint8ClampedArray): number {
  const samples: number[] = [];
  for (let p = 0; p < plane.length; p += 11) samples.push(plane[p]);
  samples.sort((a, b) => a - b);
  const lo = (samples.length * 0.1) | 0;
  const hi = Math.ceil(samples.length * 0.9);
  let s = 0;
  for (let i = lo; i < hi; i++) s += samples[i];
  return s / (hi - lo);
}

export interface ColorResult {
  frames: PlanarImage[];
  /** Per-shot channel gains [r,g,b]. */
  shotGains: [number, number, number][];
}

/** Estimate per-shot gray-world channel gains without modifying pixels. */
export function computeColorGains(
  frames: PlanarImage[],
  shots: { start: number; end: number }[],
): [number, number, number][] {
  const shotGains: [number, number, number][] = [];

  for (const shot of shots) {
    // Pool channel statistics across a subset of the shot's frames.
    let rSum = 0;
    let gSum = 0;
    let bSum = 0;
    let cnt = 0;
    const step = Math.max(1, Math.floor((shot.end - shot.start) / 4));
    for (let f = shot.start; f < shot.end; f += step) {
      const img = frames[f];
      rSum += channelRobustMean(img.r);
      gSum += channelRobustMean(img.g);
      bSum += channelRobustMean(img.b);
      cnt++;
    }
    const rMean = rSum / cnt;
    const gMean = gSum / cnt;
    const bMean = bSum / cnt;
    const target = (rMean + gMean + bMean) / 3;

    shotGains.push([
      Math.min(1.8, Math.max(0.55, target / Math.max(1, rMean))),
      Math.min(1.8, Math.max(0.55, target / Math.max(1, gMean))),
      Math.min(1.8, Math.max(0.55, target / Math.max(1, bMean))),
    ]);
  }

  return shotGains;
}

export function applyColorGains(
  frames: PlanarImage[],
  shots: { start: number; end: number }[],
  shotGains: [number, number, number][],
): PlanarImage[] {
  const out = frames.map((f) => f.clone());
  shots.forEach((shot, si) => {
    const gains = shotGains[si];
    for (let f = shot.start; f < shot.end; f++) {
      const img = out[f];
      for (let p = 0; p < img.width * img.height; p++) {
        img.r[p] = img.r[p] * gains[0];
        img.g[p] = img.g[p] * gains[1];
        img.b[p] = img.b[p] * gains[2];
      }
    }
  });
  return out;
}

export function restoreColor(frames: PlanarImage[], shots: { start: number; end: number }[]): ColorResult {
  const shotGains = computeColorGains(frames, shots);
  return { frames: applyColorGains(frames, shots, shotGains), shotGains };
}
