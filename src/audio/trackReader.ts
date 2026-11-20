import { SPECS, SoundTrackType } from '../film/specs';
import { Reel } from '../film/image';
import { mad } from '../film/util';

/**
 * Reads the optical track back out of the packed-RGBA scanner frames.
 * The soundtrack region is monochrome (R = G = B), so the red channel is the
 * photographic density.
 *
 * Both readers self-calibrate from the strip itself; nothing assumes the
 * synthesizer's constants:
 *  - Variable-area: per-row exposed width, then centered by its median and
 *    scaled by its robust spread, so no hardcoded zero-width or sensitivity.
 *  - Variable-density: per-row interior mean, mapped through black/white
 *    levels taken from near-extreme percentiles (the density signal clusters
 *    mid-range, so central quantiles would compress the dynamic range).
 */

const H = SPECS.scanH;
const INTERIOR = 6;

function mono(data: Uint8ClampedArray, width: number, x: number, y: number): number {
  return data[(y * width + x) * 4];
}

function percentile(sorted: ArrayLike<number>, q: number): number {
  return sorted[Math.min(sorted.length - 1, Math.max(0, Math.floor(q * sorted.length)))];
}

/** All strip pixel values across the reel, sorted, for level estimation. */
function stripValues(reel: Reel): Uint8ClampedArray {
  const vals: number[] = [];
  for (const frame of reel.frames) {
    for (let y = 0; y < H; y += 2) {
      for (let x = SPECS.soundX0; x < SPECS.soundX1; x += 2) {
        vals.push(mono(frame.data, reel.width, x, y));
      }
    }
  }
  vals.sort((a, b) => a - b);
  return Uint8ClampedArray.from(vals);
}

export interface TrackLevels {
  black: number;
  white: number;
}

export function estimateLevels(reel: Reel, type: SoundTrackType): TrackLevels {
  const sorted = stripValues(reel);
  if (type === 'variable-density') {
    // Near-extreme percentiles recover the achievable black/white extents even
    // though most density samples sit around the middle of the range.
    return { black: percentile(sorted, 0.005), white: percentile(sorted, 0.995) };
  }
  // Variable-area is genuinely bimodal black/white; central-extreme quantiles
  // sit safely inside each mode.
  return { black: percentile(sorted, 0.1), white: percentile(sorted, 0.9) };
}

export function readTrack(reel: Reel, type: SoundTrackType, levels: TrackLevels): Float32Array {
  const n = reel.frames.length * H;
  const out = new Float32Array(n);
  const span = Math.max(1, levels.white - levels.black);
  let write = 0;
  for (const frame of reel.frames) {
    for (let y = 0; y < H; y++) {
      let sum = 0;
      let cnt = 0;
      for (let x = SPECS.soundX0; x < SPECS.soundX1; x++) {
        sum += mono(frame.data, reel.width, x, y);
        cnt++;
      }
      if (type === 'variable-area') {
        // Integrated light over the slit equals the anti-aliased tongue width.
        out[write++] = sum / span;
      } else {
        // Density track: mean photographic density across the interior.
        let inner = 0;
        let innerN = 0;
        for (let x = SPECS.soundX0 + INTERIOR; x < SPECS.soundX1 - INTERIOR; x++) {
          inner += mono(frame.data, reel.width, x, y);
          innerN++;
        }
        out[write++] = inner / innerN;
      }
    }
  }

  // Both readers self-calibrate: median is the zero-signal level, robust spread
  // is the unit waveform scale, so no geometry or black/white constants needed.
  const sorted = Float32Array.from(out).sort();
  const zero = percentile(sorted, 0.5);
  const { mad: m } = mad(out);
  const spread = Math.max(1e-3, 1.4826 * m);
  for (let i = 0; i < out.length; i++) out[i] = (out[i] - zero) / spread;
  return out;
}
