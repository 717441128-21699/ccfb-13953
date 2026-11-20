import { PlanarImage } from '../film/image';

/**
 * Scratch / dust removal using neighbor frames, with a spatial-consistency
 * guard so genuine motion (an object moving into a pixel) is not mistaken for
 * a blemish.
 *
 * A pixel is repaired only when BOTH hold:
 *  - temporal: it disagrees with both bracketing frames while those two agree
 *  - spatial:  it also disagrees with the median of its 3x3 neighborhood in
 *              the current frame
 *
 * A moving object's interior is spatially continuous and fails the spatial
 * test; a scratch line or a dust speck is spatially isolated and passes it.
 * Corrections are committed per channel into a working copy, and a pixel is
 * only accepted as a real blemish when enough channels agree; confirmed
 * blemish pixels then get a spatial median inpaint as a fallback.
 */

const TEMPORAL_THRESHOLD = 46;
const NEIGHBOR_AGREE = 40;
const SPATIAL_THRESHOLD = 40;
const CHANNELS_REQUIRED = 2;

export interface RepairStats {
  temporalRepairs: number;
  spatialRepairs: number;
}

type Plane = Uint8ClampedArray;
const PLANES: Array<'r' | 'g' | 'b'> = ['r', 'g', 'b'];

function spatialMedian(plane: Plane, w: number, h: number, x: number, y: number): number {
  const vals: number[] = [];
  for (let dy = -1; dy <= 1; dy++) {
    for (let dx = -1; dx <= 1; dx++) {
      if (dx === 0 && dy === 0) continue;
      const nx = x + dx;
      const ny = y + dy;
      if (nx < 0 || ny < 0 || nx >= w || ny >= h) continue;
      vals.push(plane[ny * w + nx]);
    }
  }
  vals.sort((a, b) => a - b);
  return vals[vals.length >> 1];
}

export function repairBlemishes(frames: PlanarImage[]): { frames: PlanarImage[]; stats: RepairStats } {
  const out = frames.map((f) => f.clone());
  const stats: RepairStats = { temporalRepairs: 0, spatialRepairs: 0 };

  for (let f = 0; f < frames.length; f++) {
    const cur = out[f];
    const src = frames[f];
    const prev = f > 0 ? frames[f - 1] : undefined;
    const next = f < frames.length - 1 ? frames[f + 1] : undefined;
    const { width: w, height: h } = cur;
    const channelHits = new Uint8Array(w * h);

    for (const ch of PLANES) {
      for (let y = 0; y < h; y++) {
        for (let x = 0; x < w; x++) {
          const p = y * w + x;
          const v = src[ch][p];
          const a = prev ? prev[ch][p] : undefined;
          const b = next ? next[ch][p] : undefined;

          let temporalHit = false;
          let replacement = v;
          if (a !== undefined && b !== undefined) {
            const neighborMean = (a + b) / 2;
            temporalHit = Math.abs(a - b) < NEIGHBOR_AGREE && Math.abs(v - neighborMean) > TEMPORAL_THRESHOLD;
            replacement = (a + b) >> 1;
          } else if (a !== undefined || b !== undefined) {
            const ref = (a ?? b) as number;
            temporalHit = Math.abs(v - ref) > TEMPORAL_THRESHOLD;
            replacement = ref;
          }
          if (!temporalHit) continue;

          // Spatial guard: the pixel must also stand out inside its own frame.
          const med = spatialMedian(src[ch], w, h, x, y);
          if (Math.abs(v - med) <= SPATIAL_THRESHOLD) continue;

          cur[ch][p] = replacement;
          channelHits[p]++;
        }
      }
    }

    // Revert lone-channel corrections: not enough evidence for a real defect.
    for (let p = 0; p < w * h; p++) {
      if (channelHits[p] > 0 && channelHits[p] < CHANNELS_REQUIRED) {
        for (const ch of PLANES) cur[ch][p] = src[ch][p];
      }
    }

    // Confirmed blemishes: count and apply spatial inpaint where needed.
    for (let p = 0; p < w * h; p++) {
      if (channelHits[p] < CHANNELS_REQUIRED) continue;
      stats.temporalRepairs++;
      const x = p % w;
      const y = (p / w) | 0;
      for (const ch of PLANES) {
        const med = spatialMedian(cur[ch], w, h, x, y);
        if (Math.abs(cur[ch][p] - med) > SPATIAL_THRESHOLD) {
          cur[ch][p] = med;
          stats.spatialRepairs++;
        }
      }
    }
  }

  return { frames: out, stats };
}
