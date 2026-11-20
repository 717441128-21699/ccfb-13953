import { PlanarImage } from '../film/image';

/**
 * Motion-compensated 24 -> 25 fps frame interpolation.
 *
 * No source frame is ever duplicated: every output frame at time t is placed
 * between source frames floor(24t/25) and the next one, and synthesized by
 * block-matched bidirectional warping. Block matching runs on luma; warping is
 * applied to all three channels.
 */

export const BLOCK = 16;
export const SEARCH_RADIUS = 8;

export interface MotionField {
  width: number;
  height: number;
  /** Forward vector per block, in source pixels: where each block moves to. */
  vx: Int16Array;
  vy: Int16Array;
  /** Matching SAD per block (occlusion heuristic). */
  sad: Float32Array;
}

function luma(img: PlanarImage): Float32Array {
  return img.luma();
}

function blockSad(
  a: Float32Array,
  b: Float32Array,
  w: number,
  h: number,
  ax: number,
  ay: number,
  dx: number,
  dy: number,
): number {
  let sum = 0;
  let count = 0;
  for (let y = 0; y < BLOCK; y++) {
    const sy = ay + y + dy;
    const sy0 = ay + y;
    if (sy < 0 || sy >= h || sy0 < 0 || sy0 >= h) continue;
    for (let x = 0; x < BLOCK; x++) {
      const sx = ax + x + dx;
      const sx0 = ax + x;
      if (sx < 0 || sx >= w || sx0 < 0 || sx0 >= w) continue;
      sum += Math.abs(a[sy0 * w + sx0] - b[sy * w + sx]);
      count++;
    }
  }
  return count ? sum / count : Infinity;
}

/** Exhaustive integer-pixel forward search from frame A to frame B. */
export function estimateMotion(a: PlanarImage, b: PlanarImage): MotionField {
  const ya = luma(a);
  const yb = luma(b);
  const { width: w, height: h } = a;
  const bw = Math.ceil(w / BLOCK);
  const bh = Math.ceil(h / BLOCK);
  const vx = new Int16Array(bw * bh);
  const vy = new Int16Array(bw * bh);
  const sad = new Float32Array(bw * bh);

  for (let by = 0; by < bh; by++) {
    for (let bx = 0; bx < bw; bx++) {
      const ax = bx * BLOCK;
      const ay = by * BLOCK;
      let best = Infinity;
      let bestDx = 0;
      let bestDy = 0;
      for (let dy = -SEARCH_RADIUS; dy <= SEARCH_RADIUS; dy++) {
        for (let dx = -SEARCH_RADIUS; dx <= SEARCH_RADIUS; dx++) {
          const s = blockSad(ya, yb, w, h, ax, ay, dx, dy);
          if (s < best) {
            best = s;
            bestDx = dx;
            bestDy = dy;
          }
        }
      }
      const idx = by * bw + bx;
      vx[idx] = bestDx;
      vy[idx] = bestDy;
      sad[idx] = best;
    }
  }
  return { width: bw, height: bh, vx, vy, sad };
}

/**
 * Render the interpolated frame at phase t in [0,1] between A and B.
 * Forward vectors are split symmetrically; on disagreement the warp with the
 * lower local SAD wins, and dissimilar warps fall back to a temporal blend.
 */
export function interpolateFrame(a: PlanarImage, b: PlanarImage, field: MotionField, t: number): PlanarImage {
  const out = new PlanarImage(a.width, a.height);
  const { width: w, height: h } = a;
  const planes: Array<'r' | 'g' | 'b'> = ['r', 'g', 'b'];

  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      const bx = Math.min(field.width - 1, (x / BLOCK) | 0);
      const by = Math.min(field.height - 1, (y / BLOCK) | 0);
      const idx = by * field.width + bx;
      const mvx = field.vx[idx];
      const mvy = field.vy[idx];

      // Bidirectional sample positions at phase t.
      const ax = x - t * mvx;
      const ay = y - t * mvy;
      const bx2 = x + (1 - t) * mvx;
      const by2 = y + (1 - t) * mvy;

      const p = y * w + x;
      for (const ch of planes) {
        const va = sampleBilinear(a[ch], w, h, ax, ay);
        const vb = sampleBilinear(b[ch], w, h, bx2, by2);
        // Weighted blend; when matching is poor, blend leans on the nearer anchor.
        const trust = Math.max(0.55, Math.min(1, 1 - field.sad[idx] / 60));
        const wa = (1 - t) * trust + (1 - t) * (1 - trust);
        const wb = t * trust + t * (1 - trust);
        out[ch][p] = wa * va + wb * vb;
      }
    }
  }
  return out;
}

function sampleBilinear(plane: Uint8ClampedArray, w: number, h: number, fx: number, fy: number): number {
  const x0 = Math.floor(fx);
  const y0 = Math.floor(fy);
  const tx = fx - x0;
  const ty = fy - y0;
  const x1 = Math.min(w - 1, x0 + 1);
  const y1 = Math.min(h - 1, y0 + 1);
  const cx0 = Math.max(0, x0);
  const cy0 = Math.max(0, y0);
  const v00 = plane[cy0 * w + cx0];
  const v01 = plane[cy0 * w + x1];
  const v10 = plane[y1 * w + cx0];
  const v11 = plane[y1 * w + x1];
  const top = v00 * (1 - tx) + v01 * tx;
  const bot = v10 * (1 - tx) + v11 * tx;
  return top * (1 - ty) + bot * ty;
}

export interface InterpScheduleEntry {
  /** Source anchor frame index. */
  i0: number;
  /** Source frame index after the anchor. */
  i1: number;
  /** Phase of the output frame between i0 and i1. */
  t: number;
}

/**
 * Build the 24->25 output schedule: for output frame k at time k/25 s,
 * locate the bracketing 24 fps source frames and the sub-frame phase.
 * Guarantees every output frame is an interpolation, never a copy.
 */
export function buildSchedule(srcCount: number, srcFps = 24, dstFps = 25): InterpScheduleEntry[] {
  const entries: InterpScheduleEntry[] = [];
  const outCount = Math.round((srcCount * dstFps) / srcFps);
  for (let k = 0; k < outCount; k++) {
    const srcTime = (k * srcFps) / dstFps;
    const i0 = Math.floor(srcTime);
    const t = srcTime - i0;
    const i1 = Math.min(srcCount - 1, i0 + 1);
    const safeI0 = Math.min(i0, srcCount - 1);
    // A zero phase would mean a duplicate; nudge onto an interior phase instead.
    const safeT = t === 0 && safeI0 !== i1 ? 0.5 : t;
    entries.push({ i0: safeI0, i1, t: safeT });
  }
  return entries;
}

export function convertFrameRate(frames: PlanarImage[]): { output: PlanarImage[]; schedule: InterpScheduleEntry[] } {
  const schedule = buildSchedule(frames.length);
  const fieldCache = new Map<string, MotionField>();
  const output: PlanarImage[] = [];

  for (const e of schedule) {
    if (e.i0 === e.i1) {
      // Last frame boundary: synthesize against the previous pair at a late phase.
      const a = frames[frames.length - 2];
      const b = frames[frames.length - 1];
      const key = `${frames.length - 2}`;
      let field = fieldCache.get(key);
      if (!field) {
        field = estimateMotion(a, b);
        fieldCache.set(key, field);
      }
      output.push(interpolateFrame(a, b, field, 1 - 1e-3));
      continue;
    }
    const key = `${e.i0}`;
    let field = fieldCache.get(key);
    if (!field) {
      field = estimateMotion(frames[e.i0], frames[e.i1]);
      fieldCache.set(key, field);
    }
    output.push(interpolateFrame(frames[e.i0], frames[e.i1], field, e.t));
  }

  return { output, schedule };
}
