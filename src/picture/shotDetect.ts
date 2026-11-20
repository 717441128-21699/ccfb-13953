import { PlanarImage } from '../film/image';

/**
 * Blind shot boundary detection on brightness-normalized luma histograms.
 *
 * The normalization (each pixel's luma divided by the frame's robust mean) is
 * what separates a real cut from a genuine light on/off event. A light event
 * scales the whole frame by a constant factor, so the NORMALIZED histogram is
 * virtually unchanged and the distance stays low. A change of scene rearranges
 * the tonal distribution and drives the distance up regardless of exposure.
 */

const BINS = 32;
/** Normalized luma spans roughly [0, 4] around a mean of 1. */
const NORM_RANGE = 4;

/**
 * Spatially robust cut score. A real scene cut changes every region of the
 * frame at once, so normalized-histogram distance rises across ALL spatial
 * blocks. A scratch or dust speck affects only a small area, so its distance
 * is high in a block or two but low elsewhere; taking the median over blocks
 * rejects those local defects while a global cut keeps a high median.
 */
const GRID = 4; // 4x4 spatial blocks

function blockDistance(a: PlanarImage, b: PlanarImage): number {
  const bw = Math.floor(a.width / GRID);
  const bh = Math.floor(a.height / GRID);
  const scores: number[] = [];
  for (let by = 0; by < GRID; by++) {
    for (let bx = 0; bx < GRID; bx++) {
      const ha = blockHistogram(a, bx * bw, by * bh, bw, bh);
      const hb = blockHistogram(b, bx * bw, by * bh, bw, bh);
      let bc = 0;
      for (let i = 0; i < BINS; i++) bc += Math.sqrt(ha[i] * hb[i]);
      scores.push(1 - bc);
    }
  }
  scores.sort((x, y) => x - y);
  return scores[scores.length >> 1];
}

function blockHistogram(img: PlanarImage, x0: number, y0: number, w: number, h: number): Float32Array {
  const hist = new Float32Array(BINS);
  let sum = 0;
  let count = 0;
  for (let y = y0; y < y0 + h; y++) {
    for (let x = x0; x < x0 + w; x++) {
      const p = y * img.width + x;
      sum += 0.299 * img.r[p] + 0.587 * img.g[p] + 0.114 * img.b[p];
      count++;
    }
  }
  const mean = Math.max(1, sum / count);
  for (let y = y0; y < y0 + h; y++) {
    for (let x = x0; x < x0 + w; x++) {
      const p = y * img.width + x;
      const luma = 0.299 * img.r[p] + 0.587 * img.g[p] + 0.114 * img.b[p];
      const bin = Math.min(BINS - 1, Math.max(0, Math.floor((luma / mean / NORM_RANGE) * BINS)));
      hist[bin]++;
    }
  }
  for (let i = 0; i < BINS; i++) hist[i] /= count;
  return hist;
}

export interface ShotBoundary {
  start: number;
  end: number; // exclusive
}

export function detectShots(frames: PlanarImage[], cutThreshold = 0.15): ShotBoundary[] {
  if (frames.length === 0) return [];
  const starts = [0];
  for (let i = 1; i < frames.length; i++) {
    if (blockDistance(frames[i - 1], frames[i]) > cutThreshold) starts.push(i);
  }
  return starts.map((start, k) => ({ start, end: starts[k + 1] ?? frames.length }));
}
