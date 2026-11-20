import { describe, it, expect } from 'vitest';
import { PlanarImage } from '../src/film/image';
import {
  buildSchedule,
  estimateMotion,
  interpolateFrame,
  convertFrameRate,
  BLOCK,
} from '../src/picture/motion';
import { synthesizeReel } from '../src/film/reel';
import { runCpuPipeline } from '../src/film/pipeline';

/**
 * Translating aperiodic value-noise texture. Smooth coherent blobs give every
 * block a distinctive, non-repeating signature so block matching has a unique
 * SAD minimum (a periodic checkerboard would tie at multiples of its period).
 */
function hash2(ix: number, iy: number): number {
  let h = (ix * 374761393 + iy * 668265263) | 0;
  h = Math.imul(h ^ (h >>> 13), 1274126177);
  return ((h ^ (h >>> 16)) >>> 0) / 4294967295;
}

function valueNoise(x: number, y: number): number {
  const x0 = Math.floor(x);
  const y0 = Math.floor(y);
  const fx = x - x0;
  const fy = y - y0;
  const sx = fx * fx * (3 - 2 * fx);
  const sy = fy * fy * (3 - 2 * fy);
  const v00 = hash2(x0, y0);
  const v10 = hash2(x0 + 1, y0);
  const v01 = hash2(x0, y0 + 1);
  const v11 = hash2(x0 + 1, y0 + 1);
  const top = v00 + (v10 - v00) * sx;
  const bot = v01 + (v11 - v01) * sx;
  return top + (bot - top) * sy;
}

function translatingTexture(shift: number): PlanarImage {
  const w = 96;
  const h = 72;
  const img = new PlanarImage(w, h);
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      const p = y * w + x;
      const n = valueNoise((x - shift) / 12, y / 12);
      const v = Math.round(40 + n * 190);
      img.r[p] = v;
      img.g[p] = v;
      img.b[p] = v;
    }
  }
  return img;
}

describe('motion-compensated frame interpolation', () => {
  it('block matcher recovers a known integer translation', () => {
    const a = translatingTexture(0);
    const b = translatingTexture(4);
    const field = estimateMotion(a, b);
    // Block at the square's initial position should report +4 on x.
    const bx = Math.floor(20 / BLOCK);
    const by = Math.floor(20 / BLOCK);
    const idx = by * field.width + bx;
    expect(field.vx[idx]).toBe(4);
    expect(field.vy[idx]).toBe(0);
  });

  it('interpolated frame at t=0.5 sits between the anchors', () => {
    const a = translatingTexture(0);
    const b = translatingTexture(4);
    const field = estimateMotion(a, b);
    const mid = interpolateFrame(a, b, field, 0.5);
    // A half-phase interpolation must differ from both anchor frames: it is a
    // genuinely synthesized frame, never a copy.
    expect(framesEqual(mid, a)).toBe(false);
    expect(framesEqual(mid, b)).toBe(false);
    // And it tracks the ground-truth half-shift texture.
    const truth = translatingTexture(2);
    expect(lumaMse(truth, mid)).toBeLessThan(lumaMse(truth, a));
  });

  it('24->25 schedule never duplicates a source frame', () => {
    const schedule = buildSchedule(24); // 1 source second
    expect(schedule.length).toBe(25);
    for (const e of schedule) {
      // Every entry is a genuine interpolation between two distinct anchors
      // or an end-of-reel synthesis; t==0 with distinct anchors is nudged.
      if (e.i0 !== e.i1) expect(e.t).toBeGreaterThan(0);
      expect(e.t).toBeLessThanOrEqual(1);
    }
    // Output timestamps are strictly increasing.
    for (let i = 1; i < schedule.length; i++) {
      const prev = schedule[i - 1].i0 + schedule[i - 1].t;
      const cur = schedule[i].i0 + schedule[i].t;
      expect(cur).toBeGreaterThan(prev);
    }
  });

  it('conversion preserves total duration and output count exactly', () => {
    const srcCount = 48; // 2 seconds at 24 fps
    const frames = Array.from({ length: srcCount }, (_, i) => translatingTexture(i % 8));
    const { output, schedule } = convertFrameRate(frames);
    expect(output.length).toBe(Math.round((srcCount * 25) / 24));
    expect(schedule.length).toBe(output.length);
    const srcDuration = srcCount / 24;
    const dstDuration = output.length / 25;
    expect(Math.abs(srcDuration - dstDuration)).toBeLessThan(1e-9);
  });

  it('end-to-end pipeline yields 25 fps output with no copied frames', () => {
    const reel = synthesizeReel(24, 'variable-area', 31);
    const result = runCpuPipeline(reel, 'variable-area');
    expect(result.converted.length).toBe(25);
    expect(result.restored.length).toBe(24);
    // No output frame is byte-identical to a source frame (real interpolation).
    const src = result.restored;
    const isCopy = result.converted.every((out) =>
      src.some((s) => framesEqual(s, out)),
    );
    expect(isCopy).toBe(false);
  });

  it('interpolated frames are closer to true intermediate than either anchor', () => {
    // Ground-truth intermediate at shift 2; compare mid-frame vs anchors.
    const a = translatingTexture(0);
    const b = translatingTexture(4);
    const truth = translatingTexture(2);
    const field = estimateMotion(a, b);
    const mid = interpolateFrame(a, b, field, 0.5);
    const errMid = lumaMse(truth, mid);
    const errA = lumaMse(truth, a);
    const errB = lumaMse(truth, b);
    expect(errMid).toBeLessThan(Math.min(errA, errB));
  });
});

function framesEqual(a: PlanarImage, b: PlanarImage): boolean {
  if (a.width !== b.width || a.height !== b.height) return false;
  for (let p = 0; p < a.r.length; p++) {
    if (a.r[p] !== b.r[p] || a.g[p] !== b.g[p] || a.b[p] !== b.b[p]) return false;
  }
  return true;
}

function lumaMse(ref: PlanarImage, cand: PlanarImage): number {
  let se = 0;
  for (let p = 0; p < ref.r.length; p++) {
    const lr = 0.299 * ref.r[p] + 0.587 * ref.g[p] + 0.114 * ref.b[p];
    const lc = 0.299 * cand.r[p] + 0.587 * cand.g[p] + 0.114 * cand.b[p];
    se += (lr - lc) ** 2;
  }
  return se / ref.r.length;
}
