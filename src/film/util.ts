/** Deterministic PRNG so every "known defect" sequence is reproducible. */
export function mulberry32(seed: number): () => number {
  let a = seed >>> 0;
  return function next(): number {
    a |= 0;
    a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

/** Box-Muller gaussian from a uniform generator. */
export function gaussian(rand: () => number): number {
  let u = 0;
  let v = 0;
  while (u === 0) u = rand();
  while (v === 0) v = rand();
  return Math.sqrt(-2.0 * Math.log(u)) * Math.cos(2.0 * Math.PI * v);
}

export function clamp(x: number, lo: number, hi: number): number {
  return x < lo ? lo : x > hi ? hi : x;
}

/**
 * Precomputed unit-variance gaussian table for cheap per-pixel film grain.
 * Indexed by a deterministic integer hash so grain is reproducible without
 * paying for Box-Muller on every pixel.
 */
const GRAIN_TABLE_SIZE = 1 << 16;
const GRAIN_TABLE = (() => {
  const rand = mulberry32(0x6a11);
  const table = new Float32Array(GRAIN_TABLE_SIZE);
  for (let i = 0; i < GRAIN_TABLE_SIZE; i++) table[i] = gaussian(rand);
  return table;
})();

export function grain(index: number): number {
  let h = (index | 0) * 0x9e3779b1;
  h ^= h >>> 16;
  h = Math.imul(h, 0x7feb352d);
  h ^= h >>> 15;
  return GRAIN_TABLE[h & (GRAIN_TABLE_SIZE - 1)];
}

export function clampByte(x: number): number {
  return x < 0 ? 0 : x > 255 ? 255 : x | 0;
}

export function lerp(a: number, b: number, t: number): number {
  return a + (b - a) * t;
}

export function median3(a: number, b: number, c: number): number {
  return Math.max(Math.min(a, b), Math.min(Math.max(a, b), c));
}

export function mean(xs: ArrayLike<number>): number {
  let s = 0;
  for (let i = 0; i < xs.length; i++) s += xs[i];
  return xs.length ? s / xs.length : 0;
}

export function rms(xs: Float32Array | number[]): number {
  let s = 0;
  for (let i = 0; i < xs.length; i++) s += xs[i] * xs[i];
  return Math.sqrt(s / xs.length);
}

/** Median absolute deviation (robust scale estimate). */
export function mad(xs: Float32Array): { med: number; mad: number } {
  const sorted = Float32Array.from(xs).sort();
  const med = sorted[sorted.length >> 1] ?? 0;
  const dev = new Float32Array(xs.length);
  for (let i = 0; i < xs.length; i++) dev[i] = Math.abs(xs[i] - med);
  dev.sort();
  return { med, mad: dev[dev.length >> 1] ?? 0 };
}

/**
 * Signal-to-noise ratio in dB between a reference signal and a candidate,
 * after best-fit constant gain alignment (scale-only, no delay search).
 */
export function snrDb(reference: Float32Array, candidate: Float32Array): number {
  const n = Math.min(reference.length, candidate.length);
  let dot = 0;
  let refPow = 0;
  let candPow = 0;
  for (let i = 0; i < n; i++) {
    dot += reference[i] * candidate[i];
    refPow += reference[i] * reference[i];
    candPow += candidate[i] * candidate[i];
  }
  if (refPow === 0 || candPow === 0) return -Infinity;
  const gain = dot / candPow;
  let errPow = 0;
  for (let i = 0; i < n; i++) {
    const e = reference[i] - gain * candidate[i];
    errPow += e * e;
  }
  if (errPow === 0) return Infinity;
  return 10 * Math.log10(refPow / errPow);
}

/** Peak signal-to-noise ratio for 8-bit image planes. */
export function psnrByte(reference: Uint8Array | Uint8ClampedArray, candidate: Uint8Array | Uint8ClampedArray): number {
  const n = Math.min(reference.length, candidate.length);
  let se = 0;
  for (let i = 0; i < n; i++) {
    const d = reference[i] - candidate[i];
    se += d * d;
  }
  if (se === 0) return Infinity;
  const mse = se / n;
  return 10 * Math.log10((255 * 255) / mse);
}
