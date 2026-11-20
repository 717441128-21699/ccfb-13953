/** Digital signal processing primitives shared by synth and decode stages. */

export function hannWindow(n: number): Float32Array {
  const w = new Float32Array(n);
  for (let i = 0; i < n; i++) {
    w[i] = 0.5 * (1 - Math.cos((2 * Math.PI * i) / (n - 1)));
  }
  return w;
}

/**
 * In-place radix-2 FFT. Length must be a power of two.
 * re/im hold the signal on entry and the spectrum on exit.
 */
export function fft(re: Float32Array, im: Float32Array): void {
  const n = re.length;
  // Bit-reversal permutation.
  for (let i = 1, j = 0; i < n; i++) {
    let bit = n >> 1;
    for (; j & bit; bit >>= 1) j ^= bit;
    j ^= bit;
    if (i < j) {
      const tr = re[i];
      re[i] = re[j];
      re[j] = tr;
      const ti = im[i];
      im[i] = im[j];
      im[j] = ti;
    }
  }
  for (let len = 2; len <= n; len <<= 1) {
    const ang = (-2 * Math.PI) / len;
    const wr = Math.cos(ang);
    const wi = Math.sin(ang);
    const half = len >> 1;
    for (let i = 0; i < n; i += len) {
      let curR = 1;
      let curI = 0;
      for (let k = 0; k < half; k++) {
        const aR = re[i + k];
        const aI = im[i + k];
        const bR = re[i + k + half] * curR - im[i + k + half] * curI;
        const bI = re[i + k + half] * curI + im[i + k + half] * curR;
        re[i + k] = aR + bR;
        im[i + k] = aI + bI;
        re[i + k + half] = aR - bR;
        im[i + k + half] = aI - bI;
        const nr = curR * wr - curI * wi;
        curI = curR * wi + curI * wr;
        curR = nr;
      }
    }
  }
}

export function ifft(re: Float32Array, im: Float32Array): void {
  for (let i = 0; i < im.length; i++) im[i] = -im[i];
  fft(re, im);
  const n = re.length;
  for (let i = 0; i < n; i++) {
    re[i] /= n;
    im[i] = -im[i] / n;
  }
}

export interface StftConfig {
  fftSize: number;
  hop: number;
  window: Float32Array;
}

export interface StftFrame {
  index: number;
  start: number;
  re: Float32Array;
  im: Float32Array;
}

/** Forward STFT with zero padding at the edges. */
export function stft(input: Float32Array, cfg: StftConfig): StftFrame[] {
  const { fftSize, hop, window } = cfg;
  const frames: StftFrame[] = [];
  const count = Math.ceil(input.length / hop);
  for (let f = 0; f < count; f++) {
    const start = f * hop;
    const re = new Float32Array(fftSize);
    const im = new Float32Array(fftSize);
    for (let i = 0; i < fftSize; i++) {
      const s = start + i;
      if (s >= 0 && s < input.length) re[i] = input[s] * window[i];
    }
    fft(re, im);
    frames.push({ index: f, start, re, im });
  }
  return frames;
}

/**
 * Inverse STFT (overlap-add, Hann-windowed COLA with 50% hop).
 * Returns an array trimmed to outputLength.
 */
export function istft(frames: StftFrame[], cfg: StftConfig, outputLength: number): Float32Array {
  const { fftSize, window } = cfg;
  const out = new Float32Array(outputLength);
  const norm = new Float32Array(outputLength);
  const re = new Float32Array(fftSize);
  const im = new Float32Array(fftSize);
  for (const frame of frames) {
    re.set(frame.re);
    im.set(frame.im);
    ifft(re, im);
    for (let i = 0; i < fftSize; i++) {
      const s = frame.start + i;
      if (s >= 0 && s < outputLength) {
        const w = window[i];
        out[s] += re[i] * w;
        norm[s] += w * w;
      }
    }
  }
  for (let i = 0; i < outputLength; i++) {
    if (norm[i] > 1e-6) out[i] /= norm[i];
  }
  return out;
}

/** Simple one-pole DC blocker (removes offset introduced by optical scanning). */
export class DcBlocker {
  private x1 = 0;
  private y1 = 0;
  constructor(private readonly r = 0.995) {}

  process(x: number): number {
    const y = x - this.x1 + this.r * this.y1;
    this.x1 = x;
    this.y1 = y;
    return y;
  }

  processBuffer(buf: Float32Array): Float32Array {
    for (let i = 0; i < buf.length; i++) buf[i] = this.process(buf[i]);
    return buf;
  }
}

/** Resample by rational ratio using linear interpolation (used for 480->48k upsample). */
export function resampleLinear(input: Float32Array, ratio: number, outputLength?: number): Float32Array {
  const outLen = outputLength ?? Math.floor(input.length * ratio);
  const out = new Float32Array(outLen);
  for (let i = 0; i < outLen; i++) {
    const pos = i / ratio;
    const i0 = Math.floor(pos);
    const frac = pos - i0;
    const a = input[Math.min(i0, input.length - 1)] ?? 0;
    const b = input[Math.min(i0 + 1, input.length - 1)] ?? a;
    out[i] = a + (b - a) * frac;
  }
  return out;
}
