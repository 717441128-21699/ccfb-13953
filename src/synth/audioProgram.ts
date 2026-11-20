import { mulberry32, gaussian } from '../film/util';

/**
 * Clean reference audio program, generated in the optical-scan time domain
 * (one sample per scanner row, rate = scanH * srcFps = 11520 Hz), then
 * upsampled to the output rate. Tones stay well below the optical bandwidth.
 */

export const SCAN_AUDIO_RATE = 480 * 24; // 11520 Hz: one decoded sample per scan row

export interface AudioProgram {
  /** Clean signal at scan-row rate, length = frameCount * 480. */
  scanRate: Float32Array;
  /** Clean signal at output rate, length = frameCount * 2000. */
  outputRate: Float32Array;
  outputRateHz: number;
}

/** Band-limited tone mix with a known amplitude envelope and quiet leader. */
export function generateAudioProgram(frameCount: number, outputRateHz: number, seed: number): AudioProgram {
  const rand = mulberry32(seed ^ 0xa17b);
  const rowsPerFrame = 480;
  const n = frameCount * rowsPerFrame;
  const scan = new Float32Array(n);

  // Frequencies chosen safely inside the optical channel (< 4 kHz).
  const partials = [
    { f: 196 + rand() * 12, a: 0.55 },
    { f: 587 + rand() * 20, a: 0.22 },
    { f: 988 + rand() * 30, a: 0.1 },
  ];

  const leaderRows = rowsPerFrame * 6; // 6 frames of optical silence (noise-only leader)
  for (let t = 0; t < n; t++) {
    const tt = t / SCAN_AUDIO_RATE;
    // Slow musical envelope so the signal itself is never impulse-like.
    const env =
      t < leaderRows
        ? 0
        : 0.75 +
          0.2 * Math.sin(2 * Math.PI * 0.7 * tt) +
          0.08 * Math.sin(2 * Math.PI * 2.3 * tt);
    let s = 0;
    for (const p of partials) {
      s += p.a * Math.sin(2 * Math.PI * p.f * tt + p.f);
    }
    // A gentle amplitude swell mid-reel gives the denoiser varying energy.
    const swell = 1 + 0.25 * Math.sin(2 * Math.PI * 0.23 * tt + 1.7);
    scan[t] = 0.6 * env * swell * s;
  }

  const outSamplesPerFrame = outputRateHz / 24;
  const out = resampleToOutput(scan, outputRateHz, frameCount * outSamplesPerFrame);
  return { scanRate: scan, outputRate: out, outputRateHz };
}

/**
 * Windowed-sinc (Lanczos-3) resample from scan rate to the output rate.
 * Ratio = outputRate / SCAN_AUDIO_RATE (e.g. 48000/11520 = 25/6).
 */
export function resampleToOutput(input: Float32Array, outputRateHz: number, outLen: number): Float32Array {
  const ratio = outputRateHz / SCAN_AUDIO_RATE;
  const out = new Float32Array(outLen);
  const a = 3;
  for (let i = 0; i < outLen; i++) {
    const center = i / ratio;
    const left = Math.max(0, Math.floor(center) - a + 1);
    const right = Math.min(input.length - 1, Math.floor(center) + a);
    let acc = 0;
    for (let j = left; j <= right; j++) {
      const d = center - j;
      acc += input[j] * lanczos(d, a);
    }
    out[i] = acc;
  }
  return out;
}

function lanczos(x: number, a: number): number {
  if (x === 0) return 1;
  if (Math.abs(x) >= a) return 0;
  const px = Math.PI * x;
  return (a * Math.sin(px) * Math.sin(px / a)) / (px * px);
}

/** Faint broadband grain source shared by track wear simulation. */
export function grainNoise(rand: () => number, n: number, sigma: number): Float32Array {
  const out = new Float32Array(n);
  for (let i = 0; i < n; i++) out[i] = gaussian(rand) * sigma;
  return out;
}
