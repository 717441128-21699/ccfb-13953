import { hannWindow, stft, istft, StftConfig, StftFrame } from './dsp';

/**
 * Hiss / quantization-noise suppression for the worn optical track.
 *
 * Two physically motivated stages:
 *  1. Wiener spectral gate using the noise fingerprint measured on the quiet
 *     leader. The noise PSD is smoothed across frequency (film grain /
 *     quantization noise is spectrally smooth), which stabilizes the estimate
 *     from a short leader. Over-subtraction is gentle so program bins survive.
 *  2. Optical-bandwidth low-pass. An optical soundtrack cannot carry high
 *     frequencies; the synthetic program sits entirely below ~1.1 kHz, so bins
 *     above the optical cutoff contain only noise and are removed with zero
 *     damage to the program. This is what a real optical reproducer does.
 */

export interface DenoiseOptions {
  fftSize?: number;
  /** Over-subtraction factor for the Wiener stage. */
  alpha?: number;
  /** Gain floor (dB) for the Wiener stage. */
  floorDb?: number;
  /** Optical channel bandwidth in Hz at the scan-row sample rate. */
  bandwidthHz?: number;
  sampleRate?: number;
}

export function denoise(
  input: Float32Array,
  leader: { start: number; end: number },
  opts: DenoiseOptions = {},
): Float32Array {
  const fftSize = opts.fftSize ?? 1024;
  const hop = fftSize >> 1;
  const cfg: StftConfig = { fftSize, hop, window: hannWindow(fftSize) };
  const frames = stft(input, cfg);

  const rawNoise = estimateNoisePsd(frames, cfg, leader);
  const noisePsd = smoothPsd(rawNoise, Math.max(1, fftSize / 32));
  const alpha = opts.alpha ?? 0.5;
  const floorLin = 10 ** ((opts.floorDb ?? -20) / 20);
  const sampleRate = opts.sampleRate ?? 11520;
  const bandwidthHz = opts.bandwidthHz ?? 5600;
  const binHz = sampleRate / fftSize;
  const cutoffBin = bandwidthHz / binHz;

  for (const frame of frames) {
    const { re, im } = frame;
    for (let k = 0; k < fftSize; k++) {
      const mirror = fftSize - k;
      // Stage 2: hard optical-bandwidth limit (lossless for the band-limited program).
      if (k > cutoffBin && mirror > cutoffBin) {
        re[k] = 0;
        im[k] = 0;
        continue;
      }
      // Stage 1: gentle Wiener gate inside the passband.
      const power = re[k] * re[k] + im[k] * im[k];
      const nse = Math.max(1e-12, alpha * noisePsd[k]);
      let gain: number;
      if (power <= nse) {
        gain = floorLin;
      } else {
        gain = (power - nse) / power;
        gain = Math.max(floorLin, Math.min(1, gain));
      }
      re[k] *= gain;
      im[k] *= gain;
    }
  }

  return istft(frames, cfg, input.length);
}

/** Moving-average smoothing across frequency bins. */
function smoothPsd(psd: Float32Array, radius: number): Float32Array {
  const out = new Float32Array(psd.length);
  const r = Math.max(1, radius | 0);
  for (let k = 0; k < psd.length; k++) {
    let acc = 0;
    let cnt = 0;
    for (let j = Math.max(0, k - r); j <= Math.min(psd.length - 1, k + r); j++) {
      acc += psd[j];
      cnt++;
    }
    out[k] = acc / cnt;
  }
  return out;
}

function estimateNoisePsd(frames: StftFrame[], cfg: StftConfig, leader: { start: number; end: number }): Float32Array {
  const psd = new Float32Array(cfg.fftSize);
  let count = 0;
  for (const frame of frames) {
    const center = frame.start + cfg.fftSize / 2;
    if (center < leader.start || center >= leader.end) continue;
    for (let k = 0; k < cfg.fftSize; k++) {
      psd[k] += frame.re[k] * frame.re[k] + frame.im[k] * frame.im[k];
    }
    count++;
  }
  if (count === 0) {
    let best = Infinity;
    let bestFrame = frames[0];
    for (const f of frames) {
      let e = 0;
      for (let k = 0; k < cfg.fftSize; k++) e += f.re[k] ** 2 + f.im[k] ** 2;
      if (e < best) {
        best = e;
        bestFrame = f;
      }
    }
    for (let k = 0; k < cfg.fftSize; k++) psd[k] = bestFrame.re[k] ** 2 + bestFrame.im[k] ** 2;
  } else {
    for (let k = 0; k < psd.length; k++) psd[k] /= count;
  }
  return psd;
}
