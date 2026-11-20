import { describe, it, expect } from 'vitest';
import { fft, ifft, hannWindow, stft, istft, DcBlocker, resampleLinear } from '../src/audio/dsp';

describe('DSP primitives', () => {
  it('FFT/IFFT round-trips a signal', () => {
    const n = 64;
    const re = new Float32Array(n);
    const im = new Float32Array(n);
    for (let i = 0; i < n; i++) re[i] = Math.sin((2 * Math.PI * 3 * i) / n) + 0.5 * Math.cos((2 * Math.PI * 7 * i) / n);
    const orig = re.slice();
    fft(re, im);
    // Energy concentrated away from DC bin for a tone.
    let bandEnergy = 0;
    for (let k = 1; k < 12; k++) bandEnergy += re[k] ** 2 + im[k] ** 2;
    expect(bandEnergy).toBeGreaterThan(0);
    ifft(re, im);
    for (let i = 0; i < n; i++) expect(re[i]).toBeCloseTo(orig[i], 4);
  });

  it('STFT/ISTFT is a near-perfect reconstruction', () => {
    const n = 2048;
    const sig = new Float32Array(n);
    for (let i = 0; i < n; i++) sig[i] = Math.sin((2 * Math.PI * 440 * i) / 11520);
    const fftSize = 256;
    const cfg = { fftSize, hop: fftSize >> 1, window: hannWindow(fftSize) };
    const frames = stft(sig, cfg);
    const recon = istft(frames, cfg, n);
    let err = 0;
    let pow = 0;
    for (let i = fftSize; i < n - fftSize; i++) {
      err += (recon[i] - sig[i]) ** 2;
      pow += sig[i] ** 2;
    }
    expect(10 * Math.log10(pow / err)).toBeGreaterThan(25);
  });

  it('DC blocker removes a constant offset but keeps AC', () => {
    const blocker = new DcBlocker(0.95);
    const out = new Float32Array(400);
    for (let i = 0; i < 400; i++) out[i] = blocker.process(0.3 + Math.sin((2 * Math.PI * i) / 20));
    const tail = out.slice(200);
    const mean = tail.reduce((a, b) => a + b, 0) / tail.length;
    expect(Math.abs(mean)).toBeLessThan(0.02);
    const ac = tail.some((v) => Math.abs(v) > 0.5);
    expect(ac).toBe(true);
  });

  it('linear resampler hits the requested length', () => {
    const input = new Float32Array(100).map((_, i) => Math.sin(i * 0.2));
    const out = resampleLinear(input, 2.5, 250);
    expect(out.length).toBe(250);
    expect(out[0]).toBeCloseTo(input[0], 4);
  });
});
