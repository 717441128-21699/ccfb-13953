import { describe, it, expect } from 'vitest';
import { synthesizeReel } from '../src/film/reel';
import { SPECS } from '../src/film/specs';
import { readTrack, estimateLevels } from '../src/audio/trackReader';
import { declick } from '../src/audio/declicker';
import { decodeOpticalSound, extractPictureWindow } from '../src/audio/decoder';
import { resampleToOutput } from '../src/synth/audioProgram';
import { snrDb, rms } from '../src/film/util';

const FRAMES = 36;
const OFFSET = SPECS.soundOffset;

describe.each([
  ['variable-area', '变积式'],
  ['variable-density', '变密式'],
] as const)('optical soundtrack decode: %s (%s)', (trackType, _label) => {
  const reel = synthesizeReel(FRAMES, trackType, 11);

  it('produces one decoded sample per scanner row', () => {
    const levels = estimateLevels(reel.reel, trackType);
    const raw = readTrack(reel.reel, trackType, levels);
    expect(raw.length).toBe(FRAMES * SPECS.scanH);
    // Levels must be sane 8-bit values with white above black.
    expect(levels.white).toBeGreaterThan(levels.black + 40);
  });

  it('detects and removes injected clicks', () => {
    const levels = estimateLevels(reel.reel, trackType);
    const raw = readTrack(reel.reel, trackType, levels);
    const { output } = declick(raw);
    expect(output.length).toBe(raw.length);
    // The manifest recorded dozens of click events across the reel.
    const events = reel.trackManifest.clicks;
    expect(events.length).toBeGreaterThan(20);

    // Event-level recall: a click is caught if any of its rows was corrected.
    const changed = new Set<number>();
    for (let i = 0; i < raw.length; i++) if (output[i] !== raw[i]) changed.add(i);
    let caught = 0;
    for (const c of events) {
      for (let k = 0; k < c.length; k++) {
        if (changed.has(c.sample + k) || changed.has(c.sample + k - 1) || changed.has(c.sample + k + 1)) {
          caught++;
          break;
        }
      }
    }
    expect(caught / events.length).toBeGreaterThan(0.7);

    // Corrections stay sparse (a click is rare; well under 2% of all rows).
    expect(changed.size / raw.length).toBeLessThan(0.02);

    // Post-declick max residual impulse is smaller than pre-declick.
    const before = maxJump(raw);
    const after = maxJump(output);
    expect(after).toBeLessThan(before);
  });

  it('suppresses edge-wear hiss while keeping the program', () => {
    const decoded = decodeOpticalSound(reel.reel, trackType, reel.trackManifest.leaderRows);

    // Noise floor on the (silent) leader must be clearly below program level.
    const leader = decoded.scanRate.slice(
      reel.trackManifest.leaderRows.start,
      reel.trackManifest.leaderRows.end,
    );
    expect(rms(leader)).toBeLessThan(0.05);

    // Full SNR: the decoded timeline starts at physical lead, so it compares
    // against the SAME leading segment of the clean program (no offset window).
    const cleanSegment = reel.scanProgramExtended.slice(0, FRAMES * SPECS.scanH);
    const cleanOut = resampleToOutput(cleanSegment, SPECS.audioRate, decoded.outputRate.length);
    const snr = snrDb(cleanOut, decoded.outputRate);
    expect(snr).toBeGreaterThanOrEqual(40);
  });

  it('keeps timing and length exactly aligned to the 26-frame offset', () => {
    const decoded = decodeOpticalSound(reel.reel, trackType, reel.trackManifest.leaderRows);
    expect(decoded.outputRate.length).toBe(FRAMES * (SPECS.audioRate / SPECS.srcFps));
    // The windowed scan-rate signal must start exactly at offset*scanH.
    const windowed = extractPictureWindow(decoded.scanRate, FRAMES);
    expect(windowed.length).toBe(FRAMES * SPECS.scanH);
    const cleanWindow = reel.scanProgramExtended.slice(OFFSET * SPECS.scanH, (OFFSET + FRAMES) * SPECS.scanH);
    // Correlation at zero lag beats a one-row-shifted lag (alignment check).
    expect(zeroLagCorrelation(windowed, cleanWindow)).toBeGreaterThan(
      zeroLagCorrelation(windowed.slice(1), cleanWindow.slice(0, cleanWindow.length - 1)),
    );
  });  it('STFT denoiser preserves tone frequencies (no musical content destroyed)', () => {
    const decoded = decodeOpticalSound(reel.reel, trackType, reel.trackManifest.leaderRows);
    const cleanWindow = reel.scanProgramExtended.slice(0, FRAMES * SPECS.scanH);
    const cleanOut = resampleToOutput(cleanWindow, SPECS.audioRate, decoded.outputRate.length);
    const beforePower = bandPower(cleanOut, 150, 1200, 48000);
    const afterPower = bandPower(decoded.outputRate, 150, 1200, 48000);
    // Retain at least 60% of the in-band program energy.
    expect(afterPower / beforePower).toBeGreaterThan(0.6);
  });
});

function maxJump(xs: Float32Array): number {
  let m = 0;
  for (let i = 1; i < xs.length; i++) m = Math.max(m, Math.abs(xs[i] - xs[i - 1]));
  return m;
}

function zeroLagCorrelation(a: Float32Array, b: Float32Array): number {
  const n = Math.min(a.length, b.length);
  let s = 0;
  for (let i = 0; i < n; i++) s += a[i] * b[i];
  return s;
}

function bandPower(xs: Float32Array, f0: number, f1: number, rate: number): number {
  const n = xs.length;
  let power = 0;
  const steps = 48;
  for (let k = 0; k < steps; k++) {
    const f = f0 + ((f1 - f0) * k) / steps;
    let accR = 0;
    let accI = 0;
    const w = (2 * Math.PI * f) / rate;
    for (let t = 0; t < n; t++) {
      accR += xs[t] * Math.cos(w * t);
      accI -= xs[t] * Math.sin(w * t);
    }
    power += accR * accR + accI * accI;
  }
  return power / n;
}
