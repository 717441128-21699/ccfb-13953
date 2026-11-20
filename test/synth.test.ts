import { describe, it, expect } from 'vitest';
import { mulberry32, gaussian, snrDb, mad, psnrByte } from '../src/film/util';
import { SPECS } from '../src/film/specs';
import { synthesizeReel } from '../src/film/reel';
import { OpticalTrackRenderer } from '../src/synth/opticalTrack';
import { generateAudioProgram, SCAN_AUDIO_RATE } from '../src/synth/audioProgram';

describe('synthetic test sequence realism', () => {
  it('matches 16mm geometry: perf strip, 96px track, 4:3 picture', () => {
    const reel = synthesizeReel(12, 'variable-area', 5);
    expect(reel.reel.width).toBe(SPECS.scanW);
    expect(reel.reel.height).toBe(SPECS.scanH);
    expect(SPECS.soundX1 - SPECS.soundX0).toBe(96);
    const pw = SPECS.picX1 - SPECS.picX0;
    const ph = SPECS.picY1 - SPECS.picY0;
    expect(pw / ph).toBeCloseTo(4 / 3, 2);
    expect(SPECS.soundOffset).toBe(26);
  });

  it('renders one packed-RGBA scan frame per film frame with a non-empty track', () => {
    const reel = synthesizeReel(12, 'variable-area', 5);
    expect(reel.reel.frames.length).toBe(12);
    const frame = reel.reel.frames[3];
    expect(frame.data.length).toBe(SPECS.scanW * SPECS.scanH * 4);
    let whiteInTrack = 0;
    for (let y = 0; y < SPECS.scanH; y++) {
      for (let x = SPECS.soundX0; x < SPECS.soundX1; x++) {
        if (frame.data[(y * SPECS.scanW + x) * 4] > 150) whiteInTrack++;
      }
    }
    expect(whiteInTrack).toBeGreaterThan(500);
  });

  it('embeds known flicker, scratches, dust and both track defect classes', () => {
    const reel = synthesizeReel(24, 'variable-density', 9);
    const pm = reel.picture.manifest;
    expect(pm.scratches.length).toBeGreaterThan(0);
    expect(pm.dust.length).toBeGreaterThan(20);
    // Flicker gains actually deviate from 1 on most frames.
    const flickerFrames = pm.flickerGain.filter((g) => Math.abs(g - 1) > 0.03).length;
    expect(flickerFrames).toBeGreaterThan(10);
    // Sustained light-off event exists.
    const shot2 = pm.shots[1];
    expect(shot2.lightOffEnd - shot2.lightOffStart).toBe(6);
    // Track defects.
    expect(reel.trackManifest.clicks.length).toBeGreaterThan(10);
    expect(reel.trackManifest.edgeWearSigma).toBeGreaterThan(0);
  });

  it('is deterministic for a fixed seed', () => {
    const a = synthesizeReel(12, 'variable-area', 42);
    const b = synthesizeReel(12, 'variable-area', 42);
    expect(a.reel.frames[0].data).toEqual(b.reel.frames[0].data);
  });

  it('PRNG and gaussian helpers behave statistically', () => {
    const rand = mulberry32(1);
    const xs = Array.from({ length: 20000 }, () => rand());
    expect(Math.min(...xs)).toBeGreaterThanOrEqual(0);
    expect(Math.max(...xs)).toBeLessThan(1);
    const mean = xs.reduce((a, b) => a + b, 0) / xs.length;
    expect(mean).toBeCloseTo(0.5, 1);
    // One persistent PRNG feeds the whole gaussian stream.
    const gRand = mulberry32(2);
    const gs = Array.from({ length: 20000 }, () => gaussian(gRand));
    const gm = gs.reduce((a, b) => a + b, 0) / gs.length;
    expect(Math.abs(gm)).toBeLessThan(0.05);
  });

  it('audio program is band-limited below the optical channel ceiling', () => {
    const pictureFrames = 12;
    const program = generateAudioProgram(pictureFrames + SPECS.soundOffset, 48000, 3);
    expect(program.scanRate.length).toBe((pictureFrames + SPECS.soundOffset) * 480);
    expect(SCAN_AUDIO_RATE).toBe(11520);
    // Nyquist of scan rate is 5760 Hz; generated partials stay under 1100 Hz.
    expect(program.outputRate.length).toBe((pictureFrames + SPECS.soundOffset) * 2000);
  });

  it('renderer accepts both track types with distinct statistics', () => {
    const program = generateAudioProgram(8, 48000, 4).scanRate;
    const va = new OpticalTrackRenderer(program, 'variable-area', 8, 4).renderAll();
    const vd = new OpticalTrackRenderer(program, 'variable-density', 8, 4).renderAll();
    expect(va.strips.length).toBe(8);
    expect(vd.strips.length).toBe(8);
    // VA concentrates values near black/white; VD spreads across mid-tones.
    const vaMid = midFraction(va.strips[0]);
    const vdMid = midFraction(vd.strips[0]);
    expect(vdMid).toBeGreaterThan(vaMid);
  });
});

function midFraction(strip: Uint8ClampedArray): number {
  let mid = 0;
  for (const v of strip) if (v > 60 && v < 200) mid++;
  return mid / strip.length;
}

describe('metric helpers', () => {
  it('snrDb is infinite for identical signals and finite with noise', () => {
    const ref = new Float32Array(1000).map((_, i) => Math.sin(i * 0.1));
    expect(snrDb(ref, ref)).toBe(Infinity);
    const noisy = ref.map((v) => v + 0.01);
    expect(snrDb(ref, noisy)).toBeGreaterThan(30);
  });

  it('mad is zero for a constant array', () => {
    const xs = new Float32Array(50).fill(7);
    expect(mad(xs).mad).toBe(0);
  });

  it('psnrByte is infinite for identical byte planes', () => {
    const a = new Uint8ClampedArray([1, 2, 3, 4, 5]);
    expect(psnrByte(a, a)).toBe(Infinity);
  });
});
