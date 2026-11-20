import { describe, it, expect } from 'vitest';
import { synthesizeReel } from '../src/film/reel';
import { detectShots } from '../src/picture/shotDetect';
import { computeFlickerGains, removeFlicker } from '../src/picture/flicker';
import { computeColorGains, restoreColor } from '../src/picture/colorRestore';
import { repairBlemishes } from '../src/picture/blemishRepair';
import { extractPictures, runCpuPipeline } from '../src/film/pipeline';
import { psnrByte } from '../src/film/util';
import { PlanarImage } from '../src/film/image';

const FRAMES = 36;

describe('picture restoration', () => {
  const reel = synthesizeReel(FRAMES, 'variable-area', 23);
  const pictures = extractPictures(reel);
  const cleanRefs = reel.picture.manifest.cleanFrames;

  it('detects the three synthesized shots', () => {
    const shots = detectShots(pictures);
    expect(shots.length).toBe(3);
    expect(shots[0].start).toBe(0);
    // Ground-truth boundaries.
    const truth = reel.picture.manifest.shots;
    shots.forEach((s, i) => {
      expect(Math.abs(s.start - truth[i].start)).toBeLessThanOrEqual(1);
    });
  });

  it('flattens flicker but preserves the sustained light-off event', () => {
    const shots = detectShots(pictures);
    const { gain, observedLevel, sustainedLevel } = computeFlickerGains(pictures, shots);
    expect(gain.length).toBe(FRAMES);

    // Inside normal frames, gains actively compensate jumps.
    const normalFrame = 2;
    expect(Math.abs(gain[normalFrame] - 1)).toBeGreaterThan(0.02);

    // Light-off frames: observed level drops hard. The shot reference stays at
    // the normal brightness, but the detector recognizes the dimmed run and
    // applies gain 1, so the real lighting event is never brightened back.
    const shot = reel.picture.manifest.shots[1];
    const off = shot.lightOffStart + 1;
    expect(observedLevel[off]).toBeLessThan(observedLevel[shot.start] * 0.4);
    expect(sustainedLevel[off]).toBeGreaterThan(observedLevel[off] * 1.8);
    expect(gain[off]).toBe(1);

    const result = removeFlicker(pictures, shots);

    // The light-off frame stays dark after processing (not flattened away).
    const meanOff = meanLuma(result.frames[off]);
    const meanOn = meanLuma(result.frames[shot.start]);
    expect(meanOff).toBeLessThan(meanOn * 0.5);
  });

  it('restores the pink color cast per shot (gray world)', () => {
    const shots = detectShots(pictures);
    const gains = computeColorGains(pictures, shots);
    for (const g of gains) {
      // Pink fade: red boosted, blue suppressed -> correction must go the other way.
      expect(g[0]).toBeLessThan(1);
      expect(g[2]).toBeGreaterThan(1);
    }
    const restored = restoreColor(pictures, shots);
    // After correction, per-channel means are closer together.
    const before = channelSpread(pictures[4]);
    const after = channelSpread(restored.frames[4]);
    expect(after).toBeLessThan(before);
  });

  it('repairs scratches and dust using neighbor frames', () => {
    expect(reel.picture.manifest.scratches.length).toBeGreaterThan(5);
    expect(reel.picture.manifest.dust.length).toBeGreaterThan(20);
    const { frames, stats } = repairBlemishes(pictures);
    expect(stats.temporalRepairs).toBeGreaterThan(100);
    // Repaired frames are closer to the clean reference than the degraded inputs.
    const degradedLuma = lumaPlane(pictures[10]);
    const repairedLuma = lumaPlane(frames[10]);
    const cleanLuma = lumaPlane(cleanRefs[10]);
    expect(psnrByte(cleanLuma, repairedLuma)).toBeGreaterThan(psnrByte(cleanLuma, degradedLuma));
  });

  it('full CPU pipeline improves PSNR against clean ground truth on normal frames', () => {
    const result = runCpuPipeline(reel, 'variable-area');

    // The deliberate light-off event is real cinematography and must be kept,
    // so restoration quality is scored on normal frames only; the light event
    // has its own preservation assertion above.
    const lightOff = new Set<number>();
    for (const shot of reel.picture.manifest.shots) {
      for (let f = shot.lightOffStart; f < shot.lightOffEnd; f++) if (f >= 0) lightOff.add(f);
    }
    const normal = pictures.map((_, i) => i).filter((f) => !lightOff.has(f));

    const psnrBefore = psnrByte(lumaSubset(cleanRefs, normal), lumaSubset(pictures, normal));
    const psnrAfter = psnrByte(lumaSubset(cleanRefs, normal), lumaSubset(result.restored, normal));
    expect(psnrAfter).toBeGreaterThan(psnrBefore + 3);

    // Restored never makes the overall reel worse than the degraded input.
    const allBefore = psnrByte(lumaAll(cleanRefs), lumaAll(pictures));
    const allAfter = psnrByte(lumaAll(cleanRefs), lumaAll(result.restored));
    expect(allAfter).toBeGreaterThanOrEqual(allBefore);
  });
});

function lumaSubset(frames: PlanarImage[], idxs: number[]): Uint8ClampedArray {
  const n = frames[0].width * frames[0].height;
  const out = new Uint8ClampedArray(n * idxs.length);
  idxs.forEach((f, k) => out.set(lumaPlane(frames[f]), k * n));
  return out;
}

function meanLuma(img: PlanarImage): number {
  let s = 0;
  for (let p = 0; p < img.width * img.height; p++) {
    s += 0.299 * img.r[p] + 0.587 * img.g[p] + 0.114 * img.b[p];
  }
  return s / (img.width * img.height);
}

function channelSpread(img: PlanarImage): number {
  const means = [mean(img.r), mean(img.g), mean(img.b)];
  return Math.max(...means) - Math.min(...means);
}

function mean(plane: Uint8ClampedArray): number {
  let s = 0;
  for (const v of plane) s += v;
  return s / plane.length;
}

function lumaPlane(img: PlanarImage): Uint8ClampedArray {
  const out = new Uint8ClampedArray(img.width * img.height);
  for (let p = 0; p < out.length; p++) {
    out[p] = (0.299 * img.r[p] + 0.587 * img.g[p] + 0.114 * img.b[p]) | 0;
  }
  return out;
}

function lumaAll(frames: PlanarImage[]): Uint8ClampedArray {
  const n = frames[0].width * frames[0].height;
  const out = new Uint8ClampedArray(n * frames.length);
  frames.forEach((f, i) => out.set(lumaPlane(f), i * n));
  return out;
}
