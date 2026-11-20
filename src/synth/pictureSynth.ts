import { SPECS } from '../film/specs';
import { PlanarImage } from '../film/image';
import { mulberry32, gaussian, clampByte, grain } from '../film/util';

/**
 * Synthetic picture ground truth and defect injection.
 *
 * The reel is divided into three shots. Each shot gets:
 *  - a moving geometric scene (so motion-compensated interpolation has real motion)
 *  - per-frame flicker gains (spurious brightness jumps) that must be flattened
 *  - a pink-fade color cast that must be corrected per shot
 *  - vertical scratches and dust specks that must be repaired from neighbors
 *
 * One shot contains a deliberate, sustained light-off/on event: that is real
 * cinematography, not flicker, and must survive the flattening stage.
 */

export interface ShotInfo {
  start: number;
  end: number; // exclusive
  /** Ground-truth clean color multipliers used to synthesize the pink fade (r,g,b). */
  fadeGain: [number, number, number];
  /** Frames inside the shot where the scene light is genuinely switched off/on. */
  lightOffStart: number;
  lightOffEnd: number; // exclusive
}

export interface ScratchDefect {
  frame: number;
  x: number;
  y0: number;
  y1: number;
  width: number;
  polarity: number; // +1 bright (cinch), -1 dark (scratch)
}

export interface DustDefect {
  frame: number;
  x: number;
  y: number;
  radius: number;
  polarity: number;
}

export interface PictureManifest {
  shots: ShotInfo[];
  /** Per-frame brightness multipliers injected as flicker (clean -> faded base). */
  flickerGain: Float32Array;
  scratches: ScratchDefect[];
  dust: DustDefect[];
  /** Clean picture frames before any defect was applied. */
  cleanFrames: PlanarImage[];
}

export interface BuiltPicture {
  manifest: PictureManifest;
  degradedFrames: PlanarImage[];
}

const PW = SPECS.picX1 - SPECS.picX0;
const PH = SPECS.picY1 - SPECS.picY0;

export class PictureSynthesizer {
  private readonly rand: () => number;

  constructor(
    private readonly frameCount: number,
    seed: number,
  ) {
    this.rand = mulberry32(seed ^ 0x9c11);
  }

  private shots(): ShotInfo[] {
    const n = this.frameCount;
    const b1 = Math.floor(n / 3);
    const b2 = Math.floor((2 * n) / 3);
    return [
      { start: 0, end: b1, fadeGain: [1.18, 1.0, 0.82], lightOffStart: -1, lightOffEnd: -1 },
      // Sustained real light-off event in the middle shot, 6 frames long.
      {
        start: b1,
        end: b2,
        fadeGain: [1.28, 1.02, 0.74],
        lightOffStart: b1 + 4,
        lightOffEnd: b1 + 10,
      },
      { start: b2, end: n, fadeGain: [1.12, 0.97, 0.88], lightOffStart: -1, lightOffEnd: -1 },
    ];
  }

  /**
   * Deterministic scene. Each shot has a distinct tonal (luma) layout so shot
   * boundaries are real cuts, while every scene is kept roughly gray-world
   * neutral across channels: that is the premise the blind color-restoration
   * stage relies on to invert the pink fade without distorting real colors.
   * Translating bars and a moving disc provide genuine motion inside a shot.
   */
  private cleanFrame(frame: number, shotIndex: number): PlanarImage {
    const img = new PlanarImage(PW, PH);
    const shift = (frame * 7) % PW;
    // Neutral grayscale backdrops; shots differ in tonal distribution/coverage.
    const base = [70, 110, 140][shotIndex % 3];
    const span = [120, 70, 50][shotIndex % 3];

    for (let y = 0; y < PH; y++) {
      for (let x = 0; x < PW; x++) {
        const p = y * PW + x;
        const gx = x / PW;
        const gy = y / PH;
        // Static spatial texture only: no temporal phase terms, so the clean
        // frame's global mean is constant and flicker is the sole mean driver.
        const luma = base + span * (gx - 0.5) + 40 * (gy - 0.5)
          + 18 * Math.sin(0.03 * x)
          + 14 * Math.cos(0.025 * y);
        // Neutral: equal RGB, so per-channel means agree (gray-world holds).
        let r = luma;
        let g = luma;
        let b = luma;

        // Neutral landmarks with distinct coverage per shot (drives normalized
        // luma-histogram differences at cuts) but no chromatic bias.
        let inLandmark = false;
        if (shotIndex === 0) {
          inLandmark = x >= 40 && x < 110 && y >= 40 && y < 110;
        } else if (shotIndex === 1) {
          inLandmark = x >= PW - 90 && y >= 30 && y < PH - 30;
        } else {
          inLandmark = y >= PH * 0.55;
        }
        if (inLandmark) {
          const dl = shotIndex === 1 ? -70 : 80;
          r += dl;
          g += dl;
          b += dl;
        }

        // Sum-conserving wrapping bars: identical coverage every frame, so their
        // contribution to the global mean is motion-independent. The third shot
        // is held static so its within-shot histogram distance is ~0, keeping a
        // clean separation from the real cut that precedes it.
        if (shotIndex !== 2) {
          const bar1 = (x - shift + PW) % PW;
          if (bar1 < 34) {
            r += 95;
            g += 90;
            b += 85;
          }
          const bar2 = (x - (shift * 2) % PW + PW) % PW;
          if (bar2 < 16) {
            r -= 60;
            g -= 55;
            b -= 50;
          }
        }

        // A horizontally translating disc kept fully inside the frame: its
        // pixel count (and thus total luma) is constant, only its position moves.
        // Static again in the third shot.
        const discX = 70 + (shotIndex === 2 ? 0 : (frame * 9) % (PW - 140));
        const discY = PH * 0.5;
        const dx = x - discX;
        const dy = y - discY;
        if (dx * dx + dy * dy < 38 * 38) {
          r = 200;
          g = 200;
          b = 200;
        }

        img.r[p] = clampByte(r);
        img.g[p] = clampByte(g);
        img.b[p] = clampByte(b);
      }
    }
    return img;
  }

  build(): BuiltPicture {
    const shots = this.shots();
    const flickerGain = new Float32Array(this.frameCount);
    const cleanFrames: PlanarImage[] = [];
    const degradedFrames: PlanarImage[] = [];
    const scratches: ScratchDefect[] = [];
    const dust: DustDefect[] = [];

    const shotOf = (f: number) => shots.find((s) => f >= s.start && f < s.end) ?? shots[shots.length - 1];

    for (let f = 0; f < this.frameCount; f++) {
      const shot = shotOf(f);
      const shotIndex = shots.indexOf(shot);
      // Flicker: fast per-frame jumps, distinct from the sustained light event.
      let gain = 1 + 0.16 * Math.sin(f * 1.7 + shot.start) + 0.07 * gaussian(this.rand);
      const lightOff = f >= shot.lightOffStart && f < shot.lightOffEnd;
      if (lightOff) gain = 0.18; // genuine, sustained exposure change -> keep it
      flickerGain[f] = gain;

      const clean = this.cleanFrame(f, shotIndex);
      cleanFrames.push(clean);
      const degraded = clean.clone();

      // Pink fade: boost red, suppress blue (applied before flicker).
      for (let p = 0; p < PW * PH; p++) {
        degraded.r[p] = clampByte(clean.r[p] * shot.fadeGain[0]);
        degraded.g[p] = clampByte(clean.g[p] * shot.fadeGain[1]);
        degraded.b[p] = clampByte(clean.b[p] * shot.fadeGain[2]);
      }

      // Flicker gain on luminance (all channels scaled together).
      const frameSalt = f * 7919;
      for (let p = 0; p < PW * PH; p++) {
        const gr = grain(frameSalt + p * 3 + 1) * 2.5;
        const gg = grain(frameSalt + p * 3 + 2) * 2.5;
        const gb = grain(frameSalt + p * 3 + 3) * 2.5;
        degraded.r[p] = clampByte(degraded.r[p] * gain + gr);
        degraded.g[p] = clampByte(degraded.g[p] * gain + gg);
        degraded.b[p] = clampByte(degraded.b[p] * gain + gb);
      }

      // Vertical scratches: appear on ~1 frame in 4, usually lasting several frames.
      if (this.rand() < 0.25) {
        const sx = Math.floor(this.rand() * PW);
        const span = 1 + Math.floor(this.rand() * 3);
        for (let k = 0; k < span && f + k < this.frameCount; k++) {
          scratches.push({
            frame: f + k,
            x: sx,
            y0: Math.floor(this.rand() * PH * 0.2),
            y1: PH - Math.floor(this.rand() * PH * 0.2),
            width: 1 + Math.floor(this.rand() * 2),
            polarity: this.rand() > 0.35 ? -1 : 1,
          });
        }
      }

      // Dust specks: a handful per frame.
      const dustCount = 2 + Math.floor(this.rand() * 4);
      for (let d = 0; d < dustCount; d++) {
        dust.push({
          frame: f,
          x: Math.floor(this.rand() * PW),
          y: Math.floor(this.rand() * PH),
          radius: 1 + Math.floor(this.rand() * 3),
          polarity: this.rand() > 0.4 ? -1 : 1,
        });
      }

      this.applyBlemishes(degraded, f, scratches, dust);
      degradedFrames.push(degraded);
    }

    const manifest: PictureManifest = {
      shots,
      flickerGain,
      scratches,
      dust,
      cleanFrames,
    };
    return { manifest, degradedFrames };
  }

  private applyBlemishes(img: PlanarImage, frame: number, scratches: ScratchDefect[], dust: DustDefect[]): void {
    for (const s of scratches) {
      if (s.frame !== frame) continue;
      for (let y = s.y0; y < s.y1 && y < PH; y++) {
        for (let dx = 0; dx < s.width; dx++) {
          const x = s.x + dx;
          if (x < 0 || x >= PW) continue;
          const p = y * PW + x;
          const delta = s.polarity * 170;
          img.r[p] = clampByte(img.r[p] + delta);
          img.g[p] = clampByte(img.g[p] + delta);
          img.b[p] = clampByte(img.b[p] + delta);
        }
      }
    }
    for (const d of dust) {
      if (d.frame !== frame) continue;
      for (let y = d.y - d.radius; y <= d.y + d.radius; y++) {
        for (let x = d.x - d.radius; x <= d.x + d.radius; x++) {
          if (x < 0 || x >= PW || y < 0 || y >= PH) continue;
          if ((x - d.x) ** 2 + (y - d.y) ** 2 > d.radius * d.radius) continue;
          const p = y * PW + x;
          const delta = d.polarity * 150;
          img.r[p] = clampByte(img.r[p] + delta);
          img.g[p] = clampByte(img.g[p] + delta);
          img.b[p] = clampByte(img.b[p] + delta);
        }
      }
    }
  }
}

