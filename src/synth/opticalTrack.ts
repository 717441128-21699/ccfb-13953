import { SPECS, SoundTrackType } from '../film/specs';
import { mulberry32, gaussian, clampByte, clamp, grain } from '../film/util';

/**
 * Renders the optical soundtrack strip for every scanned frame.
 *
 * Two real-world track families are supported:
 *  - variable-area:   a white tongue whose width follows the waveform
 *  - variable-density: a uniform-width strip whose photographic density follows it
 *
 * Known, seeded defects are injected and recorded in the manifest so tests can
 * verify blind restoration against ground truth:
 *  - clicks  : dust specks / short scratches on the track (impulsive bursts)
 *  - wear    : frayed track edges (colored jitter / haze -> broadband hiss)
 *  - grain   : fine emulsion grain (low-level white noise)
 */

export interface ClickDefect {
  /** Absolute scan-row timeline sample index. */
  sample: number;
  /** Burst length in rows. */
  length: number;
  /** +1 bright defect, -1 dark defect. */
  polarity: number;
}

export interface TrackManifest {
  type: SoundTrackType;
  clicks: ClickDefect[];
  /** Timeline rows used as the "quiet leader" noise reference. */
  leaderRows: { start: number; end: number };
  edgeWearSigma: number;
  grainSigma: number;
}

export interface RenderedTrack {
  /** One strip per frame, row-major Uint8 values of length trackW * scanH. */
  strips: Uint8ClampedArray[];
  manifest: TrackManifest;
}

const TRACK_W = SPECS.soundX1 - SPECS.soundX0;
const H = SPECS.scanH;
const ROWS_PER_FRAME = H;
const BLACK = 18;
const WHITE = 235;
/**
 * Area modulation depth. The exposed tongue must stay strictly inside the
 * track slit at full scale: baseHalf*(1+m) < TRACK_W/2, otherwise the edge
 * clips and introduces irrecoverable nonlinear distortion.
 */
export const AREA_MODULATION = 0.55;

export class OpticalTrackRenderer {
  private readonly rand: () => number;
  private readonly totalRows: number;
  private readonly edgeNoiseL: Float32Array;
  private readonly edgeNoiseR: Float32Array;
  private readonly hazeNoise: Float32Array;
  readonly manifest: TrackManifest;
  private readonly clickMap: Int8Array;

  constructor(
    private readonly clean: Float32Array,
    private readonly type: SoundTrackType,
    frameCount: number,
    seed: number,
  ) {
    this.rand = mulberry32(seed ^ 0x504e);
    this.totalRows = frameCount * ROWS_PER_FRAME;

    // Colored edge jitter / haze sit inside the signal's own low-frequency band
    // and cannot be removed by spectral filtering without touching the program,
    // so a well-preserved readable track keeps them very small. White emulsion
    // grain (below) is independent per pixel: it is what the hiss suppressor
    // removes, and it doubles as dither that de-correlates 8-bit quantization.
    const wearSigma = type === 'variable-area' ? 0.02 : 0.1;
    this.edgeNoiseL = this.coloredNoise(this.totalRows, wearSigma, 0.82);
    this.edgeNoiseR = this.coloredNoise(this.totalRows, wearSigma, 0.82);
    this.hazeNoise = this.coloredNoise(this.totalRows, type === 'variable-density' ? 0.12 : 0.03, 0.9);

    // One click every ~900 rows on average, never inside the leader.
    const clicks: ClickDefect[] = [];
    this.clickMap = new Int8Array(this.totalRows);
    const leaderEnd = ROWS_PER_FRAME * SPECS.leaderFrames;
    for (let t = leaderEnd + 40; t < this.totalRows - 20; ) {
      t += 350 + Math.floor(this.rand() * 250);
      if (t >= this.totalRows - 10) break;
      // Most dust strikes are single scanner lines (rebuilt almost losslessly
      // on a band-limited signal); longer scratches are the minority.
      const roll = this.rand();
      const len = roll < 0.8 ? 1 : roll < 0.94 ? 2 : roll < 0.99 ? 3 : 4;
      const pol = this.rand() > 0.5 ? 1 : -1;
      clicks.push({ sample: t, length: len, polarity: pol });
      for (let k = 0; k < len && t + k < this.totalRows; k++) this.clickMap[t + k] = pol;
    }

    this.manifest = {
      type,
      clicks,
      leaderRows: { start: 0, end: leaderEnd },
      edgeWearSigma: wearSigma,
      grainSigma: type === 'variable-area' ? 0.6 : 0.8,
    };
  }

  /** AR(1) colored noise: the frayed-edge / haze process is correlated along film travel. */
  private coloredNoise(n: number, sigma: number, rho: number): Float32Array {
    const out = new Float32Array(n);
    let v = 0;
    const scale = sigma * Math.sqrt(1 - rho * rho);
    for (let i = 0; i < n; i++) {
      v = rho * v + scale * gaussian(this.rand);
      out[i] = v;
    }
    return out;
  }

  renderAll(): RenderedTrack {
    const strips: Uint8ClampedArray[] = [];
    for (let f = 0; f < this.totalRows / ROWS_PER_FRAME; f++) {
      strips.push(this.renderFrame(f));
    }
    return { strips, manifest: this.manifest };
  }

  renderFrame(frame: number): Uint8ClampedArray {
    const strip = new Uint8ClampedArray(TRACK_W * H);
    const base = frame * ROWS_PER_FRAME;
    const center = TRACK_W / 2;
    const baseHalf = TRACK_W * 0.3;

    for (let row = 0; row < H; row++) {
      const t = base + row;
      const s = clamp(this.clean[t], -1, 1);
      const click = this.clickMap[t];
      const grainAt = (x: number): number => grain((t * TRACK_W + x) * 3 + 1) * this.manifest.grainSigma;

      // A real dust speck/scratch sits across the scanner slit and drives the
      // ENTIRE track width on that row to a bright/dark extreme. That full-width
      // transient is what makes an optical click impulsive (and detectable).
      if (click !== 0) {
        const extreme = click > 0 ? 252 : 4;
        for (let x = 0; x < TRACK_W; x++) {
          strip[row * TRACK_W + x] = clampByte(extreme + grainAt(x) * 0.4);
        }
        continue;
      }

      if (this.type === 'variable-area') {
        const half = baseHalf * (1 + AREA_MODULATION * s);
        const left = center - half + this.edgeNoiseL[t];
        const right = center + half + this.edgeNoiseR[t];
        // Photographic exposure integrates light across each pixel, so a
        // fractional tongue edge is an anti-aliased coverage ramp, not a hard
        // binary column. Coverage of column [x, x+1) by the tongue [left,right].
        for (let x = 0; x < TRACK_W; x++) {
          const coverage = Math.max(0, Math.min(x + 1, right) - Math.max(x, left));
          const v = BLACK + (WHITE - BLACK) * coverage + grainAt(x);
          strip[row * TRACK_W + x] = clampByte(v);
        }
      } else {
        // Variable density: value across the strip tracks the waveform.
        const target = (BLACK + WHITE) / 2 + s * ((WHITE - BLACK) / 2) * 0.97;
        const haze = this.hazeNoise[t];
        for (let x = 0; x < TRACK_W; x++) {
          // Edge wear: haze strongest at the two frayed boundaries.
          const edgeDist = Math.min(x, TRACK_W - 1 - x);
          const edgeGain = edgeDist < 6 ? 1 + (6 - edgeDist) * 0.45 : 1;
          const v = target + haze * edgeGain + grainAt(x) * edgeGain;
          strip[row * TRACK_W + x] = clampByte(v);
        }
      }
    }
    return strip;
  }
}
