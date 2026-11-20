import { SPECS } from './specs';
import { Reel, ScanFrame } from './image';
import { mulberry32, clampByte } from './util';
import { generateAudioProgram, SCAN_AUDIO_RATE, AudioProgram } from '../synth/audioProgram';
import { OpticalTrackRenderer, TrackManifest } from '../synth/opticalTrack';
import { PictureSynthesizer, BuiltPicture } from '../synth/pictureSynth';

/**
 * Full synthetic reel assembly.
 *
 * Physical 16mm prints carry the soundtrack SOUND_OFFSET (26) frames ahead of
 * the picture. The audio program is therefore generated with a 26-frame
 * trailer; strip f on the scan encodes timeline rows [f*480,(f+1)*480), and
 * the picture-aligned output audio window starts at row 26*480.
 *
 * Scan frames are packed RGBA: the perf and soundtrack regions are monochrome
 * (R=G=B); the picture region keeps the film's real color so the color-fade
 * restoration stage has actual chrominance to work on.
 */

export interface SyntheticReel {
  reel: Reel;
  picture: BuiltPicture;
  audio: AudioProgram;
  /** Extended scan-rate program including the 26-frame offset trailer. */
  scanProgramExtended: Float32Array;
  trackManifest: TrackManifest;
  /** Scan-rate timeline row where picture-aligned output audio begins. */
  audioWindowStart: number;
  audioWindowRows: number;
}

const W = SPECS.scanW;
const H = SPECS.scanH;
const OFFSET = SPECS.soundOffset;

export function synthesizeReel(frameCount: number, trackType: 'variable-area' | 'variable-density', seed: number): SyntheticReel {
  const rand = mulberry32(seed ^ 0xbeef);
  const picture = new PictureSynthesizer(frameCount, seed).build();

  // Extended program: frameCount picture frames + offset trailer.
  const audio = generateAudioProgram(frameCount + OFFSET, SPECS.audioRate, seed);
  const extended = audio.scanRate;

  const renderer = new OpticalTrackRenderer(extended, trackType, frameCount, seed);
  const { strips, manifest } = renderer.renderAll();

  const frames: ScanFrame[] = [];
  for (let f = 0; f < frameCount; f++) {
    const data = new Uint8ClampedArray(W * H * 4);
    // Film base (dark, near-neutral).
    for (let p = 0; p < W * H; p++) {
      data[p * 4] = 26;
      data[p * 4 + 1] = 26;
      data[p * 4 + 2] = 26;
      data[p * 4 + 3] = 255;
    }
    drawPerforations(data, f, rand);
    blitTrack(data, strips[f]);
    blitPicture(data, picture.degradedFrames[f]);
    frames.push({ index: f, data });
  }

  return {
    reel: { width: W, height: H, frames },
    picture,
    audio,
    scanProgramExtended: extended,
    trackManifest: manifest,
    audioWindowStart: OFFSET * H,
    audioWindowRows: frameCount * H,
  };
}

function setMono(data: Uint8ClampedArray, x: number, y: number, value: number): void {
  const i = (y * W + x) * 4;
  const v = clampByte(value);
  data[i] = v;
  data[i + 1] = v;
  data[i + 2] = v;
}

function drawPerforations(data: Uint8ClampedArray, frame: number, rand: () => number): void {
  const holeH = 52;
  const holeW = 18;
  const cx = 14;
  for (const cy of [H * 0.22, H * 0.78]) {
    const jitter = (rand() - 0.5) * 2;
    for (let y = Math.floor(cy - holeH / 2); y < cy + holeH / 2; y++) {
      for (let x = cx - holeW / 2; x < cx + holeW / 2; x++) {
        if (x < 0 || y < 0 || x >= W || y >= H) continue;
        setMono(data, x, y, 6 + jitter);
      }
    }
  }
  const shade = 22 + 4 * Math.sin(frame * 0.9);
  for (let y = 0; y < H; y++) {
    for (let x = 0; x < SPECS.perfX1; x++) {
      const i = (y * W + x) * 4;
      if (data[i] > 15) setMono(data, x, y, shade);
    }
  }
}

function blitTrack(data: Uint8ClampedArray, strip: Uint8ClampedArray): void {
  const tw = SPECS.soundX1 - SPECS.soundX0;
  for (let y = 0; y < H; y++) {
    for (let x = 0; x < tw; x++) {
      setMono(data, SPECS.soundX0 + x, y, strip[y * tw + x]);
    }
  }
}

function blitPicture(
  data: Uint8ClampedArray,
  img: { width: number; height: number; r: Uint8ClampedArray; g: Uint8ClampedArray; b: Uint8ClampedArray },
): void {
  for (let y = 0; y < img.height; y++) {
    for (let x = 0; x < img.width; x++) {
      const p = y * img.width + x;
      const i = ((SPECS.picY0 + y) * W + SPECS.picX0 + x) * 4;
      data[i] = img.r[p];
      data[i + 1] = img.g[p];
      data[i + 2] = img.b[p];
      data[i + 3] = 255;
    }
  }
}

export { SCAN_AUDIO_RATE };
