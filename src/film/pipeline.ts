import { SPECS, SoundTrackType } from './specs';
import { SyntheticReel } from './reel';
import { PlanarImage } from './image';
import { decodeOpticalSound } from '../audio/decoder';
import { detectShots } from '../picture/shotDetect';
import { removeFlicker, computeFlickerGains } from '../picture/flicker';
import { restoreColor, computeColorGains } from '../picture/colorRestore';
import { repairBlemishes } from '../picture/blemishRepair';
import { convertFrameRate, buildSchedule, estimateMotion, InterpScheduleEntry, MotionField } from '../picture/motion';

/**
 * End-to-end CPU reference pipeline. The browser product runs the same
 * sequence with the pixel stages replaced by GpuPipeline shaders; this path
 * stays available for headless unit tests and as the numerical ground truth.
 */

/** Extract the colored picture window of each scan into an RGB planar image. */
export function extractPictures(reel: SyntheticReel): PlanarImage[] {
  const pw = SPECS.picX1 - SPECS.picX0;
  const ph = SPECS.picY1 - SPECS.picY0;
  const stride = reel.reel.width;
  return reel.reel.frames.map((frame) => {
    const img = new PlanarImage(pw, ph);
    for (let y = 0; y < ph; y++) {
      for (let x = 0; x < pw; x++) {
        const si = ((SPECS.picY0 + y) * stride + SPECS.picX0 + x) * 4;
        const p = y * pw + x;
        img.r[p] = frame.data[si];
        img.g[p] = frame.data[si + 1];
        img.b[p] = frame.data[si + 2];
      }
    }
    return img;
  });
}

export interface CpuPipelineResult {
  restored: PlanarImage[];
  converted: PlanarImage[];
  schedule: InterpScheduleEntry[];
  audioOutput: Float32Array;
  clicksRemoved: number;
  shots: { start: number; end: number }[];
  shotColorGains: [number, number, number][];
}

export function runCpuPipeline(
  reel: SyntheticReel,
  trackType: SoundTrackType,
): CpuPipelineResult {
  // 1. Sound recovery.
  const decoded = decodeOpticalSound(reel.reel, trackType, reel.trackManifest.leaderRows);

  // 2. Picture extraction and restoration.
  const pictures = extractPictures(reel);
  const shots = detectShots(pictures);
  const flicker = removeFlicker(pictures, shots);
  const color = restoreColor(flicker.frames, shots);
  const repair = repairBlemishes(color.frames);

  // 3. Motion-compensated 24 -> 25 fps conversion.
  const { output: converted, schedule } = convertFrameRate(repair.frames);

  return {
    restored: repair.frames,
    converted,
    schedule,
    audioOutput: decoded.outputRate,
    clicksRemoved: decoded.clicksRemoved,
    shots,
    shotColorGains: color.shotGains,
  };
}

/**
 * GPU path: CPU only computes shot boundaries, per-frame/shot gains and block
 * motion vectors; every per-pixel operation (flicker, color, blemish repair,
 * warping) executes inside GpuPipeline fragment shaders.
 */
export interface GpuStage {
  restoreFrame: (
    cur: PlanarImage,
    neighbors: { prev?: PlanarImage; next?: PlanarImage },
    flickerGain: number,
    colorGain: [number, number, number],
  ) => PlanarImage;
  repairFrame: (cur: PlanarImage, neighbors: { prev?: PlanarImage; next?: PlanarImage }) => PlanarImage;
  interpolate: (a: PlanarImage, b: PlanarImage, field: MotionField, t: number) => PlanarImage;
}

export async function runGpuPipeline(
  reel: SyntheticReel,
  gpu: GpuStage,
  trackType: SoundTrackType,
): Promise<CpuPipelineResult> {
  const decoded = decodeOpticalSound(reel.reel, trackType, reel.trackManifest.leaderRows);
  const pictures = extractPictures(reel);
  const shots = detectShots(pictures);

  // Statistics on CPU; pixels are only modified inside shaders.
  const { gain: flickerGain } = computeFlickerGains(pictures, shots);
  const shotGains = computeColorGains(pictures, shots);
  const shotGainAt = (f: number): [number, number, number] => {
    const idx = shots.findIndex((s) => f >= s.start && f < s.end);
    return shotGains[idx < 0 ? shotGains.length - 1 : idx];
  };

  // Pass 1: per-frame flicker + color shaders.
  const corrected: PlanarImage[] = pictures.map((img, f) =>
    gpu.restoreFrame(img, {}, flickerGain[f], shotGainAt(f)),
  );

  // Pass 2: temporal/spatial blemish shader with neighbor frames.
  const repaired: PlanarImage[] = corrected.map((img, f) =>
    gpu.repairFrame(img, {
      prev: f > 0 ? corrected[f - 1] : undefined,
      next: f < corrected.length - 1 ? corrected[f + 1] : undefined,
    }),
  );

  // Pass 3: motion-compensated interpolation shader.
  const schedule = buildSchedule(repaired.length);
  const fieldCache = new Map<string, MotionField>();
  const converted: PlanarImage[] = [];
  for (const e of schedule) {
    const key = `${e.i0}`;
    let field = fieldCache.get(key);
    if (!field) {
      field = estimateMotion(repaired[e.i0], repaired[e.i1]);
      fieldCache.set(key, field);
    }
    converted.push(gpu.interpolate(repaired[e.i0], repaired[e.i1], field, e.t));
  }

  return {
    restored: repaired,
    converted,
    schedule,
    audioOutput: decoded.outputRate,
    clicksRemoved: decoded.clicksRemoved,
    shots,
    shotColorGains: shotGains,
  };
}
