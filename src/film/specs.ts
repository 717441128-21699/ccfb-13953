/**
 * 16mm film geometry and process constants.
 *
 * Scanner image layout (one scan per film frame), width W x height H:
 *
 *   x: 0..PERF_W-1            perf strip (sprocket holes)
 *      PERF_W..SOUND_X0-1     gap
 *      SOUND_X0..SOUND_X1-1   optical soundtrack (the wavy white track)
 *      SOUND_X1..PIC_X0-1     gap
 *      PIC_X0..PIC_X1-1       projected picture (4:3)
 *
 * Film physical note: on real 16mm prints the soundtrack sits 26 frames
 * ahead of the picture it accompanies (SOUND_OFFSET). Every synthetic
 * sequence and the decode pipeline honor this offset.
 */

export const SPECS = {
  // Scanner frame (downscaled but proportional to a real 16mm telecine frame).
  scanW: 640,
  scanH: 480,

  // Geometry, x-ranges inclusive-start/exclusive-end.
  perfX0: 0,
  perfX1: 28,
  soundX0: 40,
  soundX1: 136, // 96 px wide track region
  picX0: 160,
  picX1: 624, // 464 px wide
  picY0: 30,
  picY1: 378, // 348 px tall -> 464:348 = 4:3

  // Real 16mm sound/picture separation, in frames.
  soundOffset: 26,

  // Quiet, noise-only film leader (frames) used to fingerprint the hiss floor.
  leaderFrames: 6,

  // Frame rates.
  srcFps: 24,
  dstFps: 25,

  // Audio: samples per scanned picture-frame row period at 24 fps.
  audioRate: 48000,
} as const;

export const SAMPLES_PER_FRAME = SPECS.audioRate / SPECS.srcFps; // 2000
export const SOUND_ROWS = SPECS.scanH; // one decoded audio sample per scan row

/** Total audio samples covered by N scanned frames (each frame contributes scanH rows). */
export function totalAudioSamples(frameCount: number): number {
  return frameCount * SOUND_ROWS;
}

/** Picture frame index -> first audio sample index it physically contains. */
export function pictureFrameToAudioStart(pictureFrame: number): number {
  // Sound for picture frame k lives on frame k+SOUND_OFFSET.
  return (pictureFrame + SPECS.soundOffset) * SOUND_ROWS;
}

/** Audio sample index (in the whole-reel timeline) -> (scan frame, row). */
export function audioSampleToScanPos(sampleIndex: number): { frame: number; row: number } {
  return { frame: Math.floor(sampleIndex / SOUND_ROWS), row: sampleIndex % SOUND_ROWS };
}

export type SoundTrackType = 'variable-area' | 'variable-density';

export interface SequenceConfig {
  frames: number;
  trackType: SoundTrackType;
  seed: number;
  /** Master output audio sample rate target (decode rate is mapped to this). */
  outputRate?: number;
}
