import { SPECS, SoundTrackType } from '../film/specs';
import { Reel } from '../film/image';
import { estimateLevels, readTrack } from './trackReader';
import { declick } from './declicker';
import { denoise } from './denoiser';
import { resampleToOutput } from '../synth/audioProgram';
import { SCAN_AUDIO_RATE } from '../synth/audioProgram';
import { rms, mean } from '../film/util';

/**
 * End-to-end optical-sound recovery:
 *   strip pixels -> raw waveform -> impulse removal -> hiss suppression
 *   -> one-shot DC offset removal -> upsample to output rate.
 *
 * DC offset is removed by subtracting the global mean, never with a high-pass
 * DC blocker: a one-pole blocker has a long settling transient at the head of
 * the clip that destroys SNR on a short reel, and the self-calibrating reader
 * has already centered the waveform, so only a constant offset remains.
 */

export interface DecodeResult {
  /** Decoded scan-rate waveform over the full reel timeline (frameCount*480). */
  scanRate: Float32Array;
  /** Picture-aligned audio at the output sample rate (length frameCount*2000). */
  outputRate: Float32Array;
  clicksRemoved: number;
  levels: { black: number; white: number };
}

export function decodeOpticalSound(
  reel: Reel,
  type: SoundTrackType,
  leaderRows: { start: number; end: number },
  outputRateHz = SPECS.audioRate,
): DecodeResult {
  const levels = estimateLevels(reel, type);
  const raw = readTrack(reel, type, levels);

  const { output: declicked, clicks } = declick(raw);
  const denoised = denoise(declicked, leaderRows);

  // One-shot constant-offset removal (transient-free).
  const offset = mean(denoised);
  for (let i = 0; i < denoised.length; i++) denoised[i] -= offset;

  // Gentle global level normalization to unit RMS (gain-only; never touches timing).
  const level = rms(denoised);
  if (level > 1e-9) {
    const target = 0.22;
    const g = target / level;
    for (let i = 0; i < denoised.length; i++) denoised[i] *= g;
  }

  const samplesPerFrame = outputRateHz / SPECS.srcFps;
  const outLen = reel.frames.length * samplesPerFrame;
  const output = resampleToOutput(denoised, outputRateHz, outLen);

  return { scanRate: denoised, outputRate: output, clicksRemoved: clicks, levels };
}

/**
 * Align the decoded full-timeline waveform against the picture using the
 * known physical offset: returns the window corresponding to picture frames.
 */
export function extractPictureWindow(
  scanRate: Float32Array,
  frameCount: number,
  offsetFrames: number = SPECS.soundOffset,
): Float32Array {
  const start = offsetFrames * SPECS.scanH;
  const len = frameCount * SPECS.scanH;
  const windowed = new Float32Array(len);
  for (let i = 0; i < len; i++) {
    const s = start + i;
    windowed[i] = s < scanRate.length ? scanRate[s] : 0;
  }
  return windowed;
}

export { SCAN_AUDIO_RATE };
