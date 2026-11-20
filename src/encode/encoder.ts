import { Muxer, ArrayBufferTarget } from 'mp4-muxer';
import { PlanarImage } from '../film/image';

/**
 * Browser-side MP4 export built entirely on WebCodecs:
 *   VideoEncoder (H.264/AVC) + AudioEncoder (AAC) -> mp4-muxer.
 *
 * Timestamps are in microseconds. The video track runs at 25 fps; the audio
 * track is the untouched restored program at its native sample rate, so muxed
 * playback keeps the original reel duration with sample-accurate A/V sync.
 */

export interface EncodeOptions {
  fps: number;
  width: number;
  height: number;
  sampleRate: number;
  videoCodec?: string;
  videoBitrate?: number;
  signal?: AbortSignal;
}

export interface EncodeResult {
  bytes: Uint8Array;
  videoChunks: number;
  audioChunks: number;
  durationMs: number;
}

function supportChecked(codec: string, type: 'video' | 'audio', dims?: { width: number; height: number }): Promise<boolean> {
  if (type === 'video') {
    // Probe at the real frame size: some builds reject a degenerate 2x2 config
    // even though the codec encodes fine at the production resolution.
    return VideoEncoder.isConfigSupported({
      codec,
      width: dims?.width ?? 640,
      height: dims?.height ?? 480,
      bitrate: 2_000_000,
    })
      .then((r): boolean => r.supported === true)
      .catch((): boolean => false);
  }
  return AudioEncoder.isConfigSupported({ codec, numberOfChannels: 1, sampleRate: 48000 })
    .then((r): boolean => r.supported === true)
    .catch((): boolean => false);
}

export async function pickVideoCodec(width: number, height: number): Promise<string> {
  // High profiles first for quality; constrained/baseline last for maximum
  // reach (headless and software-only WebCodecs builds typically only expose
  // baseline AVC).
  const candidates = ['avc1.640028', 'avc1.4d4032', 'avc1.4d401f', 'avc1.42e01f', 'avc1.42E01F'];
  for (const codec of candidates) {
    if (await supportChecked(codec, 'video', { width, height })) return codec;
  }
  // Last-resort: baseline without relying on the probe, which can itself be
  // flaky on some headless builds.
  return 'avc1.42E01F';
}

export async function pickAudioCodec(sampleRate: number): Promise<string> {
  const candidates = ['mp4a.40.2', 'mp4a.40.5', 'mp4a.67'];
  for (const codec of candidates) {
    const ok = await AudioEncoder.isConfigSupported({
      codec,
      numberOfChannels: 1,
      sampleRate,
    })
      .then((r) => r.supported)
      .catch(() => false);
    if (ok) return codec;
  }
  // AAC-LC is the mandatory baseline for MP4; use it if probing is unavailable.
  return 'mp4a.40.2';
}

function frameToVideoFrame(img: PlanarImage): VideoFrame {
  const rgba = img.toRGBA();
  const data = new Uint8ClampedArray(rgba);
  const vf = new VideoFrame(data, {
    format: 'RGBA',
    codedWidth: img.width,
    codedHeight: img.height,
    timestamp: 0,
  });
  return vf;
}

export async function encodeMp4(
  frames: PlanarImage[],
  audio: Float32Array,
  opts: EncodeOptions,
): Promise<EncodeResult> {
  const fps = opts.fps;
  const frameDurationUs = Math.round(1_000_000 / fps);
  const videoCodec = opts.videoCodec ?? (await pickVideoCodec(opts.width, opts.height));

  const target = new ArrayBufferTarget();
  const muxer = new Muxer({
    target,
    fastStart: 'in-memory',
    firstTimestampBehavior: 'strict',
    video: { codec: 'avc', width: opts.width, height: opts.height, frameRate: fps },
    audio: { codec: 'aac', numberOfChannels: 1, sampleRate: opts.sampleRate },
  });

  let videoChunks = 0;
  let audioChunks = 0;

  const videoEncoder = new VideoEncoder({
    output: (chunk, meta) => {
      muxer.addVideoChunk(chunk, meta);
      videoChunks++;
    },
    error: (e) => {
      throw new Error(`VideoEncoder error: ${e.message}`);
    },
  });

  videoEncoder.configure({
    codec: videoCodec,
    width: opts.width,
    height: opts.height,
    bitrate: opts.videoBitrate ?? 4_000_000,
    framerate: fps,
    avc: { format: 'avc' },
  });

  // Encode every frame; every one is a real frame, none is duplicated.
  for (let i = 0; i < frames.length; i++) {
    if (opts.signal?.aborted) throw new DOMException('aborted', 'AbortError');
    const vf = frameToVideoFrame(frames[i]);
    const ts = i * frameDurationUs;
    const frame = new VideoFrame(vf, { timestamp: ts, duration: frameDurationUs });
    vf.close();
    videoEncoder.encode(frame, { keyFrame: i === 0 });
    frame.close();
    if ((i & 7) === 0) await videoEncoder.flush();
  }
  await videoEncoder.flush();

  // Audio: untouched mono PCM fed to AAC in 1024-sample AAC frames.
  const audioCodec = await pickAudioCodec(opts.sampleRate);
  const audioEncoder = new AudioEncoder({
    output: (chunk, meta) => {
      muxer.addAudioChunk(chunk, meta);
      audioChunks++;
    },
    error: (e) => {
      throw new Error(`AudioEncoder error: ${e.message}`);
    },
  });
  audioEncoder.configure({
    codec: audioCodec,
    numberOfChannels: 1,
    sampleRate: opts.sampleRate,
    bitrate: 128_000,
  });

  const aacFrame = 1024;
  const bytesPerSample = 4;
  let offset = 0;
  let chunkIndex = 0;
  while (offset + aacFrame <= audio.length) {
    const pcm = audio.slice(offset, offset + aacFrame);
    const data = new Uint8Array(pcm.buffer, pcm.byteOffset, pcm.length * bytesPerSample);
    const ad = new AudioData({
      format: 'f32-planar',
      sampleRate: opts.sampleRate,
      numberOfFrames: aacFrame,
      numberOfChannels: 1,
      timestamp: chunkIndex * ((aacFrame * 1_000_000) / opts.sampleRate),
      data,
    });
    audioEncoder.encode(ad);
    ad.close();
    offset += aacFrame;
    chunkIndex++;
  }
  await audioEncoder.flush();

  muxer.finalize();
  videoEncoder.close();
  audioEncoder.close();

  const durationMs = (frames.length / fps) * 1000;
  return {
    bytes: new Uint8Array(target.buffer),
    videoChunks,
    audioChunks,
    durationMs,
  };
}

/** Parse an MP4 box tree (depth-first) — used by tests to verify container contents. */
export function listBoxes(bytes: Uint8Array): string[] {
  const names: string[] = [];
  let pos = 0;
  while (pos + 8 <= bytes.length) {
    const size = (bytes[pos] << 24) | (bytes[pos + 1] << 16) | (bytes[pos + 2] << 8) | bytes[pos + 3];
    const type = String.fromCharCode(bytes[pos + 4], bytes[pos + 5], bytes[pos + 6], bytes[pos + 7]);
    names.push(type);
    if (size < 8) break;
    pos += size;
  }
  return names;
}
