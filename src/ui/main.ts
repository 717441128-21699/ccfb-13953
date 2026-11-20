import { synthesizeReel } from '../film/reel';
import { runCpuPipeline, runGpuPipeline, CpuPipelineResult } from '../film/pipeline';
import { GpuPipeline } from '../gpu/GpuPipeline';
import { encodeMp4 } from '../encode/encoder';
import { snrDb, psnrByte } from '../film/util';
import { resampleToOutput } from '../synth/audioProgram';
import { PlanarImage } from '../film/image';
import { drawScan, drawPlanar, drawWaveform } from './canvasRender';

interface RunState {
  result: CpuPipelineResult;
  reel: ReturnType<typeof synthesizeReel>;
  trackType: 'variable-area' | 'variable-density';
}

let state: RunState | null = null;

const $ = <T extends HTMLElement>(id: string): T => document.getElementById(id) as T;
const statusEl = $('status');

function setStatus(msg: string): void {
  statusEl.textContent = msg;
}

async function run(): Promise<void> {
  const frameCount = Number($<HTMLInputElement>('frameCount').value);
  const trackType = $<HTMLSelectElement>('trackType').value as 'variable-area' | 'variable-density';
  const seed = Number($<HTMLInputElement>('seed').value);
  const backend = $<HTMLSelectElement>('backend').value;

  setStatus('合成测试序列（真实 16mm 几何 + 已知缺陷）…');
  await frameYield();

  const reel = synthesizeReel(frameCount, trackType, seed);
  drawScan($<HTMLCanvasElement>('scanCanvas'), reel.reel.frames[0].data, reel.reel.width, reel.reel.height);

  setStatus(backend === 'gpu' ? 'GPU 着色器修复中…' : 'CPU 参考流水线处理中…');
  await frameYield();

  let result: CpuPipelineResult;
  if (backend === 'gpu') {
    const gpu = new GpuPipeline();
    try {
      result = await runGpuPipeline(
        reel,
        {
          restoreFrame: (cur, _n, fg, cg) => gpu.correctFrame(cur, fg, cg),
          repairFrame: (cur, n) => gpu.repairFrame(cur, n),
          interpolate: (a, b, field, t) => gpu.interpolate(a, b, field, t),
        },
        trackType,
      );
    } finally {
      gpu.dispose();
    }
  } else {
    result = runCpuPipeline(reel, trackType);
  }

  const cleanRefs = reel.picture.manifest.cleanFrames;

  drawPlanar($<HTMLCanvasElement>('restoredCanvas'), result.restored[0]);
  drawPlanar($<HTMLCanvasElement>('interpCanvas'), result.converted[Math.min(2, result.converted.length - 1)]);
  drawWaveform($<HTMLCanvasElement>('waveCanvas'), result.audioOutput);

  // Metrics: audio SNR vs. the same leading clean program segment (decoded
  // timeline starts at physical lead; no picture-offset window at this stage).
  const windowRows = frameCount * 480;
  const cleanSegment = reel.scanProgramExtended.slice(0, windowRows);
  const cleanOut = resampleToOutput(cleanSegment, 48000, result.audioOutput.length);
  const snr = snrDb(cleanOut, result.audioOutput);

  // Picture PSNR of restored 24fps frames against clean reference (luma planes).
  const refLuma = concatLuma(cleanRefs);
  const gotLuma = concatLuma(result.restored);
  const psnr = psnrByte(refLuma, gotLuma);
  $('metrics').hidden = false;
  $<HTMLElement>('m-shots').textContent = String(result.shots.length);
  $<HTMLElement>('m-clicks').textContent = String(result.clicksRemoved);
  $<HTMLElement>('m-snr').textContent = `${snr.toFixed(1)} dB`;
  $<HTMLElement>('m-psnr').textContent = `${psnr.toFixed(1)} dB`;
  $<HTMLElement>('m-frames').textContent = `${frameCount} → ${result.converted.length}`;
  $<HTMLElement>('m-duration').textContent =
    `${(frameCount / 24).toFixed(3)}s → ${(result.converted.length / 25).toFixed(3)}s`;

  state = { result, reel, trackType };
  $<HTMLButtonElement>('exportBtn').disabled = false;
  setStatus(`完成：${backend.toUpperCase()} 路径，${result.converted.length} 个输出帧全部为运动补偿插帧，无复制帧。`);
}

function concatLuma(frames: PlanarImage[]): Uint8ClampedArray {
  const n = frames[0].width * frames[0].height;
  const out = new Uint8ClampedArray(n * frames.length);
  frames.forEach((f, i) => {
    for (let p = 0; p < n; p++) {
      out[i * n + p] = (0.299 * f.r[p] + 0.587 * f.g[p] + 0.114 * f.b[p]) | 0;
    }
  });
  return out;
}async function exportMp4(): Promise<void> {
  if (!state) return;
  const btn = $<HTMLButtonElement>('exportBtn');
  btn.disabled = true;
  setStatus('WebCodecs 编码 H.264 + AAC 并封装 MP4…');
  await frameYield();
  try {
    const enc = await encodeMp4(state.result.converted, state.result.audioOutput, {
      fps: 25,
      width: state.result.converted[0].width,
      height: state.result.converted[0].height,
      sampleRate: 48000,
    });
    const blob = new Blob([enc.bytes.slice().buffer], { type: 'video/mp4' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `restored-${state.trackType}-${enc.videoChunks}f.mp4`;
    a.click();
    URL.revokeObjectURL(url);
    setStatus(`MP4 已导出：${(blob.size / 1024).toFixed(1)} KiB，${enc.videoChunks} 视频块 / ${enc.audioChunks} 音频块，时长 ${enc.durationMs.toFixed(0)} ms。`);
  } catch (e) {
    setStatus(`编码失败：${(e as Error).message}`);
  } finally {
    btn.disabled = false;
  }
}

function frameYield(): Promise<void> {
  return new Promise((r) => setTimeout(r, 0));
}

$<HTMLButtonElement>('run').addEventListener('click', () => {
  run().catch((e) => setStatus(`运行失败：${(e as Error).message}`));
});
$<HTMLButtonElement>('exportBtn').addEventListener('click', () => {
  exportMp4().catch((e) => setStatus(`导出失败：${(e as Error).message}`));
});
