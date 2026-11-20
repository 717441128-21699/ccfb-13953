import { PlanarImage } from '../film/image';

export function drawScan(canvas: HTMLCanvasElement, data: Uint8ClampedArray, w: number, h: number): void {
  const ctx = canvas.getContext('2d')!;
  const img = new ImageData(new Uint8ClampedArray(data), w, h);
  ctx.putImageData(img, 0, 0);
}

export function drawPlanar(canvas: HTMLCanvasElement, img: PlanarImage): void {
  const ctx = canvas.getContext('2d')!;
  const out = ctx.createImageData(img.width, img.height);
  img.toRGBA(out.data);
  ctx.putImageData(out, 0, 0);
}

export function drawWaveform(canvas: HTMLCanvasElement, audio: Float32Array): void {
  const ctx = canvas.getContext('2d')!;
  const { width, height } = canvas;
  ctx.fillStyle = '#0c0f13';
  ctx.fillRect(0, 0, width, height);
  let peak = 1e-6;
  for (const v of audio) peak = Math.max(peak, Math.abs(v));
  const step = Math.max(1, Math.floor(audio.length / width));
  ctx.strokeStyle = '#7fb2ff';
  ctx.lineWidth = 1;
  ctx.beginPath();
  for (let x = 0; x < width; x++) {
    let max = 0;
    for (let i = x * step; i < Math.min(audio.length, (x + 1) * step); i++) {
      max = Math.max(max, Math.abs(audio[i]));
    }
    const y = height / 2 - (max / peak) * (height / 2 - 4);
    if (x === 0) ctx.moveTo(x, y);
    else ctx.lineTo(x, y);
  }
  ctx.stroke();
  ctx.strokeStyle = 'rgba(224,164,88,0.4)';
  ctx.beginPath();
  ctx.moveTo(0, height / 2);
  ctx.lineTo(width, height / 2);
  ctx.stroke();
}
