/**
 * Planar 8-bit RGB image plus optional RGBA packed view.
 * All CPU-side reference algorithms operate on the planar buffers;
 * the GPU pipeline uploads the same data into textures.
 */
export class PlanarImage {
  readonly width: number;
  readonly height: number;
  readonly r: Uint8ClampedArray;
  readonly g: Uint8ClampedArray;
  readonly b: Uint8ClampedArray;

  constructor(width: number, height: number, buffers?: [Uint8ClampedArray, Uint8ClampedArray, Uint8ClampedArray]) {
    this.width = width;
    this.height = height;
    const n = width * height;
    this.r = buffers?.[0] ?? new Uint8ClampedArray(n);
    this.g = buffers?.[1] ?? new Uint8ClampedArray(n);
    this.b = buffers?.[2] ?? new Uint8ClampedArray(n);
  }

  static fromRGBA(data: Uint8ClampedArray, width: number, height: number): PlanarImage {
    const n = width * height;
    const r = new Uint8ClampedArray(n);
    const g = new Uint8ClampedArray(n);
    const b = new Uint8ClampedArray(n);
    for (let i = 0, p = 0; i < data.length; i += 4, p++) {
      r[p] = data[i];
      g[p] = data[i + 1];
      b[p] = data[i + 2];
    }
    return new PlanarImage(width, height, [r, g, b]);
  }

  toRGBA(out?: Uint8ClampedArray): Uint8ClampedArray {
    const n = this.width * this.height;
    const rgba = out ?? new Uint8ClampedArray(n * 4);
    for (let p = 0, i = 0; p < n; p++, i += 4) {
      rgba[i] = this.r[p];
      rgba[i + 1] = this.g[p];
      rgba[i + 2] = this.b[p];
      rgba[i + 3] = 255;
    }
    return rgba;
  }

  clone(): PlanarImage {
    return new PlanarImage(this.width, this.height, [
      this.r.slice(),
      this.g.slice(),
      this.b.slice(),
    ]);
  }

  luma(out?: Float32Array): Float32Array {
    const n = this.width * this.height;
    const y = out ?? new Float32Array(n);
    for (let p = 0; p < n; p++) {
      y[p] = 0.299 * this.r[p] + 0.587 * this.g[p] + 0.114 * this.b[p];
    }
    return y;
  }
}

/**
 * One scanned film frame: the full scanner image (perf + soundtrack + picture)
 * stored as packed RGBA. Perf and soundtrack regions are monochrome
 * (R = G = B); the picture region carries the film's real color.
 */
export interface ScanFrame {
  index: number;
  data: Uint8ClampedArray;
}

export interface Reel {
  width: number;
  height: number;
  frames: ScanFrame[];
}

/** Luma of one RGBA pixel. */
export function rgbaLuma(data: Uint8ClampedArray, index: number): number {
  const i = index * 4;
  return 0.299 * data[i] + 0.587 * data[i + 1] + 0.114 * data[i + 2];
}
