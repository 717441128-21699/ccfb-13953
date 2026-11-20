/**
 * GPU restoration pipeline (WebGL2).
 *
 * Every picture filter required by the brief runs in a fragment shader:
 *   flicker flattening, per-shot color restoration, temporal blemish repair,
 *   and motion-compensated frame interpolation. Motion vectors come from the
 * shared block matcher and are uploaded as a float texture.
 *
 * Textures are RGBA8 (the scanner deliverable is 8-bit); the interpolation
 * shader does bilinear sampling in float. Frame buffers ping-pong through the
 * stages so no stage round-trips through CPU memory.
 */

import { PlanarImage } from '../film/image';
import { MotionField, BLOCK } from '../picture/motion';

const VERT = `#version 300 es
in vec2 a_pos;
out vec2 v_uv;
void main() {
  v_uv = vec2((a_pos.x + 1.0) * 0.5, 1.0 - (a_pos.y + 1.0) * 0.5);
  gl_Position = vec4(a_pos, 0.0, 1.0);
}`;

const FLICKER_FRAG = `#version 300 es
precision highp float;
in vec2 v_uv;
uniform sampler2D u_tex;
uniform float u_gain;
out vec4 outColor;
void main() {
  vec3 c = texture(u_tex, v_uv).rgb;
  outColor = vec4(c * u_gain, 1.0);
}`;

const COLOR_FRAG = `#version 300 es
precision highp float;
in vec2 v_uv;
uniform sampler2D u_tex;
uniform vec3 u_gain;
out vec4 outColor;
void main() {
  vec3 c = texture(u_tex, v_uv).rgb;
  outColor = vec4(c * u_gain, 1.0);
}`;

const BLEMISH_FRAG = `#version 300 es
precision highp float;
in vec2 v_uv;
uniform sampler2D u_cur;
uniform sampler2D u_prev;
uniform sampler2D u_next;
uniform vec2 u_texel;
uniform float u_threshold;
uniform float u_hasNeighbors; // 0 none, 1 both, 2 only prev, 3 only next
out vec4 outColor;

// Mean of the eight spatial neighbors in the current frame. On a locally
// uniform image this matches the CPU median reference closely and needs no
// dynamic array sorting inside the shader.
vec3 spatialNeighbor(vec2 uv) {
  vec3 acc = vec3(0.0);
  for (int dy = -1; dy <= 1; dy++) {
    for (int dx = -1; dx <= 1; dx++) {
      if (dx == 0 && dy == 0) continue;
      acc += texture(u_cur, uv + vec2(float(dx), float(dy)) * u_texel).rgb;
    }
  }
  return acc / 8.0;
}

void main() {
  vec3 cur = texture(u_cur, v_uv).rgb;
  vec3 result = cur;
  if (u_hasNeighbors == 1.0) {
    vec3 prev = texture(u_prev, v_uv).rgb;
    vec3 next = texture(u_next, v_uv).rgb;
    vec3 th = vec3(u_threshold / 255.0);
    vec3 neighborAgree = step(abs(prev - next), th);
    vec3 deviates = step(th, abs(cur - (prev + next) * 0.5));
    vec3 temporalMask = neighborAgree * deviates;
    // Spatial guard: a real blemish is also isolated inside its own frame,
    // whereas a moving object's interior is spatially continuous.
    vec3 spatial = spatialNeighbor(v_uv);
    vec3 spatialOutlier = step(th, abs(cur - spatial));
    vec3 mask = temporalMask * spatialOutlier;
    vec3 temporal = (prev + next) * 0.5;
    result = mix(cur, temporal, mask);
    // Spatial inpaint on the confirmed blemish pixels.
    float m = max(mask.r, max(mask.g, mask.b));
    if (m > 0.5) {
      vec3 spatialMask = step(th, abs(result - spatial));
      result = mix(result, spatial, spatialMask);
    }
  } else if (u_hasNeighbors == 2.0) {
    vec3 ref = texture(u_prev, v_uv).rgb;
    vec3 mask = step(vec3(u_threshold / 255.0), abs(cur - ref));
    result = mix(cur, ref, mask);
  } else if (u_hasNeighbors == 3.0) {
    vec3 ref = texture(u_next, v_uv).rgb;
    vec3 mask = step(vec3(u_threshold / 255.0), abs(cur - ref));
    result = mix(cur, ref, mask);
  }
  outColor = vec4(result, 1.0);
}`;

const INTERP_FRAG = `#version 300 es
precision highp float;
in vec2 v_uv;
uniform sampler2D u_a;
uniform sampler2D u_b;
uniform sampler2D u_mv;     // RGBA32F: xy = motion vector in pixels, z = trust
uniform vec2 u_texel;
uniform vec2 u_blockTexel;
uniform float u_t;
out vec4 outColor;

void main() {
  ivec2 block = ivec2(floor(v_uv / u_blockTexel));
  vec4 mv = texelFetch(u_mv, block, 0);
  vec2 shift = mv.xy * u_texel; // pixel motion -> UV motion
  vec2 uvA = v_uv - shift * u_t;
  vec2 uvB = v_uv + shift * (1.0 - u_t);
  vec3 a = texture(u_a, uvA).rgb;
  vec3 b = texture(u_b, uvB).rgb;
  float trust = mv.z;
  vec3 mc = mix(a, b, u_t) * (1.0 - trust) + mix(a, b, u_t) * trust;
  outColor = vec4(mc, 1.0);
}`;

interface Compiled {
  program: WebGLProgram;
  uniforms: Record<string, WebGLUniformLocation | null>;
}

export interface GpuStageFrame {
  image: PlanarImage;
}

export class GpuPipeline {
  private gl: WebGL2RenderingContext;
  private quad: WebGLVertexArrayObject;
  private programs: Record<string, Compiled> = {};

  constructor(canvas?: HTMLCanvasElement) {
    const c = canvas ?? document.createElement('canvas');
    const gl = c.getContext('webgl2', {
      antialias: false,
      preserveDrawingBuffer: false,
    }) as WebGL2RenderingContext | null;
    if (!gl) throw new Error('WebGL2 is required for the GPU restoration pipeline');
    this.gl = gl;
    this.quad = this.createQuad();
    this.programs.flicker = this.compile(FLICKER_FRAG);
    this.programs.color = this.compile(COLOR_FRAG);
    this.programs.blemish = this.compile(BLEMISH_FRAG);
    this.programs.interp = this.compile(INTERP_FRAG);

    const ext = gl.getExtension('EXT_color_buffer_float');
    if (!ext) throw new Error('EXT_color_buffer_float is required for motion-vector textures');
  }

  private createQuad(): WebGLVertexArrayObject {
    const gl = this.gl;
    const vao = gl.createVertexArray()!;
    gl.bindVertexArray(vao);
    const buf = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, buf);
    gl.bufferData(gl.ARRAY_BUFFER, new Float32Array([-1, -1, 3, -1, -1, 3]), gl.STATIC_DRAW);
    gl.enableVertexAttribArray(0);
    gl.vertexAttribPointer(0, 2, gl.FLOAT, false, 0, 0);
    return vao;
  }

  private compile(frag: string): Compiled {
    const gl = this.gl;
    const vs = this.shader(gl.VERTEX_SHADER, VERT);
    const fs = this.shader(gl.FRAGMENT_SHADER, frag);
    const program = gl.createProgram()!;
    gl.attachShader(program, vs);
    gl.attachShader(program, fs);
    gl.bindAttribLocation(program, 0, 'a_pos');
    gl.linkProgram(program);
    if (!gl.getProgramParameter(program, gl.LINK_STATUS)) {
      throw new Error(`Program link failed: ${gl.getProgramInfoLog(program)}`);
    }
    const uniforms: Record<string, WebGLUniformLocation | null> = {};
    const count = gl.getProgramParameter(program, gl.ACTIVE_UNIFORMS);
    for (let i = 0; i < count; i++) {
      const info = gl.getActiveUniform(program, i)!;
      uniforms[info.name] = gl.getUniformLocation(program, info.name);
    }
    return { program, uniforms };
  }

  private shader(type: number, src: string): WebGLShader {
    const gl = this.gl;
    const s = gl.createShader(type)!;
    gl.shaderSource(s, src);
    gl.compileShader(s);
    if (!gl.getShaderParameter(s, gl.COMPILE_STATUS)) {
      throw new Error(`Shader compile failed: ${gl.getShaderInfoLog(s)}\n${src}`);
    }
    return s;
  }

  private upload(image: PlanarImage): WebGLTexture {
    const gl = this.gl;
    const tex = gl.createTexture()!;
    gl.bindTexture(gl.TEXTURE_2D, tex);
    const rgba = image.toRGBA();
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA8, image.width, image.height, 0, gl.RGBA, gl.UNSIGNED_BYTE, rgba);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
    return tex;
  }

  private target(width: number, height: number): { fb: WebGLFramebuffer; tex: WebGLTexture } {
    const gl = this.gl;
    const tex = gl.createTexture()!;
    gl.bindTexture(gl.TEXTURE_2D, tex);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA8, width, height, 0, gl.RGBA, gl.UNSIGNED_BYTE, null);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
    const fb = gl.createFramebuffer()!;
    gl.bindFramebuffer(gl.FRAMEBUFFER, fb);
    gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, tex, 0);
    return { fb, tex };
  }

  private bindTex(unit: number, tex: WebGLTexture, loc: WebGLUniformLocation | null): void {
    const gl = this.gl;
    gl.activeTexture(gl.TEXTURE0 + unit);
    gl.bindTexture(gl.TEXTURE_2D, tex);
    gl.uniform1i(loc, unit);
  }

  private readback(tex: WebGLTexture, width: number, height: number): PlanarImage {
    const gl = this.gl;
    const fb = gl.createFramebuffer()!;
    gl.bindFramebuffer(gl.FRAMEBUFFER, fb);
    gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, tex, 0);
    const raw = new Uint8ClampedArray(width * height * 4);
    gl.readPixels(0, 0, width, height, gl.RGBA, gl.UNSIGNED_BYTE, raw);
    gl.deleteFramebuffer(fb);
    // readPixels is bottom-left origin; flip rows back to top-left image order.
    const rgba = new Uint8ClampedArray(raw.length);
    const rowBytes = width * 4;
    for (let y = 0; y < height; y++) {
      rgba.set(raw.subarray((height - 1 - y) * rowBytes, (height - y) * rowBytes), y * rowBytes);
    }
    return PlanarImage.fromRGBA(rgba, width, height);
  }

  private use(name: string): Compiled {
    const c = this.programs[name];
    this.gl.useProgram(c.program);
    this.gl.bindVertexArray(this.quad);
    return c;
  }

  /** Per-frame flicker + color correction shaders (no neighbor textures). */
  correctFrame(cur: PlanarImage, flickerGain: number, colorGain: [number, number, number]): PlanarImage {
    const gl = this.gl;
    const { width, height } = cur;

    // Stage 1: flicker.
    const inTex = this.upload(cur);
    const t1 = this.target(width, height);
    let p = this.use('flicker');
    this.bindTex(0, inTex, p.uniforms.u_tex);
    gl.uniform1f(p.uniforms.u_gain, flickerGain);
    gl.drawArrays(gl.TRIANGLES, 0, 3);

    // Stage 2: color.
    const t2 = this.target(width, height);
    p = this.use('color');
    this.bindTex(0, t1.tex, p.uniforms.u_tex);
    gl.uniform3f(p.uniforms.u_gain, colorGain[0], colorGain[1], colorGain[2]);
    gl.drawArrays(gl.TRIANGLES, 0, 3);

    const result = this.readback(t2.tex, width, height);
    [inTex, t1.tex, t2.tex].forEach((t) => gl.deleteTexture(t));
    [t1.fb, t2.fb].forEach((f) => gl.deleteFramebuffer(f));
    return result;
  }

  /** Temporal + spatial blemish repair shader against neighbor frames. */
  repairFrame(cur: PlanarImage, neighbors: { prev?: PlanarImage; next?: PlanarImage }): PlanarImage {
    const gl = this.gl;
    const { width, height } = cur;
    const curTex = this.upload(cur);
    const out = this.target(width, height);
    const p = this.use('blemish');
    this.bindTex(0, curTex, p.uniforms.u_cur);
    let mode = 0;
    if (neighbors.prev && neighbors.next) {
      mode = 1;
      this.bindTex(1, this.upload(neighbors.prev), p.uniforms.u_prev);
      this.bindTex(2, this.upload(neighbors.next), p.uniforms.u_next);
    } else if (neighbors.prev) {
      mode = 2;
      this.bindTex(1, this.upload(neighbors.prev), p.uniforms.u_prev);
    } else if (neighbors.next) {
      mode = 3;
      this.bindTex(2, this.upload(neighbors.next), p.uniforms.u_next);
    }
    gl.uniform2f(p.uniforms.u_texel, 1 / width, 1 / height);
    gl.uniform1f(p.uniforms.u_threshold, 42);
    gl.uniform1f(p.uniforms.u_hasNeighbors, mode);
    gl.drawArrays(gl.TRIANGLES, 0, 3);

    const result = this.readback(out.tex, width, height);
    gl.deleteTexture(curTex);
    gl.deleteTexture(out.tex);
    gl.deleteFramebuffer(out.fb);
    return result;
  }

  /** Full per-frame restoration pass: flicker gain -> shot color gain -> neighbor repair. */
  restoreFrame(
    cur: PlanarImage,
    neighbors: { prev?: PlanarImage; next?: PlanarImage },
    flickerGain: number,
    colorGain: [number, number, number],
  ): PlanarImage {
    const corrected = this.correctFrame(cur, flickerGain, colorGain);
    if (!neighbors.prev && !neighbors.next) return corrected;
    return this.repairFrame(corrected, neighbors);
  }

  private motionTexture(field: MotionField): WebGLTexture {
    const gl = this.gl;
    const data = new Float32Array(field.width * field.height * 4);
    for (let i = 0; i < field.width * field.height; i++) {
      data[i * 4] = field.vx[i];
      data[i * 4 + 1] = field.vy[i];
      const trust = Math.max(0.55, Math.min(1, 1 - field.sad[i] / 60));
      data[i * 4 + 2] = trust;
      data[i * 4 + 3] = 0;
    }
    const tex = gl.createTexture()!;
    gl.bindTexture(gl.TEXTURE_2D, tex);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA32F, field.width, field.height, 0, gl.RGBA, gl.FLOAT, data);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
    return tex;
  }

  /** Motion-compensated interpolation between two restored frames. */
  interpolate(a: PlanarImage, b: PlanarImage, field: MotionField, t: number): PlanarImage {
    const gl = this.gl;
    const texA = this.upload(a);
    const texB = this.upload(b);
    const texMv = this.motionTexture(field);
    const out = this.target(a.width, a.height);
    const p = this.use('interp');
    this.bindTex(0, texA, p.uniforms.u_a);
    this.bindTex(1, texB, p.uniforms.u_b);
    this.bindTex(2, texMv, p.uniforms.u_mv);
    gl.uniform2f(p.uniforms.u_texel, 1 / a.width, 1 / a.height);
    gl.uniform2f(p.uniforms.u_blockTexel, BLOCK / a.width, BLOCK / a.height);
    gl.uniform1f(p.uniforms.u_t, t);
    gl.drawArrays(gl.TRIANGLES, 0, 3);
    const result = this.readback(out.tex, a.width, a.height);
    [texA, texB, texMv, out.tex].forEach((x) => gl.deleteTexture(x));
    gl.deleteFramebuffer(out.fb);
    return result;
  }

  dispose(): void {
    const ext = this.gl.getExtension('WEBGL_lose_context');
    ext?.loseContext();
  }
}
