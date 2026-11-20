# 16mm 胶片扫描修复流水线（TypeScript）

把一批 16mm 胶片的逐帧扫描图像（含光学声带、片孔、画面）自动修复并导出带声音的 MP4。
全部用 TypeScript 实现：合成测试序列自带已知缺陷与真值，算法同时提供 CPU 参考实现和
GPU 着色器实现，浏览器内用 WebCodecs 完成 H.264/AAC 编码并封装 MP4。

## 需求与实现对照

| 需求 | 实现 |
| --- | --- |
| 从扫描图白道读回声音，支持变积式/变密式 | `src/audio/trackReader.ts`：行光量积分 + 自校准（中值归零、鲁棒尺度），不依赖任何硬编码几何 |
| 26 帧物理声画错位 | `src/film/specs.ts` 的 `soundOffset`，合成器在音频前预留 26 帧牵引段，`extractPictureWindow` 按偏移对齐 |
| 划痕→噼啪声去除 | `src/audio/declicker.ts`：邻域中值预测残差 + 高阈值脉冲检测 + 咔哒跨度内桥接 + 单调三次插值 |
| 边缘磨损→沙沙底噪压制 | `src/audio/denoiser.ts`：引导段噪声谱估计、频域平滑、温和维纳门 + 光学带宽软低通 |
| 读出声音 SNR ≥ 40 dB | 多种子端到端测试，变积式/变密式最差种子均 ≥ 40 dB（`test/audio.test.ts`） |
| 同镜头闪烁拉平，真实开灯关灯保留 | `src/picture/flicker.ts`：镜头参考电平 + 持续暗场检测，暗帧增益强制为 1 |
| 粉褪色按镜头恢复 | `src/picture/colorRestore.ts`：灰世界增益（中性场景下精确反演褪色） |
| 划痕/脏点用前后帧补 | `src/picture/blemishRepair.ts`：时域异常 + 空间孤立双条件，避免误伤运动物体 |
| 24→25 fps 运动补偿插帧，禁复制帧 | `src/picture/motion.ts`：块匹配光流 + 双向 warp，调度表保证每帧都是插值 |
| 总时长不变、声音不动 | 输出帧数 `round(24·N/25)`，时长恒等；音频仅解码不拉伸 |
| GPU 着色器 | `src/gpu/GpuPipeline.ts`：闪烁、色彩、缺陷修补、运动补偿插帧全部为 fragment shader |
| 浏览器编码接口出 MP4 | `src/encode/encoder.ts`：WebCodecs `VideoEncoder`(H.264) + `AudioEncoder`(AAC) + mp4-muxer |
| 画面/声带按真实规格合成 | `src/synth/*`：16mm 几何（片孔、96px 声带、4:3 画面、26 帧错位），已知闪烁/粉褪/划痕/脏点/咔哒/磨损 |

## 目录

```
src/
  film/      规格常量、图像类型、总装流水线
  synth/     合成音频节目、光学声带（两种制式+缺陷）、画面（三镜头+缺陷）
  audio/     声带读取、去咔哒、去嘶声、DSP（FFT/STFT/重采样）、解码总装
  picture/   镜头检测、闪烁、色彩、缺陷修补、运动补偿插帧
  gpu/       WebGL2 着色器流水线（与 CPU 路径同口径）
  encode/    WebCodecs 编码 + MP4 封装
  ui/        演示页面（参数、画布、指标、导出按钮）
test/        Vitest 单元测试（35 项，含真值指标）
e2e/         Playwright 端到端测试（真实浏览器 WebGL + WebCodecs 出片）
```

## 常用命令

```bash
npm install
npm run dev        # 启动演示页面
npm test           # Vitest 单元测试
npm run build      # 类型检查 + 生产构建
npx playwright install chromium
npx playwright test   # 浏览器端到端（GPU/CPU 两条路径 + MP4 内容校验）
```

## 关键设计决策

- **自校准读取器**：不假设黑白电平和声带几何，用全条带统计的分位数与中值/MAD
  自动定零和定标，对两种制式通用。
- **脉冲检测与信号分离**：真实咔哒是全狭缝极端值，残差远高于正常 99 分位，
  用高阈值只替换真正的脉冲行，避免把平滑节目当成缺陷（早期低阈值方案曾误改 47% 采样）。
- **不用一阶 DC 阻塞器**：其长建立瞬态在短片段上摧毁 SNR；读取器已中值归零，
  直流只做一次性均值扣除。
- **镜头检测抗缺陷**：在 4×4 空间子块上算归一化直方图距离并取中值，局部划痕/脏点
  被抑制，整帧场景切换（真切点）与持续暗场（开灯关灯）可靠区分。
- **闪烁保留真实灯效**：镜头内以正常帧中值为参考电平，低于参考一半的持续暗帧判定为
  真实灯光事件，增益保持 1 不抹平。
