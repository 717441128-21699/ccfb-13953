import { test, expect, chromium } from '@playwright/test';

/**
 * End-to-end browser test. Runs the real product path:
 *  - GPU fragment-shader restoration (WebGL2)
 *  - WebCodecs H.264/AAC encoding through the browser
 *  - MP4 muxing with an unchanged audio track
 * Headless Chromium exposes WebGL2 and WebCodecs with the GL/SwiftShader flags.
 */

test.beforeAll(async () => {
  // Browser availability is exercised by the launch itself; no extra setup.
});

test('GPU pipeline restores frames and exports a valid MP4', async () => {
  const browser = await chromium.launch({
    headless: true,
    args: [
      '--use-gl=angle',
      '--use-angle=swiftshader',
      '--enable-unsafe-swiftshader',
      '--ignore-gpu-blocklist',
    ],
  });
  const page = await browser.newPage();
  const errors: string[] = [];
  page.on('console', (msg) => {
    if (msg.type() === 'error') errors.push(msg.text());
  });
  page.on('pageerror', (e) => errors.push(String(e)));

  await page.goto('/');

  // Use a 36-frame reel: long enough that the recovered SNR clears 40 dB while
  // the headless run stays fast.
  await page.fill('#frameCount', '36');
  await page.selectOption('#trackType', 'variable-area');
  await page.selectOption('#backend', 'gpu');
  await page.click('#run');

  // Wait for completion status (the run button path reports a final message).
  await page.waitForFunction(
    () => {
      const el = document.getElementById('status');
      return !!el && /完成|失败/.test(el.textContent ?? '');
    },
    { timeout: 120000 },
  );
  const status = (await page.textContent('#status')) ?? '';
  expect(status).toContain('完成');
  expect(errors).toEqual([]);

  // Metrics panel populated with a passing audio SNR and 24->25 frame count.
  expect(await page.textContent('#m-snr')).toMatch(/\d+\.\d dB/);
  const snrText = (await page.textContent('#m-snr')) ?? '';
  const snr = parseFloat(snrText);
  expect(snr).toBeGreaterThanOrEqual(40);
  expect(await page.textContent('#m-frames')).toContain('36 → 38');

  // The MP4 export button is enabled and produces a downloadable file.
  const downloadPromise = page.waitForEvent('download', { timeout: 120000 });
  await page.click('#exportBtn');
  const download = await downloadPromise;
  expect(download.suggestedFilename()).toMatch(/\.mp4$/);
  const path = await download.path();
  expect(path).toBeTruthy();
  if (path) {
    const fs = await import('node:fs/promises');
    const bytes = await fs.readFile(path);
    // ISO-BMFF containers begin with an ftyp box marked 'isom'/'mp42'/'iso5'.
    const head = bytes.subarray(4, 12).toString('latin1');
    expect(head).toMatch(/ftyp(isom|mp42|iso5|avc1|dash)/);
    // Both an H.264 video track (avc1) and an AAC audio track (mp4a) are present.
    const body = bytes.toString('latin1');
    expect(body).toContain('avc1');
    expect(body).toContain('mp4a');
  }

  await browser.close();
});

test('CPU reference path also completes with 25 fps output', async () => {
  const browser = await chromium.launch({
    headless: true,
    args: ['--use-gl=angle', '--use-angle=swiftshader', '--enable-unsafe-swiftshader'],
  });
  const page = await browser.newPage();
  await page.goto('/');
  await page.fill('#frameCount', '36');
  await page.selectOption('#trackType', 'variable-density');
  await page.selectOption('#backend', 'cpu');
  await page.click('#run');
  await page.waitForFunction(
    () => {
      const el = document.getElementById('status');
      return !!el && /完成|失败/.test(el.textContent ?? '');
    },
    { timeout: 120000 },
  );
  expect(await page.textContent('#status')).toContain('完成');
  expect(await page.textContent('#m-frames')).toContain('36 → 38');
  expect(await page.textContent('#m-shots')).toBe('3');
  await browser.close();
});
