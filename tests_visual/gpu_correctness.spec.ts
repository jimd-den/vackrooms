import { expect, test } from '@playwright/test';

type RenderProbe = {
  meanLuminance: number;
  brightPixelRatio: number;
  nonBlackPixelRatio: number;
  colorBuckets: number;
};

const referenceBackends = [
  {
    name: 'surface-direct-reference',
    renderer: 'surface',
    toggles: 'rt_bake=0&rt_cull=0&rt_shadows=0&rt_dither=0&rt_timer=0',
  },
  {
    name: 'raymarch-direct-dda-reference',
    renderer: 'raymarch',
    toggles: 'rt_bake=0&rt_skip=0&rt_f2b=0&rt_dither=0&rt_timer=0',
  },
] as const;

async function probeScreenshot(
  page: import('@playwright/test').Page,
  png: Buffer,
): Promise<RenderProbe> {
  const dataUrl = `data:image/png;base64,${png.toString('base64')}`;
  return page.evaluate(async encodedImage => {
    const response = await fetch(encodedImage);
    const bitmap = await createImageBitmap(await response.blob());
    const sample = document.createElement('canvas');
    sample.width = 80;
    sample.height = 45;
    const context = sample.getContext('2d', { willReadFrequently: true });
    if (!context) throw new Error('2D render probe is unavailable');
    context.drawImage(bitmap, 0, 0, sample.width, sample.height);

    const pixels = context.getImageData(0, 0, sample.width, sample.height).data;
    const colors = new Set<number>();
    let luminanceSum = 0;
    let bright = 0;
    let nonBlack = 0;
    const count = pixels.length / 4;
    for (let offset = 0; offset < pixels.length; offset += 4) {
      const red = pixels[offset];
      const green = pixels[offset + 1];
      const blue = pixels[offset + 2];
      const luminance = red * 0.2126 + green * 0.7152 + blue * 0.0722;
      luminanceSum += luminance;
      if (luminance >= 48) bright++;
      if (luminance >= 3) nonBlack++;
      colors.add(((red >> 4) << 8) | ((green >> 4) << 4) | (blue >> 4));
    }
    const result = {
      meanLuminance: luminanceSum / count,
      brightPixelRatio: bright / count,
      nonBlackPixelRatio: nonBlack / count,
      colorBuckets: colors.size,
    };
    bitmap.close();
    return result;
  }, dataUrl);
}

for (const backend of referenceBackends) {
  test(`${backend.name} renders emissive fixtures and their room`, async ({ page }, testInfo) => {
    test.setTimeout(90_000);
    const browserErrors: string[] = [];
    page.on('pageerror', error => browserErrors.push(error.message));
    page.on('console', message => {
      if (message.type() === 'error' || /GL_INVALID|shader .*error/i.test(message.text())) {
        browserErrors.push(message.text());
      }
    });

    await page.setViewportSize({ width: 320, height: 180 });
    await page.goto(
      `/index.html?seed=42&level=0&workers=0&capture=1&smoke=1` +
      `&renderer=${backend.renderer}&${backend.toggles}` +
      '&camera=6,1.7,31.2&yaw=-1.5708&pitch=-0.05',
    );
    await page.waitForFunction(
      () => {
        const status = document.getElementById('status-msg')?.textContent ?? '';
        if (status.includes('Failed to start engine')) throw new Error(status);
        return (window as typeof window & { __sceneReady?: boolean }).__sceneReady === true;
      },
      null,
      { timeout: 60_000 },
    );

    await page.evaluate(() => {
      for (const id of ['overlay', 'hud', 'hud-section', 'debug-overlay']) {
        const element = document.getElementById(id);
        if (element) element.style.display = 'none';
      }
    });
    const screenshotPath = testInfo.outputPath(`${backend.name}.png`);
    const screenshot = await page.locator('#view').screenshot({ path: screenshotPath });
    const probe = await probeScreenshot(page, screenshot);

    // A dead panel path produces a nearly black canvas with only a few
    // emissive texels. Require both visible fixtures and illuminated room
    // geometry, while keeping the contract independent of exact pixels.
    expect(probe.colorBuckets).toBeGreaterThan(8);
    expect(probe.brightPixelRatio).toBeGreaterThan(0.0005);
    expect(probe.nonBlackPixelRatio).toBeGreaterThan(0.08);
    expect(probe.meanLuminance).toBeGreaterThan(4);
    expect(browserErrors).toEqual([]);
  });
}
