import { expect, test } from '@playwright/test';

type RenderProbe = {
  meanLuminance: number;
  brightPixelRatio: number;
  nonBlackPixelRatio: number;
  colorBuckets: number;
};

type PixelDifference = {
  changedPixels: number;
  maximumChannelDelta: number;
};

const referenceBackends = [
  {
    name: 'surface-direct-reference',
    renderer: 'surface',
    toggles: 'rt_bake=0&rt_cull=0&rt_shadows=0&rt_dither=0&rt_timer=0',
    equivalentOptimizations: ['cull'],
    viewport: { width: 160, height: 90 },
    timeoutMs: 90_000,
  },
  {
    name: 'raymarch-direct-dda-reference',
    renderer: 'raymarch',
    toggles: 'rt_bake=0&rt_skip=0&rt_f2b=0&rt_shadows=0&rt_dither=0&rt_timer=0',
    equivalentOptimizations: ['skip', 'f2b'],
    viewport: { width: 160, height: 90 },
    // The correctness baseline deliberately disables empty-space skipping.
    // Software WebGL in CI can need more than the interactive-path timeout
    // to capture the two live optimization comparisons.
    timeoutMs: 180_000,
  },
  {
    name: 'raymarch-occluded-direct-reference',
    renderer: 'raymarch',
    toggles: 'rt_bake=0&rt_skip=1&rt_f2b=1&rt_shadows=1&rt_dither=0&rt_timer=0',
    equivalentOptimizations: [],
    viewport: { width: 128, height: 72 },
    timeoutMs: 90_000,
  },
  {
    name: 'cpu-splat-direct-reference',
    renderer: 'cpu',
    toggles:
      'rt_bake=0&rt_hiz=0&rt_mips=0&rt_beam_occlusion=0&rt_f2b=0&rt_cull=0',
    equivalentOptimizations: ['hiz', 'f2b', 'cull'],
    // Balanced CPU quality renders at one quarter of this backing size and
    // CSS presents the complete result over the viewport.
    viewport: { width: 320, height: 180 },
    timeoutMs: 90_000,
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

async function compareScreenshots(
  page: import('@playwright/test').Page,
  left: Buffer,
  right: Buffer,
): Promise<PixelDifference> {
  const encode = (png: Buffer) => `data:image/png;base64,${png.toString('base64')}`;
  return page.evaluate(async ([leftUrl, rightUrl]) => {
    const decode = async (url: string) => {
      const response = await fetch(url);
      return createImageBitmap(await response.blob());
    };
    const [leftImage, rightImage] = await Promise.all([decode(leftUrl), decode(rightUrl)]);
    if (leftImage.width !== rightImage.width || leftImage.height !== rightImage.height) {
      throw new Error('render comparison dimensions differ');
    }
    const canvas = document.createElement('canvas');
    canvas.width = leftImage.width;
    canvas.height = leftImage.height;
    const context = canvas.getContext('2d', { willReadFrequently: true });
    if (!context) throw new Error('2D comparison context is unavailable');
    context.drawImage(leftImage, 0, 0);
    const leftPixels = context.getImageData(0, 0, canvas.width, canvas.height).data;
    context.clearRect(0, 0, canvas.width, canvas.height);
    context.drawImage(rightImage, 0, 0);
    const rightPixels = context.getImageData(0, 0, canvas.width, canvas.height).data;
    let changedPixels = 0;
    let maximumChannelDelta = 0;
    for (let offset = 0; offset < leftPixels.length; offset += 4) {
      let pixelChanged = false;
      for (let channel = 0; channel < 3; channel++) {
        const delta = Math.abs(leftPixels[offset + channel] - rightPixels[offset + channel]);
        maximumChannelDelta = Math.max(maximumChannelDelta, delta);
        pixelChanged ||= delta !== 0;
      }
      changedPixels += Number(pixelChanged);
    }
    leftImage.close();
    rightImage.close();
    return { changedPixels, maximumChannelDelta };
  }, [encode(left), encode(right)] as const);
}

for (const backend of referenceBackends) {
  test(`${backend.name} renders emissive fixtures and their room`, async ({ page }, testInfo) => {
    test.setTimeout(backend.timeoutMs);
    const browserErrors: string[] = [];
    page.on('pageerror', error => browserErrors.push(error.message));
    page.on('console', message => {
      if (message.type() === 'error' || /GL_INVALID|shader .*error/i.test(message.text())) {
        browserErrors.push(message.text());
      }
    });

    await page.setViewportSize(backend.viewport);
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
    // Keep PNG analysis off the page continuously executing the deliberately
    // slow reference shader; software WebGL can otherwise starve evaluation.
    const probePage = await page.context().newPage();
    const probe = await probeScreenshot(probePage, screenshot);

    // A dead panel path produces a nearly black canvas with only a few
    // emissive texels. Require both visible fixtures and illuminated room
    // geometry, while keeping the contract independent of exact pixels.
    expect(probe.colorBuckets).toBeGreaterThan(8);
    expect(probe.brightPixelRatio).toBeGreaterThan(0.0005);
    expect(probe.nonBlackPixelRatio).toBeGreaterThan(0.08);
    expect(probe.meanLuminance).toBeGreaterThan(4);
    expect(browserErrors).toEqual([]);

    // Correctness-preserving optimizations are required to be image-neutral.
    // Flip them live so both captures use the identical resident world and
    // camera, then compare the canvas bytes rather than a subjective metric.
    for (const name of backend.equivalentOptimizations) {
      await page.evaluate(toggleName => {
        const input = document.getElementById(
          `setting-rt-${toggleName}`,
        ) as HTMLInputElement | null;
        if (!input) throw new Error(`missing render toggle: ${toggleName}`);
        input.checked = true;
        input.dispatchEvent(new Event('change', { bubbles: true }));
      }, name);
      await page.evaluate(() => new Promise<void>(resolve => {
        requestAnimationFrame(() => requestAnimationFrame(() => resolve()));
      }));
      const optimized = await page.locator('#view').screenshot({
        path: testInfo.outputPath(`${backend.name}-${name}.png`),
      });
      expect(
        await compareScreenshots(probePage, screenshot, optimized),
        `${name} must preserve every rendered pixel`,
      ).toEqual({ changedPixels: 0, maximumChannelDelta: 0 });
    }
    await probePage.close();
  });
}
