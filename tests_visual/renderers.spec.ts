import { expect, test } from '@playwright/test';

const backends = [
  {
    query: 'renderer=surface&rt_bake=0&rt_cull=0&rt_shadows=0&rt_dither=0&rt_timer=0',
    label: 'GPU surfaces',
    disabled: ['bake', 'cull', 'shadows', 'dither', 'timer'],
    viewport: { width: 320, height: 240 },
  },
  {
    query:
      'renderer=splat&rt_cull=0&rt_shadows=0&rt_cells=0&rt_budget=0&rt_dither=0&rt_timer=0',
    label: 'GPU face splats',
    disabled: ['cull', 'shadows', 'cells', 'budget', 'dither', 'timer'],
    viewport: { width: 320, height: 240 },
  },
  {
    query: 'renderer=raymarch&rt_bake=0&rt_skip=0&rt_f2b=0&rt_dither=0&rt_timer=0',
    label: 'GPU raymarch (debug)',
    disabled: ['bake', 'skip', 'f2b', 'dither', 'timer'],
    // Finest-cell DDA is the intentionally slow diagnostic path. Its image
    // correctness is covered at 160x90 in gpu_correctness.spec.ts; this case
    // only needs enough pixels to prove the backend boots and draws.
    viewport: { width: 96, height: 54 },
  },
  {
    query: 'renderer=cpu&rt_hiz=0&rt_f2b=0&rt_mips=0&rt_beam_occlusion=0&rt_cull=0',
    label: 'CPU splat',
    disabled: ['hiz', 'f2b', 'mips', 'beam_occlusion', 'cull'],
    viewport: { width: 320, height: 240 },
  },
] as const;

async function renderedImageStats(page: import('@playwright/test').Page) {
  const png = await page.locator('#view').screenshot();
  const dataUrl = `data:image/png;base64,${png.toString('base64')}`;

  return page.evaluate(async encodedImage => {
    const response = await fetch(encodedImage);
    const bitmap = await createImageBitmap(await response.blob());
    const sample = document.createElement('canvas');
    sample.width = 64;
    sample.height = 48;
    const context = sample.getContext('2d', { willReadFrequently: true });
    if (!context) throw new Error('2D screenshot probe is unavailable');
    context.drawImage(bitmap, 0, 0, sample.width, sample.height);

    const pixels = context.getImageData(0, 0, sample.width, sample.height).data;
    const colors = new Set<number>();
    let darkest = 255;
    let brightest = 0;
    for (let i = 0; i < pixels.length; i += 4) {
      const red = pixels[i];
      const green = pixels[i + 1];
      const blue = pixels[i + 2];
      const luminance = Math.round(red * 0.2126 + green * 0.7152 + blue * 0.0722);
      darkest = Math.min(darkest, luminance);
      brightest = Math.max(brightest, luminance);
      colors.add(((red >> 4) << 8) | ((green >> 4) << 4) | (blue >> 4));
    }
    bitmap.close();
    return { colorBuckets: colors.size, luminanceRange: brightest - darkest };
  }, dataUrl);
}

for (const backend of backends) {
  test(`${backend.label} reference path boots and renders`, async ({ page }) => {
    test.setTimeout(90_000);
    const pageErrors: string[] = [];
    const consoleErrors: string[] = [];
    page.on('pageerror', error => pageErrors.push(error.message));
    page.on('console', message => {
      const text = message.text();
      if (message.type() === 'error' || /GL_INVALID|shader .*error/i.test(text)) {
        consoleErrors.push(text);
      }
    });

    // Reference paths intentionally do more work. A small but still useful
    // framebuffer keeps the all-off raymarch/CPU cases deterministic in CI.
    await page.setViewportSize(backend.viewport);

    await page.goto(
      `/index.html?seed=42&level=0&workers=0&capture=1&smoke=1&${backend.query}` +
      '&camera=6,1.7,31.2&yaw=-1.5708&pitch=-0.05',
    );

    await page.waitForFunction(
      () => {
        const status = document.getElementById('status-msg')?.textContent ?? '';
        if (status.includes('Failed to start engine')) throw new Error(status);
        return (window as typeof window & { __sceneReady?: boolean }).__sceneReady === true;
      },
      null,
      { timeout: 75_000 },
    );

    await expect(page.locator('#hud-renderer')).toHaveText(backend.label);
    await expect(page.locator('#status-msg')).not.toContainText('Failed to start engine');

    const effectiveToggles = await page.evaluate(async names => {
      const moduleUrl = new URL('./pkg/wasm_frontend.js', window.location.href).href;
      const wasm = (await import(moduleUrl)) as {
        render_toggle_enabled(name: string): boolean | undefined;
      };
      return names.map(name => wasm.render_toggle_enabled(name));
    }, backend.disabled);
    expect(effectiveToggles).toEqual(backend.disabled.map(() => false));

    // A successful boot label is not proof of a successful shader/draw. The
    // compositor screenshot must contain real scene variation rather than a
    // uniform clear/error canvas.
    const image = await renderedImageStats(page);
    expect(image.colorBuckets).toBeGreaterThan(3);
    expect(image.luminanceRange).toBeGreaterThan(4);
    expect(consoleErrors).toEqual([]);
    expect(pageErrors).toEqual([]);
  });
}
