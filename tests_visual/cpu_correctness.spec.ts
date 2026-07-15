import { expect, test } from '@playwright/test';

type SceneProbe = {
  meanLuminance: number;
  nonBlackRatio: number;
  roomLitRatio: number;
  brightPixelRatio: number;
  upperBrightPixels: number;
  quadrantNonBlackRatios: number[];
};

type FlashlightProbe = {
  changedPixelRatio: number;
  positiveEnergy: number;
  coneEnergyRatio: number;
  outerMeanEnergy: number;
  beamCenterX: number;
  beamCenterY: number;
  redToBlueEnergy: number;
};

const CPU_REFERENCE_QUERY =
  'seed=42&level=0&workers=0&renderer=cpu&cpu_preset=balanced' +
  '&rt_bake=0&rt_hiz=0&rt_mips=0&rt_beam_occlusion=0&rt_deferred=0&rt_ao=0' +
  '&rt_f2b=0&rt_cull=0';

async function waitForCapturedScene(page: import('@playwright/test').Page) {
  await page.waitForFunction(
    () => {
      const status = document.getElementById('status-msg')?.textContent ?? '';
      if (status.includes('Failed to start engine')) throw new Error(status);
      return (window as typeof window & { __sceneReady?: boolean }).__sceneReady === true;
    },
    null,
    { timeout: 75_000 },
  );
}

async function hideChrome(page: import('@playwright/test').Page) {
  await page.evaluate(() => {
    for (const id of ['overlay', 'hud', 'hud-section', 'debug-overlay', 'touch-ui']) {
      const element = document.getElementById(id);
      if (element) element.style.display = 'none';
    }
  });
}

function encoded(png: Buffer) {
  return `data:image/png;base64,${png.toString('base64')}`;
}

async function probeScene(
  page: import('@playwright/test').Page,
  screenshot: Buffer,
): Promise<SceneProbe> {
  return page.evaluate(async url => {
    const response = await fetch(url);
    const bitmap = await createImageBitmap(await response.blob());
    const canvas = document.createElement('canvas');
    canvas.width = bitmap.width;
    canvas.height = bitmap.height;
    const context = canvas.getContext('2d', { willReadFrequently: true });
    if (!context) throw new Error('CPU scene probe cannot create a 2D context');
    context.drawImage(bitmap, 0, 0);
    const pixels = context.getImageData(0, 0, canvas.width, canvas.height).data;

    let luminanceSum = 0;
    let nonBlack = 0;
    let roomLit = 0;
    let bright = 0;
    let upperBrightPixels = 0;
    const quadrantNonBlack = [0, 0, 0, 0];
    const quadrantCounts = [0, 0, 0, 0];
    for (let y = 0; y < canvas.height; y++) {
      for (let x = 0; x < canvas.width; x++) {
        const offset = (y * canvas.width + x) * 4;
        const luminance =
          pixels[offset] * 0.2126 + pixels[offset + 1] * 0.7152 + pixels[offset + 2] * 0.0722;
        const quadrant = Number(x >= canvas.width / 2) + 2 * Number(y >= canvas.height / 2);
        luminanceSum += luminance;
        nonBlack += Number(luminance >= 3);
        roomLit += Number(luminance >= 10);
        bright += Number(luminance >= 48);
        upperBrightPixels += Number(y < canvas.height * 0.55 && luminance >= 48);
        quadrantNonBlack[quadrant] += Number(luminance >= 3);
        quadrantCounts[quadrant]++;
      }
    }
    bitmap.close();
    const pixelCount = canvas.width * canvas.height;
    return {
      meanLuminance: luminanceSum / pixelCount,
      nonBlackRatio: nonBlack / pixelCount,
      roomLitRatio: roomLit / pixelCount,
      brightPixelRatio: bright / pixelCount,
      upperBrightPixels,
      quadrantNonBlackRatios: quadrantNonBlack.map(
        (count, index) => count / quadrantCounts[index],
      ),
    };
  }, encoded(screenshot));
}

async function probeFlashlight(
  page: import('@playwright/test').Page,
  offBefore: Buffer,
  flashlightOn: Buffer,
  offAfter: Buffer,
): Promise<FlashlightProbe> {
  return page.evaluate(async urls => {
    const decode = async (url: string) => {
      const response = await fetch(url);
      return createImageBitmap(await response.blob());
    };
    const images = await Promise.all(urls.map(decode));
    const [first] = images;
    if (images.some(image => image.width !== first.width || image.height !== first.height)) {
      throw new Error('flashlight comparison dimensions changed');
    }
    const canvas = document.createElement('canvas');
    canvas.width = first.width;
    canvas.height = first.height;
    const context = canvas.getContext('2d', { willReadFrequently: true });
    if (!context) throw new Error('flashlight probe cannot create a 2D context');
    const buffers = images.map(image => {
      context.clearRect(0, 0, canvas.width, canvas.height);
      context.drawImage(image, 0, 0);
      return context.getImageData(0, 0, canvas.width, canvas.height).data;
    });
    const [before, on, after] = buffers;

    // Average captures bracketing the flashlight frame. This rejects the
    // fluorescent fixture's time-varying flicker without weakening the beam
    // shape contract.
    let changedPixels = 0;
    let positiveEnergy = 0;
    let coneEnergy = 0;
    let outerEnergy = 0;
    let outerPixels = 0;
    let weightedX = 0;
    let weightedY = 0;
    let redEnergy = 0;
    let blueEnergy = 0;
    for (let y = 0; y < canvas.height; y++) {
      for (let x = 0; x < canvas.width; x++) {
        const offset = (y * canvas.width + x) * 4;
        const channelDelta = [0, 1, 2].map(
          channel => on[offset + channel] - (before[offset + channel] + after[offset + channel]) / 2,
        );
        const luminanceDelta =
          channelDelta[0] * 0.2126 + channelDelta[1] * 0.7152 + channelDelta[2] * 0.0722;
        const energy = Math.max(0, luminanceDelta);
        const nx = (x + 0.5) / canvas.width;
        const ny = (y + 0.5) / canvas.height;
        const inConeEnvelope = nx >= 0.25 && nx <= 0.75 && ny >= 0.12 && ny <= 0.88;
        const inOuterFrame = nx < 0.14 || nx > 0.86 || ny < 0.07 || ny > 0.93;

        changedPixels += Number(luminanceDelta >= 3);
        positiveEnergy += energy;
        coneEnergy += inConeEnvelope ? energy : 0;
        if (inOuterFrame) {
          outerEnergy += energy;
          outerPixels++;
        }
        weightedX += nx * energy;
        weightedY += ny * energy;
        redEnergy += Math.max(0, channelDelta[0]);
        blueEnergy += Math.max(0, channelDelta[2]);
      }
    }
    images.forEach(image => image.close());
    const pixelCount = canvas.width * canvas.height;
    return {
      changedPixelRatio: changedPixels / pixelCount,
      positiveEnergy,
      coneEnergyRatio: coneEnergy / Math.max(positiveEnergy, 1),
      outerMeanEnergy: outerEnergy / Math.max(outerPixels, 1),
      beamCenterX: weightedX / Math.max(positiveEnergy, 1),
      beamCenterY: weightedY / Math.max(positiveEnergy, 1),
      redToBlueEnergy: redEnergy / Math.max(blueEnergy, 1),
    };
  }, [encoded(offBefore), encoded(flashlightOn), encoded(offAfter)]);
}

test('CPU panels illuminate a visible room across the complete canvas', async ({ page }, testInfo) => {
  test.setTimeout(90_000);
  await page.setViewportSize({ width: 320, height: 180 });
  await page.goto(
    `/index.html?${CPU_REFERENCE_QUERY}&capture=1` +
      '&camera=6,1.7,31.2&yaw=-1.5708&pitch=-0.05',
  );
  await waitForCapturedScene(page);
  await hideChrome(page);

  const canvasGeometry = await page.locator('#view').evaluate(canvas => {
    const element = canvas as HTMLCanvasElement;
    const bounds = element.getBoundingClientRect();
    return {
      backingWidth: element.width,
      backingHeight: element.height,
      cssWidth: bounds.width,
      cssHeight: bounds.height,
    };
  });
  expect(canvasGeometry).toEqual({
    backingWidth: 80,
    backingHeight: 45,
    cssWidth: 320,
    cssHeight: 180,
  });

  const screenshot = await page.locator('#view').screenshot();
  await testInfo.attach('cpu-balanced-lit-room', { body: screenshot, contentType: 'image/png' });
  const scene = await probeScene(page, screenshot);

  // Bright upper pixels are the emissive ceiling panels. The broader lit
  // ratio and mean require their analytic light to reach room surfaces, not
  // merely render a handful of white emitter texels in an otherwise black
  // image. Every quadrant guards the reduced-framebuffer presentation path.
  expect(scene.upperBrightPixels).toBeGreaterThan(8);
  expect(scene.brightPixelRatio).toBeGreaterThan(0.001);
  expect(scene.roomLitRatio).toBeGreaterThan(0.08);
  expect(scene.nonBlackRatio).toBeGreaterThan(0.25);
  expect(scene.meanLuminance).toBeGreaterThan(4);
  for (const ratio of scene.quadrantNonBlackRatios) expect(ratio).toBeGreaterThan(0.12);
});

test('CPU flashlight is a centered finite warm cone and toggles cleanly', async ({ page }, testInfo) => {
  test.setTimeout(90_000);
  await page.setViewportSize({ width: 320, height: 180 });
  // Keep ordinary-play input listeners, but make its render clock as stable
  // as capture mode. Generation still advances once per animation frame;
  // only simulation time (and therefore fluorescent flicker) is frozen.
  await page.addInitScript(() => {
    const scheduleFrame = window.requestAnimationFrame.bind(window);
    window.requestAnimationFrame = callback => scheduleFrame(() => callback(1_000));
  });
  await page.goto(
    '/index.html?seed=42&level=0&workers=0&renderer=cpu&cpu_preset=balanced' +
      '&rt_bake=0&rt_beam_occlusion=1',
  );

  // Capture mode intentionally has no input listeners. In ordinary play the
  // hidden loading overlay is the engine's deterministic ready signal; wait
  // for all nine low-spec chunks to refine before comparing live frames.
  await page.waitForFunction(
    () => {
      const status = document.getElementById('status-msg')?.textContent ?? '';
      if (status.includes('Failed to start engine')) throw new Error(status);
      const text = document.getElementById('hud-chunks')?.textContent ?? '';
      const match = text.match(/^(\d+)\/(\d+)$/);
      return match !== null && Number(match[1]) === Number(match[2]) && Number(match[2]) >= 9;
    },
    null,
    { timeout: 75_000 },
  );
  await page.evaluate(async () => {
    const moduleUrl = new URL('./pkg/wasm_frontend.js', window.location.href).href;
    const wasm = (await import(moduleUrl)) as { set_render_scale(scale: number): void };
    wasm.set_render_scale(1);
    await new Promise<void>(resolve => requestAnimationFrame(() => requestAnimationFrame(() => resolve())));
  });
  await hideChrome(page);

  const offBefore = await page.locator('#view').screenshot();
  await page.keyboard.press('KeyF');
  await page.evaluate(() => new Promise<void>(resolve => {
    requestAnimationFrame(() => requestAnimationFrame(() => resolve()));
  }));
  const flashlightOn = await page.locator('#view').screenshot();
  await page.keyboard.press('KeyF');
  await page.evaluate(() => new Promise<void>(resolve => {
    requestAnimationFrame(() => requestAnimationFrame(() => resolve()));
  }));
  const offAfter = await page.locator('#view').screenshot();
  await testInfo.attach('cpu-flashlight-off', { body: offBefore, contentType: 'image/png' });
  await testInfo.attach('cpu-flashlight-on', { body: flashlightOn, contentType: 'image/png' });

  expect(offAfter.equals(offBefore), 'turning the flashlight off restores the same frame').toBe(true);
  const beam = await probeFlashlight(page, offBefore, flashlightOn, offAfter);
  expect(beam.changedPixelRatio).toBeGreaterThan(0.002);
  expect(beam.positiveEnergy).toBeGreaterThan(1_000);
  expect(beam.coneEnergyRatio).toBeGreaterThan(0.65);
  expect(beam.outerMeanEnergy).toBeLessThan(1.5);
  expect(beam.beamCenterX).toBeGreaterThan(0.36);
  expect(beam.beamCenterX).toBeLessThan(0.64);
  expect(beam.beamCenterY).toBeGreaterThan(0.20);
  expect(beam.beamCenterY).toBeLessThan(0.82);
  expect(beam.redToBlueEnergy).toBeGreaterThan(1.05);
});

test('CPU preset and custom URL settings form one live validated contract', async ({ page }) => {
  test.setTimeout(90_000);
  await page.setViewportSize({ width: 320, height: 180 });
  await page.goto(
    '/index.html?seed=42&workers=0&renderer=cpu&capture=1&smoke=1&cpu_preset=quality',
  );
  await waitForCapturedScene(page);

  const values = await page.evaluate(() => {
    const value = (id: string) => {
      const control = document.getElementById(id);
      if (!(control instanceof HTMLInputElement || control instanceof HTMLSelectElement)) {
        throw new Error(`missing settings control #${id}`);
      }
      return control.value;
    };
    return {
      preset: value('setting-cpu-preset'),
      scale: value('setting-cpu-scale'),
      lod: value('setting-cpu-lod'),
      splat: value('setting-cpu-splat'),
      depth: value('setting-cpu-depth'),
      occupancy: value('setting-cpu-mip-occupancy'),
      distance: value('setting-cpu-dist'),
      shadows: value('setting-cpu-shadows'),
    };
  });
  expect(values).toEqual({
    preset: 'quality',
    scale: '1.5',
    lod: '0.65',
    splat: '4',
    depth: '6',
    occupancy: '0.12',
    distance: '128',
    shadows: 'hero',
  });

  const canvasPresentation = async () => page.locator('#view').evaluate(canvas => {
    const element = canvas as HTMLCanvasElement;
    const bounds = element.getBoundingClientRect();
    return {
      backingWidth: element.width,
      backingHeight: element.height,
      cssLeft: bounds.left,
      cssTop: bounds.top,
      cssWidth: bounds.width,
      cssHeight: bounds.height,
    };
  });
  expect(await canvasPresentation()).toEqual({
    backingWidth: 120,
    backingHeight: 67,
    cssLeft: 0,
    cssTop: 0,
    cssWidth: 320,
    cssHeight: 180,
  });

  // A live preset switch crosses the DOM -> WASM boundary and resizes the
  // CPU backing store without changing its full-viewport CSS presentation.
  await page.click('#settings-btn');
  await page.locator('[data-tab="tab-cpu"]').click();
  await page.selectOption('#setting-cpu-preset', 'performance');
  await page.waitForFunction(() => {
    const canvas = document.getElementById('view') as HTMLCanvasElement | null;
    return canvas?.width === 40 && canvas.height === 22;
  });
  expect(await canvasPresentation()).toEqual({
    backingWidth: 40,
    backingHeight: 22,
    cssLeft: 0,
    cssTop: 0,
    cssWidth: 320,
    cssHeight: 180,
  });

  // Exercise an odd, non-16:9 page size so integer truncation and a stale
  // one-size backing store cannot accidentally satisfy the assertion. The
  // performance profile is 12.5% per axis, while CSS must still cover every
  // viewport pixel from the top-left origin.
  await page.setViewportSize({ width: 997, height: 611 });
  await page.waitForFunction(() => {
    const canvas = document.getElementById('view') as HTMLCanvasElement | null;
    if (!canvas) return false;
    const bounds = canvas.getBoundingClientRect();
    return canvas.width === 124 && canvas.height === 76 &&
      bounds.left === 0 && bounds.top === 0 &&
      bounds.width === 997 && bounds.height === 611;
  });
  expect(await canvasPresentation()).toEqual({
    backingWidth: 124,
    backingHeight: 76,
    cssLeft: 0,
    cssTop: 0,
    cssWidth: 997,
    cssHeight: 611,
  });

  const setRange = async (id: string, value: string) => {
    await page.locator(id).evaluate((input, next) => {
      const element = input as HTMLInputElement;
      element.value = next;
      element.dispatchEvent(new Event('input', { bubbles: true }));
    }, value);
  };
  await setRange('#setting-cpu-scale', '1.25');
  await page.waitForFunction(() => {
    const canvas = document.getElementById('view') as HTMLCanvasElement | null;
    return canvas?.width === 311 && canvas.height === 190;
  });
  expect(await canvasPresentation()).toEqual({
    backingWidth: 311,
    backingHeight: 190,
    cssLeft: 0,
    cssTop: 0,
    cssWidth: 997,
    cssHeight: 611,
  });
  await setRange('#setting-cpu-lod', '0.75');
  await setRange('#setting-cpu-splat', '3.5');
  await setRange('#setting-cpu-depth', '7');
  await setRange('#setting-cpu-mip-occupancy', '0.08');
  await setRange('#setting-cpu-dist', '144');
  await page.selectOption('#setting-cpu-shadows', 'full');
  await expect(page.locator('#setting-cpu-preset')).toHaveValue('custom');

  await Promise.all([
    page.waitForURL(url => new URL(url).searchParams.get('cpu_preset') === 'custom'),
    page.click('#apply-settings-btn'),
  ]);
  const params = new URL(page.url()).searchParams;
  expect(Object.fromEntries([
    'cpu_preset',
    'cpu_scale',
    'cpu_lod_px',
    'cpu_splat_radius_px',
    'cpu_virtual_depth',
    'cpu_mip_occupancy',
    'cpu_range',
    'cpu_shadows',
  ].map(key => [key, params.get(key)]))).toEqual({
    cpu_preset: 'custom',
    cpu_scale: '1.25',
    cpu_lod_px: '0.75',
    cpu_splat_radius_px: '3.5',
    cpu_virtual_depth: '7',
    cpu_mip_occupancy: '0.08',
    cpu_range: '144',
    cpu_shadows: 'full',
  });
});

test('CPU URL overrides stay active without leaking into device preferences', async ({ page }) => {
  test.setTimeout(90_000);
  await page.addInitScript(() => localStorage.removeItem('vackrooms_prefs'));
  await page.goto(
    '/index.html?seed=42&workers=0&renderer=cpu&capture=1&smoke=1' +
      '&cpu_preset=custom&cpu_scale=1.25&cpu_shadows=hero',
  );
  await waitForCapturedScene(page);

  await expect(page.locator('#setting-cpu-preset')).toHaveValue('custom');
  await expect(page.locator('#setting-cpu-scale')).toHaveValue('1.25');
  await expect(page.locator('#setting-cpu-shadows')).toHaveValue('hero');

  // Saving an unrelated live preference must serialize the device defaults,
  // not the CPU profile borrowed from this shared diagnostic URL.
  await page.locator('#setting-sensitivity').evaluate(input => {
    const element = input as HTMLInputElement;
    element.value = '1.1';
    element.dispatchEvent(new Event('input', { bubbles: true }));
  });
  const stored = await page.evaluate(() =>
    JSON.parse(localStorage.getItem('vackrooms_prefs') ?? '{}'),
  );
  expect({
    preset: stored.cpuPreset,
    scale: stored.cpuScale,
    shadows: stored.cpuShadows,
  }).toEqual({ preset: 'balanced', scale: 1, shadows: 'off' });
});
