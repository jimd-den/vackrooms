# Instructions

- Following Playwright test failed.
- Explain why, be concise, respect Playwright best practices.
- Provide a snippet of code with the fix, if possible.

# Test info

- Name: renderers.spec.ts >> GPU face splats reference path boots and renders
- Location: tests_visual/renderers.spec.ts:60:7

# Error details

```
Test timeout of 60000ms exceeded.
```

```
Error: locator.screenshot: Test timeout of 60000ms exceeded.
Call log:
  - taking element screenshot
  - waiting for fonts to load...
  - fonts loaded
  - attempting scroll into view action
    - waiting for element to be stable

```

# Page snapshot

```yaml
- generic [active] [ref=e1]:
  - generic:
    - text: DISTANCE 0 m
    - text: Renderer GPU face splats
    - text: FPS —
    - text: Chunks 0
    - text: SVO nodes 0
    - text: Res scale —
    - text: GPU —
  - generic: …
  - generic [ref=e3] [cursor=pointer]:
    - heading "VACKROOMS" [level=1] [ref=e4]
    - paragraph [ref=e5]:
      - text: Click to enter — WASD move · mouse look · F flashlight · G drop flare · ESC release
      - text: "Touch: tap to enter — left thumb move · right thumb look · 🔥 drop flare"
    - button "Settings" [ref=e6]
```

# Test source

```ts
  1   | import { expect, test } from '@playwright/test';
  2   | 
  3   | const backends = [
  4   |   {
  5   |     query: 'renderer=surface&rt_cull=0&rt_shadows=0&rt_dither=0&rt_timer=0',
  6   |     label: 'GPU surfaces',
  7   |     disabled: ['cull', 'shadows', 'dither', 'timer'],
  8   |   },
  9   |   {
  10  |     query:
  11  |       'renderer=splat&rt_cull=0&rt_shadows=0&rt_cells=0&rt_budget=0&rt_dither=0&rt_timer=0',
  12  |     label: 'GPU face splats',
  13  |     disabled: ['cull', 'shadows', 'cells', 'budget', 'dither', 'timer'],
  14  |   },
  15  |   {
  16  |     query: 'renderer=raymarch&rt_f2b=0&rt_dither=0&rt_timer=0',
  17  |     label: 'GPU raymarch (debug)',
  18  |     disabled: ['f2b', 'dither', 'timer'],
  19  |   },
  20  |   {
  21  |     query: 'renderer=cpu&rt_hiz=0&rt_f2b=0&rt_mips=0&rt_beam_occlusion=0&rt_cull=0',
  22  |     label: 'CPU splat',
  23  |     disabled: ['hiz', 'f2b', 'mips', 'beam_occlusion', 'cull'],
  24  |   },
  25  | ] as const;
  26  | 
  27  | async function renderedImageStats(page: import('@playwright/test').Page) {
> 28  |   const png = await page.locator('#view').screenshot();
      |                                           ^ Error: locator.screenshot: Test timeout of 60000ms exceeded.
  29  |   const dataUrl = `data:image/png;base64,${png.toString('base64')}`;
  30  | 
  31  |   return page.evaluate(async encodedImage => {
  32  |     const response = await fetch(encodedImage);
  33  |     const bitmap = await createImageBitmap(await response.blob());
  34  |     const sample = document.createElement('canvas');
  35  |     sample.width = 64;
  36  |     sample.height = 48;
  37  |     const context = sample.getContext('2d', { willReadFrequently: true });
  38  |     if (!context) throw new Error('2D screenshot probe is unavailable');
  39  |     context.drawImage(bitmap, 0, 0, sample.width, sample.height);
  40  | 
  41  |     const pixels = context.getImageData(0, 0, sample.width, sample.height).data;
  42  |     const colors = new Set<number>();
  43  |     let darkest = 255;
  44  |     let brightest = 0;
  45  |     for (let i = 0; i < pixels.length; i += 4) {
  46  |       const red = pixels[i];
  47  |       const green = pixels[i + 1];
  48  |       const blue = pixels[i + 2];
  49  |       const luminance = Math.round(red * 0.2126 + green * 0.7152 + blue * 0.0722);
  50  |       darkest = Math.min(darkest, luminance);
  51  |       brightest = Math.max(brightest, luminance);
  52  |       colors.add(((red >> 4) << 8) | ((green >> 4) << 4) | (blue >> 4));
  53  |     }
  54  |     bitmap.close();
  55  |     return { colorBuckets: colors.size, luminanceRange: brightest - darkest };
  56  |   }, dataUrl);
  57  | }
  58  | 
  59  | for (const backend of backends) {
  60  |   test(`${backend.label} reference path boots and renders`, async ({ page }) => {
  61  |     test.setTimeout(60_000);
  62  |     const pageErrors: string[] = [];
  63  |     const consoleErrors: string[] = [];
  64  |     page.on('pageerror', error => pageErrors.push(error.message));
  65  |     page.on('console', message => {
  66  |       const text = message.text();
  67  |       if (message.type() === 'error' || /GL_INVALID|shader .*error/i.test(text)) {
  68  |         consoleErrors.push(text);
  69  |       }
  70  |     });
  71  | 
  72  |     // Reference paths intentionally do more work. A small but still useful
  73  |     // framebuffer keeps the all-off raymarch/CPU cases deterministic in CI.
  74  |     await page.setViewportSize({ width: 320, height: 240 });
  75  | 
  76  |     await page.goto(
  77  |       `/index.html?seed=42&level=0&workers=0&capture=1&smoke=1&${backend.query}` +
  78  |       '&camera=6,1.7,31.2&yaw=-1.5708&pitch=-0.05',
  79  |     );
  80  | 
  81  |     await page.waitForFunction(
  82  |       () => {
  83  |         const status = document.getElementById('status-msg')?.textContent ?? '';
  84  |         if (status.includes('Failed to start engine')) throw new Error(status);
  85  |         return (window as typeof window & { __sceneReady?: boolean }).__sceneReady === true;
  86  |       },
  87  |       null,
  88  |       { timeout: 45_000 },
  89  |     );
  90  | 
  91  |     await expect(page.locator('#hud-renderer')).toHaveText(backend.label);
  92  |     await expect(page.locator('#status-msg')).not.toContainText('Failed to start engine');
  93  | 
  94  |     const effectiveToggles = await page.evaluate(async names => {
  95  |       const moduleUrl = new URL('./pkg/wasm_frontend.js', window.location.href).href;
  96  |       const wasm = (await import(moduleUrl)) as {
  97  |         render_toggle_enabled(name: string): boolean | undefined;
  98  |       };
  99  |       return names.map(name => wasm.render_toggle_enabled(name));
  100 |     }, backend.disabled);
  101 |     expect(effectiveToggles).toEqual(backend.disabled.map(() => false));
  102 | 
  103 |     // A successful boot label is not proof of a successful shader/draw. The
  104 |     // compositor screenshot must contain real scene variation rather than a
  105 |     // uniform clear/error canvas.
  106 |     const image = await renderedImageStats(page);
  107 |     expect(image.colorBuckets).toBeGreaterThan(3);
  108 |     expect(image.luminanceRange).toBeGreaterThan(4);
  109 |     expect(consoleErrors).toEqual([]);
  110 |     expect(pageErrors).toEqual([]);
  111 |   });
  112 | }
  113 | 
```