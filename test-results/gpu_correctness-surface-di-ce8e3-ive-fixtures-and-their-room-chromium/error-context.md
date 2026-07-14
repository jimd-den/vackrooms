# Instructions

- Following Playwright test failed.
- Explain why, be concise, respect Playwright best practices.
- Provide a snippet of code with the fix, if possible.

# Test info

- Name: gpu_correctness.spec.ts >> surface-direct-reference renders emissive fixtures and their room
- Location: tests_visual/gpu_correctness.spec.ts:66:7

# Error details

```
Error: expect(received).toBeGreaterThan(expected)

Expected: > 8
Received:   6
```

# Test source

```ts
  5   |   brightPixelRatio: number;
  6   |   nonBlackPixelRatio: number;
  7   |   colorBuckets: number;
  8   | };
  9   | 
  10  | const referenceBackends = [
  11  |   {
  12  |     name: 'surface-direct-reference',
  13  |     renderer: 'surface',
  14  |     toggles: 'rt_bake=0&rt_cull=0&rt_shadows=0&rt_dither=0&rt_timer=0',
  15  |   },
  16  |   {
  17  |     name: 'raymarch-direct-dda-reference',
  18  |     renderer: 'raymarch',
  19  |     toggles: 'rt_bake=0&rt_skip=0&rt_f2b=0&rt_dither=0&rt_timer=0',
  20  |   },
  21  | ] as const;
  22  | 
  23  | async function probeScreenshot(
  24  |   page: import('@playwright/test').Page,
  25  |   png: Buffer,
  26  | ): Promise<RenderProbe> {
  27  |   const dataUrl = `data:image/png;base64,${png.toString('base64')}`;
  28  |   return page.evaluate(async encodedImage => {
  29  |     const response = await fetch(encodedImage);
  30  |     const bitmap = await createImageBitmap(await response.blob());
  31  |     const sample = document.createElement('canvas');
  32  |     sample.width = 80;
  33  |     sample.height = 45;
  34  |     const context = sample.getContext('2d', { willReadFrequently: true });
  35  |     if (!context) throw new Error('2D render probe is unavailable');
  36  |     context.drawImage(bitmap, 0, 0, sample.width, sample.height);
  37  | 
  38  |     const pixels = context.getImageData(0, 0, sample.width, sample.height).data;
  39  |     const colors = new Set<number>();
  40  |     let luminanceSum = 0;
  41  |     let bright = 0;
  42  |     let nonBlack = 0;
  43  |     const count = pixels.length / 4;
  44  |     for (let offset = 0; offset < pixels.length; offset += 4) {
  45  |       const red = pixels[offset];
  46  |       const green = pixels[offset + 1];
  47  |       const blue = pixels[offset + 2];
  48  |       const luminance = red * 0.2126 + green * 0.7152 + blue * 0.0722;
  49  |       luminanceSum += luminance;
  50  |       if (luminance >= 48) bright++;
  51  |       if (luminance >= 3) nonBlack++;
  52  |       colors.add(((red >> 4) << 8) | ((green >> 4) << 4) | (blue >> 4));
  53  |     }
  54  |     const result = {
  55  |       meanLuminance: luminanceSum / count,
  56  |       brightPixelRatio: bright / count,
  57  |       nonBlackPixelRatio: nonBlack / count,
  58  |       colorBuckets: colors.size,
  59  |     };
  60  |     bitmap.close();
  61  |     return result;
  62  |   }, dataUrl);
  63  | }
  64  | 
  65  | for (const backend of referenceBackends) {
  66  |   test(`${backend.name} renders emissive fixtures and their room`, async ({ page }, testInfo) => {
  67  |     test.setTimeout(90_000);
  68  |     const browserErrors: string[] = [];
  69  |     page.on('pageerror', error => browserErrors.push(error.message));
  70  |     page.on('console', message => {
  71  |       if (message.type() === 'error' || /GL_INVALID|shader .*error/i.test(message.text())) {
  72  |         browserErrors.push(message.text());
  73  |       }
  74  |     });
  75  | 
  76  |     await page.setViewportSize({ width: 320, height: 180 });
  77  |     await page.goto(
  78  |       `/index.html?seed=42&level=0&workers=0&capture=1&smoke=1` +
  79  |       `&renderer=${backend.renderer}&${backend.toggles}` +
  80  |       '&camera=6,1.7,31.2&yaw=-1.5708&pitch=-0.05',
  81  |     );
  82  |     await page.waitForFunction(
  83  |       () => {
  84  |         const status = document.getElementById('status-msg')?.textContent ?? '';
  85  |         if (status.includes('Failed to start engine')) throw new Error(status);
  86  |         return (window as typeof window & { __sceneReady?: boolean }).__sceneReady === true;
  87  |       },
  88  |       null,
  89  |       { timeout: 60_000 },
  90  |     );
  91  | 
  92  |     await page.evaluate(() => {
  93  |       for (const id of ['overlay', 'hud', 'hud-section', 'debug-overlay']) {
  94  |         const element = document.getElementById(id);
  95  |         if (element) element.style.display = 'none';
  96  |       }
  97  |     });
  98  |     const screenshotPath = testInfo.outputPath(`${backend.name}.png`);
  99  |     const screenshot = await page.locator('#view').screenshot({ path: screenshotPath });
  100 |     const probe = await probeScreenshot(page, screenshot);
  101 | 
  102 |     // A dead panel path produces a nearly black canvas with only a few
  103 |     // emissive texels. Require both visible fixtures and illuminated room
  104 |     // geometry, while keeping the contract independent of exact pixels.
> 105 |     expect(probe.colorBuckets).toBeGreaterThan(8);
      |                                ^ Error: expect(received).toBeGreaterThan(expected)
  106 |     expect(probe.brightPixelRatio).toBeGreaterThan(0.0005);
  107 |     expect(probe.nonBlackPixelRatio).toBeGreaterThan(0.08);
  108 |     expect(probe.meanLuminance).toBeGreaterThan(4);
  109 |     expect(browserErrors).toEqual([]);
  110 |   });
  111 | }
  112 | 
```