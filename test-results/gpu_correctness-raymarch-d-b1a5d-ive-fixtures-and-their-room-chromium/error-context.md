# Instructions

- Following Playwright test failed.
- Explain why, be concise, respect Playwright best practices.
- Provide a snippet of code with the fix, if possible.

# Test info

- Name: gpu_correctness.spec.ts >> raymarch-direct-dda-reference renders emissive fixtures and their room
- Location: tests_visual/gpu_correctness.spec.ts:124:7

# Error details

```
Error: skip must preserve every rendered pixel

expect(received).toEqual(expected) // deep equality

- Expected  - 2
+ Received  + 2

  Object {
-   "changedPixels": 0,
-   "maximumChannelDelta": 0,
+   "changedPixels": 1,
+   "maximumChannelDelta": 55,
  }
```

# Test source

```ts
  90  |       return createImageBitmap(await response.blob());
  91  |     };
  92  |     const [leftImage, rightImage] = await Promise.all([decode(leftUrl), decode(rightUrl)]);
  93  |     if (leftImage.width !== rightImage.width || leftImage.height !== rightImage.height) {
  94  |       throw new Error('render comparison dimensions differ');
  95  |     }
  96  |     const canvas = document.createElement('canvas');
  97  |     canvas.width = leftImage.width;
  98  |     canvas.height = leftImage.height;
  99  |     const context = canvas.getContext('2d', { willReadFrequently: true });
  100 |     if (!context) throw new Error('2D comparison context is unavailable');
  101 |     context.drawImage(leftImage, 0, 0);
  102 |     const leftPixels = context.getImageData(0, 0, canvas.width, canvas.height).data;
  103 |     context.clearRect(0, 0, canvas.width, canvas.height);
  104 |     context.drawImage(rightImage, 0, 0);
  105 |     const rightPixels = context.getImageData(0, 0, canvas.width, canvas.height).data;
  106 |     let changedPixels = 0;
  107 |     let maximumChannelDelta = 0;
  108 |     for (let offset = 0; offset < leftPixels.length; offset += 4) {
  109 |       let pixelChanged = false;
  110 |       for (let channel = 0; channel < 3; channel++) {
  111 |         const delta = Math.abs(leftPixels[offset + channel] - rightPixels[offset + channel]);
  112 |         maximumChannelDelta = Math.max(maximumChannelDelta, delta);
  113 |         pixelChanged ||= delta !== 0;
  114 |       }
  115 |       changedPixels += Number(pixelChanged);
  116 |     }
  117 |     leftImage.close();
  118 |     rightImage.close();
  119 |     return { changedPixels, maximumChannelDelta };
  120 |   }, [encode(left), encode(right)] as const);
  121 | }
  122 | 
  123 | for (const backend of referenceBackends) {
  124 |   test(`${backend.name} renders emissive fixtures and their room`, async ({ page }, testInfo) => {
  125 |     test.setTimeout(90_000);
  126 |     const browserErrors: string[] = [];
  127 |     page.on('pageerror', error => browserErrors.push(error.message));
  128 |     page.on('console', message => {
  129 |       if (message.type() === 'error' || /GL_INVALID|shader .*error/i.test(message.text())) {
  130 |         browserErrors.push(message.text());
  131 |       }
  132 |     });
  133 | 
  134 |     await page.setViewportSize(backend.viewport);
  135 |     await page.goto(
  136 |       `/index.html?seed=42&level=0&workers=0&capture=1&smoke=1` +
  137 |       `&renderer=${backend.renderer}&${backend.toggles}` +
  138 |       '&camera=6,1.7,31.2&yaw=-1.5708&pitch=-0.05',
  139 |     );
  140 |     await page.waitForFunction(
  141 |       () => {
  142 |         const status = document.getElementById('status-msg')?.textContent ?? '';
  143 |         if (status.includes('Failed to start engine')) throw new Error(status);
  144 |         return (window as typeof window & { __sceneReady?: boolean }).__sceneReady === true;
  145 |       },
  146 |       null,
  147 |       { timeout: 60_000 },
  148 |     );
  149 | 
  150 |     await page.evaluate(() => {
  151 |       for (const id of ['overlay', 'hud', 'hud-section', 'debug-overlay']) {
  152 |         const element = document.getElementById(id);
  153 |         if (element) element.style.display = 'none';
  154 |       }
  155 |     });
  156 |     const screenshotPath = testInfo.outputPath(`${backend.name}.png`);
  157 |     const screenshot = await page.locator('#view').screenshot({ path: screenshotPath });
  158 |     const probe = await probeScreenshot(page, screenshot);
  159 | 
  160 |     // A dead panel path produces a nearly black canvas with only a few
  161 |     // emissive texels. Require both visible fixtures and illuminated room
  162 |     // geometry, while keeping the contract independent of exact pixels.
  163 |     expect(probe.colorBuckets).toBeGreaterThan(8);
  164 |     expect(probe.brightPixelRatio).toBeGreaterThan(0.0005);
  165 |     expect(probe.nonBlackPixelRatio).toBeGreaterThan(0.08);
  166 |     expect(probe.meanLuminance).toBeGreaterThan(4);
  167 |     expect(browserErrors).toEqual([]);
  168 | 
  169 |     // Correctness-preserving optimizations are required to be image-neutral.
  170 |     // Flip them live so both captures use the identical resident world and
  171 |     // camera, then compare the canvas bytes rather than a subjective metric.
  172 |     for (const name of backend.equivalentOptimizations) {
  173 |       await page.evaluate(toggleName => {
  174 |         const input = document.getElementById(
  175 |           `setting-rt-${toggleName}`,
  176 |         ) as HTMLInputElement | null;
  177 |         if (!input) throw new Error(`missing render toggle: ${toggleName}`);
  178 |         input.checked = true;
  179 |         input.dispatchEvent(new Event('change', { bubbles: true }));
  180 |       }, name);
  181 |       await page.evaluate(() => new Promise<void>(resolve => {
  182 |         requestAnimationFrame(() => requestAnimationFrame(() => resolve()));
  183 |       }));
  184 |       const optimized = await page.locator('#view').screenshot({
  185 |         path: testInfo.outputPath(`${backend.name}-${name}.png`),
  186 |       });
  187 |       expect(
  188 |         await compareScreenshots(page, screenshot, optimized),
  189 |         `${name} must preserve every rendered pixel`,
> 190 |       ).toEqual({ changedPixels: 0, maximumChannelDelta: 0 });
      |         ^ Error: skip must preserve every rendered pixel
  191 |     }
  192 |   });
  193 | }
  194 | 
```