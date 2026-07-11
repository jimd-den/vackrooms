# Instructions

- Following Playwright test failed.
- Explain why, be concise, respect Playwright best practices.
- Provide a snippet of code with the fix, if possible.

# Test info

- Name: visual.spec.ts >> main spine scene matches approved visual
- Location: tests_visual/visual.spec.ts:3:5

# Error details

```
TimeoutError: page.waitForFunction: Timeout 15000ms exceeded.
```

# Test source

```ts
  1  | import { test, expect } from '@playwright/test';
  2  | 
  3  | test("main spine scene matches approved visual", async ({ page }) => {
  4  |   // Listen for console logs and errors in the browser page
  5  |   page.on('console', msg => {
  6  |     console.log(`[BROWSER CONSOLE ${msg.type()}]: ${msg.text()}`);
  7  |   });
  8  |   page.on('pageerror', err => {
  9  |     console.error(`[BROWSER PAGE ERROR]: ${err.message}`);
  10 |   });
  11 | 
  12 |   // Navigate to our main game page with the capture configuration
  13 |   await page.goto(
  14 |     "/index.html?seed=42&level=0&renderer=surface&workers=0" +
  15 |     "&camera=5,1.7,5&yaw=1.5708&pitch=-0.05&capture=1"
  16 |   );
  17 | 
  18 |   // Wait for the WASM WebGL scene to fully load and compile
> 19 |   await page.waitForFunction(() => (window as any).__sceneReady === true, null, { timeout: 15000 });
     |              ^ TimeoutError: page.waitForFunction: Timeout 15000ms exceeded.
  20 | 
  21 |   // Assert that the rendered canvas matches the approved baseline image
  22 |   await expect(page.locator("#view")).toHaveScreenshot(
  23 |     "main-spine-scene.png",
  24 |     { maxDiffPixelRatio: 0.001 }
  25 |   );
  26 | });
  27 | 
```