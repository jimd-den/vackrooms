import { test, expect } from '@playwright/test';

test("main spine scene matches approved visual", async ({ page }) => {
  // Listen for console logs and errors in the browser page
  page.on('console', msg => {
    console.log(`[BROWSER CONSOLE ${msg.type()}]: ${msg.text()}`);
  });
  page.on('pageerror', err => {
    console.error(`[BROWSER PAGE ERROR]: ${err.message}`);
  });

  // Navigate to our main game page with the capture configuration.
  // Camera = the real spawn: on the main corridor of region (0,0)
  // (see vackrooms::use_cases::region_plan::spawn_point), facing east
  // down the corridor like the player does on their first frame.
  await page.goto(
    "/index.html?seed=42&level=0&renderer=surface&workers=0" +
    "&camera=6,1.7,31.2&yaw=-1.5708&pitch=-0.05&capture=1"
  );

  // Wait for the WASM WebGL scene to fully load and compile
  await page.waitForFunction(() => (window as any).__sceneReady === true, null, { timeout: 15000 });

  // Assert that the rendered canvas matches the approved baseline image
  await expect(page.locator("#view")).toHaveScreenshot(
    "main-spine-scene.png",
    { maxDiffPixelRatio: 0.001 }
  );
});
