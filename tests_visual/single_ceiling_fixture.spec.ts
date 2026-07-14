import { test } from '@playwright/test';
import * as fs from 'fs';
import * as path from 'path';

const gpuRenderers = [
  {
    query: 'renderer=surface&rt_cull=0&rt_shadows=0&rt_dither=0',
    name: 'gpu_surface',
  },
  {
    query: 'renderer=splat&rt_cull=0&rt_shadows=0&rt_cells=0&rt_budget=0&rt_dither=0',
    name: 'gpu_splat',
  },
  {
    query: 'renderer=raymarch&rt_f2b=1&rt_dither=0',
    name: 'gpu_raymarch',
  },
];

for (const renderer of gpuRenderers) {
  test(`Render ${renderer.name} at 1080p`, async ({ page }) => {
    test.setTimeout(120_000); // Higher timeout for CI/server runs
    
    page.on('console', msg => {
      console.log(`[BROWSER CONSOLE ${msg.type()}]: ${msg.text()}`);
    });
    page.on('pageerror', err => {
      console.error(`[BROWSER PAGE ERROR]: ${err.message}`);
    });

    // Set 1080p viewport
    await page.setViewportSize({ width: 1920, height: 1080 });

    // Navigate to the exact same scene and camera as the CPU tests
    await page.goto(
      `/index.html?seed=42&level=0&workers=0&capture=1&${renderer.query}` +
      '&camera=5,1.7,5&yaw=0&pitch=-0.2',
    );

    // Wait for WASM WebGL scene to fully load
    await page.waitForFunction(
      () => {
        const status = document.getElementById('status-msg')?.textContent ?? '';
        if (status.includes('Failed to start engine')) throw new Error(status);
        return (window as any).__sceneReady === true;
      },
      null,
      { timeout: 90_000 },
    );

    // Take screenshot and write it directly to the native test golden directory
    const screenshot = await page.locator('#view').screenshot();
    const goldenPath = path.join(
      __dirname,
      '../wasm_frontend/tests/golden',
      `single_ceiling_fixture_${renderer.name}.png`
    );
    
    fs.mkdirSync(path.dirname(goldenPath), { recursive: true });
    fs.writeFileSync(goldenPath, screenshot);
  });
}
