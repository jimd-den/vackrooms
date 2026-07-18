import { test, expect } from '@playwright/test';

test('level 0 boots with the body trace and level 1 boots with sector readout', async ({ page }) => {
  test.setTimeout(180000);
  const errors: string[] = [];
  page.on('pageerror', (e) => errors.push(String(e)));

  await page.goto('/?level=0&seed=42');
  // The body trace is the only permanent vital presence in the new shell.
  await expect(page.locator('#body-trace')).toBeVisible();
  // The presenter stamps the beat period as an inline custom property on
  // the first HUD refresh (every 30 frames, which under software GL can
  // take a while) — that write is the proof body telemetry reached the DOM.
  await page.waitForFunction(
    () =>
      (document.getElementById('body-trace') as HTMLElement | null)?.style
        .getPropertyValue('--beat-period')
        .endsWith('ms'),
    { timeout: 120000 }
  );
  // Steps are a whole-number session record, present from the first frame.
  const steps = await page.locator('#hud-steps').textContent();
  expect(steps).toMatch(/^\d+$/);

  // Level 1 boots directly and identifies itself.
  await page.goto('/?level=1&seed=42');
  await page.waitForFunction(
    () => document.getElementById('hud-section')?.textContent?.includes('HABITABLE ZONE'),
    { timeout: 120000 }
  );

  expect(errors, errors.join('\n')).toEqual([]);
});
