import { test, expect } from '@playwright/test';

test('level 0 boots with vitals HUD and level 1 boots with sector readout', async ({ page }) => {
  test.setTimeout(180000);
  const errors: string[] = [];
  page.on('pageerror', (e) => errors.push(String(e)));

  await page.goto('/?level=0&seed=42');
  await expect(page.locator('#vitals')).toBeVisible();
  // Vitals bars are driven by the engine HUD refresh (every 30 frames,
  // which under software GL can take a while).
  await page.waitForFunction(
    () =>
      (document.getElementById('vitals-thirst-fill') as HTMLElement | null)?.style.width.endsWith(
        '%'
      ),
    { timeout: 120000 }
  );
  const inv = await page.locator('#vitals-inv').textContent();
  expect(inv).toContain('°C');

  // Level 1 boots directly and identifies itself.
  await page.goto('/?level=1&seed=42');
  await page.waitForFunction(
    () => document.getElementById('hud-section')?.textContent?.includes('HABITABLE ZONE'),
    { timeout: 120000 }
  );

  expect(errors, errors.join('\n')).toEqual([]);
});
