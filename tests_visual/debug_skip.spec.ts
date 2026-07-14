import { test } from '@playwright/test';

test('report skip hit record', async ({ page }) => {
  test.setTimeout(90_000);
  await page.addInitScript(() => {
    const original = WebGL2RenderingContext.prototype.shaderSource;
    WebGL2RenderingContext.prototype.shaderSource = function (shader, source) {
      const marker = '    vec3 radiance = shadeVoxel(';
      if (source.includes(marker)) {
        source = source.replace(marker, `
    uint debugDistance = uint(clamp(round(closest.distance * 1000.0), 0.0, 65535.0));
    float debugNormalCode = closest.normal.x < -0.5 ? 1.0
        : closest.normal.x > 0.5 ? 2.0
        : closest.normal.y < -0.5 ? 3.0
        : closest.normal.y > 0.5 ? 4.0
        : closest.normal.z < -0.5 ? 5.0
        : closest.normal.z > 0.5 ? 6.0 : 0.0;
    uint debugMetadata = uint(debugNormalCode) * 32u + min(closest.material, 31u);
    fragColor = vec4(
        float((debugDistance >> 8u) & 255u),
        float(debugDistance & 255u),
        float(debugMetadata),
        255.0
    ) * (1.0 / 255.0);
    return;

${marker}`);
      }
      return original.call(this, shader, source);
    };
  });

  await page.setViewportSize({ width: 160, height: 90 });
  await page.goto(
    '/index.html?seed=42&level=0&workers=0&capture=1&smoke=1' +
    '&renderer=raymarch&rt_bake=0&rt_skip=0&rt_f2b=0&rt_shadows=0&rt_dither=0&rt_timer=0' +
    '&camera=6,1.7,31.2&yaw=-1.5708&pitch=-0.05',
  );
  await page.waitForFunction(() => (window as typeof window & { __sceneReady?: boolean }).__sceneReady === true);

  const read = () => page.evaluate(() => new Promise<number[]>((resolve, reject) => {
    requestAnimationFrame(() => {
      const canvas = document.getElementById('view') as HTMLCanvasElement;
      const gl = canvas.getContext('webgl2');
      if (!gl) return reject(new Error('missing WebGL context'));
      const pixel = new Uint8Array(4);
      gl.readPixels(2, canvas.height - 1 - 6, 1, 1, gl.RGBA, gl.UNSIGNED_BYTE, pixel);
      resolve(Array.from(pixel));
    });
  }));
  const dda = await read();
  await page.evaluate(() => {
    const input = document.getElementById('setting-rt-skip') as HTMLInputElement;
    input.checked = true;
    input.dispatchEvent(new Event('change', { bubbles: true }));
  });
  await page.evaluate(() => new Promise<void>(resolve => requestAnimationFrame(() => requestAnimationFrame(() => resolve()))));
  const skip = await read();
  console.log(JSON.stringify({ dda, skip }));
});
