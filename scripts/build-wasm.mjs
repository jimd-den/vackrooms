#!/usr/bin/env node

/**
 * Build the browser WebAssembly package without exposing a half-written
 * `static/pkg` directory to the dev server.
 *
 * wasm-pack reads an existing output package before replacing it, and older
 * wasm-pack releases cannot parse the newer `files: [...]` manifest they just
 * generated. A fresh staging directory avoids that version-skew trap. Only a
 * successful build is copied over the live browser assets.
 */

import { cp, mkdir, mkdtemp, readdir, rm } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { dirname, join, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import { spawn } from 'node:child_process';

const root = resolve(dirname(fileURLToPath(import.meta.url)), '..');
const output = join(root, 'static', 'pkg');
const staging = await mkdtemp(join(tmpdir(), 'vackrooms-wasm-'));

function run(command, args) {
  return new Promise((resolveRun, rejectRun) => {
    const child = spawn(command, args, { cwd: root, stdio: 'inherit' });
    child.once('error', rejectRun);
    child.once('exit', (code, signal) => {
      if (code === 0) resolveRun();
      else rejectRun(new Error(`${command} failed (${signal ?? `exit ${code}`})`));
    });
  });
}

try {
  await run('wasm-pack', [
    'build',
    'wasm_frontend',
    '--target',
    'web',
    '--release',
    '--out-dir',
    staging,
  ]);

  await mkdir(output, { recursive: true });
  for (const entry of await readdir(staging, { withFileTypes: true })) {
    await cp(join(staging, entry.name), join(output, entry.name), {
      force: true,
      recursive: entry.isDirectory(),
    });
  }
} finally {
  await rm(staging, { force: true, recursive: true });
}
