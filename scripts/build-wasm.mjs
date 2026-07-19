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

import { cp, mkdir, mkdtemp, readdir, rename, rm, writeFile } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { dirname, join, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import { spawn } from 'node:child_process';

import { wasmSourceHash } from './wasm-source-hash.mjs';

const root = resolve(dirname(fileURLToPath(import.meta.url)), '..');
const output = join(root, 'static', 'pkg');
const staging = await mkdtemp(join(tmpdir(), 'vackrooms-wasm-'));
await mkdir(dirname(output), { recursive: true });
const replacement = await mkdtemp(join(dirname(output), '.pkg-next-'));

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

  await writeFile(join(staging, '.source-sha256'), `${await wasmSourceHash(root)}\n`);

  // wasm-pack emits a `.gitignore` containing `*`, meant for pkg dirs that
  // are build output. This repo deliberately tracks static/pkg (including
  // the freshness stamp CI verifies), so that file must not survive.
  await rm(join(staging, '.gitignore'), { force: true });

  for (const entry of await readdir(staging, { withFileTypes: true })) {
    await cp(join(staging, entry.name), join(replacement, entry.name), {
      force: true,
      recursive: entry.isDirectory(),
    });
  }

  // Swap one complete package directory for another. This also removes files
  // retired by wasm-bindgen instead of leaving stale glue beside the new ABI.
  const previous = `${replacement}-previous`;
  let hadPrevious = false;
  try {
    await rename(output, previous);
    hadPrevious = true;
  } catch (error) {
    if (error?.code !== 'ENOENT') throw error;
  }
  try {
    await rename(replacement, output);
  } catch (error) {
    if (hadPrevious) await rename(previous, output);
    throw error;
  }
  if (hadPrevious) await rm(previous, { force: true, recursive: true });
} finally {
  await rm(staging, { force: true, recursive: true });
  await rm(replacement, { force: true, recursive: true });
}
