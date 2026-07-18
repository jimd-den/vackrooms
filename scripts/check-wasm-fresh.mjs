#!/usr/bin/env node

import { readFile } from 'node:fs/promises';
import { dirname, join, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';

import { wasmSourceHash } from './wasm-source-hash.mjs';

const root = resolve(dirname(fileURLToPath(import.meta.url)), '..');
const stamp = join(root, 'static', 'pkg', '.source-sha256');
const expected = await wasmSourceHash(root);
let actual;
try {
  actual = (await readFile(stamp, 'utf8')).trim();
} catch {
  throw new Error('static/pkg has no source fingerprint; run `npm run build:wasm`');
}

if (actual !== expected) {
  throw new Error('static/pkg is stale for the current Rust/worker sources; run `npm run build:wasm`');
}

console.log(`static/pkg matches source ${expected.slice(0, 12)}`);
