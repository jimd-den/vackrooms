import { createHash } from 'node:crypto';
import { readdir, readFile, stat } from 'node:fs/promises';
import { join, relative } from 'node:path';

const INPUTS = Object.freeze([
  'Cargo.lock',
  'Cargo.toml',
  'src',
  'wasm_frontend/assets',
  'wasm_frontend/Cargo.toml',
  'wasm_frontend/src',
  'static/worker.js',
]);

async function filesBelow(path) {
  const entries = await readdir(path, { withFileTypes: true });
  const files = [];
  for (const entry of entries.sort((left, right) => left.name.localeCompare(right.name))) {
    const child = join(path, entry.name);
    if (entry.isDirectory()) files.push(...await filesBelow(child));
    else if (entry.isFile()) files.push(child);
  }
  return files;
}

/** Fingerprints every source that can change the browser wasm contract. */
export async function wasmSourceHash(root) {
  const hash = createHash('sha256');
  for (const input of INPUTS) {
    const path = join(root, input);
    const files = (await stat(path)).isDirectory() ? await filesBelow(path) : [path];
    for (const file of files) {
      hash.update(relative(root, file));
      hash.update('\0');
      hash.update(await readFile(file));
      hash.update('\0');
    }
  }
  return hash.digest('hex');
}
