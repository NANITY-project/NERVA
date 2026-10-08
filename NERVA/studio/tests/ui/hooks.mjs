// Node ESM resolve hook mirroring the browser import map in web/index.html.
import { pathToFileURL } from 'node:url';
import path from 'node:path';
const nm = path.resolve(path.dirname(new URL(import.meta.url).pathname), 'node_modules/three');
export async function resolve(specifier, context, next) {
  if (specifier === 'three') return { url: pathToFileURL(path.join(nm, 'build/three.module.js')).href, shortCircuit: true };
  if (specifier.startsWith('three/addons/')) return { url: pathToFileURL(path.join(nm, 'examples/jsm', specifier.slice(13))).href, shortCircuit: true };
  return next(specifier, context);
}
