// Loads the JSON written by the project's Python code. Each file is fetched once and cached;
// per-dataset files are fetched only when a section first needs them.
const cache = new Map();

export function getJSON(path) {
  if (!cache.has(path)) {
    cache.set(path, fetch(`data/${path}`).then((r) => {
      if (!r.ok) throw new Error(`could not load data/${path} (${r.status})`);
      return r.json();
    }));
  }
  return cache.get(path);
}

export const datasetIndex = () => getJSON("datasets_index.json");

export async function dataset(name) {
  const index = await datasetIndex();
  const entry = index.find((d) => d.name === name);
  if (!entry) throw new Error(`unknown dataset ${name}`);
  return getJSON(`datasets/${entry.slug}.json`);
}

// rows of a table file that match every key/value pair in `where`
export async function rows(file, where = {}) {
  const all = await getJSON(file);
  return all.filter((r) => Object.entries(where).every(([k, v]) => r[k] === v));
}
