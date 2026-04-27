/**
 * board-fs.js — File System Access persistence for board-shaped JSON.
 *
 * Each "board" is one JSON file the user picks from disk via the File System
 * Access API. The browser hands back a FileSystemFileHandle which we persist
 * in IndexedDB, keyed by an internal id, so the user can reopen recent boards
 * without re-picking.
 *
 * Persistence model
 * ─────────────────
 *   • The file on disk is the single source of truth — no copy in the browser.
 *   • Cross-device sync is the OS-level cloud-drive client's job (iCloud Drive,
 *     Dropbox, …). The user picks a file inside such a folder; sync follows.
 *   • Cross-tab sync is intentionally NOT implemented here — deferred (would
 *     use BroadcastChannel + permission re-checks).
 *
 * Atomicity
 * ─────────
 * `createWritable()` writes to a browser-managed swap file in Chromium; the
 * visible file is replaced only on `.close()`. That gives crash-safety against
 * partial writes for free.
 *
 * Security
 * ────────
 * No app-level encryption. The threat model is one user with their own files
 * in their own cloud-drive folder; OS file permissions plus the cloud drive's
 * encryption are the boundary. AES-GCM + passphrase would be straightforward
 * but is YAGNI for v1.
 *
 * Browser support
 * ───────────────
 * Chromium-only (Chrome, Edge, Brave, Opera). Firefox and Safari do not
 * implement FSA. Use `isSupported` to feature-detect; every other call
 * throws if FSA is unavailable.
 *
 * Reusable across future personal-tool apps — depends only on browser globals,
 * no app-specific imports.
 */

export const isSupported =
  typeof window !== 'undefined' &&
  'showOpenFilePicker' in window &&
  'showSaveFilePicker' in window;

const DB_NAME    = 'board-fs';
const DB_VERSION = 1;
const STORE      = 'recents';

const PICKER_TYPES = [
  { description: 'Board (JSON)', accept: { 'application/json': ['.json'] } },
];

// ── IndexedDB helpers ───────────────────────────────────────────────────────
// Tiny inlined wrapper rather than a dependency. Records:
//   { id: string (uuid), name: string, handle: FileSystemFileHandle, lastOpenedAt: number }

function openIdb() {
  return new Promise((resolve, reject) => {
    const req = indexedDB.open(DB_NAME, DB_VERSION);
    req.onupgradeneeded = () => {
      const db = req.result;
      if (!db.objectStoreNames.contains(STORE)) {
        db.createObjectStore(STORE, { keyPath: 'id' });
      }
    };
    req.onsuccess = () => resolve(req.result);
    req.onerror   = () => reject(req.error);
  });
}

function idbAll() {
  return openIdb().then((db) => new Promise((resolve, reject) => {
    const tx  = db.transaction(STORE, 'readonly');
    const req = tx.objectStore(STORE).getAll();
    req.onsuccess = () => resolve(req.result);
    req.onerror   = () => reject(req.error);
  }));
}

function idbPut(record) {
  return openIdb().then((db) => new Promise((resolve, reject) => {
    const tx = db.transaction(STORE, 'readwrite');
    tx.objectStore(STORE).put(record);
    tx.oncomplete = () => resolve();
    tx.onerror    = () => reject(tx.error);
  }));
}

function idbDelete(id) {
  return openIdb().then((db) => new Promise((resolve, reject) => {
    const tx = db.transaction(STORE, 'readwrite');
    tx.objectStore(STORE).delete(id);
    tx.oncomplete = () => resolve();
    tx.onerror    = () => reject(tx.error);
  }));
}

async function findRecordByHandle(handle) {
  const all = await idbAll();
  for (const rec of all) {
    if (await rec.handle.isSameEntry(handle)) return rec;
  }
  return null;
}

async function rememberBoard(handle) {
  const existing = await findRecordByHandle(handle);
  const now      = Date.now();
  if (existing) {
    existing.lastOpenedAt = now;
    existing.name         = handle.name;
    await idbPut(existing);
    return existing.id;
  }
  const id = crypto.randomUUID();
  await idbPut({ id, name: handle.name, handle, lastOpenedAt: now });
  return id;
}

function ensureSupported() {
  if (!isSupported) {
    throw new Error(
      'File System Access API is not available in this browser. ' +
      'Use a Chromium-based browser (Chrome, Edge, Brave, Opera).'
    );
  }
}

// ── Public API ──────────────────────────────────────────────────────────────

/**
 * Ensures the current page has the requested permission for `handle`.
 * If the permission is `prompt`, this calls `requestPermission`, which
 * REQUIRES a user activation (must be invoked from a click handler etc.).
 *
 * @param {FileSystemFileHandle} handle
 * @param {'read'|'readwrite'} mode
 * @returns {Promise<true>}
 * @throws if permission is denied
 */
export async function ensurePermission(handle, mode = 'readwrite') {
  if ((await handle.queryPermission({ mode })) === 'granted') return true;
  if ((await handle.requestPermission({ mode })) === 'granted') return true;
  throw new Error(`Permission "${mode}" denied for ${handle.name}`);
}

/**
 * Prompts the user to pick an existing board file. Reads + parses it,
 * records it in recents, and returns the handle and parsed data.
 * Caller must invoke this from a user gesture.
 *
 * @returns {Promise<{ id: string, handle: FileSystemFileHandle, data: unknown }>}
 * @throws AbortError if the user cancels the picker
 */
export async function openBoard() {
  ensureSupported();
  const [handle] = await window.showOpenFilePicker({
    types: PICKER_TYPES,
    multiple: false,
    excludeAcceptAllOption: false,
  });
  const data = await readBoard(handle);
  const id   = await rememberBoard(handle);
  return { id, handle, data };
}

/**
 * Prompts the user for a save location, writes `defaultData` to it, and
 * records it in recents. Caller must invoke from a user gesture.
 *
 * @param {unknown} defaultData
 * @param {string} suggestedName
 * @returns {Promise<{ id: string, handle: FileSystemFileHandle, data: unknown }>}
 * @throws AbortError if the user cancels the picker
 */
export async function createBoard(defaultData, suggestedName = 'board.json') {
  ensureSupported();
  const handle = await window.showSaveFilePicker({
    suggestedName,
    types: PICKER_TYPES,
    excludeAcceptAllOption: false,
  });
  await writeBoard(handle, defaultData);
  const id = await rememberBoard(handle);
  return { id, handle, data: defaultData };
}

/**
 * Reads and parses the JSON content of a board file.
 * Requests `read` permission if not already granted (needs user gesture).
 *
 * @param {FileSystemFileHandle} handle
 * @returns {Promise<unknown>}
 */
export async function readBoard(handle) {
  ensureSupported();
  await ensurePermission(handle, 'read');
  const file = await handle.getFile();
  const text = await file.text();
  return JSON.parse(text);
}

/**
 * Atomically writes JSON to the given board file (Chromium swap-file
 * implementation; visible file replaced only on close).
 * Touches `lastOpenedAt` if the handle is in recents.
 *
 * @param {FileSystemFileHandle} handle
 * @param {unknown} data
 */
export async function writeBoard(handle, data) {
  ensureSupported();
  await ensurePermission(handle, 'readwrite');
  const writable = await handle.createWritable();
  await writable.write(JSON.stringify(data, null, 2) + '\n');
  await writable.close();

  const existing = await findRecordByHandle(handle);
  if (existing) {
    existing.lastOpenedAt = Date.now();
    await idbPut(existing);
  }
}

/**
 * Returns recent boards, most-recently-used first.
 * Empty array if FSA is unsupported (no records can exist).
 *
 * @returns {Promise<Array<{ id: string, name: string, handle: FileSystemFileHandle, lastOpenedAt: number }>>}
 */
export async function listRecentBoards() {
  if (!isSupported) return [];
  const all = await idbAll();
  return all.sort((a, b) => b.lastOpenedAt - a.lastOpenedAt);
}

/**
 * Removes a board from recents. Accepts either an internal id (string) or
 * a FileSystemFileHandle. Does not delete the file on disk.
 *
 * @param {string | FileSystemFileHandle} idOrHandle
 */
export async function forgetBoard(idOrHandle) {
  if (typeof idOrHandle === 'string') {
    await idbDelete(idOrHandle);
    return;
  }
  const rec = await findRecordByHandle(idOrHandle);
  if (rec) await idbDelete(rec.id);
}
