/**
 * board-file.js — Manual download/upload persistence for board JSON.
 *
 * Each board is a single JSON file on the user's disk. The user explicitly:
 *   • imports a file via a hidden <input type=file>
 *   • downloads changes via Blob + <a download>
 *
 * No auto-save, no browser-stored handles, no IndexedDB. The trade-off is
 * deliberate: works in every browser (including Brave with default flags,
 * Firefox, Safari, mobile), and cross-device sync is whatever the user does
 * with the file (drop it in iCloud Drive, email it to themselves, etc.).
 *
 * Privacy: the file never leaves the browser unless the user explicitly
 * downloads it; the upload path reads from a file the user picked.
 *
 * Designed for extraction across future personal-tool apps — depends only
 * on browser globals, no app-specific imports.
 */

/**
 * Triggers a browser download for JSON-serialisable data.
 * @param {unknown} data
 * @param {string} [suggestedName='board.json']
 */
export function downloadBoard(data, suggestedName = 'board.json') {
  const json = JSON.stringify(data, null, 2) + '\n';
  const blob = new Blob([json], { type: 'application/json' });
  const url  = URL.createObjectURL(blob);
  const a    = document.createElement('a');
  a.href     = url;
  a.download = suggestedName;
  document.body.appendChild(a);
  a.click();
  document.body.removeChild(a);
  // Revoke after a tick — the click handler needs the URL alive briefly.
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}

/**
 * Prompts the user for a JSON file and returns parsed contents + filename.
 * Must be called from a user gesture (input.click() requires activation).
 *
 * @returns {Promise<{ data: unknown, name: string }>}
 * @throws AbortError on cancel; SyntaxError on malformed JSON.
 */
export function pickBoardFile() {
  return new Promise((resolve, reject) => {
    const input  = document.createElement('input');
    input.type   = 'file';
    input.accept = '.json,application/json';

    const cancelErr = () => {
      const err = new Error('File picker cancelled');
      err.name  = 'AbortError';
      return err;
    };

    // Modern browsers fire 'cancel'; older ones leave the input dormant
    // and the change handler's empty-files branch covers that case.
    input.addEventListener('cancel', () => reject(cancelErr()));
    input.addEventListener('change', async () => {
      const file = input.files?.[0];
      if (!file) { reject(cancelErr()); return; }
      try {
        const text = await file.text();
        const data = JSON.parse(text);
        resolve({ data, name: file.name });
      } catch (err) {
        reject(err);
      }
    });

    input.click();
  });
}
