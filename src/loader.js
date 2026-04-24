/**
 * loader.js — JSON loading and schema validation
 *
 * Fetches the sticky notes dataset from the bundled data file.
 * Validates each note against the expected schema before returning.
 */

const REQUIRED_FIELDS = ['id', 'text', 'x', 'y', 'author', 'color'];

/**
 * Validates a single note object against the required schema.
 * @param {unknown} note
 * @param {number} index
 * @returns {string|null} error message or null if valid
 */
function validateNote(note, index) {
  if (typeof note !== 'object' || note === null) {
    return `Note at index ${index} is not an object`;
  }
  for (const field of REQUIRED_FIELDS) {
    if (!(field in note)) {
      return `Note at index ${index} (id: ${note.id ?? '?'}) missing field: "${field}"`;
    }
  }
  if (typeof note.text !== 'string' || note.text.trim() === '') {
    return `Note at index ${index} has empty or non-string "text"`;
  }
  if (typeof note.x !== 'number' || typeof note.y !== 'number') {
    return `Note at index ${index} has non-numeric x/y coordinates`;
  }
  return null;
}

/**
 * Loads and validates the sticky notes JSON from the given URL.
 * @param {string} url
 * @returns {Promise<Array>} validated array of note objects
 */
export async function loadNotes(url) {
  let data;
  try {
    const res = await fetch(url);
    if (!res.ok) throw new Error(`HTTP ${res.status}: ${res.statusText}`);
    data = await res.json();
  } catch (err) {
    throw new Error(`Failed to load sticky notes: ${err.message}`);
  }

  if (!Array.isArray(data)) {
    throw new Error('Sticky notes JSON must be a top-level array');
  }
  if (data.length === 0) {
    throw new Error('Sticky notes JSON is empty');
  }

  const errors = data.map(validateNote).filter(Boolean);
  if (errors.length > 0) {
    throw new Error(`Schema validation failed:\n${errors.join('\n')}`);
  }

  return data;
}
