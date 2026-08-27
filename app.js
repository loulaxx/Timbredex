'use strict';

const CONFIDENCE_THRESHOLD = 40; // recalibrated for cosine similarity, see notes below
const MAX_FILE_SIZE = 10 * 1024 * 1024;

let stamps = [], selectedImage = null, selectedFile = null;
let mobilenetModel = null;
let useFallbackHash = false; // becomes true if MobileNet fails to load

const $ = (id) => document.getElementById(id);

// ---------- Image loading ----------
function loadImage(src) {
  return new Promise((resolve, reject) => {
    const img = new Image();
    img.onload = () => resolve(img);
    img.onerror = () => reject(new Error(`Could not load image: ${src}`));
    img.src = src;
  });
}

// ---------- Fallback: original aHash (kept as safety net) ----------
function computeHash(img, size = 16) {
  const canvas = document.createElement('canvas');
  canvas.width = size; canvas.height = size;
  const ctx = canvas.getContext('2d', { willReadFrequently: true });
  ctx.drawImage(img, 0, 0, size, size);
  const data = ctx.getImageData(0, 0, size, size).data;
  let total = 0;
  const gray = [];
  for (let i = 0; i < data.length; i += 4) {
    const value = Math.round(.299 * data[i] + .587 * data[i+1] + .114 * data[i+2]);
    gray.push(value);
    total += value;
  }
  const mean = total / gray.length;
  return gray.map(value => value >= mean ? 1 : 0);
}

function hammingDistance(a, b) {
  if (!a || !b || a.length !== b.length) throw new Error('Hash lengths do not match');
  return a.reduce((distance, bit, index) => distance + (bit !== b[index] ? 1 : 0), 0);
}

// ---------- CNN embedding (MobileNet) ----------
async function loadModel() {
  try {
    mobilenetModel = await mobilenet.load({ version: 2, alpha: 1.0 });
    useFallbackHash = false;
    console.log('MobileNet loaded successfully.');
  } catch (error) {
    console.error('MobileNet failed to load, falling back to aHash:', error);
    useFallbackHash = true;
  }
}

async function computeEmbedding(img) {
  // model.infer(img, true) returns the internal feature vector
  // instead of the final 1000-class classification
  const embeddingTensor = mobilenetModel.infer(img, true);
  const embeddingArray = await embeddingTensor.data(); // Float32Array
  embeddingTensor.dispose(); // free GPU/CPU memory — important, tensors don't auto garbage-collect
  return Array.from(embeddingArray);
}

// Unified function: returns either a CNN embedding or a fallback hash,
// so the rest of the code doesn't need to know which one is active.
async function fingerprint(img) {
  if (useFallbackHash) return { type: 'hash', data: computeHash(img) };
  return { type: 'embedding', data: await computeEmbedding(img) };
}

// ---------- Similarity metrics ----------
function cosineSimilarity(a, b) {
  let dot = 0, normA = 0, normB = 0;
  for (let i = 0; i < a.length; i++) {
    dot += a[i] * b[i];
    normA += a[i] * a[i];
    normB += b[i] * b[i];
  }
  return dot / (Math.sqrt(normA) * Math.sqrt(normB) || 1);
}

// Returns a 0–100 similarity score regardless of which fingerprint type is used
function compareFingerprints(a, b) {
  if (a.type === 'hash') {
    const distance = hammingDistance(a.data, b.data);
    return Math.max(0, (1 - distance / a.data.length) * 100);
  }
  const sim = cosineSimilarity(a.data, b.data); // roughly 0..1, can dip slightly negative
  return Math.max(0, sim * 100);
}

// ---------- UI helpers ----------
function setError(message) { $('uploadError').textContent = message || ''; }

function renderCatalogue() {
  $('catalogue').innerHTML = stamps.map(stamp => `
    <article class="stamp-card">
      <img src="${stamp.image}" alt="${stamp.name} placeholder stamp">
      <h4>${stamp.name}</h4>
      <p>${stamp.country} · ${stamp.year}</p>
      <p>${stamp.denomination} · ${stamp.catalogNumber}</p>
    </article>`).join('');
}

// ---------- Catalogue + model initialization ----------
async function init() {
  try {
    $('catalogue').innerHTML = '<p>Loading recognition model…</p>';

    // Load the CNN model first (or flag fallback)
    await loadModel();

    const response = await fetch('stamps.json');
    if (!response.ok) throw new Error(`stamps.json returned ${response.status}`);
    stamps = await response.json();
    if (!Array.isArray(stamps) || stamps.length !== 5) {
      throw new Error('Catalogue must contain exactly five stamps');
    }

    await Promise.all(stamps.map(async stamp => {
      const image = await loadImage(stamp.image);
      stamp.fingerprint = await fingerprint(image);
    }));

    renderCatalogue();
  } catch (error) {
    console.error('Catalogue loading failed:', error);
    $('catalogue').innerHTML = '<p class="error">The reference shelf could not load. Check that stamps.json and the stamps folder are present.</p>';
  }
}

// ---------- Upload handling ----------
function resetUpload() {
  selectedImage = null;
  selectedFile = null;
  $('fileInput').value = '';
  $('previewWrap').hidden = true;
  $('previewImg').removeAttribute('src');
  $('analyzeBtn').disabled = true;
  setError('');
  $('result').hidden = true;
}

function handleFile(file) {
  setError('');
  if (!file) return;
  if (!['image/jpeg', 'image/png'].includes(file.type)) {
    setError('Please choose a JPEG or PNG image.');
    return;
  }
  if (file.size > MAX_FILE_SIZE) {
    setError('That image is over 10 MB. Please choose a smaller file.');
    return;
  }
  const reader = new FileReader();
  reader.onload = async () => {
    try {
      selectedImage = await loadImage(reader.result);
      selectedFile = file;
      $('previewImg').src = reader.result;
      $('previewWrap').hidden = false;
      $('analyzeBtn').disabled = false;
    } catch (error) {
      console.error('Preview failed:', error);
      setError('We could not read that image. Please try another file.');
    }
  };
  reader.onerror = () => setError('We could not read that file. Please try again.');
  reader.readAsDataURL(file);
}

// ---------- Result rendering ----------
function showResult(best, confidence) {
  $('result').innerHTML = `
    <div class="result-content">
      <img class="result-image" src="${best.image}" alt="${best.name} stamp">
      <div>
        <p class="result-kicker">CLOSEST MATCH · ${best.catalogNumber}</p>
        <h3 class="result-name">${best.name}</h3>
        <p class="result-details">${best.country} · ${best.year} · ${best.denomination}</p>
        <p class="result-description">${best.description}</p>
        <div class="confidence">
          <div class="confidence-label"><span>VISUAL CONFIDENCE</span><strong>${confidence}%</strong></div>
          <div class="confidence-track"><div class="confidence-fill" style="width:${confidence}%"></div></div>
        </div>
      </div>
    </div>`;
  $('result').hidden = false;
}

// ---------- Main analysis ----------
async function analyze() {
  if (!selectedImage || !stamps.length) return;
  $('analyzeBtn').disabled = true;
  $('loading').hidden = false;
  $('result').hidden = true;

  try {
    const inputFingerprint = await fingerprint(selectedImage);

    const ranked = stamps
      .map(stamp => ({ stamp, score: compareFingerprints(inputFingerprint, stamp.fingerprint) }))
      .sort((a, b) => b.score - a.score); // higher score = better match

    const best = ranked[0];
    const second = ranked[1];

    // Ambiguity penalty: if top two matches are very close, reduce confidence
    const margin = second ? (best.score - second.score) : 100;
    const ambiguityPenalty = margin < 3 ? 15 : 0;

    const confidence = Math.max(0, Math.round(best.score - ambiguityPenalty));

    $('loading').hidden = true;

    if (confidence >= CONFIDENCE_THRESHOLD) {
      showResult(best.stamp, confidence);
    } else {
      $('result').innerHTML = `
        <div>
          <p class="result-kicker">NO CONFIDENT MATCH</p>
          <h3 class="result-name">Not quite sure yet.</h3>
          <p class="result-description">We could not confidently identify this stamp. Try a clearer, front-facing image with even lighting.</p>
          <button class="primary-button" id="tryAgainBtn" type="button">Try another image ↗</button>
        </div>`;
      $('result').hidden = false;
      $('tryAgainBtn').addEventListener('click', resetUpload);
    }
  } catch (error) {
    console.error('Recognition failed:', error);
    $('loading').hidden = true;
    setError('Something went wrong while recognizing the image. Please try again.');
  } finally {
    $('analyzeBtn').disabled = !selectedFile;
  }
}

// ---------- Event wiring ----------
function wireEvents() {
  const dropzone = $('dropzone');
  $('chooseBtn').addEventListener('click', e => { e.stopPropagation(); $('fileInput').click(); });
  $('fileInput').addEventListener('change', e => handleFile(e.target.files[0]));
  dropzone.addEventListener('click', e => { if (e.target !== $('chooseBtn')) $('fileInput').click(); });
  dropzone.addEventListener('keydown', e => { if (e.key === 'Enter' || e.key === ' ') $('fileInput').click(); });
  ['dragenter', 'dragover'].forEach(type => dropzone.addEventListener(type, e => {
    e.preventDefault(); dropzone.classList.add('dragover');
  }));
  ['dragleave', 'drop'].forEach(type => dropzone.addEventListener(type, e => {
    e.preventDefault(); dropzone.classList.remove('dragover');
  }));
  dropzone.addEventListener('drop', e => handleFile(e.dataTransfer.files[0]));
  $('removeBtn').addEventListener('click', resetUpload);
  $('analyzeBtn').addEventListener('click', analyze);
}

document.addEventListener('DOMContentLoaded', () => { wireEvents(); init(); });
