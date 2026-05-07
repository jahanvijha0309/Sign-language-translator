// ===== ELEMENTS =====
const uploadTab    = document.getElementById('uploadTab');
const liveTab      = document.getElementById('liveTab');
const uploadSection = document.getElementById('uploadSection');
const liveSection  = document.getElementById('liveSection');
const dropZone     = document.getElementById('dropZone');
const mediaUpload  = document.getElementById('mediaUpload');
const preview      = document.getElementById('preview');
const predictBtn   = document.getElementById('predictBtn');
const loader       = document.getElementById('loader');
const liveText     = document.getElementById('liveText');
const webcam       = document.getElementById('webcam');

let currentMode = 'upload';   // 'upload' | 'live'
let selectedFile = null;
let liveActive = false;

// ===== INIT =====
liveText.classList.add('placeholder');

// ===== TABS =====
uploadTab.addEventListener('click', () => {
  currentMode = 'upload';
  uploadTab.classList.add('active');
  liveTab.classList.remove('active');
  uploadSection.classList.remove('hidden');
  liveSection.classList.add('hidden');
  stopLive();
  resetResult();
});

liveTab.addEventListener('click', () => {
  currentMode = 'live';
  liveTab.classList.add('active');
  uploadTab.classList.remove('active');
  liveSection.classList.remove('hidden');
  uploadSection.classList.add('hidden');
  selectedFile = null;
  resetResult();
  startLive();
});

// ===== DROP ZONE =====
dropZone.addEventListener('click', () => mediaUpload.click());

dropZone.addEventListener('dragover', (e) => {
  e.preventDefault();
  dropZone.classList.add('dragover');
});

dropZone.addEventListener('dragleave', () => {
  dropZone.classList.remove('dragover');
});

dropZone.addEventListener('drop', (e) => {
  e.preventDefault();
  dropZone.classList.remove('dragover');
  const file = e.dataTransfer.files[0];
  if (file) handleFile(file);
});

mediaUpload.addEventListener('change', () => {
  if (mediaUpload.files[0]) handleFile(mediaUpload.files[0]);
});

function handleFile(file) {
  selectedFile = file;
  preview.innerHTML = '';

  const url = URL.createObjectURL(file);

  if (file.type.startsWith('image/')) {
    const img = document.createElement('img');
    img.src = url;
    preview.appendChild(img);
  } else if (file.type.startsWith('video/')) {
    const vid = document.createElement('video');
    vid.src = url;
    vid.controls = true;
    preview.appendChild(vid);
  }

  dropZone.querySelector('p').textContent = '✅ ' + file.name;
  resetResult();
}

// ===== PREDICT BUTTON =====
predictBtn.addEventListener('click', () => {
  if (currentMode === 'upload') {
    if (!selectedFile) {
      alert('Please upload an image or video first.');
      return;
    }
    predictFile(selectedFile);
  } else {
    // In live mode button triggers a single snapshot prediction
    predictLiveSnapshot();
  }
});

// ===== UPLOAD PREDICT =====
async function predictFile(file) {
  const isImage = file.type.startsWith('image/');
  const endpoint = isImage ? '/predict_image' : '/predict_video';

  showLoader(true);
  setResult('---', null);

  const formData = new FormData();
  formData.append('file', file);

  try {
    const res = await fetch(endpoint, { method: 'POST', body: formData });
    const data = await res.json();

    if (data.error) {
      setResult('Error: ' + data.error, null);
    } else if (isImage) {
      setResult(data.prediction, data.confidence);
    } else {
      setResult(data.prediction || 'No signs detected', null);
    }
  } catch (err) {
    setResult('Network error. Is the server running?', null);
  } finally {
    showLoader(false);
  }
}

// ===== LIVE WEBCAM =====
function startLive() {
  liveActive = true;
  // Show the server's streaming feed via /predict_live
  webcam.src = '/predict_live';
  predictBtn.textContent = '📸 Snapshot';
}

function stopLive() {
  liveActive = false;
  webcam.src = '';
  predictBtn.textContent = 'Translate';
}

// Snapshot: grab current frame from the live stream image and send to /predict_image
async function predictLiveSnapshot() {
  showLoader(true);
  setResult('---', null);

  if (!webcam.complete || webcam.naturalWidth === 0 || webcam.naturalHeight === 0) {
    setResult('Waiting for live feed...', null);
    showLoader(false);
    return;
  }

  const canvas = document.createElement('canvas');
  canvas.width = webcam.naturalWidth;
  canvas.height = webcam.naturalHeight;
  canvas.getContext('2d').drawImage(webcam, 0, 0, canvas.width, canvas.height);

  try {
    const blob = await new Promise((resolve) => canvas.toBlob(resolve, 'image/jpeg'));
    if (!blob) {
      throw new Error('Could not capture live frame');
    }

    const file = new File([blob], 'snapshot.jpg', { type: 'image/jpeg' });
    const formData = new FormData();
    formData.append('file', file);

    const predRes = await fetch('/predict_image', { method: 'POST', body: formData });
    const data = await predRes.json();

    if (data.error) {
      setResult('Error: ' + data.error, null);
    } else {
      setResult(data.prediction, data.confidence);
    }
  } catch (err) {
    setResult('Could not capture frame.', null);
  } finally {
    showLoader(false);
  }
}

// ===== HELPERS =====
function showLoader(show) {
  loader.classList.toggle('hidden', !show);
  predictBtn.disabled = show;
}

function setResult(text, confidence) {
  liveText.classList.remove('placeholder');
  liveText.textContent = text;

  // Remove old badge if any
  const oldBadge = document.getElementById('confidenceBadge');
  if (oldBadge) oldBadge.remove();

  if (confidence !== null && confidence !== undefined) {
    const badge = document.createElement('div');
    badge.id = 'confidenceBadge';
    badge.textContent = `Confidence: ${(confidence * 100).toFixed(1)}%`;
    document.querySelector('.output').appendChild(badge);
  }
}

function resetResult() {
  liveText.textContent = '---';
  liveText.classList.add('placeholder');
  const badge = document.getElementById('confidenceBadge');
  if (badge) badge.remove();
  showLoader(false);
}
