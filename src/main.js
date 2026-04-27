import { pipeline, env } from '@huggingface/transformers';
import { createIcons, Mic, Square, Copy, Check } from 'lucide';

env.allowLocalModels = false;

const MODEL_WEBGPU = 'onnx-community/whisper-large-v3-turbo';
const MODEL_WASM = 'Xenova/whisper-base';

const INSTALLED_KEY = 'webwhisper:installed';

const installView = document.getElementById('install-view');
const installBtn = document.getElementById('install-button');
const micView = document.getElementById('mic-view');
const micBtn = document.getElementById('mic-button');
const langEl = document.getElementById('language');
const statusEl = document.getElementById('status');
const loadingEl = document.getElementById('loading');
const loadingFill = document.getElementById('loading-fill');
const loadingLabel = document.getElementById('loading-label');
const resultsEl = document.getElementById('results');
const cardTpl = document.getElementById('card-template');

let transcriber = null;
let modelLoadingPromise = null;
let recorder = null;
let recordedChunks = [];
let activeStream = null;
let recorderMime = '';

renderIcons();

const setStatus = (text) => {
  statusEl.textContent = text;
};

function renderIcons(root) {
  createIcons({
    icons: { Mic, Square, Copy, Check },
    attrs: { 'stroke-linecap': 'round', 'stroke-linejoin': 'round' },
    ...(root ? { root } : {}),
  });
}

function showLoadingUI(label = 'Carregando modelo…') {
  loadingEl.classList.remove('hidden');
  loadingLabel.textContent = label;
  loadingFill.style.width = '0%';
}

function hideLoadingUI() {
  loadingEl.classList.add('hidden');
}

function updateLoadingProgress(p) {
  if (!p) return;
  if (p.status === 'progress' && typeof p.progress === 'number') {
    const pct = Math.max(0, Math.min(100, Math.round(p.progress)));
    loadingFill.style.width = pct + '%';
    const file = p.file ? p.file.split('/').pop() : 'modelo';
    loadingLabel.textContent = `Baixando ${file} · ${pct}%`;
  } else if (p.status === 'ready') {
    loadingFill.style.width = '100%';
    loadingLabel.textContent = 'Modelo pronto';
  } else if (p.status === 'done') {
    loadingFill.style.width = '100%';
  } else if (p.status === 'initiate') {
    loadingLabel.textContent = `Preparando ${p.file?.split('/').pop() || 'modelo'}…`;
  }
}

async function ensureTranscriber() {
  if (transcriber) return transcriber;
  if (modelLoadingPromise) return modelLoadingPromise;

  showLoadingUI();

  const supportsWebGPU = 'gpu' in navigator;

  modelLoadingPromise = (async () => {
    try {
      if (supportsWebGPU) {
        return await pipeline('automatic-speech-recognition', MODEL_WEBGPU, {
          device: 'webgpu',
          dtype: {
            encoder_model: 'fp16',
            decoder_model_merged: 'q4',
          },
          progress_callback: updateLoadingProgress,
        });
      }
    } catch (err) {
      console.warn('WebGPU falhou, caindo para WASM:', err);
    }
    return await pipeline('automatic-speech-recognition', MODEL_WASM, {
      device: 'wasm',
      progress_callback: updateLoadingProgress,
    });
  })();

  try {
    transcriber = await modelLoadingPromise;
  } finally {
    hideLoadingUI();
    modelLoadingPromise = null;
  }
  return transcriber;
}

function pickMimeType() {
  const candidates = [
    'audio/webm;codecs=opus',
    'audio/webm',
    'audio/mp4',
    'audio/ogg;codecs=opus',
  ];
  for (const m of candidates) {
    if (typeof MediaRecorder !== 'undefined' && MediaRecorder.isTypeSupported(m)) {
      return m;
    }
  }
  return '';
}

async function startRecording() {
  micBtn.disabled = true;
  if (!transcriber) {
    setStatus('Carregando modelo…');
    try {
      await ensureTranscriber();
    } catch (err) {
      console.error(err);
      setStatus('Falha ao carregar modelo. Recarregue a página.');
      micBtn.disabled = false;
      return;
    }
  }

  try {
    activeStream = await navigator.mediaDevices.getUserMedia({
      audio: {
        channelCount: 1,
        echoCancellation: false,
        noiseSuppression: false,
        autoGainControl: false,
      },
    });
  } catch (err) {
    console.error(err);
    setStatus('Permissão de microfone negada.');
    micBtn.disabled = false;
    return;
  }

  recorderMime = pickMimeType();
  recorder = recorderMime
    ? new MediaRecorder(activeStream, { mimeType: recorderMime })
    : new MediaRecorder(activeStream);
  recordedChunks = [];

  recorder.addEventListener('dataavailable', (e) => {
    if (e.data && e.data.size > 0) recordedChunks.push(e.data);
  });
  recorder.addEventListener('stop', handleRecorderStop);
  recorder.start();

  micBtn.classList.add('recording');
  micBtn.disabled = false;
  setStatus('Gravando… clique para parar');
}

function stopRecording() {
  if (!recorder || recorder.state === 'inactive') return;
  recorder.stop();
  if (activeStream) {
    activeStream.getTracks().forEach((t) => t.stop());
    activeStream = null;
  }
  micBtn.classList.remove('recording');
  micBtn.classList.add('processing');
  micBtn.disabled = true;
  setStatus('Transcrevendo…');
}

async function handleRecorderStop() {
  try {
    const mime = recorderMime || (recorder && recorder.mimeType) || 'audio/webm';
    const blob = new Blob(recordedChunks, { type: mime });
    if (blob.size === 0) {
      addCard('');
      setStatus('Pronto para gravar');
      return;
    }
    const audio = await blobToMonoFloat32At16k(blob);
    const language = langEl.value || null;
    const result = await transcriber(audio, {
      task: 'transcribe',
      language,
      return_timestamps: false,
      chunk_length_s: 30,
      stride_length_s: 5,
      no_repeat_ngram_size: 3,
      temperature: 0,
    });
    addCard(result?.text ?? '');
    setStatus('Pronto para gravar');
  } catch (err) {
    console.error(err);
    setStatus('Erro ao transcrever: ' + (err?.message || 'falha desconhecida'));
  } finally {
    micBtn.classList.remove('processing');
    micBtn.disabled = false;
    recordedChunks = [];
  }
}

async function blobToMonoFloat32At16k(blob) {
  const arrayBuffer = await blob.arrayBuffer();
  const decodeCtx = new (window.AudioContext || window.webkitAudioContext)();
  let audioBuffer;
  try {
    audioBuffer = await decodeCtx.decodeAudioData(arrayBuffer.slice(0));
  } finally {
    decodeCtx.close();
  }

  const targetRate = 16000;
  const channels = audioBuffer.numberOfChannels;
  const sourceRate = audioBuffer.sampleRate;

  if (channels === 1 && sourceRate === targetRate) {
    return audioBuffer.getChannelData(0).slice();
  }

  const length = Math.ceil(audioBuffer.duration * targetRate);
  const offline = new OfflineAudioContext(1, length, targetRate);
  const src = offline.createBufferSource();

  if (channels > 1) {
    const mono = offline.createBuffer(1, audioBuffer.length, sourceRate);
    const out = mono.getChannelData(0);
    for (let c = 0; c < channels; c++) {
      const ch = audioBuffer.getChannelData(c);
      for (let i = 0; i < ch.length; i++) out[i] += ch[i] / channels;
    }
    src.buffer = mono;
  } else {
    src.buffer = audioBuffer;
  }

  src.connect(offline.destination);
  src.start();
  const rendered = await offline.startRendering();
  return rendered.getChannelData(0).slice();
}

function addCard(text) {
  const trimmed = (text || '').trim();
  const node = cardTpl.content.firstElementChild.cloneNode(true);
  const timeEl = node.querySelector('.card-time');
  const textEl = node.querySelector('.card-text');
  const copyBtn = node.querySelector('.copy-btn');

  timeEl.textContent = new Date().toLocaleTimeString([], {
    hour: '2-digit',
    minute: '2-digit',
    second: '2-digit',
  });

  if (!trimmed) {
    textEl.textContent = 'Sem fala detectada.';
    node.classList.add('empty');
    copyBtn.disabled = true;
  } else {
    textEl.textContent = trimmed;
  }

  copyBtn.addEventListener('click', async () => {
    if (!trimmed) return;
    try {
      await navigator.clipboard.writeText(trimmed);
      copyBtn.classList.add('copied');
      setTimeout(() => copyBtn.classList.remove('copied'), 1500);
    } catch (err) {
      console.error('Clipboard:', err);
    }
  });

  resultsEl.prepend(node);
  renderIcons(node);
}

micBtn.addEventListener('click', () => {
  if (micBtn.classList.contains('recording')) {
    stopRecording();
  } else {
    startRecording();
  }
});

window.addEventListener('keydown', (e) => {
  if (e.code === 'Space' && document.activeElement === document.body) {
    if (micView.classList.contains('hidden')) return;
    e.preventDefault();
    micBtn.click();
  }
});

installBtn.addEventListener('click', async () => {
  installBtn.disabled = true;
  installView.classList.add('hidden');
  showLoadingUI('Baixando Whisper (~800 MB, uma vez só)…');
  try {
    await ensureTranscriber();
    localStorage.setItem(INSTALLED_KEY, 'true');
    micView.classList.remove('hidden');
    setStatus('Pronto para gravar');
  } catch (err) {
    console.error(err);
    installView.classList.remove('hidden');
    installBtn.disabled = false;
  } finally {
    hideLoadingUI();
  }
});

async function init() {
  const installed = localStorage.getItem(INSTALLED_KEY) === 'true';
  if (!installed) {
    installView.classList.remove('hidden');
    return;
  }
  micView.classList.remove('hidden');
  setStatus('Carregando modelo…');
  try {
    await ensureTranscriber();
    setStatus('Pronto para gravar');
  } catch (err) {
    console.error(err);
    localStorage.removeItem(INSTALLED_KEY);
    micView.classList.add('hidden');
    installView.classList.remove('hidden');
    setStatus('');
  }
}

init();
