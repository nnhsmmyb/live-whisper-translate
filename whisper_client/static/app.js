const $ = (id) => document.getElementById(id);

const btnStart = $("btn-start");
const btnStop = $("btn-stop");
const btnSettings = $("btn-settings");
const statusBadge = $("status-badge");
const audioSource = $("audio-source");
const settingsPanel = $("settings-panel");
const settingsLockedNote = $("settings-locked-note");
const settingsAutoNote = $("settings-auto-note");
const transcriptFeed = $("transcript-feed");
const transcriptTitle = $("transcript-title");
const audioMeter = $("audio-meter");
const serverStatus = $("server-status");
const presetSelect = $("preset-select");
const presetName = $("preset-name");
const btnPresetApply = $("btn-preset-apply");
const btnPresetSave = $("btn-preset-save");
const btnPresetDelete = $("btn-preset-delete");

const configFields = [
  "cfg-lang",
  "cfg-translate-tgt-lang",
  "cfg-whisper-model",
  "cfg-chunk-sec",
  "cfg-whisper-beam",
  "cfg-min-chars",
  "cfg-buffer-chars",
  "cfg-chunk-flush-chars",
  "cfg-translate-timeout",
  "cfg-max-feed-entries",
];

const SRC_LABEL = { en: "EN", es: "ES", ja: "JA" };
const TGT_LABEL = {
  jpn_Jpan: "JA",
  eng_Latn: "EN",
  spa_Latn: "ES",
};

function updateTranscriptTitle(lang, tgtLang) {
  const src = SRC_LABEL[lang] || lang.toUpperCase();
  const tgt = TGT_LABEL[tgtLang] || tgtLang;
  transcriptTitle.textContent = `文字起こし (${src}) / 翻訳 (${tgt})`;
}

let running = false;
const pendingEntries = new Map();
let serverStatusTimer = null;
let maxFeedEntries = 20;

function renderServerStatus(servers) {
  serverStatus.innerHTML = "";
  if (!servers.length) {
    const pill = document.createElement("span");
    pill.className = "server-pill ng";
    pill.textContent = "サーバ未設定";
    serverStatus.appendChild(pill);
    return;
  }

  for (const server of servers) {
    const pill = document.createElement("span");
    pill.className = `server-pill ${server.ok ? "ok" : "ng"}`;
    const title = server.ok
      ? `${server.health.model} / ${server.health.device}`
      : server.error;
    pill.title = title;
    pill.innerHTML = `<span class="dot"></span>${escapeHtml(server.label)}`;
    serverStatus.appendChild(pill);
  }
}

async function refreshServerStatus() {
  try {
    const data = await api("/api/translate-servers");
    renderServerStatus(data.servers);
  } catch {
    renderServerStatus([{ label: "?", ok: false, error: "確認失敗" }]);
  }
}

function startServerStatusPolling() {
  if (serverStatusTimer) {
    clearInterval(serverStatusTimer);
  }
  refreshServerStatus();
  serverStatusTimer = setInterval(refreshServerStatus, 10000);
}

function stopServerStatusPolling() {
  if (serverStatusTimer) {
    clearInterval(serverStatusTimer);
    serverStatusTimer = null;
  }
}

function setRunning(value) {
  running = value;
  btnStart.disabled = value;
  btnStop.disabled = !value;
  statusBadge.textContent = value ? "実行中" : "停止中";
  statusBadge.className = `badge ${value ? "running" : "idle"}`;
  setSettingsEditable(!value);
  if (value) {
    stopServerStatusPolling();
  } else {
    startServerStatusPolling();
  }
}

function setSettingsEditable(editable) {
  for (const id of configFields) {
    $(id).disabled = !editable;
  }
  audioSource.disabled = !editable;
  presetSelect.disabled = !editable;
  presetName.disabled = !editable;
  btnPresetApply.disabled = !editable;
  btnPresetSave.disabled = !editable;
  btnPresetDelete.disabled = !editable;
  settingsPanel.classList.toggle("is-locked", !editable);
  settingsLockedNote.classList.toggle("hidden", editable);
  settingsAutoNote.classList.toggle("hidden", !editable);
  updateSettingsButtonLabel();
}

function updateSettingsButtonLabel() {
  const open = !settingsPanel.classList.contains("hidden");
  btnSettings.setAttribute("aria-expanded", open ? "true" : "false");
  btnSettings.textContent = open ? "設定パネルを隠す" : "設定パネルを表示";
}

function scrollFeed() {
  transcriptFeed.scrollTop = 0;
}

function removePendingEntry(el) {
  for (const [key, pendingEl] of pendingEntries) {
    if (pendingEl === el) {
      pendingEntries.delete(key);
      return;
    }
  }
}

function trimFeed() {
  while (transcriptFeed.children.length > maxFeedEntries) {
    const el = transcriptFeed.lastElementChild;
    removePendingEntry(el);
    el.remove();
  }
}

function createEntry(sourceText) {
  const el = document.createElement("div");
  el.className = "entry";
  el.innerHTML = `
    <div class="source">${escapeHtml(sourceText)}</div>
    <div class="translation pending">翻訳中...</div>
    <div class="meta"></div>
  `;
  transcriptFeed.prepend(el);
  trimFeed();
  scrollFeed();
  return el;
}

function appendError(message) {
  const el = document.createElement("div");
  el.className = "entry error";
  el.innerHTML = `
    <div class="meta">エラー</div>
    <div class="translation">${escapeHtml(message)}</div>
  `;
  transcriptFeed.prepend(el);
  trimFeed();
  scrollFeed();
}

async function api(path, options = {}) {
  const res = await fetch(path, {
    headers: { "Content-Type": "application/json" },
    ...options,
  });
  if (!res.ok) {
    const text = await res.text();
    let message = text;
    try {
      const json = JSON.parse(text);
      if (json.detail) {
        message = typeof json.detail === "string" ? json.detail : JSON.stringify(json.detail);
      }
    } catch {
      // plain text error
    }
    throw new Error(message);
  }
  return res.json();
}

function readConfigFromForm() {
  return {
    lang: $("cfg-lang").value,
    translate_tgt_lang: $("cfg-translate-tgt-lang").value,
    chunk_sec: Number($("cfg-chunk-sec").value),
    whisper_model: $("cfg-whisper-model").value,
    whisper_beam: Number($("cfg-whisper-beam").value),
    min_chars: Number($("cfg-min-chars").value),
    buffer_chars: Number($("cfg-buffer-chars").value),
    chunk_flush_chars: Number($("cfg-chunk-flush-chars").value),
    translate_timeout: Number($("cfg-translate-timeout").value),
    max_feed_entries: Number($("cfg-max-feed-entries").value) || 20,
  };
}

function applyConfigToForm(config) {
  $("cfg-lang").value = config.lang;
  $("cfg-translate-tgt-lang").value = config.translate_tgt_lang || "jpn_Jpan";
  $("cfg-whisper-model").value = config.whisper_model;
  $("cfg-chunk-sec").value = config.chunk_sec;
  $("cfg-whisper-beam").value = config.whisper_beam;
  $("cfg-min-chars").value = config.min_chars;
  $("cfg-buffer-chars").value = config.buffer_chars;
  $("cfg-chunk-flush-chars").value = config.chunk_flush_chars ?? 0;
  $("cfg-translate-timeout").value = config.translate_timeout;
  $("cfg-max-feed-entries").value = config.max_feed_entries > 0 ? config.max_feed_entries : 20;
  maxFeedEntries = Number($("cfg-max-feed-entries").value) || 20;
  trimFeed();
  updateTranscriptTitle(config.lang, config.translate_tgt_lang || "jpn_Jpan");
}

async function loadPresets(selected = "") {
  const { names } = await api("/api/presets");
  presetSelect.innerHTML = '<option value="">—</option>';
  for (const name of names) {
    const opt = document.createElement("option");
    opt.value = name;
    opt.textContent = name;
    presetSelect.appendChild(opt);
  }
  if (selected && names.includes(selected)) {
    presetSelect.value = selected;
    presetName.value = selected;
  }
}

async function loadConfig() {
  const config = await api("/api/config");
  applyConfigToForm(config);
}

async function loadAudioSources() {
  const data = await api("/api/audio-sources");
  audioSource.innerHTML = "";
  for (const src of data.sources) {
    const opt = document.createElement("option");
    opt.value = src.name;
    const suffix = src.is_default ? " [default]" : "";
    opt.textContent = `${src.description} (${src.name})${suffix}`;
    audioSource.appendChild(opt);
  }
  if (data.selected) {
    audioSource.value = data.selected;
  }
}

function showError(err) {
  appendError(String(err.message));
}

async function saveConfig() {
  if (running) {
    return;
  }
  const body = readConfigFromForm();
  await api("/api/config", { method: "PUT", body: JSON.stringify(body) });
  maxFeedEntries = body.max_feed_entries || 20;
  trimFeed();
  updateTranscriptTitle(body.lang, body.translate_tgt_lang);
  refreshServerStatus();
}

async function savePreset() {
  if (running) {
    return;
  }
  const name = presetName.value.trim();
  if (!name) {
    throw new Error("プリセット名を入力してください");
  }
  const body = readConfigFromForm();
  await api(`/api/presets/${encodeURIComponent(name)}`, {
    method: "PUT",
    body: JSON.stringify(body),
  });
  await loadPresets(name);
}

async function applyPreset() {
  if (running) {
    return;
  }
  const name = presetSelect.value;
  if (!name) {
    throw new Error("プリセットを選択してください");
  }
  const config = await api(`/api/presets/${encodeURIComponent(name)}/apply`, {
    method: "POST",
  });
  applyConfigToForm(config);
  presetName.value = name;
  refreshServerStatus();
}

function handleEvent(data) {
  if (data.type === "status") {
    setRunning(!!data.running);
  }
  if (data.type === "transcription") {
    const el = createEntry(data.text);
    pendingEntries.set(data.text, el);
  }
  if (data.type === "translation") {
    let el = pendingEntries.get(data.source);
    if (!el) {
      el = createEntry(data.source);
    }
    const translationEl = el.querySelector(".translation");
    translationEl.textContent = data.text;
    translationEl.classList.remove("pending");
    el.querySelector(".meta").textContent =
      `${data.elapsed}s / ${data.source.length}字 / ${data.gpu}`;
    pendingEntries.delete(data.source);
    transcriptFeed.prepend(el);
    trimFeed();
    scrollFeed();
  }
  if (data.type === "audio_level") {
    audioMeter.style.width = `${Math.round(data.level * 100)}%`;
  }
  if (data.type === "error") {
    appendError(data.message);
  }
}

function connectEvents() {
  const es = new EventSource("/api/events");
  es.onmessage = (event) => handleEvent(JSON.parse(event.data));
  es.onerror = () => {
    es.close();
    setTimeout(connectEvents, 2000);
  };
}

function escapeHtml(text) {
  return text
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;");
}

btnStart.addEventListener("click", async () => {
  try {
    await saveConfig();
    await api("/api/start", { method: "POST" });
    setRunning(true);
  } catch (err) {
    appendError(String(err.message));
  }
});

btnStop.addEventListener("click", async () => {
  await api("/api/stop", { method: "POST" });
  setRunning(false);
});

btnSettings.addEventListener("click", () => {
  settingsPanel.classList.toggle("hidden");
  updateSettingsButtonLabel();
});

for (const id of configFields) {
  $(id).addEventListener("change", () => {
    saveConfig().catch(showError);
  });
}

audioSource.addEventListener("change", () => {
  api("/api/audio-source", {
    method: "PUT",
    body: JSON.stringify({ audio_source: audioSource.value }),
  }).catch(showError);
});

presetSelect.addEventListener("change", () => {
  if (presetSelect.value) {
    presetName.value = presetSelect.value;
  }
});

btnPresetApply.addEventListener("click", () => {
  applyPreset().catch(showError);
});

btnPresetSave.addEventListener("click", () => {
  savePreset().catch(showError);
});

btnPresetDelete.addEventListener("click", async () => {
  if (running) {
    return;
  }
  const name = presetSelect.value || presetName.value.trim();
  if (!name) {
    showError(new Error("削除するプリセットを選択してください"));
    return;
  }
  if (!window.confirm(`プリセット「${name}」を削除しますか？`)) {
    return;
  }
  try {
    await api(`/api/presets/${encodeURIComponent(name)}`, { method: "DELETE" });
    presetSelect.value = "";
    presetName.value = "";
    await loadPresets();
  } catch (err) {
    showError(err);
  }
});

(async () => {
  await loadConfig();
  await loadAudioSources();
  await loadPresets();
  const status = await api("/api/status");
  setRunning(status.running);
  connectEvents();
})();
