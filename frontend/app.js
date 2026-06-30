/* ═══════════════════════════════════════════════════
   EmotionLens — App Logic  (v4 · all English UI)
   One model, many lenses.
   ═══════════════════════════════════════════════════ */
(() => {
  'use strict';

  /* ─── Constants ─── */
  const WS_URL      = `ws://${location.hostname || 'localhost'}:8000/ws`;
  const CAPTURE_W   = 640;
  const CAPTURE_H   = 480;
  const JPEG_Q      = 0.6;
  const SEND_INTERVAL = 100;   // ~10 fps
  const LERP_RATE   = 0.35;

  // Order MUST match backend EMO_CLASSES (config.py).
  // datasets[7] marker index in m3 timeline chart relies on length === 7.
  const EMOTIONS = ['neutral','happiness','surprise','sadness','anger','disgust','fear'];

  const EMOJI_MAP = {
    neutral:'\u{1f610}', happiness:'\u{1f604}', surprise:'\u{1f632}',
    sadness:'\u{1f622}', anger:'\u{1f620}', disgust:'\u{1f922}', fear:'\u{1f628}'
  };

  const TYPE_NAMES = {
    happiness:'ELECTRIC', neutral:'NORMAL', surprise:'PSYCHIC',
    sadness:'WATER', anger:'FIRE', disgust:'POISON', fear:'GHOST'
  };

  const EMO_COLORS = {
    happiness:'#FFD700', neutral:'#A8A878', surprise:'#F85888',
    sadness:'#6890F0', fear:'#705898', disgust:'#A040A0', anger:'#F08030'
  };

  /* Lens definitions — maps to backend config.LENS_DEFS */
  const LENS_DEFS = {
    m0: { title:'Live',               icon:'\u{1f4ca}', timer:false, durations:[] },
    m1: { title:'Cafeteria Mood',     icon:'\u{1f37d}\u{fe0f}', timer:true,  durations:[30,60,180,600] },
    m2: { title:'CODE RED',           icon:'\u{1f6a8}', timer:false, durations:[] },
    m3: { title:'Audience Reactions', icon:'\u{1f3ac}', timer:false, durations:[] },
    m4: { title:'Speech Coach',       icon:'\u{1f3a4}', timer:true,  durations:[10,30,60,180] },
    m5: { title:'Mimic Game',         icon:'\u{1f3ae}', timer:false, durations:[] },
  };

  /* ═══════════════ DOM refs ═══════════════ */
  const $loading      = document.getElementById('loading-overlay');
  const $connBadge    = document.getElementById('conn-badge');
  const $connLabel    = $connBadge.querySelector('.conn-label');
  const $fpsBadge     = document.getElementById('fps-badge');
  const $modelSelect  = document.getElementById('model-select');
  const $video        = document.getElementById('webcam');
  const $overlay      = document.getElementById('overlay');
  const $captureCanvas= document.getElementById('capture-canvas');
  const $placeholder  = document.getElementById('cam-placeholder');
  const $faceCount    = document.getElementById('face-count');
  const $lensTitle    = document.getElementById('lens-title');
  const $dominantDisplay = document.querySelector('.dominant-display');
  const $dominantEmoji= document.getElementById('dominant-emoji');
  const $dominantName = document.getElementById('dominant-name');
  const $dominantBadge= document.getElementById('dominant-badge');
  const $dominantPct  = document.getElementById('dominant-pct');
  const $emoBars      = document.getElementById('emotion-bars');
  const $modePanelTitle = document.getElementById('mode-panel-title');
  const $modeContent  = document.getElementById('mode-content');
  const $camRes       = document.getElementById('cam-res');

  /* ═══════════════ State ═══════════════ */
  let ws = null;
  let currentMode = 'm0';
  let connected = false;
  let cameraReady = false;
  let latestFaces = [];
  let latestModeOutput = {};
  let trackedFaces = {};

  // FPS
  let frameCount = 0;
  let lastFpsTime = performance.now();
  let displayFps = 0;

  // Sidebar animation targets
  let animatedProbs = {};
  EMOTIONS.forEach(e => animatedProbs[e] = 0);
  let animatedDominantPct = 0;
  let currentDominantEmo = 'neutral';

  // Lens control state
  let lensControlState = {};  // per-mode: { state:'idle', duration:0 }

  // Face selection (Phase 3) — selectedTrackId persists across mode switches.
  // selectedLostSince marks when the selected face vanished; >5s → auto-clear.
  let selectedTrackId = null;
  let selectedLostSince = null;
  let lastSelectedFace = null;     // sticky snapshot for sidebar while Lost
  const LOST_TIMEOUT_MS = 5000;

  // Frame dimensions from the most recent result (drives canvas internal size).
  let frameW = 0, frameH = 0;

  // Chart.js instances
  let pieChartL2 = null;
  let pieChartL3 = null;
  let lineChartL3 = null;

  /* ═══════════════ Share hint ─────────────────── */
  function updateShareHint() {
    const $hint = document.getElementById('share-hint');
    if (!$hint) return;
    const isLocal = location.hostname === 'localhost' || location.hostname === '127.0.0.1';
    const isSecure = location.protocol === 'https:';
    if (isLocal) {
      $hint.textContent = '\u{1f517} Share: check terminal for public URL';
      $hint.style.color = 'var(--text-muted)';
    } else if (isSecure) {
      $hint.textContent = '\u{1f310} ' + location.origin;
      $hint.style.color = 'var(--c-happiness)';
      $hint.style.fontWeight = '900';
    } else {
      $hint.textContent = '⚠️ HTTP — camera may be blocked';
      $hint.style.color = 'var(--c-anger)';
    }
  }

  /* ═══════════════ INIT ═══════════════ */
  document.addEventListener('DOMContentLoaded', () => {
    updateShareHint();
    buildEmotionBars();
    initSwiper();
    setupModelPicker();
    setupCanvasClick();
    startCamera();
    connectWS();
    requestAnimationFrame(renderLoop);
  });

  /* ═══════════════ MODEL PICKER ═══════════════ */
  async function setupModelPicker() {
    if (!$modelSelect) return;
    try {
      const res = await fetch('/models');
      const data = await res.json();
      const models = data.models || [];
      $modelSelect.innerHTML = models.map(m => {
        const sel = m.key === data.active ? ' selected' : '';
        const dis = m.available ? '' : ' disabled';
        const tag = m.available ? '' : ' (missing)';
        return `<option value="${m.key}"${sel}${dis}>${m.label}${tag}</option>`;
      }).join('');
      $modelSelect.disabled = false;
      $modelSelect.addEventListener('change', () => {
        const key = $modelSelect.value;
        if (!key) return;
        $modelSelect.disabled = true;
        $modelSelect.dataset.pending = '1';
        wsSend({ type: 'set_model', key });
      });
    } catch (err) {
      console.warn('Model list fetch failed:', err);
      $modelSelect.innerHTML = '<option>—</option>';
    }
  }

  /* ─── Emotion bars ─── */
  function buildEmotionBars() {
    $emoBars.innerHTML = EMOTIONS.map(emo => `
      <div class="emo-row" data-emo="${emo}">
        <span class="emo-row__emoji">${EMOJI_MAP[emo]}</span>
        <span class="emo-row__label">${emo.slice(0,8)}</span>
        <div class="emo-row__track">
          <div class="emo-row__fill" id="bar-${emo}" style="width:0%"></div>
        </div>
        <span class="emo-row__val" id="val-${emo}">0%</span>
      </div>
    `).join('');
  }

  /* ─── Swiper ─── */
  let swiperInstance;
  function initSwiper() {
    swiperInstance = new Swiper('.mode-swiper', {
      slidesPerView: 'auto', centeredSlides: true, spaceBetween: 14,
      effect: 'coverflow',
      coverflowEffect: { rotate:0, stretch:0, depth:80, modifier:1.2, slideShadows:false },
      navigation: { prevEl:'.swiper-button-prev', nextEl:'.swiper-button-next' },
      on: {
        click(swiper, e) {
          const slide = e.target.closest('.mode-slide');
          if (!slide) return;
          const mode = slide.dataset.mode;
          if (mode && mode !== currentMode) selectMode(mode);
        }
      }
    });
  }

  function selectMode(mode) {
    currentMode = mode;
    document.body.dataset.activeMode = mode;

    // Update carousel
    document.querySelectorAll('.mode-slide__inner').forEach(el => el.classList.remove('active'));
    const activeSlide = document.querySelector(`.mode-slide[data-mode="${mode}"] .mode-slide__inner`);
    if (activeSlide) activeSlide.classList.add('active');

    // Update lens title
    const def = LENS_DEFS[mode] || { title: mode };
    $lensTitle.textContent = def.title || mode;
    $modePanelTitle.textContent = def.title || mode;

    // Notify server
    wsSend({ type: 'set_mode', mode });

    // Reset & build panel
    latestModeOutput = {};
    lensControlState[mode] = lensControlState[mode] || { state:'idle', duration:def.durations?.[0]||0 };
    destroyCharts();
    buildModePanel();
    updateModePanel();
  }

  /* ═══════════════ CAMERA ═══════════════ */
  async function startCamera() {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { width:{ideal:CAPTURE_W}, height:{ideal:CAPTURE_H}, facingMode:'user' },
        audio: false
      });
      $video.srcObject = stream;
      await $video.play();
      cameraReady = true;
      $placeholder.classList.add('hidden');
      const settings = stream.getVideoTracks()[0].getSettings();
      $camRes.textContent = `${settings.width||CAPTURE_W}×${settings.height||CAPTURE_H}`;
      setTimeout(() => $loading.classList.add('hidden'), 600);
      setInterval(sendFrame, SEND_INTERVAL);
    } catch (err) {
      console.error('Camera error:', err);
      $placeholder.querySelector('p').textContent = `Camera error: ${err.message}`;
      $loading.classList.add('hidden');
    }
  }

  function sendFrame() {
    if (!cameraReady || !connected || ws.readyState !== WebSocket.OPEN) return;
    $captureCanvas.width  = CAPTURE_W;
    $captureCanvas.height = CAPTURE_H;
    const ctx = $captureCanvas.getContext('2d');
    ctx.drawImage($video, 0, 0, CAPTURE_W, CAPTURE_H);
    const dataUrl = $captureCanvas.toDataURL('image/jpeg', JPEG_Q);
    wsSend({ type:'frame', ts: Date.now()/1000, data: dataUrl });
  }

  /* ═══════════════ OVERLAY DRAWING ═══════════════ */
  function drawOverlay(faces) {
    if (!$overlay || !frameW || !frameH) return;
    if ($overlay.width !== frameW)  $overlay.width = frameW;
    if ($overlay.height !== frameH) $overlay.height = frameH;
    const ctx = $overlay.getContext('2d');
    ctx.clearRect(0, 0, frameW, frameH);

    // Backend bbox is in raw (non-mirrored) frame coords. The visible <video>
    // is CSS-mirrored, so we mirror bbox X here to land on the right face.
    for (const f of faces) {
      const [x, y, w, h] = f.bbox;
      const dx = frameW - x - w;
      const color = EMO_COLORS[f.dominant] || '#fff';
      const isSel = (f.track_id === selectedTrackId);

      ctx.lineWidth = isSel ? 4 : 2;
      ctx.strokeStyle = color;
      ctx.strokeRect(dx, y, w, h);

      // Top label: emoji + dominant emotion + %
      const emoji = EMOJI_MAP[f.dominant] || '';
      const pct = Math.round((f.probs?.[f.dominant] || f.conf || 0) * 100);
      const labelFont = isSel ? 'bold 22px Inter, sans-serif' : 'bold 16px Inter, sans-serif';
      const labelText = `${emoji}  ${f.dominant}  ${pct}%`;
      ctx.font = labelFont;
      const tw = ctx.measureText(labelText).width + 16;
      const th = isSel ? 30 : 24;
      ctx.fillStyle = color;
      ctx.fillRect(dx, y - th, tw, th);
      ctx.fillStyle = '#000';
      ctx.fillText(labelText, dx + 8, y - (isSel ? 9 : 7));

      // Selected face: track-id badge top-left
      if (isSel) {
        const badge = `#${f.track_id}`;
        ctx.font = 'bold 18px "Space Grotesk", monospace';
        const bw = ctx.measureText(badge).width + 12;
        ctx.fillStyle = '#000';
        ctx.fillRect(dx, y, bw, 24);
        ctx.fillStyle = '#fff';
        ctx.fillText(badge, dx + 6, y + 18);
      }
    }

    // "Lost · 2.3s" indicator if selected face has temporarily vanished
    if (selectedTrackId !== null && selectedLostSince !== null) {
      const lostSec = (performance.now() - selectedLostSince) / 1000;
      ctx.font = 'bold 18px Inter, sans-serif';
      const txt = `Lost · ${lostSec.toFixed(1)}s`;
      ctx.fillStyle = 'rgba(0,0,0,0.7)';
      const w = ctx.measureText(txt).width + 16;
      ctx.fillRect(12, 12, w, 28);
      ctx.fillStyle = '#FFD700';
      ctx.fillText(txt, 20, 32);
    }
  }

  /* ═══════════════ CLICK-TO-SELECT ═══════════════ */
  function setupCanvasClick() {
    if (!$overlay) return;
    $overlay.addEventListener('click', (e) => {
      if (!frameW || !frameH || !latestFaces.length) return;
      const rect = $overlay.getBoundingClientRect();
      // CSS-mirrored video → canvas itself is NOT transformed, but bbox draw
      // mirrors X. So click in viewport → map back to canvas raw coords by
      // mirroring X again.
      const clickXVisual = (e.clientX - rect.left) * (frameW / rect.width);
      const clickY       = (e.clientY - rect.top)  * (frameH / rect.height);
      const clickX       = frameW - clickXVisual;  // undo CSS mirror

      // Hit-test against raw bboxes
      for (const f of latestFaces) {
        const [x, y, w, h] = f.bbox;
        if (clickX >= x && clickX <= x + w && clickY >= y && clickY <= y + h) {
          selectedTrackId = f.track_id;
          selectedLostSince = null;
          lastSelectedFace = f;
          wsSend({ type: 'set_focus', track_id: f.track_id });
          return;
        }
      }
      // Clicked empty space → clear selection
      selectedTrackId = null;
      selectedLostSince = null;
      lastSelectedFace = null;
      wsSend({ type: 'set_focus', track_id: null });
    });
  }

  /* ═══════════════ WEBSOCKET ═══════════════ */
  function connectWS() {
    if (ws && (ws.readyState === WebSocket.OPEN || ws.readyState === WebSocket.CONNECTING)) return;
    ws = new WebSocket(WS_URL);
    ws.onopen = () => {
      connected = true;
      $connBadge.classList.add('connected');
      $connLabel.textContent = 'Connected';
      wsSend({ type:'set_mode', mode: currentMode });
    };
    ws.onclose = () => {
      connected = false;
      $connBadge.classList.remove('connected');
      $connLabel.textContent = 'Disconnected';
      setTimeout(connectWS, 2000);
    };
    ws.onerror = () => { ws.close(); };
    ws.onmessage = (evt) => {
      try {
        const msg = JSON.parse(evt.data);
        if (msg.type === 'result') handleResult(msg);
        else if (msg.type === 'model_changed') handleModelChanged(msg);
        else if (msg.type === 'model_error') handleModelError(msg);
      } catch(e) {}
    };
  }

  function handleModelChanged(msg) {
    if ($modelSelect) {
      if (msg.key) $modelSelect.value = msg.key;
      $modelSelect.disabled = false;
      delete $modelSelect.dataset.pending;
    }
  }

  function handleModelError(msg) {
    console.error('Model switch failed:', msg.error);
    if ($modelSelect) {
      $modelSelect.disabled = false;
      delete $modelSelect.dataset.pending;
    }
    // brief flash to indicate failure
    if ($modelSelect) {
      $modelSelect.classList.add('error');
      setTimeout(() => $modelSelect.classList.remove('error'), 1200);
    }
  }

  function wsSend(obj) {
    if (ws && ws.readyState === WebSocket.OPEN) ws.send(JSON.stringify(obj));
  }

  /* ─── Handle result ─── */
  function handleResult(msg) {
    latestFaces = msg.faces || [];
    latestModeOutput = msg.mode_output || {};
    if (msg.frame_width)  frameW = msg.frame_width;
    if (msg.frame_height) frameH = msg.frame_height;
    drawOverlay(latestFaces);

    const now = performance.now();
    const seenIds = new Set();
    for (const face of latestFaces) {
      seenIds.add(face.track_id);
      if (trackedFaces[face.track_id]) {
        trackedFaces[face.track_id].targetBbox = [...face.bbox];
        trackedFaces[face.track_id].dominant = face.dominant;
        trackedFaces[face.track_id].probs = face.probs || {};
        trackedFaces[face.track_id].conf = face.conf;
        trackedFaces[face.track_id].lastSeen = now;
        if (trackedFaces[face.track_id].emoji !== EMOJI_MAP[face.dominant]) {
          trackedFaces[face.track_id].emoji = EMOJI_MAP[face.dominant] || '\u{1f610}';
          trackedFaces[face.track_id].emojiBounce = now;
        }
      } else {
        trackedFaces[face.track_id] = {
          bbox:[...face.bbox], targetBbox:[...face.bbox],
          dominant:face.dominant, probs:face.probs||{}, conf:face.conf,
          emoji:EMOJI_MAP[face.dominant]||'\u{1f610}', emojiBounce:now, lastSeen:now
        };
      }
    }
    for (const id of Object.keys(trackedFaces)) {
      if (!seenIds.has(parseInt(id)) && !seenIds.has(id)) {
        if (now - trackedFaces[id].lastSeen > 1500) delete trackedFaces[id];
      }
    }

    updateSidebar();
    updateModePanel();
  }

  /* ═══════════════ SIDEBAR ═══════════════ */
  function pickSidebarFace() {
    // 1. If a track is explicitly selected, prefer it
    if (selectedTrackId !== null) {
      const live = latestFaces.find(f => f.track_id === selectedTrackId);
      if (live) {
        selectedLostSince = null;
        lastSelectedFace = live;
        return live;
      }
      // Selected face missing — sticky on last snapshot, start/continue Lost timer
      if (selectedLostSince === null) selectedLostSince = performance.now();
      if (performance.now() - selectedLostSince > LOST_TIMEOUT_MS) {
        // Auto-clear after timeout
        selectedTrackId = null;
        selectedLostSince = null;
        lastSelectedFace = null;
        wsSend({ type: 'set_focus', track_id: null });
      } else if (lastSelectedFace) {
        return lastSelectedFace;
      }
    }
    // 2. Default: first detected face
    return latestFaces[0] || null;
  }

  function updateSidebar() {
    const face = pickSidebarFace();
    if (!face) return;
    const probs = face.probs || {};
    const dominant = face.dominant || 'neutral';

    EMOTIONS.forEach(emo => {
      const target = (probs[emo] || 0) * 100;
      animatedProbs[emo] = target;
    });
    const pct = Math.round((probs[dominant] || 0) * 100);
    animatedDominantPct = pct;

    if (dominant !== currentDominantEmo) {
      currentDominantEmo = dominant;
      $dominantEmoji.textContent = EMOJI_MAP[dominant] || '\u{1f610}';
      $dominantEmoji.classList.add('bounce');
      setTimeout(() => $dominantEmoji.classList.remove('bounce'), 350);
    }
    // Bind sidebar border color to current emotion (color-coded selection cue)
    if ($dominantDisplay) {
      $dominantDisplay.style.borderColor = EMO_COLORS[dominant] || '#000';
    }
    $dominantName.textContent = dominant;
    $dominantBadge.textContent = TYPE_NAMES[dominant] || dominant.toUpperCase();
    $dominantBadge.dataset.type = dominant;
    $dominantPct.textContent = pct + '%';

    EMOTIONS.forEach(emo => {
      const val = Math.round((probs[emo] || 0) * 100);
      const fillEl = document.getElementById(`bar-${emo}`);
      const valEl  = document.getElementById(`val-${emo}`);
      if (fillEl) fillEl.style.width = val + '%';
      if (valEl)  valEl.textContent = val + '%';
    });

    $faceCount.textContent = latestFaces.length === 0 ? 'No faces detected'
      : latestFaces.length === 1 ? '1 face detected'
      : `${latestFaces.length} faces detected`;
  }

  /* ═══════════════ CONTROL PROTOCOL ═══════════════ */
  function sendControl(mode, action, duration) {
    wsSend({ type:'control', mode, action, duration: duration || undefined });
    if (!lensControlState[mode]) lensControlState[mode] = {};
    lensControlState[mode].state = action === 'start' ? 'running'
      : action === 'reset' ? 'idle' : lensControlState[mode].state;
    if (duration) lensControlState[mode].duration = duration;
  }

  /* ═══════════════ BUILD MODE PANELS ═══════════════ */
  function buildModePanel() {
    const def = LENS_DEFS[currentMode] || {};
    const hasTimer = def.timer;
    const durations = def.durations || [];
    const cs = lensControlState[currentMode] || { state:'idle', duration:durations[0]||0 };

    // ── Timer controls (shared by L1, L4) ──
    let timerHTML = '';
    if (hasTimer && durations.length) {
      const pills = durations.map(d => {
        const label = d >= 60 ? (d/60)+'m' : d+'s';
        const sel = cs.duration === d ? ' selected' : '';
        return `<button class="dur-pill${sel}" data-dur="${d}">${label}</button>`;
      }).join('');
      timerHTML = `
        <div class="timer-ctrl">
          <div class="dur-pills">${pills}</div>
          <div class="timer-main">
            <span class="timer-big" id="timer-display">--:--</span>
            <button class="ctrl-btn ctrl-start" id="ctrl-start">Start</button>
          </div>
        </div>`;
    }

    let bodyHTML = '';
    switch (currentMode) {
      case 'm0':
        bodyHTML = `<div class="m0-desc"><p>Real-time emotion probabilities for all detected faces.</p></div>`;
        break;

      case 'm1':
        bodyHTML = timerHTML + `
          <div class="settle-card" id="settle-m1" style="display:none">
            <div class="settle-ring-wrap">
              <svg viewBox="0 0 80 80"><circle class="sat-ring-bg" cx="40" cy="40" r="33"/>
                <circle class="sat-ring-fg" id="m1-ring" cx="40" cy="40" r="33" stroke-dasharray="207.345" stroke-dashoffset="207.345"/></svg>
              <div class="sat-ring-val" id="m1-val">0%</div>
            </div>
            <div class="settle-bars">
              <div class="sat-bar-row"><span class="sat-bar-label">POS</span><div class="sat-bar-track"><div class="sat-bar-fill pos" id="m1-pos" style="width:0%"></div></div><span id="m1-pos-v">0%</span></div>
              <div class="sat-bar-row"><span class="sat-bar-label">NEU</span><div class="sat-bar-track"><div class="sat-bar-fill neu" id="m1-neu" style="width:0%"></div></div><span id="m1-neu-v">0%</span></div>
              <div class="sat-bar-row"><span class="sat-bar-label">NEG</span><div class="sat-bar-track"><div class="sat-bar-fill neg" id="m1-neg" style="width:0%"></div></div><span id="m1-neg-v">0%</span></div>
            </div>
            <div class="settle-main" id="m1-main"></div>
            <div class="settle-verdict" id="m1-verdict"></div>
            <div class="settle-dist" id="m1-dist"></div>
          </div>`;
        break;

      case 'm2':
        bodyHTML = `
          <div class="codered-grid">
            <div class="pie-chart-wrap"><canvas id="pie-l2"></canvas></div>
            <div class="codered-right">
              <div class="risk-label"><span>LOW</span><span id="m2-risk-t">Risk: 0%</span><span>HIGH</span></div>
              <div class="risk-meter"><div class="risk-meter__fill" id="m2-risk-f" style="width:0%"></div></div>
              <div class="alarm-indicator" id="m2-alarm">✅ No Alarm</div>
              <p class="codered-threat">Threat Level: <strong id="m2-threat">Normal</strong></p>
            </div>
          </div>
          <div class="banner-overlay" id="banner-overlay"><span id="banner-text"></span></div>`;
        break;

      case 'm3':
        bodyHTML = `
          <div class="audience-top-row">
            <div class="audience-dominant"><span class="audience-dom-label">DOMINANT</span><span class="audience-dom-val" id="m3-dom">Neutral</span><span class="audience-dom-valence">V: <strong id="m3-val">0.000</strong></span></div>
            <div class="pie-chart-wrap pie-l3-wrap"><canvas id="pie-l3"></canvas></div>
            <button class="ctrl-btn ctrl-clear" id="m3-clear" title="Reset timeline & pie">Clear</button>
          </div>
          <div class="timeline-chart-wrap">
            <canvas id="timeline-l3"></canvas>
          </div>
          <div class="timeline-legend" id="m3-legend"></div>`;
        break;

      case 'm4':
        bodyHTML = timerHTML + `
          <div class="settle-card" id="settle-m4" style="display:none">
            <div class="coach-stats">
              <div class="coach-stat"><div class="coach-stat__label">Positivity</div><div class="coach-stat__val" id="m4-pos">0%</div></div>
              <div class="coach-stat"><div class="coach-stat__label">Anxiety</div><div class="coach-stat__val" id="m4-anx">0%</div></div>
              <div class="coach-stat"><div class="coach-stat__label">Expressiveness</div><div class="coach-stat__val" id="m4-exp">0%</div></div>
              <div class="coach-stat"><div class="coach-stat__label">Neutral %</div><div class="coach-stat__val" id="m4-neu">0%</div></div>
            </div>
            <div class="coach-advice" id="m4-adv">\u{1f4a1} Start a session to get feedback.</div>
            <div class="coach-source" id="m4-source"></div>
            <div class="coach-timeline" id="m4-timeline"></div>
          </div>`;
        break;

      case 'm5':
        bodyHTML = `
          <div class="mimic-panel">
            <p class="mimic-label">MIMIC THIS</p>
            <div class="mimic-target" id="m5-tgt">\u{1f604}</div>
            <div class="mimic-ring-wrap">
              <svg viewBox="0 0 72 72"><circle class="mimic-ring-bg" cx="36" cy="36" r="28"/>
                <circle class="mimic-ring-fg" id="m5-ring" cx="36" cy="36" r="28" stroke-dasharray="175.929" stroke-dashoffset="175.929" style="stroke:#FFD700"/></svg>
              <div class="mimic-ring-val" id="m5-val">0%</div>
            </div>
            <div class="mimic-stats">
              <span>Score <strong id="m5-score">0</strong></span>
              <span>Combo <strong id="m5-combo">x0</strong></span>
              <span>Round <strong id="m5-round">0</strong></span>
            </div>
            <div class="mimic-timer" id="m5-time">0:00</div>
            <button class="ctrl-btn ctrl-start" id="m5-start-btn">Start Game</button>
            <div class="game-over-card" id="m5-gameover" style="display:none">
              <h3>Game Over</h3>
              <p>Final Score: <strong id="m5-final">0</strong></p>
              <p>Best Streak: <strong id="m5-best">x0</strong></p>
              <button class="ctrl-btn ctrl-start" id="m5-replay">Play Again</button>
            </div>
          </div>`;
        break;
    }

    $modeContent.innerHTML = (timerHTML ? timerHTML + bodyHTML.replace(timerHTML, '') : bodyHTML);

    // Bind control events
    bindControlEvents();
  }

  /* ═══════════════ BIND CONTROL EVENTS ═══════════════ */
  function bindControlEvents() {
    // Duration pills
    document.querySelectorAll('.dur-pill').forEach(btn => {
      btn.addEventListener('click', () => {
        const dur = parseInt(btn.dataset.dur);
        const cs = lensControlState[currentMode] || {};
        cs.duration = dur;
        lensControlState[currentMode] = cs;
        document.querySelectorAll('.dur-pill').forEach(b => b.classList.remove('selected'));
        btn.classList.add('selected');
      });
    });

    // Start button
    const startBtn = document.getElementById('ctrl-start');
    if (startBtn) {
      startBtn.addEventListener('click', () => {
        const cs = lensControlState[currentMode] || {};
        const dur = cs.duration || (LENS_DEFS[currentMode]?.durations?.[0] || 30);
        sendControl(currentMode, 'start', dur);
        startBtn.textContent = 'Running...';
        startBtn.disabled = true;
      });
    }

    // M3 Clear button → reset audience-reactions timeline/markers/pie
    const m3clear = document.getElementById('m3-clear');
    if (m3clear) {
      m3clear.addEventListener('click', () => {
        sendControl('m3', 'reset');
        // Clear chart locally so the user sees immediate reset
        if (lineChartL3) {
          lineChartL3.data.labels = [];
          lineChartL3.data.datasets.forEach(d => { d.data = []; });
          lineChartL3.update('none');
        }
        if (pieChartL3) {
          pieChartL3.data.datasets[0].data = EMOTIONS.map(() => 0);
          pieChartL3.update('none');
        }
      });
    }

    // M5 start/replay
    const m5start = document.getElementById('m5-start-btn');
    if (m5start) {
      m5start.addEventListener('click', () => {
        sendControl('m5', 'start', 60);
        m5start.style.display = 'none';
        const go = document.getElementById('m5-gameover');
        if (go) go.style.display = 'none';
      });
    }
    const m5replay = document.getElementById('m5-replay');
    if (m5replay) {
      m5replay.addEventListener('click', () => {
        sendControl('m5', 'start', 60);
        const go = document.getElementById('m5-gameover');
        if (go) go.style.display = 'none';
        m5replay.style.display = 'none';
      });
    }
  }

  /* ═══════════════ DESTROY CHARTS ═══════════════ */
  function destroyCharts() {
    if (pieChartL2) { pieChartL2.destroy(); pieChartL2 = null; }
    if (pieChartL3) { pieChartL3.destroy(); pieChartL3 = null; }
    if (lineChartL3) { lineChartL3.destroy(); lineChartL3 = null; }
  }

  /* ═══════════════ CREATE / UPDATE PIE CHART ═══════════════ */
  function ensurePieChart(canvasId, existingChart) {
    const canvas = document.getElementById(canvasId);
    if (!canvas) return null;
    if (existingChart) return existingChart;
    const ctx = canvas.getContext('2d');
    return new Chart(ctx, {
      type: 'doughnut',
      data: {
        labels: EMOTIONS.map(e => e.slice(0,4)),
        datasets: [{
          data: EMOTIONS.map(() => 1),
          backgroundColor: EMOTIONS.map(e => EMO_COLORS[e]),
          borderColor: '#000',
          borderWidth: 2
        }]
      },
      options: {
        responsive: true, maintainAspectRatio: true,
        plugins: {
          legend: { display: true, position:'bottom', labels:{ color:'#000', font:{ weight:'bold' }, padding:8 } }
        },
        animation: { duration: 150 }
      }
    });
  }

  function updatePieChart(chart, distribution) {
    if (!chart || !distribution) return;
    const data = EMOTIONS.map(e => distribution[e] || 0);
    chart.data.datasets[0].data = data;
    chart.update('none');
  }

  /* ─── L3 Timeline line chart ─── */
  function ensureTimelineChart(canvasId, existingChart) {
    const canvas = document.getElementById(canvasId);
    if (!canvas) return null;
    if (existingChart) return existingChart;
    const ctx = canvas.getContext('2d');

    const datasets = EMOTIONS.map((emo, idx) => ({
      label: emo.slice(0, 4),
      data: [],
      borderColor: EMO_COLORS[emo],
      backgroundColor: EMO_COLORS[emo] + '33',
      borderWidth: 2,
      pointRadius: 0,
      tension: 0.3,
      fill: false,
    }));

    // Extra dataset for dominant-switch markers (triangles)
    datasets.push({
      label: 'Switch',
      data: [],
      pointStyle: 'triangle',
      pointRadius: 10,
      pointBackgroundColor: '#000',
      pointBorderColor: '#FFD700',
      pointBorderWidth: 2,
      showLine: false,
      borderWidth: 0,
    });

    return new Chart(ctx, {
      type: 'line',
      data: { labels: [], datasets },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        animation: { duration: 200 },
        scales: {
          x: {
            title: { display: true, text: 'Time (seconds)', color: '#000', font: { weight: 'bold' } },
            ticks: { color: '#000', maxTicksLimit: 10 },
            grid: { color: '#ddd' },
            min: 0,
          },
          y: {
            title: { display: true, text: 'Probability (%)', color: '#000', font: { weight: 'bold' } },
            min: 0, max: 100,
            ticks: { color: '#000', stepSize: 20 },
            grid: { color: '#ddd' },
          }
        },
        plugins: {
          legend: { display: false },  // We use custom legend chips
          tooltip: {
            mode: 'index',
            intersect: false,
          }
        },
        interaction: {
          mode: 'nearest',
          axis: 'x',
          intersect: false,
        }
      }
    });
  }

  /* ═══════════════ UPDATE MODE PANELS ═══════════════ */
  function updateModePanel() {
    const o = latestModeOutput || {};

    switch (currentMode) {
      /* ── M1: Cafeteria Mood ── */
      case 'm1': {
        const state = o.state || 'idle';
        const remaining = o.remaining || 0;
        const timerEl = document.getElementById('timer-display');
        const startBtn = document.getElementById('ctrl-start');
        const settleCard = document.getElementById('settle-m1');

        // Timer display
        if (timerEl) {
          if (state === 'running') {
            timerEl.textContent = formatTimer(remaining);
          } else if (state === 'idle') {
            timerEl.textContent = '--:--';
          } else if (state === 'done') {
            timerEl.textContent = 'Done!';
          }
        }

        // Reset button state
        if (startBtn && state !== 'running') {
          startBtn.textContent = state === 'done' ? 'Retry' : 'Start';
          startBtn.disabled = false;
        }

        // Settlement card
        if (state === 'done' && o.result && settleCard) {
          settleCard.style.display = '';
          const r = o.result;
          const sat = (r.satisfaction || 0) / 100;
          const circ = 207.345;
          const ring = document.getElementById('m1-ring');
          if (ring) ring.style.strokeDashoffset = circ * (1 - sat);
          const valEl = document.getElementById('m1-val');
          if (valEl) valEl.textContent = Math.round(r.satisfaction || 0) + '%';

          document.getElementById('m1-pos').style.width = Math.round((r.positive_ratio||0)*100) + '%';
          document.getElementById('m1-pos-v').textContent = Math.round((r.positive_ratio||0)*100) + '%';
          document.getElementById('m1-neu').style.width = Math.round((r.neutral_ratio||0)*100) + '%';
          document.getElementById('m1-neu-v').textContent = Math.round((r.neutral_ratio||0)*100) + '%';
          document.getElementById('m1-neg').style.width = Math.round((r.negative_ratio||0)*100) + '%';
          document.getElementById('m1-neg-v').textContent = Math.round((r.negative_ratio||0)*100) + '%';

          const mainEl = document.getElementById('m1-main');
          if (mainEl) mainEl.innerHTML = `<span class="type-badge" data-type="${r.main_emotion||'neutral'}">${(r.main_emotion||'neutral').toUpperCase()}</span>`;
          const verdictEl = document.getElementById('m1-verdict');
          if (verdictEl) verdictEl.textContent = r.verdict || '';
          const distEl = document.getElementById('m1-dist');
          if (distEl && r.distribution) {
            distEl.innerHTML = EMOTIONS.map(e =>
              `<span class="dist-chip" style="background:${EMO_COLORS[e]}">${e.slice(0,4)} ${Math.round((r.distribution[e]||0)*100)}%</span>`
            ).join('');
          }
        } else if (settleCard && state !== 'done') {
          settleCard.style.display = 'none';
        }
        break;
      }

      /* ── M2: CODE RED ── */
      case 'm2': {
        const risk = o.risk_level || 0;
        const riskPct = Math.round(risk * 100);
        const alarm = !!o.alarm;
        const trigger = o.trigger;
        const banner = o.banner_text || '';

        document.getElementById('m2-risk-t').textContent = 'Risk: ' + riskPct + '%';
        document.getElementById('m2-risk-f').style.width = Math.min(100, riskPct) + '%';
        const alarmEl = document.getElementById('m2-alarm');
        alarmEl.className = 'alarm-indicator' + (alarm ? ' alarm-on' : '');
        alarmEl.textContent = alarm ? '\u{1f6a8} ALARM: ' + (trigger||'active') : '✅ No Alarm';
        document.getElementById('m2-threat').textContent = alarm ? 'HIGH THREAT' : 'Normal';

        // Explosive effects
        const bannerOverlay = document.getElementById('banner-overlay');
        const bannerText = document.getElementById('banner-text');
        if (alarm && banner) {
          document.body.classList.add('code-red-active');
          if (bannerOverlay) { bannerOverlay.classList.add('show'); bannerText.textContent = banner; }
        } else {
          document.body.classList.remove('code-red-active');
          if (bannerOverlay) bannerOverlay.classList.remove('show');
        }

        // Pie chart
        pieChartL2 = ensurePieChart('pie-l2', pieChartL2);
        updatePieChart(pieChartL2, o.distribution);
        break;
      }

      /* ── M3: Audience Reactions ── */
      case 'm3': {
        const dominant = o.dominant || 'neutral';
        const valence = o.valence || 0;
        const timeline = o.timeline || [];
        const markers = o.markers || [];

        // Dominant display
        document.getElementById('m3-dom').textContent = dominant;
        document.getElementById('m3-dom').style.color = EMO_COLORS[dominant] || '#000';
        document.getElementById('m3-val').textContent = valence.toFixed(3);

        // Pie chart (cumulative distribution)
        pieChartL3 = ensurePieChart('pie-l3', pieChartL3);
        updatePieChart(pieChartL3, o.distribution);

        // ── Timeline line chart ──
        lineChartL3 = ensureTimelineChart('timeline-l3', lineChartL3);
        if (lineChartL3 && timeline.length > 0) {
          // Update labels (x = time)
          lineChartL3.data.labels = timeline.map(p => p.t.toFixed(1) + 's');
          // Update each emotion's dataset
          EMOTIONS.forEach((emo, idx) => {
            lineChartL3.data.datasets[idx].data = timeline.map(p => (p.probs[emo] || 0) * 100);
          });

          // Add marker annotations as vertical lines at dominant-switch times
          // (Simple approach: use scatter dataset for markers)
          if (markers.length > 0) {
            const markerDataset = lineChartL3.data.datasets[7]; // marker dataset
            if (markerDataset) {
              markerDataset.data = markers.map(m => ({ x: m.t.toFixed(1)+'s', y: 100 }));
            }
          }

          lineChartL3.update('none');
        }

        // Build legend
        const legend = document.getElementById('m3-legend');
        if (legend) {
          legend.innerHTML = EMOTIONS.map(e =>
            `<span class="legend-chip" style="background:${EMO_COLORS[e]}">${e.slice(0,4)}</span>`
          ).join('');
        }
        break;
      }

      /* ── M4: Speech Coach ── */
      case 'm4': {
        const state = o.state || 'idle';
        const remaining = o.remaining || 0;
        const timerEl = document.getElementById('timer-display');
        const startBtn = document.getElementById('ctrl-start');
        const settleCard = document.getElementById('settle-m4');

        if (timerEl) {
          if (state === 'running') timerEl.textContent = formatTimer(remaining);
          else if (state === 'idle') timerEl.textContent = '--:--';
          else if (state === 'generating') timerEl.textContent = 'Analyzing...';
          else if (state === 'done') timerEl.textContent = 'Done!';
        }
        if (startBtn && state !== 'running' && state !== 'generating') {
          startBtn.textContent = state === 'done' ? 'Retry' : 'Start';
          startBtn.disabled = false;
        }

        if ((state === 'done' || state === 'generating') && o.result && settleCard) {
          settleCard.style.display = '';
          const r = o.result;
          document.getElementById('m4-pos').textContent = Math.round((r.positivity||0)*100) + '%';
          document.getElementById('m4-anx').textContent = Math.round((r.anxiety||0)*100) + '%';
          document.getElementById('m4-exp').textContent = Math.round((r.expressiveness||0)*100) + '%';
          document.getElementById('m4-neu').textContent = Math.round((r.neutral_pct||0)*100) + '%';
          document.getElementById('m4-adv').textContent = '\u{1f4a1} ' + (r.advice || 'Keep practicing!');
          document.getElementById('m4-source').textContent = r.advice_source === 'llm' ? 'AI Coach' : 'Rule-based tips';

          // Simple timeline sparkline
          if (r.timeline && r.timeline.length > 0) {
            const tl = r.timeline;
            const maxT = tl[tl.length-1].t || 1;
            const points = tl.map(p => {
              const x = (p.t / maxT * 100).toFixed(1);
              const y = ((1 - (p.valence + 1) / 2) * 100).toFixed(1);
              return `${x},${y}`;
            }).join(' ');
            document.getElementById('m4-timeline').innerHTML = `
              <svg viewBox="0 0 100 100" class="coach-sparkline">
                <polyline points="${points}" fill="none" stroke="var(--c-happiness)" stroke-width="2"/>
              </svg>`;
          }
        } else if (settleCard && state !== 'done' && state !== 'generating') {
          settleCard.style.display = 'none';
        }
        break;
      }

      /* ── M5: Mimic Game ── */
      case 'm5': {
        const tgt = o.target || 'happiness';
        const pTgt = o.p_target || 0;
        const score = o.score || 0;
        const combo = o.combo || 0;
        const timeLeft = o.time_left || 0;
        const round = o.round || 0;
        const gameOver = o.game_over;
        const best = o.best || 0;

        document.getElementById('m5-tgt').textContent = EMOJI_MAP[tgt] || '\u{1f604}';
        const circ = 175.929;
        const ring = document.getElementById('m5-ring');
        if (ring) {
          ring.style.strokeDashoffset = circ * (1 - Math.min(1, pTgt));
          ring.style.stroke = EMO_COLORS[tgt] || '#FFD700';
        }
        document.getElementById('m5-val').textContent = Math.round(pTgt * 100) + '%';
        document.getElementById('m5-score').textContent = score;
        document.getElementById('m5-combo').textContent = 'x' + combo;
        document.getElementById('m5-round').textContent = round;
        document.getElementById('m5-time').textContent = formatTimer(timeLeft);

        // Game over
        const goCard = document.getElementById('m5-gameover');
        if (gameOver && goCard) {
          goCard.style.display = '';
          document.getElementById('m5-final').textContent = score;
          document.getElementById('m5-best').textContent = 'x' + best;
          // Show replay button
          const replay = document.getElementById('m5-replay');
          if (replay) replay.style.display = '';
          const startBtn = document.getElementById('m5-start-btn');
          if (startBtn) startBtn.style.display = 'none';
        }
        break;
      }
    }
  }

  function formatTimer(seconds) {
    const m = Math.floor(Math.max(0, seconds) / 60);
    const s = Math.floor(Math.max(0, seconds) % 60);
    return `${m}:${s.toString().padStart(2,'0')}`;
  }

  /* ═══════════════ 60fps RENDER LOOP ═══════════════ */
  function renderLoop(time) {
    requestAnimationFrame(renderLoop);
    frameCount++;
    if (time - lastFpsTime >= 1000) {
      displayFps = frameCount; frameCount = 0; lastFpsTime = time;
      $fpsBadge.textContent = displayFps + ' FPS';
    }
  }

})();
