/* ZoneWatch review console — vanilla JS, no build step, no dependencies.
 * Layout of this file:
 *   1. labels (Korean domain terms)   2. small DOM/format utils   3. API client
 *   4. state (single object)          5. polling + diffing        6. render functions (one per panel)
 *   7. actions (select / review / explain / commands)             8. keyboard + init
 */
'use strict';

// ---------------------------------------------------------------- 1. labels
const L = {
  review: { needs_review: '확인 필요', in_review: '검토 중', resolved: '처리 완료' },
  resolution: { confirmed: '침입 확인', dismissed: '오탐 처리', escalated: '상위 보고', no_action: '조치 불필요' },
  kind: { EVENT_KIND_ZONE_ENTRY: '진입', EVENT_KIND_ZONE_DWELL: '체류' },
  closeReason: { CLOSE_REASON_EXITED: '구역 이탈', CLOSE_REASON_TRACK_LOST: '추적 끊김', CLOSE_REASON_STREAM_ENDED: '영상 종료', CLOSE_REASON_SHUTDOWN: '분석 중지' },
  camState: { CAMERA_STATE_CONNECTING: '연결 중', CAMERA_STATE_STREAMING: '분석 중', CAMERA_STATE_RECONNECTING: '재연결 중', CAMERA_STATE_ENDED: '영상 종료', CAMERA_STATE_STOPPED: '정지', CAMERA_STATE_ERROR: '오류', CAMERA_STATE_UNSPECIFIED: '알 수 없음' },
  assessment: { needs_attention: '확인 필요', likely_benign: '정상 가능성 높음', unclear: '판단 어려움' },
  zoneRelation: { clearly_inside: '구역 안', near_boundary: '경계 부근', outside: '구역 밖', unclear: '불명확' },
  vlmStatus: { queued: '대기', running: '실행 중', done: '완료', failed: '실패', invalid: '형식 오류', timeout: '시간 초과', skipped: '건너뜀' },
  frameRole: { context_before: '진입 전', entry: '진입', dwell: '체류', exit: '이탈', context_after: '이탈 후' },
  eventStatus: { EVENT_STATUS_OPEN: '진행 중', EVENT_STATUS_CLOSED: '종료됨' },
};
const ROLES = ['context_before', 'entry', 'dwell', 'exit', 'context_after'];
const RETRYABLE = ['failed', 'invalid', 'timeout', 'skipped'];
// Static replay (GitHub Pages): when window.ZW_STATIC_BASE is set (e.g. 'data'), every GET reads the API's JSON
// from files under that base and every write is disabled. Data never changes there, so polling is slow.
const STATIC_BASE = typeof window.ZW_STATIC_BASE === 'string' && window.ZW_STATIC_BASE.trim() ? window.ZW_STATIC_BASE.trim().replace(/\/+$/, '') : null;
const IS_STATIC = STATIC_BASE !== null;
const STATIC_MSG = '정적 재생 모드에서는 변경할 수 없습니다';
const POLL_MS = IS_STATIC ? 30000 : 2000;

// ---------------------------------------------------------------- 2. utils
const $ = (sel, root = document) => root.querySelector(sel);
const SVG_NS = 'http://www.w3.org/2000/svg';

function nodes(x) {
  return [].concat(x).flat(Infinity).filter((n) => n != null && n !== false)
    .map((n) => (n.nodeType ? n : document.createTextNode(String(n))));
}
/** h('div', {class:'a', onclick:fn, dataset:{id:1}}, ...children) — safe DOM builder (no innerHTML). */
function h(tag, attrs, ...kids) {
  const el = document.createElement(tag);
  if (attrs) {
    for (const [k, v] of Object.entries(attrs)) {
      if (v == null || v === false) continue;
      if (k === 'class') el.className = v;
      else if (k === 'dataset') Object.assign(el.dataset, v);
      else if (k.startsWith('on') && typeof v === 'function') el.addEventListener(k.slice(2), v);
      else el.setAttribute(k, v === true ? '' : v);
    }
  }
  el.append(...nodes(kids));
  return el;
}
function svgEl(tag, attrs) {
  const el = document.createElementNS(SVG_NS, tag);
  for (const [k, v] of Object.entries(attrs || {})) el.setAttribute(k, v);
  return el;
}
const dt = (t) => h('dt', null, t);
const dd = (...c) => h('dd', null, ...c);
const btn = (label, cls, onclick, extra) => h('button', { class: `btn btn-sm ${cls || ''}`, type: 'button', onclick, ...(extra || {}) }, label);
const pad2 = (n) => String(n).padStart(2, '0');
const num = (v) => { const n = Number(v); return Number.isFinite(n) ? n : 0; };
function fmtClock(ts) {
  if (ts == null) return '—';
  const d = new Date(ts * 1000);
  return `${pad2(d.getHours())}:${pad2(d.getMinutes())}:${pad2(d.getSeconds())}`;
}
function fmtDateTime(ts) {
  if (ts == null) return '—';
  const d = new Date(ts * 1000);
  return `${d.getFullYear()}-${pad2(d.getMonth() + 1)}-${pad2(d.getDate())} ${fmtClock(ts)}`;
}
function fmtMedia(s, frac = false) {
  if (s == null || !Number.isFinite(Number(s))) return '—';
  const v = Math.max(0, Number(s));
  const m = Math.floor(v / 60); const sec = v - m * 60;
  return frac ? `${pad2(m)}:${sec < 10 ? '0' : ''}${sec.toFixed(1)}` : `${pad2(m)}:${pad2(Math.floor(sec))}`;
}
const fmtSec = (s) => (s == null ? '—' : `${Number(s).toFixed(1)}초`);
const fmtMs = (ms) => (ms == null ? '—' : `${Math.round(ms)} ms`);
function fmtBytes(b) {
  b = num(b);
  if (b < 1024) return `${b} B`;
  if (b < 1048576) return `${(b / 1024).toFixed(1)} KB`;
  return `${(b / 1048576).toFixed(1)} MB`;
}
const kindLabel = (k) => L.kind[k] || (k ? k.replace('EVENT_KIND_', '') : '—');
const kindClass = (k) => (k === 'EVENT_KIND_ZONE_ENTRY' ? 'kind-entry' : k === 'EVENT_KIND_ZONE_DWELL' ? 'kind-dwell' : 'kind-other');
const camStateKey = (s) => (s || 'CAMERA_STATE_UNSPECIFIED').replace('CAMERA_STATE_', '');
const camStateLabel = (s) => L.camState[s || 'CAMERA_STATE_UNSPECIFIED'] || s;
/** "file://data/videos/worker-zone.mp4" → "worker-zone.mp4" (null when not a file source). */
function originalVideoName(source) {
  const m = /^file:\/\/.*?([^/\\]+\.mp4)$/i.exec(source || '');
  return m ? m[1] : null;
}
/** Server media URLs (/api/evidence/<path>, /api/videos/<name>) → files under STATIC_BASE in static mode. The one place that rewrites them. */
function mediaUrl(u) {
  if (!u || !IS_STATIC) return u;
  if (u.startsWith('/api/evidence/')) return `${STATIC_BASE}/evidence/${u.slice('/api/evidence/'.length)}`;
  if (u.startsWith('/api/videos/')) return `${STATIC_BASE}/videos/${u.slice('/api/videos/'.length)}`;
  return u;
}
const rawEventUrl = (id) => (IS_STATIC ? `${STATIC_BASE}/events/${encodeURIComponent(id)}.json` : `/api/events/${encodeURIComponent(id)}`);
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
/** Re-render `el` only when `sig` differs from the last render. Returns true when it re-rendered. */
function renderIfChanged(el, sig, build) {
  if (el.dataset.sig === sig) return false;
  el.dataset.sig = sig;
  el.replaceChildren(...nodes(build()));
  return true;
}
function toast(msg, kind = '') {
  const el = h('div', { class: `toast ${kind}` }, msg);
  $('#toasts').append(el);
  setTimeout(() => el.remove(), 3600);
}

// ---------------------------------------------------------------- 3. API
async function api(path, opts = {}) {
  const ctrl = new AbortController();
  const timer = setTimeout(() => ctrl.abort(), opts.timeout || 6000);
  try {
    const init = { method: opts.method || 'GET', signal: ctrl.signal, headers: {} };
    if (opts.body !== undefined) { init.body = JSON.stringify(opts.body); init.headers['content-type'] = 'application/json'; }
    const res = await fetch(path, init);
    const text = await res.text();
    let data = null;
    try { data = text ? JSON.parse(text) : null; } catch { data = null; }
    if (!res.ok) {
      const detail = data && (typeof data.detail === 'string' ? data.detail : JSON.stringify(data.detail || data));
      const err = new Error(detail || `HTTP ${res.status}`);
      err.status = res.status;
      throw err;
    }
    return data;
  } catch (e) {
    if (e.name === 'AbortError') throw new Error('응답 시간 초과');
    throw e;
  } finally { clearTimeout(timer); }
}
/** Static mode: read a JSON file once; a missing file (404) is remembered and treated as empty, never re-requested. */
const missingStatic = new Set();
async function staticJson(name) {
  const path = `${STATIC_BASE}/${name}`;
  if (missingStatic.has(path)) return null;
  try { return await api(path); } catch (e) { if (e.status === 404) { missingStatic.add(path); return null; } throw e; }
}
const staticWrite = async () => { const e = new Error(STATIC_MSG); e.status = 403; e.static = true; throw e; };
const API_STATIC = {
  status: async () => (await staticJson('status.json')) || { mode: 'replay', devices: {}, vlm: {}, ingest: {}, events: {}, missing: true },
  events: async () => (await staticJson('events.json')) || { items: [], counts: {} },   // filters are applied client-side (filteredEvents)
  event: (id) => staticJson(`events/${encodeURIComponent(id)}.json`),
  timeline: async (cameraId) => { const r = await staticJson('timeline.json'); return { items: ((r && r.items) || []).filter((i) => !cameraId || i.camera_id === cameraId) }; },
  audit: async () => (await staticJson('audit.json')) || { items: [] },
  devices: async () => (await staticJson('devices.json')) || { items: [] },
  videos: async () => (await staticJson('videos.json')) || { items: [] },
  displayed: async () => ({ ok: true, static: true }),   // silent no-op: one toast per rendered row would be noise
  review: staticWrite, explain: staticWrite, command: staticWrite, commandStatus: staticWrite,
};
const API_LIVE = {
  status: () => api('/api/status'),
  events: () => api('/api/events?limit=500'),
  event: (id) => api(`/api/events/${encodeURIComponent(id)}`),
  review: (id, body) => api(`/api/events/${encodeURIComponent(id)}/review`, { method: 'POST', body }),
  explain: (id) => api(`/api/events/${encodeURIComponent(id)}/explain`, { method: 'POST' }),
  displayed: (id) => api(`/api/events/${encodeURIComponent(id)}/displayed`, { method: 'POST', body: { client_ts: Date.now() / 1000 } }),
  videos: () => api('/api/videos'),
  command: (deviceId, body) => api(`/api/devices/${encodeURIComponent(deviceId)}/commands`, { method: 'POST', body }),
  commandStatus: (id) => api(`/api/commands/${encodeURIComponent(id)}`),
  timeline: (cameraId) => api(`/api/timeline?camera_id=${encodeURIComponent(cameraId)}`),
  audit: () => api('/api/audit?limit=500'),
  devices: () => api('/api/devices'),
};
const API = IS_STATIC ? API_STATIC : API_LIVE;

// ---------------------------------------------------------------- 4. state
const state = {
  status: null, events: [], eventsById: new Map(), counts: {},
  loaded: false, serverDown: false, lastError: '', lastOk: null,
  videos: new Set(), videosLoaded: false,
  filters: { camera: '', state: '', kind: '', q: '' },
  selectedId: null, detail: null, detailSig: null, detailLoading: null, detailError: null,
  tab: 'cv', annot: true, frameRole: 'entry', engOpen: false,
  videoMode: 'clip',      // user preference: 'clip' | 'original'
  stageMode: 'none',      // what is actually shown: 'clip' | 'original' | 'frames' | 'none'
  pendingSeek: null,      // {id, play}
  timeline: { camera: null, items: [], sig: '', domain: 60, inflightCam: null },
  commands: {},           // camera_id -> {phase:'wait'|'ok'|'err', id, message}
  auditOpen: false, audit: [], auditSig: '',
  displayed: new Set(),   // event ids already reported via /displayed in this session
  form: { open: false, resolution: 'confirmed', note: '' },
  actor: '담당자', reviewError: null, reviewBusy: false, explainBusy: false,
};
const sigOf = (ev) => [ev.event_id, ev.revision, ev.review && ev.review.state, ev.explanation ? ev.explanation.status : 'none', ev.evidence_ready ? 1 : 0, ev.evidence && ev.evidence.clip_available ? 1 : 0].join('|');

const video = $('#video');
const frameImg = $('#frame-img');

/** The selected event: detail (full) when fresh, else the list row, else a merge that keeps detail-only extras. */
function currentEvent() {
  const id = state.selectedId;
  if (!id) return null;
  const lst = state.eventsById.get(id);
  const det = state.detail && state.detail.event_id === id ? state.detail : null;
  if (!det) return lst || null;
  if (!lst || sigOf(lst) === sigOf(det)) return det;
  const expl = lst.explanation
    ? { ...(det.explanation || {}), ...lst.explanation, raw_text: det.explanation && det.explanation.id === lst.explanation.id ? det.explanation.raw_text : null }
    : null;
  return { ...det, ...lst, explanation: expl, review_log: det.review_log, explanations: det.explanations, evidence_files: det.evidence_files, trajectory: det.trajectory };
}
const hasFullDetail = (ev) => !!(ev && state.detail && state.detail.event_id === ev.event_id);

function filteredEvents() {
  const f = state.filters; const q = f.q.trim().toLowerCase();
  return state.events.filter((e) => (!f.camera || e.camera_id === f.camera)
    && (!f.state || e.review.state === f.state)
    && (!f.kind || e.kind === f.kind)
    && (!q || (e.zone_name || '').toLowerCase().includes(q) || e.event_id.toLowerCase().includes(q) || (e.camera_name || '').toLowerCase().includes(q)));
}
/** Cameras known from device heartbeats plus cameras seen in events. */
function cameraIndex() {
  const map = new Map();
  const devs = (state.status && state.status.devices) || {};
  for (const d of Object.values(devs)) {
    for (const c of (d.status && d.status.cameras) || []) {
      map.set(c.camera_id, { camera_id: c.camera_id, name: c.name || c.camera_id, device_id: d.device_id, device: d, cam: c, eventCount: 0, source: c.source || '' });
    }
  }
  for (const e of state.events) {
    let x = map.get(e.camera_id);
    if (!x) {
      x = { camera_id: e.camera_id, name: e.camera_name || e.camera_id, device_id: e.device_id, device: devs[e.device_id] || null, cam: null, eventCount: 0, source: e.source || '' };
      map.set(e.camera_id, x);
    }
    x.eventCount += 1;
    if (!x.source) x.source = e.source || '';
  }
  return [...map.values()];
}
const frameByRole = (ev, role) => ((ev.evidence && ev.evidence.key_frames) || []).find((k) => k.role === role) || null;
function frameFor(ev, role) {
  const base = frameByRole(ev, role); const an = frameByRole(ev, `${role}_annot`);
  return { role, base, annot: an, pick: state.annot ? an || base : base || an };
}
const frameCount = (ev) => ((ev.evidence && ev.evidence.key_frames) || []).filter((k) => !k.role.endsWith('_annot') && k.available).length;

// ---------------------------------------------------------------- 5. polling
let pollTimer = null;
async function poll() {
  clearTimeout(pollTimer);
  try {
    const [status, events] = await Promise.all([API.status(), API.events()]);
    const wasDown = state.serverDown;
    state.status = status;
    applyEvents(events);
    state.serverDown = false; state.lastError = ''; state.lastOk = Date.now();
    const first = !state.loaded;
    state.loaded = true;
    if (!state.videosLoaded) loadVideos();
    if (first && !state.selectedId && state.events.length) {
      const vis = filteredEvents();
      if (vis.length) selectEvent(vis[0].event_id, { seek: true, play: false });
    }
    renderAll();
    refreshTimeline();
    maybeRefreshDetail();
    if (state.auditOpen) refreshAudit();
    if (wasDown) toast('서버 연결이 복구되었습니다', 'ok');
  } catch (e) {
    state.serverDown = true; state.lastError = e.message || String(e);
    renderBanner(); renderTopbar();
  }
  pollTimer = setTimeout(poll, POLL_MS);
}
function applyEvents(res) {
  const items = (res && res.items) || [];
  items.sort((a, b) => (num(b.entry_ts) - num(a.entry_ts)) || (num(b.received_at) - num(a.received_at)));
  state.events = items;
  state.counts = (res && res.counts) || {};
  state.eventsById = new Map(items.map((e) => [e.event_id, e]));
}
async function loadVideos() {
  try {
    const r = await API.videos();
    state.videos = new Set((r.items || []).map((i) => i.name));
    state.videosLoaded = true;
    renderStage(); renderCameras();
  } catch { /* retried on the next poll */ }
}
function maybeRefreshDetail() {
  const id = state.selectedId; if (!id) return;
  const ev = state.eventsById.get(id); if (!ev) return;
  if (!state.detail || state.detail.event_id !== id || sigOf(ev) !== state.detailSig) loadDetail(id);
}

// ---------------------------------------------------------------- 6. render
function renderAll() {
  renderTopbar(); renderBanner(); renderCameras(); renderUplink(); renderVlm();
  renderFilters(); renderList(); renderSelHeader(); renderStage(); renderDetail(); updateExportLinks();
}

function renderTopbar() {
  const st = state.status; const badge = $('#mode-badge');
  const mode = IS_STATIC ? 'static' : st ? st.mode : null;
  const cls = mode === 'live' ? 'live' : mode === 'replay' || mode === 'static' ? 'replay' : 'unknown';
  const text = mode === 'live' ? '● 실시간 분석' : mode === 'static' ? '▶ 저장된 실행 결과 재생 (정적)' : mode === 'replay' ? '▶ 저장된 실행 결과 재생' : '● 상태 확인 중';
  const title = mode === 'static' ? '서버 없이 저장된 실행 결과 파일(JSON·영상)을 재생합니다. 처리 상태 변경·해석 요청·장치 명령은 할 수 없습니다'
    : mode === 'replay' ? '실시간 추론이 아니라 저장된 실행 결과를 재생합니다'
      : mode === 'live' ? '엣지 장치가 보내는 실시간 분석 결과를 수신합니다' : '';
  if (badge.className !== `mode-badge ${cls}`) badge.className = `mode-badge ${cls}`;
  if (badge.textContent !== text) badge.textContent = text;
  if (badge.title !== title) badge.title = title;
  const c = state.counts || {};
  const upd = state.lastOk ? fmtClock(state.lastOk / 1000) : '—';
  const sig = JSON.stringify([c, st && st.version, state.serverDown, upd]);
  renderIfChanged($('#top-stats'), sig, () => [
    h('span', null, '사건 ', h('b', null, c.total ?? 0)), h('span', { class: 'sep' }, '·'),
    h('span', null, '확인 필요 ', h('b', null, c.needs_review ?? 0)), h('span', { class: 'sep' }, '·'),
    h('span', null, '검토 중 ', h('b', null, c.in_review ?? 0)), h('span', { class: 'sep' }, '·'),
    h('span', null, '처리 완료 ', h('b', null, c.resolved ?? 0)), h('span', { class: 'sep' }, '·'),
    h('span', { title: '마지막으로 서버 응답을 받은 시각' }, state.serverDown ? '⚠ 갱신 중단' : `갱신 ${upd}`),
    st && st.version ? [h('span', { class: 'sep' }, '·'), h('span', null, `서버 v${st.version}`)] : null,
  ]);
}
function renderBanner() {
  $('#banner').classList.toggle('hidden', !state.serverDown);
  $('#banner-text').textContent = IS_STATIC ? '⚠ 데이터 파일을 불러올 수 없습니다. 마지막으로 읽은 데이터를 표시하고 있습니다.'
    : '⚠ 서버에 연결할 수 없습니다. 마지막으로 받은 데이터를 표시하고 있습니다.';
  $('#banner-detail').textContent = state.serverDown ? `(${state.lastError})` : '';
}

/** Keyed reconciliation: rebuild a child only when its `sig` changes, otherwise `patch` it in place. */
function reconcile(root, want) {
  const existing = new Map();
  for (const el of root.children) if (el.dataset && el.dataset.key) existing.set(el.dataset.key, el);
  const kids = want.map((w) => {
    let el = existing.get(w.key);
    if (!el || el.dataset.sig !== w.sig) { el = w.build(); el.dataset.key = w.key; el.dataset.sig = w.sig; }
    if (w.patch) w.patch(el);
    return el;
  });
  const same = kids.length === root.children.length && kids.every((n, i) => n === root.children[i]);
  if (!same) root.replaceChildren(...kids);
}
function renderCameras() {
  const root = $('#cameras'); if (!state.loaded) return;
  const st = state.status; const replay = st.mode === 'replay';
  const devs = st.devices || {}; const cams = cameraIndex();
  const devList = Object.values(devs);
  // Rows are keyed by camera/device id. Heartbeat counters (fps, lag, queue …) change every second,
  // so they are patched in place instead of rebuilding the row: the 분석 시작/정지 buttons stay stable under the mouse.
  const want = [];
  if (!devList.length) {
    want.push({ key: 'note', sig: `note|${replay}`, build: () => h('p', { class: 'note', style: 'margin:0 0 8px' }, IS_STATIC
      ? '정적 재생 모드: 저장된 사건의 카메라만 표시합니다. 장치 명령은 보낼 수 없습니다.' : replay
      ? '재생 모드: 연결된 장치가 없습니다. 저장된 사건의 카메라만 표시합니다.'
      : '연결된 엣지 장치가 없습니다 (하트비트 없음). 사건에 나타난 카메라만 표시합니다.') });
  }
  for (const d of devList) {
    want.push({ key: `dev|${d.device_id}`, sig: JSON.stringify(['dev', d.device_id, d.online, d.gateway_version]),
      build: () => h('div', { class: 'dev-head' },
        h('span', { class: d.online ? 'online' : 'offline' }, d.online ? '● 온라인' : '○ 오프라인'),
        h('b', null, d.device_id), h('span', { class: 'muted dev-age' })),
      patch: (el) => { const t = `${Math.round(d.age_s)}초 전 · gw ${d.gateway_version || '?'}`; const a = el.querySelector('.dev-age'); if (a.textContent !== t) a.textContent = t; } });
    for (const c of cams.filter((x) => x.device_id === d.device_id)) want.push(camEntry(c, replay, true));
  }
  const orphan = cams.filter((c) => !devs[c.device_id]);
  if (orphan.length && devList.length) want.push({ key: 'orphan', sig: 'orphan', build: () => h('div', { class: 'dev-head' }, h('span', { class: 'muted' }, '장치 미연결 (사건 기록 기준)')) });
  for (const c of orphan) want.push(camEntry(c, replay, false));
  if (!cams.length) want.push({ key: 'none', sig: 'none', build: () => h('p', { class: 'note' }, '아직 카메라가 없습니다.') });
  reconcile(root, want);
}
function camEntry(c, replay, deviceOnline) {
  const cs = c.cam; const cmd = state.commands[c.camera_id];
  const orig = originalVideoName(c.source); const hasOrig = !!(orig && state.videos.has(orig));
  const sig = JSON.stringify(['cam', c.camera_id, c.name, c.device_id, c.eventCount, !!cs, cs && cs.state, !!(cs && cs.last_error),
    replay, deviceOnline, state.filters.camera === c.camera_id, cmd, hasOrig, orig]);
  return { key: `cam|${c.camera_id}`, sig, build: () => buildCamRow(c, replay, deviceOnline, hasOrig, orig), patch: (el) => patchCamStats(el, cs) };
}
function camStatValues(cs) {
  return { fps: `${num(cs.input_fps).toFixed(1)} / ${num(cs.processed_fps).toFixed(1)}`, lag: `${Math.round(num(cs.analysis_lag_ms))} ms`,
    queue: `${num(cs.queue_depth)} / ${num(cs.queue_capacity)}`, dropped: String(num(cs.dropped_frames)), reconnect: `${num(cs.reconnect_count)}회`,
    media: fmtMedia(cs.media_time_s, true), err: cs.last_error || '' };
}
function patchCamStats(row, cs) {
  if (!cs) return;
  const vals = camStatValues(cs);
  for (const el of row.querySelectorAll('dd[data-k]')) { const v = vals[el.dataset.k]; if (v != null && el.textContent !== v) el.textContent = v; }
}
function buildCamRow(c, replay, deviceOnline, hasOrig, orig) {
  const active = state.filters.camera === c.camera_id;
  const cs = c.cam; const stKey = cs ? camStateKey(cs.state) : 'none';
  const row = h('div', { class: `cam-row${active ? ' active' : ''}`, role: 'button', tabindex: '0',
    title: active ? '클릭하면 카메라 필터를 해제합니다' : '클릭하면 이 카메라의 사건만 표시하고 타임라인을 보여줍니다',
    onclick: () => toggleCameraFilter(c.camera_id),
    onkeydown: (e) => { if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); toggleCameraFilter(c.camera_id); } } });
  row.append(h('div', { class: 'cam-head' },
    h('span', { class: 'cam-name' }, c.name),
    h('span', { class: `cam-state cs-${stKey}` }, cs ? `● ${camStateLabel(cs.state)}` : '○ 상태 정보 없음')));
  row.append(h('div', { class: 'cam-meta' }, `사건 ${c.eventCount}건` + (c.camera_id !== c.name ? ` · ${c.camera_id}` : '') + (hasOrig ? ` · 원본 영상 ${orig}` : '')));
  if (cs) {
    const v = camStatValues(cs);
    row.append(h('dl', { class: 'kv' },
      dt('입력 / 처리 fps'), h('dd', { dataset: { k: 'fps' } }, v.fps),
      dt('분석 지연'), h('dd', { dataset: { k: 'lag' } }, v.lag),
      dt('큐 (깊이/용량)'), h('dd', { dataset: { k: 'queue' } }, v.queue),
      dt('드롭 프레임'), h('dd', { dataset: { k: 'dropped' } }, v.dropped),
      dt('재연결'), h('dd', { dataset: { k: 'reconnect' } }, v.reconnect),
      dt('영상 위치'), h('dd', { dataset: { k: 'media' } }, v.media),
      cs.last_error ? [dt('마지막 오류'), h('dd', { class: 'err', dataset: { k: 'err' } }, v.err)] : null));
  }
  const canCmd = !IS_STATIC && !replay && deviceOnline;
  const why = IS_STATIC ? '정적 재생 모드에서는 장치 명령을 보낼 수 없습니다' : replay ? '재생 모드에서는 장치 명령을 보낼 수 없습니다 (실시간 장치 없음)'
    : !deviceOnline ? '장치 하트비트가 없어 명령을 보낼 수 없습니다' : '';
  row.append(h('div', { class: 'cam-actions' },
    h('button', { class: 'btn btn-sm btn-primary', type: 'button', disabled: !canCmd, title: why || '이 카메라의 분석을 시작합니다',
      onclick: (e) => { e.stopPropagation(); sendCommand(c, 'START_CAMERA'); } }, '▶ 분석 시작'),
    h('button', { class: 'btn btn-sm', type: 'button', disabled: !canCmd, title: why || '이 카메라의 분석을 정지합니다',
      onclick: (e) => { e.stopPropagation(); sendCommand(c, 'STOP_CAMERA'); } }, '■ 정지')));
  if (why) row.append(h('div', { class: 'note' }, why));
  const cmd = state.commands[c.camera_id];
  if (cmd) row.append(h('div', { class: `cmd-result ${cmd.phase}` }, cmd.message));
  return row;
}

function renderUplink() {
  const root = $('#uplink'); if (!state.loaded) return;
  const devs = Object.values(state.status.devices || {}); const ing = state.status.ingest || {};
  const sig = JSON.stringify([devs.map((d) => [d.device_id, d.online, d.status && d.status.uplink]), ing]);
  renderIfChanged(root, sig, () => {
    const out = [];
    if (!devs.length) out.push(h('p', { class: 'note', style: 'margin:0' }, '하트비트를 보내는 장치가 없습니다.'));
    for (const d of devs) {
      const u = (d.status && d.status.uplink) || {};
      out.push(h('div', { class: 'dev-head' }, h('b', null, d.device_id),
        h('span', { class: u.connected ? 'online' : 'offline' }, u.connected ? '● 연결됨' : '○ 끊김')));
      out.push(h('dl', { class: 'kv' },
        dt('대기 사건'), dd(String(num(u.pending_events))),
        dt('유실 사건'), dd(String(num(u.dropped_events))),
        dt('전송 / 확인'), dd(`${num(u.sent_events)} / ${num(u.acked_events)}`),
        dt('재연결'), dd(`${num(u.reconnect_count)}회`),
        dt('대기 업로드'), dd(String(num(u.pending_uploads))),
        u.last_error ? [dt('마지막 오류'), h('dd', { class: 'err' }, u.last_error)] : null));
    }
    out.push(h('p', { class: 'note' }, `서버 수신 누계: 사건 ${num(ing.events)} · 중복 ${num(ing.duplicates)} · 업로드 ${num(ing.uploads)} (${fmtBytes(ing.upload_bytes)}) · 하트비트 ${num(ing.heartbeats)}`));
    return out;
  });
}

function renderVlm() {
  const root = $('#vlm'); if (!state.loaded) return;
  const v = state.status.vlm || {};
  renderIfChanged(root, JSON.stringify(v), () => {
    const out = [];
    if (v.provider === 'mock') out.push(h('div', { class: 'warn-chip', title: '모의(mock) 제공자는 프레임을 해석하지 않고 응답 형식만 흉내 냅니다' }, '⚠ VLM 모의 응답(mock): 실제 해석 아님'));
    if (IS_STATIC) out.push(h('div', { class: 'warn-chip' }, '정적 재생: 저장된 해석만 표시하며 새 해석은 생성되지 않습니다'));
    else if (!v.provider || v.provider === 'none') out.push(h('div', { class: 'warn-chip' }, '⚠ VLM 비활성화: 해석이 생성되지 않습니다'));
    out.push(h('dl', { class: 'kv' },
      dt('제공자'), dd(v.provider || '—'), dt('모델'), dd(v.model || '—'),
      dt('프롬프트 / 스키마'), dd(`${v.prompt_version || '—'} / ${v.schema_version || '—'}`),
      dt('실행 중'), dd(String(num(v.inflight))),
      dt('제한'), dd(`${v.timeout_s ?? '—'}s · 최대 ${v.max_attempts ?? '—'}회`),
      dt('요청 시점'), dd(v.trigger || '—')));
    const db = v.db || {}; const order = ['queued', 'running', 'done', 'failed', 'invalid', 'timeout', 'skipped'];
    const keys = order.filter((k) => db[k] != null).concat(Object.keys(db).filter((k) => !order.includes(k)));
    out.push(h('div', { class: 'chips' }, keys.length
      ? keys.map((k) => h('span', { class: `chip vlm-chip vlm-${k}` }, `${L.vlmStatus[k] || k} ${db[k]}`))
      : h('span', { class: 'note' }, '해석 작업 없음')));
    if (v.provider === 'anthropic') {
      const pct = num(v.budget_usd) > 0 ? Math.min(100, (100 * num(v.spend_usd)) / num(v.budget_usd)) : 0;
      out.push(h('div', { class: 'note' }, `비용 $${num(v.spend_usd).toFixed(4)} / 예산 $${num(v.budget_usd).toFixed(2)}`),
        h('div', { class: 'budget-bar', title: `예산의 ${pct.toFixed(0)}% 사용` }, h('i', { style: `width:${pct}%` })));
    }
    return out;
  });
}

function renderFilters() {
  const sel = $('#f-camera'); const cams = cameraIndex();
  const sig = JSON.stringify(cams.map((c) => [c.camera_id, c.name]));
  if (sel.dataset.sig !== sig) {
    sel.dataset.sig = sig;
    sel.replaceChildren(h('option', { value: '' }, '모든 카메라'), ...cams.map((c) => h('option', { value: c.camera_id }, c.name)));
    sel.value = state.filters.camera;
  }
}
function renderCounts() {
  const c = state.counts || {};
  const sig = JSON.stringify([c, state.filters.state]);
  renderIfChanged($('#counts'), sig, () => ['needs_review', 'in_review', 'resolved'].map((k) =>
    h('span', { class: `chip st-${k}${state.filters.state === k ? ' on' : ''}`, role: 'button', tabindex: '0',
      title: state.filters.state === k ? '필터 해제' : `${L.review[k]} 사건만 보기`,
      onclick: () => setFilter('state', state.filters.state === k ? '' : k) }, `${L.review[k]} ${c[k] ?? 0}`)));
}

function buildRow(ev) {
  const x = ev.explanation; const vs = x ? x.status : null;
  const row = h('div', { class: 'ev-row', role: 'option', tabindex: '-1', dataset: { id: ev.event_id, sig: sigOf(ev) },
    onclick: () => selectEvent(ev.event_id, { seek: true, play: true }) });
  row.append(h('div', { class: 'ev-l1' },
    h('span', { class: 'ev-time', title: fmtDateTime(ev.entry_ts) }, fmtClock(ev.entry_ts)),
    h('span', { class: 'ev-media', title: '원본 영상 기준 시간' }, `영상 ${fmtMedia(ev.media_entry_s)}`),
    h('span', { class: `kind-badge ${kindClass(ev.kind)}` }, kindLabel(ev.kind)),
    ev.ingest_source === 'replay' ? h('span', { class: 'tag tag-replay', title: '저장된 실행 결과에서 재생된 사건' }, '재생') : null,
    h('span', { class: 'ev-dwell' }, `체류 ${fmtSec(ev.dwell_seconds)}`)));
  row.append(h('div', { class: 'ev-l2' }, h('span', { class: 'ev-cam' }, ev.camera_name || ev.camera_id), h('span', { class: 'ev-zone' }, ev.zone_name || ev.zone_id)));
  const retry = !IS_STATIC && (vs == null || RETRYABLE.includes(vs));
  row.append(h('div', { class: 'ev-l3' },
    h('span', { class: `chip st-${ev.review.state}` }, `● ${L.review[ev.review.state] || ev.review.state}`),
    h('span', { class: `chip vlm-chip vlm-${vs || 'none'}`, title: x ? `VLM ${L.vlmStatus[vs] || vs}${x.error ? ': ' + x.error : ''}` : '해석이 요청되지 않았습니다' },
      `VLM ${vs ? L.vlmStatus[vs] || vs : '미요청'}`,
      retry ? h('button', { class: 'mini-btn', type: 'button', title: '다시 해석 요청', 'aria-label': '다시 해석 요청',
        onclick: (e) => { e.stopPropagation(); requestExplain(ev.event_id); } }, '↻') : null),
    h('span', { class: `ev-evid${ev.evidence_ready ? '' : ' no'}`, title: ev.evidence_ready ? '근거(프레임/클립) 수신 완료' : '근거 업로드 대기 중' },
      ev.evidence_ready ? '▣ 근거 준비됨' : '▢ 근거 대기')));
  if (!state.displayed.has(ev.event_id)) {
    state.displayed.add(ev.event_id);
    API.displayed(ev.event_id).catch(() => {});
  }
  return row;
}
function renderList() {
  const list = $('#event-list'); if (!state.loaded) return;
  const visible = filteredEvents();
  $('#list-shown').textContent = `${visible.length} / ${state.events.length}건 표시`;
  renderCounts();
  if (!state.events.length) {
    renderIfChanged(list, 'empty', () => h('div', { class: 'list-msg' }, h('b', null, '아직 검토할 사건이 없습니다'), h('br'),
      '엣지 장치가 연결되면 왼쪽 카메라 목록에서 ', h('b', null, '분석 시작'), '을 누르세요.', h('br'),
      '저장된 실행 결과를 보려면 서버를 ', h('code', null, '--replay-bundle'), ' 옵션으로 실행합니다.'));
    return;
  }
  if (!visible.length) {
    renderIfChanged(list, 'nofilter', () => h('div', { class: 'list-msg' }, '조건에 맞는 사건이 없습니다. 필터를 바꿔 보세요.'));
    return;
  }
  list.dataset.sig = 'rows';
  const existing = new Map();
  for (const el of list.children) if (el.dataset && el.dataset.id) existing.set(el.dataset.id, el);
  const rows = visible.map((ev) => {
    const old = existing.get(ev.event_id);
    return old && old.dataset.sig === sigOf(ev) ? old : buildRow(ev);
  });
  const same = rows.length === list.children.length && rows.every((n, i) => n === list.children[i]);
  if (!same) { const top = list.scrollTop; list.replaceChildren(...rows); list.scrollTop = top; }
  for (const n of rows) {
    const sel = n.dataset.id === state.selectedId;
    n.classList.toggle('selected', sel); n.setAttribute('aria-selected', sel ? 'true' : 'false');
  }
}

function renderSelHeader() {
  const root = $('#sel-header'); const ev = currentEvent();
  if (!ev) {
    renderIfChanged(root, state.loaded && !state.events.length ? 'none-empty' : 'none', () => [
      h('div', { class: 'sel-line' }, h('span', { class: 'sel-title' }, state.loaded && !state.events.length ? '검토할 사건이 없습니다' : '사건을 선택하세요')),
      h('div', { class: 'sel-sub' }, '오른쪽 목록에서 사건을 고르면 영상 · 근거 · 해석 · 처리 상태가 함께 표시됩니다.')]);
    return;
  }
  const x = ev.explanation; const vs = x ? x.status : null; const rv = ev.review;
  const sig = JSON.stringify([sigOf(ev), state.stageMode, rv.resolution, state.videosLoaded]);
  const arrow = () => h('span', { class: 'flow-arrow' }, '→');
  renderIfChanged(root, sig, () => [
    h('div', { class: 'sel-line' },
      h('span', { class: `kind-badge ${kindClass(ev.kind)}` }, kindLabel(ev.kind)),
      h('span', { class: 'sel-title' }, ev.zone_name || ev.zone_id),
      h('span', { class: 'muted' }, '·'), h('span', null, ev.camera_name || ev.camera_id),
      ev.ingest_source === 'replay' ? h('span', { class: 'tag tag-replay', title: '저장된 실행 결과에서 재생된 사건' }, '재생') : null,
      h('span', { style: 'flex:1' }),
      h('span', { class: `chip chip-lg st-${rv.state}`, role: 'button', tabindex: '0', title: '처리 상태 탭 열기', onclick: () => setTab('review') }, `● ${L.review[rv.state] || rv.state}`),
      rv.state === 'resolved' && rv.resolution ? h('span', { class: 'chip chip-lg' }, L.resolution[rv.resolution] || rv.resolution) : null),
    h('div', { class: 'sel-sub' },
      `${fmtDateTime(ev.entry_ts)} 진입 (영상 ${fmtMedia(ev.media_entry_s)}) → ${ev.exit_ts ? fmtClock(ev.exit_ts) : '—'} 이탈 (영상 ${fmtMedia(ev.media_exit_s)})`
      + ` · 체류 ${fmtSec(ev.dwell_seconds)} · ${L.closeReason[ev.close_reason] || '종료 사유 없음'} · 트랙 #${ev.track_id ?? '—'}`),
    h('div', { class: 'sel-flow' },
      h('span', { class: `flow-step ${state.stageMode === 'frames' ? 'warn' : 'ok'}` },
        '영상: ' + (state.stageMode === 'original' ? '원본 영상' : state.stageMode === 'clip' ? '근거 클립' : '클립 대기 · 프레임만')),
      arrow(),
      h('span', { class: `flow-step ${ev.evidence_ready ? 'ok' : 'warn'}`, role: 'button', tabindex: '0', onclick: () => setTab('cv') },
        `근거: 프레임 ${frameCount(ev)}장` + (ev.evidence_ready ? '' : ' (업로드 대기)')),
      arrow(),
      h('span', { class: `flow-step ${vs === 'done' ? 'ok' : vs && RETRYABLE.includes(vs) ? 'bad' : 'warn'}`, role: 'button', tabindex: '0', onclick: () => setTab('vlm') },
        `해석: ${vs ? L.vlmStatus[vs] || vs : '미요청'}`),
      arrow(),
      h('span', { class: `flow-step ${rv.state === 'resolved' ? 'ok' : rv.state === 'in_review' ? 'warn' : 'bad'}`, role: 'button', tabindex: '0', onclick: () => setTab('review') },
        `처리: ${L.review[rv.state] || rv.state}`)),
  ]);
}

// ---- stage (video / frames / overlay)
function setVideoSrc(src) {
  if ((video.dataset.src || '') === (src || '')) return;
  video.dataset.src = src || '';
  if (src) video.src = src; else { video.removeAttribute('src'); video.load(); }
}
function renderStage() {
  const ev = currentEvent();
  const empty = $('#stage-empty'); const notice = $('#stage-notice'); const nav = $('#frame-nav');
  if (!ev) {
    state.stageMode = 'none';
    empty.classList.remove('hidden'); video.classList.add('hidden'); frameImg.classList.add('hidden');
    notice.classList.add('hidden'); nav.classList.add('hidden'); $('#stage-span').classList.add('hidden');
    if (state.loaded && !state.events.length) empty.replaceChildren(h('b', null, '아직 사건이 없습니다'), h('span', null, '카메라 분석이 시작되고 통제구역 진입이 감지되면 여기에 근거 영상이 표시됩니다.'));
    setVideoSrc(''); renderStageBar(null); drawOverlay();
    return;
  }
  empty.classList.add('hidden');
  const orig = originalVideoName(ev.source); const hasOrig = !!(orig && state.videos.has(orig));
  if (state.videoMode === 'original' && !hasOrig) state.videoMode = 'clip';
  const mode = state.videoMode === 'original' ? 'original' : ev.evidence && ev.evidence.clip_available ? 'clip' : 'frames';
  state.stageMode = mode;
  if (mode === 'frames') {
    video.classList.add('hidden'); if (!video.paused) video.pause();
    frameImg.classList.remove('hidden'); nav.classList.remove('hidden');
    notice.classList.remove('hidden');
    notice.textContent = ev.evidence_ready ? '근거 클립 없음 — 키 프레임만 표시' : '근거 영상 준비 중 (업로드 대기)';
    syncStageFrame(ev);
  } else {
    frameImg.classList.add('hidden'); notice.classList.add('hidden'); nav.classList.add('hidden'); video.classList.remove('hidden');
    setVideoSrc(mediaUrl(mode === 'original' ? `/api/videos/${encodeURIComponent(orig)}` : ev.evidence.clip_url));
    applyPendingSeek(ev, mode);
  }
  renderStageBar(ev, hasOrig, mode); drawOverlay(); updateSpanLabel(); updatePlayhead();
}
function applyPendingSeek(ev, mode) {
  const ps = state.pendingSeek; if (!ps || ps.id !== ev.event_id) return;
  const target = mode === 'original' ? Math.max(0, num(ev.media_entry_s) - 3) : 0;
  const doSeek = () => {
    if (state.pendingSeek !== ps) return;
    state.pendingSeek = null;
    try { video.currentTime = target; } catch { /* not seekable yet */ }
    if (ps.play) video.play().catch(() => {});
  };
  if (video.readyState >= 1) doSeek(); else video.addEventListener('loadedmetadata', doSeek, { once: true });
}
function renderStageBar(ev, hasOrig, mode) {
  const bar = $('#stage-bar');
  if (!ev) { renderIfChanged(bar, 'none', () => h('span', { class: 'muted' }, '영상 소스: — (사건을 선택하세요)')); return; }
  const sig = JSON.stringify([ev.event_id, hasOrig, mode, state.annot, ev.evidence.clip_start_media_s, ev.evidence.clip_end_media_s]);
  renderIfChanged(bar, sig, () => {
    const seg = h('div', { class: 'seg', role: 'group', 'aria-label': '영상 소스' },
      h('button', { class: `btn btn-sm${mode !== 'original' ? ' active' : ''}`, type: 'button', title: '사건별 근거 클립 (업로드된 증거 영상)', onclick: () => setVideoMode('clip') }, '근거 클립'),
      hasOrig ? h('button', { class: `btn btn-sm${mode === 'original' ? ' active' : ''}`, type: 'button', title: '원본 영상에서 사건 3초 전부터 재생합니다', onclick: () => setVideoMode('original') }, '원본 영상에서 보기') : null);
    const info = mode === 'original' ? `원본 ${originalVideoName(ev.source)} · 사건 구간 ${fmtMedia(ev.media_entry_s)}–${fmtMedia(ev.media_exit_s)}`
      : mode === 'clip' ? `근거 클립 · 영상 ${fmtMedia(ev.evidence.clip_start_media_s)}–${fmtMedia(ev.evidence.clip_end_media_s)} (${ev.evidence.clip_fps || '?'} fps)`
        : '근거 클립 없음 · 키 프레임 표시';
    return [seg, h('span', { class: 'muted' }, info), h('span', { class: 'spacer' }),
      h('span', { class: 'muted', title: '영상 위에 그려지는 표시' }, '빨강 = 통제구역' + (mode === 'clip' ? ' · 노랑 = 추적 대상' : '')),
      h('button', { class: 'btn btn-sm', type: 'button', title: '키 프레임 원본/주석 전환 (r 키)', onclick: () => setAnnot(!state.annot) }, state.annot ? '주석 프레임 (r)' : '원본 프레임 (r)')];
  });
}
function setImgSrc(img, url) {
  if ((img.dataset.src || '') === (url || '')) return;
  img.dataset.src = url || '';
  if (url) img.src = url; else img.removeAttribute('src');
}
function syncStageFrame(ev) {
  const f = frameFor(ev, state.frameRole);
  setImgSrc(frameImg, f.pick && f.pick.available ? mediaUrl(f.pick.url) : '');
  renderIfChanged($('#frame-nav'), JSON.stringify([ev.event_id, state.frameRole]), () => ROLES.map((r) =>
    h('button', { class: `btn${r === state.frameRole ? ' active' : ''}`, type: 'button', onclick: () => setFrameRole(r) }, L.frameRole[r])));
}
function contentRect(media, ev) {
  const stage = $('#stage'); const W = stage.clientWidth; const H = stage.clientHeight;
  let iw = media === video ? video.videoWidth : media.naturalWidth;
  let ih = media === video ? video.videoHeight : media.naturalHeight;
  if (!iw || !ih) { iw = ev.frame_width || 16; ih = ev.frame_height || 9; }
  const s = Math.min(W / iw, H / ih); const w = iw * s; const hh = ih * s;
  return { left: (W - w) / 2, top: (H - hh) / 2, width: w, height: hh };
}
function currentTrackBox(ev) {
  if (state.stageMode !== 'clip' || !ev.trajectory || !ev.trajectory.length || !ev.frame_width || !ev.frame_height) return null;
  const t = num(ev.evidence.clip_start_media_s) + video.currentTime;
  let best = null; let bd = 0.35;
  for (const p of ev.trajectory) { const d = Math.abs(num(p.media_time_s) - t); if (d < bd) { bd = d; best = p; } }
  if (!best || !best.box) return null;
  const b = best.box; const fw = ev.frame_width; const fh = ev.frame_height;
  return { x: b.x1 / fw, y: b.y1 / fh, w: (b.x2 - b.x1) / fw, h: (b.y2 - b.y1) / fh };
}
function drawOverlay() {
  const ev = currentEvent(); const svg = $('#overlay'); const label = $('#zone-label');
  const poly = ev && ev.zone_polygon;
  if (!ev || !poly || !poly.length || state.stageMode === 'none') { svg.replaceChildren(); label.classList.add('hidden'); return; }
  const r = contentRect(state.stageMode === 'frames' ? frameImg : video, ev);
  Object.assign(svg.style, { left: `${r.left}px`, top: `${r.top}px`, width: `${r.width}px`, height: `${r.height}px` });
  const kids = [svgEl('polygon', { points: poly.map((p) => `${(p.x * 1000).toFixed(1)},${(p.y * 1000).toFixed(1)}`).join(' ') })];
  const box = currentTrackBox(ev);
  if (box) kids.push(svgEl('rect', { class: 'track', x: (box.x * 1000).toFixed(1), y: (box.y * 1000).toFixed(1), width: (box.w * 1000).toFixed(1), height: (box.h * 1000).toFixed(1) }));
  svg.replaceChildren(...kids);
  const minx = Math.min(...poly.map((p) => p.x)); const miny = Math.min(...poly.map((p) => p.y));
  label.textContent = `통제구역 · ${ev.zone_name || ev.zone_id}`;
  label.classList.remove('hidden');
  label.style.left = `${r.left + minx * r.width + 4}px`;
  label.style.top = `${Math.max(r.top + 4, r.top + miny * r.height - 20)}px`;
}
function updateSpanLabel() {
  const ev = currentEvent(); const el = $('#stage-span'); const stage = $('#stage');
  if (!ev || state.stageMode !== 'original') { el.classList.add('hidden'); stage.classList.remove('in-span'); return; }
  const t = video.currentTime; const inSpan = t >= num(ev.media_entry_s) - 0.05 && t <= num(ev.media_exit_s) + 0.05;
  el.classList.remove('hidden'); el.classList.toggle('in', inSpan); stage.classList.toggle('in-span', inSpan);
  const span = `${fmtMedia(ev.media_entry_s)}–${fmtMedia(ev.media_exit_s)}`;
  el.textContent = inSpan ? `● 사건 구간 재생 중 (${span})` : `사건 구간 ${span} · 현재 ${fmtMedia(t)}`;
}

// ---- timeline
function timelineCamera() {
  const ev = currentEvent();
  if (ev) return ev.camera_id;
  if (state.filters.camera) return state.filters.camera;
  return state.events.length ? state.events[0].camera_id : null;
}
async function refreshTimeline() {
  const cam = timelineCamera();
  if (!cam) { if (state.timeline.camera) { state.timeline = { ...state.timeline, camera: null, items: [], sig: '' }; renderTimeline(); } return; }
  if (state.timeline.inflightCam === cam) return;
  state.timeline.inflightCam = cam;
  try {
    const r = await API.timeline(cam);
    if (timelineCamera() !== cam) return; // the selection moved to another camera meanwhile
    const items = r.items || [];
    const sig = cam + JSON.stringify(items.map((i) => [i.event_id, i.review_state, i.media_entry_s, i.media_exit_s]));
    if (sig !== state.timeline.sig) { state.timeline = { ...state.timeline, camera: cam, items, sig }; renderTimeline(); }
  } catch { /* keep the last timeline */ } finally { if (state.timeline.inflightCam === cam) state.timeline.inflightCam = null; }
}
function tickStep(d) { for (const s of [5, 10, 15, 30, 60, 120, 300, 600]) if (d / s <= 12) return s; return 1200; }
function renderTimeline() {
  const head = $('#tl-head'); const root = $('#timeline'); const tl = state.timeline; const ev = currentEvent();
  const cam = tl.camera;
  if (!cam) {
    renderIfChanged(head, 'none', () => [h('span', null, '타임라인'), h('span', { class: 'muted' }, '사건이나 카메라를 선택하면 영상 시간축에 사건이 표시됩니다')]);
    root.dataset.sig = ''; root.replaceChildren(h('div', { class: 'tl-empty' }, '표시할 타임라인이 없습니다.'));
    return;
  }
  const camName = (cameraIndex().find((c) => c.camera_id === cam) || {}).name || cam;
  const dur = state.stageMode === 'original' && Number.isFinite(video.duration) ? video.duration : 0;
  const maxExit = Math.max(0, ...tl.items.map((i) => num(i.media_exit_s ?? i.media_entry_s)));
  const domain = Math.max(10, Math.ceil(Math.max(maxExit + 2, dur)));
  tl.domain = domain;
  const sig = JSON.stringify([tl.sig, domain, state.selectedId, state.stageMode, camName]);
  if (root.dataset.sig === sig) return;
  root.dataset.sig = sig;
  const leg = (cls, label) => h('span', null, h('i', { class: cls === 'rs-needs_review' ? 'rs-needs_review' : cls, style: cls === 'rs-needs_review' ? 'background:#e45560' : cls === 'rs-in_review' ? 'background:#f0a92e' : 'background:#3fae6a' }), label);
  head.dataset.sig = '';
  head.replaceChildren(h('span', null, '타임라인'), h('b', null, camName), h('span', { class: 'muted' }, `사건 ${tl.items.length}건 · 영상 시간(분:초) 기준 · 막대를 클릭하면 선택·이동`),
    h('div', { class: 'tl-legend' }, leg('rs-needs_review', '확인 필요'), leg('rs-in_review', '검토 중'), leg('rs-resolved', '처리 완료'),
      h('span', null, h('i', { style: 'background:#9aa5b1;background-image:repeating-linear-gradient(135deg, rgba(255,255,255,.5) 0 2px, transparent 2px 4px)' }), '빗금 = 체류')));
  // Overlapping events (two tracks in the zone at once) go to separate lanes so no bar hides another.
  const laneEnd = []; const laneOf = new Map();
  for (const it of tl.items.slice().sort((a, b) => num(a.media_entry_s) - num(b.media_entry_s))) {
    const s0 = num(it.media_entry_s); const e0 = Math.max(s0 + 0.5, num(it.media_exit_s ?? it.media_entry_s));
    let lane = laneEnd.findIndex((end) => end <= s0);
    if (lane < 0) { lane = laneEnd.length; laneEnd.push(0); }
    laneEnd[lane] = e0; laneOf.set(it.event_id, lane);
  }
  const lanes = Math.max(1, laneEnd.length); const barH = lanes > 1 ? 14 : 22; const barGap = 3;
  const track = h('div', { class: 'tl-track', style: `height:${Math.max(62, 8 + lanes * (barH + barGap) + 22)}px` });
  const step = tickStep(domain);
  for (let t = 0; t <= domain; t += step) track.append(h('div', { class: 'tl-tick', style: `left:${(t / domain) * 100}%` }, h('span', null, fmtMedia(t))));
  if (ev && state.stageMode === 'original' && ev.camera_id === cam) {
    track.append(h('div', { class: 'tl-span', style: `left:${(num(ev.media_entry_s) / domain) * 100}%;width:${Math.max(0.2, ((num(ev.media_exit_s) - num(ev.media_entry_s)) / domain) * 100)}%` }));
  }
  for (const it of tl.items) {
    const l = (num(it.media_entry_s) / domain) * 100;
    const w = Math.max(0.4, ((num(it.media_exit_s ?? it.media_entry_s) - num(it.media_entry_s)) / domain) * 100);
    const tip = `${kindLabel(it.kind)} · ${fmtMedia(it.media_entry_s, true)}–${fmtMedia(it.media_exit_s, true)} · 체류 ${fmtSec(it.dwell_seconds)} · ${L.review[it.review_state] || it.review_state}`;
    track.append(h('div', { class: `tl-bar rs-${it.review_state}${it.kind === 'EVENT_KIND_ZONE_DWELL' ? ' kind-dwell-bar' : ''}${it.event_id === state.selectedId ? ' selected' : ''}`,
      style: `left:${l}%;width:${w}%;top:${8 + laneOf.get(it.event_id) * (barH + barGap)}px;height:${barH}px`, role: 'button', tabindex: '0', 'aria-label': tip, dataset: { id: it.event_id },
      onclick: () => selectEvent(it.event_id, { seek: true, play: true, scroll: true }),
      onkeydown: (e) => { if (e.key === 'Enter') selectEvent(it.event_id, { seek: true, play: true, scroll: true }); },
      onmouseenter: (e) => showTip(track, e.currentTarget, tip), onmouseleave: hideTip }));
  }
  track.append(h('div', { id: 'tl-playhead', class: 'tl-playhead hidden' }, h('span', { style: `top:${8 + lanes * (barH + barGap) + 1}px` })));
  root.replaceChildren(track);
  updatePlayhead();
}
function showTip(track, bar, text) {
  hideTip();
  const tip = h('div', { class: 'tl-tip', id: 'tl-tip' }, text);
  tip.style.left = `${bar.offsetLeft + bar.offsetWidth / 2}px`; tip.style.top = `${bar.offsetTop}px`;
  track.append(tip);
}
function hideTip() { const t = $('#tl-tip'); if (t) t.remove(); }
function updatePlayhead() {
  const ph = $('#tl-playhead'); if (!ph) return;
  const ev = currentEvent();
  if (!ev || state.stageMode === 'frames' || state.stageMode === 'none' || ev.camera_id !== state.timeline.camera || video.readyState < 1) { ph.classList.add('hidden'); return; }
  const t = state.stageMode === 'original' ? video.currentTime : num(ev.evidence.clip_start_media_s) + video.currentTime;
  ph.classList.remove('hidden');
  ph.style.left = `${Math.min(100, Math.max(0, (t / (state.timeline.domain || 1)) * 100))}%`;
  ph.firstChild.textContent = fmtMedia(t);
}

// ---- detail panel (tabs)
function renderDetail() {
  const root = $('#detail'); const ev = currentEvent();
  if (!ev) {
    renderIfChanged(root, 'none', () => h('div', { class: 'detail-empty' }, '오른쪽 목록에서 사건을 선택하면 CV 근거, VLM 설명, 처리 상태가 여기에 표시됩니다.'));
    return;
  }
  const full = hasFullDetail(ev);
  const prov = state.status && state.status.vlm ? state.status.vlm.provider : '';
  const sig = JSON.stringify([sigOf(ev), full, state.tab, state.form.open, state.reviewError, state.reviewBusy, state.explainBusy, state.detailError,
    (ev.review_log || []).length, prov, ev.review.updated_at]);
  renderIfChanged(root, sig, () => [buildTabs(ev), buildTabBody(ev, full), buildEng(ev, full)]);
  syncFrames(ev);
}
function buildTabs(ev) {
  const x = ev.explanation; const vs = x ? x.status : null;
  const tabs = [['cv', 'CV 근거', `프레임 ${frameCount(ev)}`], ['vlm', 'VLM 설명', vs ? L.vlmStatus[vs] || vs : '미요청'], ['review', '처리 상태', L.review[ev.review.state] || ev.review.state]];
  return h('div', { class: 'tabs', role: 'tablist' }, tabs.map(([id, label, hint]) =>
    h('button', { class: 'tab', role: 'tab', type: 'button', 'aria-selected': state.tab === id ? 'true' : 'false', dataset: { tab: id }, onclick: () => setTab(id) }, label, h('span', { class: 'cnt' }, hint))));
}
function buildTabBody(ev, full) {
  if (state.tab === 'vlm') return buildVlmTab(ev);
  if (state.tab === 'review') return buildReviewTab(ev, full);
  return buildCvTab(ev);
}
function buildCvTab(ev) {
  const fields = h('dl', { class: 'fields' },
    dt('구역'), dd(h('b', null, ev.zone_name || '—'), h('span', { class: 'muted' }, ` (${ev.zone_id || '—'})`)),
    dt('카메라'), dd(`${ev.camera_name || ev.camera_id} · 장치 ${ev.device_id || '—'}`),
    dt('사건 종류'), dd(h('span', { class: `kind-badge ${kindClass(ev.kind)}` }, kindLabel(ev.kind)), ' ', h('span', { class: 'muted' }, L.eventStatus[ev.status] || ev.status || '')),
    dt('트랙 ID'), dd(`#${ev.track_id ?? '—'}`),
    dt('진입'), dd(`${fmtDateTime(ev.entry_ts)} · 영상 ${fmtMedia(ev.media_entry_s, true)}`),
    dt('이탈'), dd(ev.exit_ts ? `${fmtDateTime(ev.exit_ts)} · 영상 ${fmtMedia(ev.media_exit_s, true)}` : '— (진행 중)'),
    dt('체류 시간'), dd(fmtSec(ev.dwell_seconds)),
    dt('종료 사유'), dd(L.closeReason[ev.close_reason] || ev.close_reason || '—'),
    dt('근거 상태'), dd(ev.evidence_ready
      ? `✓ 수신 완료 · 프레임 ${frameCount(ev)}장 · ${ev.evidence.clip_available ? '클립 있음' : '클립 없음'}`
      : '⏳ 업로드 대기 중 (프레임/클립이 아직 도착하지 않았습니다)'),
    dt('감지 / 수신'), dd(`${fmtDateTime(ev.detected_at)} 감지 · ${fmtClock(ev.first_received_at)} 서버 수신`));
  const head = h('div', { class: 'frames-head' }, h('h3', null, '키 프레임'),
    h('span', { class: 'muted small' }, '클릭: 크게 보기 + 클립을 해당 시점으로 이동'),
    h('div', { class: 'seg', role: 'group', 'aria-label': '프레임 종류' },
      h('button', { id: 'annot-off', class: 'btn btn-sm', type: 'button', onclick: () => setAnnot(false) }, '원본'),
      h('button', { id: 'annot-on', class: 'btn btn-sm', type: 'button', title: '탐지 상자·구역이 그려진 프레임 (r 키로 전환)', onclick: () => setAnnot(true) }, '주석')));
  const view = h('div', { class: 'frame-view', id: 'frame-view' }, h('img', { alt: '선택한 키 프레임' }), h('div', { class: 'cap' }), h('div', { class: 'nopic hidden' }, '이 프레임은 아직 업로드되지 않았습니다'));
  const thumbs = h('div', { class: 'thumbs', id: 'thumbs' }, ROLES.map((r) =>
    h('button', { class: 'thumb', type: 'button', dataset: { role: r }, title: `${L.frameRole[r]} 프레임 보기`, onclick: () => { setFrameRole(r); seekToFrame(ev, r); } },
      h('img', { alt: '' }), h('div', { class: 'ph hidden' }, '없음'),
      h('div', { class: 'lab' }, h('span', null, L.frameRole[r]), h('span', null, '')))));
  return h('div', { class: 'tab-body' }, fields, head, view, thumbs);
}
/** Applies state.annot / state.frameRole to the CV tab and the stage carousel without rebuilding the tab. */
function syncFrames(ev) {
  const thumbs = $('#thumbs');
  if (thumbs) {
    for (const b of thumbs.children) {
      const r = b.dataset.role; const f = frameFor(ev, r); const img = b.querySelector('img');
      const url = f.pick && f.pick.available ? mediaUrl(f.pick.url) : '';
      setImgSrc(img, url);
      b.classList.toggle('active', r === state.frameRole); b.classList.toggle('missing', !url);
      b.querySelector('.ph').classList.toggle('hidden', !!url); img.classList.toggle('hidden', !url);
      b.querySelector('.lab span:last-child').textContent = f.pick ? fmtMedia(f.pick.media_time_s, true) : '—';
    }
  }
  const view = $('#frame-view');
  if (view) {
    const f = frameFor(ev, state.frameRole); const img = view.querySelector('img');
    const url = f.pick && f.pick.available ? mediaUrl(f.pick.url) : '';
    setImgSrc(img, url);
    view.querySelector('.nopic').classList.toggle('hidden', !!url); img.classList.toggle('hidden', !url);
    view.querySelector('.cap').textContent = `${L.frameRole[state.frameRole]} · ${state.annot ? '주석' : '원본'}`
      + (f.pick ? ` · 영상 ${fmtMedia(f.pick.media_time_s, true)} · 프레임 #${f.pick.frame_index}` : '');
  }
  const off = $('#annot-off'); const on = $('#annot-on');
  if (off) off.classList.toggle('active', !state.annot);
  if (on) on.classList.toggle('active', state.annot);
  if (state.stageMode === 'frames') syncStageFrame(ev);
}
function seekToFrame(ev, role) {
  const f = frameFor(ev, role); if (!f.pick) return;
  const mt = num(f.pick.media_time_s);
  if (state.stageMode === 'original') { video.currentTime = mt; return; }
  if (state.stageMode === 'clip') {
    const t = mt - num(ev.evidence.clip_start_media_s);
    const end = Number.isFinite(video.duration) ? video.duration : num(ev.evidence.clip_end_media_s) - num(ev.evidence.clip_start_media_s);
    if (t >= 0 && t <= end + 0.5) { video.currentTime = Math.min(t, Math.max(0, end - 0.05)); video.pause(); }
    else toast(`이 프레임(${fmtMedia(mt, true)})은 근거 클립 구간 밖입니다. "원본 영상에서 보기"로 확인하세요.`);
  }
}
const sec = (title, body) => h('div', { class: 'vlm-sec' }, h('h3', null, title), body);
const ul = (arr) => (arr && arr.length ? h('ul', null, arr.map((t) => h('li', null, String(t)))) : h('p', { class: 'muted', style: 'margin:2px 0' }, '없음'));
function buildVlmTab(ev) {
  const x = ev.explanation; const vs = x ? x.status : null; const out = [];
  const prov = state.status && state.status.vlm ? state.status.vlm.provider : '';
  if ((x && x.provider === 'mock') || (!x && prov === 'mock')) out.push(h('div', { class: 'warn-chip', style: 'margin-bottom:10px' }, '⚠ VLM 모의 응답(mock): 실제 해석 아님'));
  const retryBtn = (label = '다시 해석') => h('button', { class: 'btn btn-sm btn-primary', type: 'button', disabled: IS_STATIC || state.explainBusy || !ev.evidence_ready,
    title: IS_STATIC ? STATIC_MSG : !ev.evidence_ready ? '근거 업로드가 끝나야 해석을 요청할 수 있습니다' : '서버에 해석을 다시 요청합니다', onclick: () => requestExplain(ev.event_id) }, `↻ ${label}`);
  if (!x) {
    out.push(h('div', { class: 'vlm-none' }, h('div', null, '해석이 요청되지 않았습니다.'), h('div', { style: 'margin-top:6px' }, retryBtn('해석 요청'))));
  } else if (vs === 'queued' || vs === 'running') {
    out.push(h('div', { class: 'vlm-wait' }, h('span', { class: 'spinner' }), vs === 'queued' ? `해석 대기 중 (시도 ${x.attempts || 0}회)` : `해석 중… (${x.model || x.provider})`));
  } else if (vs !== 'done') {
    out.push(h('div', { class: 'vlm-error' }, h('div', null, h('b', null, `해석 ${L.vlmStatus[vs] || vs}`), ` · 시도 ${x.attempts || 0}회`),
      h('div', { class: 'small', style: 'margin-top:4px' }, x.error || '(오류 메시지 없음)'), h('div', { style: 'margin-top:8px' }, retryBtn())));
  } else if (x.result) {
    const r = x.result; const p = r.person || {};
    out.push(h('div', { class: 'vlm-summary' }, r.scene_summary || '(요약 없음)'));
    out.push(h('div', { class: 'review-now' },
      h('span', { class: 'muted' }, '판단'), h('span', { class: `assess assess-${r.assessment || 'unclear'}` }, L.assessment[r.assessment] || r.assessment || '—'),
      h('span', { class: 'muted' }, '구역 관계'), h('b', null, L.zoneRelation[r.zone_relation] || r.zone_relation || '—'),
      h('span', { class: 'muted' }, '행동'), h('b', null, r.activity || '—')));
    out.push(h('div', { class: 'grid2' },
      sec('관찰 (프레임별)', r.observations && r.observations.length
        ? h('ul', null, r.observations.map((o) => h('li', null, h('span', { class: 'obs-frame' }, L.frameRole[o.frame] || o.frame), o.text)))
        : h('p', { class: 'muted' }, '없음')),
      sec('사람 정보', h('dl', { class: 'fields' },
        dt('구역 내 인원'), dd(p.count_in_zone == null ? '불명' : String(p.count_in_zone)),
        dt('외형'), dd(p.appearance || '불명'),
        dt('보호구'), dd(p.ppe_visible == null ? '불명' : p.ppe_visible ? '착용 보임' : '안 보임'))),
      sec('판단 근거', ul(r.reasons)),
      h('div', { class: 'vlm-sec limits' }, h('h3', null, '판단의 한계'), ul(r.uncertainties))));
    if (r.operator_question) out.push(sec('담당자에게 묻는 질문', h('p', { style: 'margin:2px 0' }, `❓ ${r.operator_question}`)));
    out.push(h('div', { class: 'review-actions' }, retryBtn('다시 해석')));
  } else {
    out.push(h('div', { class: 'vlm-error' }, '해석은 완료됐지만 결과 데이터가 비어 있습니다. ', retryBtn()));
  }
  if (x) out.push(h('p', { class: 'note' }, `${x.provider} · ${x.model} · 프롬프트 ${x.prompt_version} · ${x.latency_ms != null ? `${Math.round(x.latency_ms)} ms` : '—'} · 완료 ${x.completed_at ? fmtClock(x.completed_at) : '—'}`));
  out.push(h('div', { class: 'vlm-footer' }, '모델 응답은 참고용이며 사건의 정답이 아닙니다. 확률·신뢰도는 표시하지 않습니다.'));
  return h('div', { class: 'tab-body' }, ...out);
}
function buildReviewTab(ev, full) {
  const rv = ev.review; const st = rv.state; const out = [];
  out.push(h('div', { class: 'review-now' }, h('span', { class: 'muted' }, '현재 상태'),
    h('span', { class: `chip chip-lg st-${st}` }, `● ${L.review[st] || st}`),
    st === 'resolved' && rv.resolution ? h('span', { class: 'chip chip-lg', style: 'background:var(--panel-2)' }, `결과: ${L.resolution[rv.resolution] || rv.resolution}`) : null,
    rv.updated_at ? h('span', { class: 'muted small' }, `${fmtDateTime(rv.updated_at)} · ${rv.actor || '—'}`) : null));
  if (rv.note) out.push(h('p', { class: 'small', style: 'margin:0 0 8px' }, '메모: ', rv.note));
  if (!IS_STATIC) out.push(h('div', { class: 'actor-row' }, h('label', { for: 'actor' }, '담당자'),
    h('input', { id: 'actor', type: 'text', value: state.actor, maxlength: '80', placeholder: '이름 또는 ID',
      oninput: (e) => { state.actor = e.target.value; try { localStorage.setItem('zw.actor', state.actor); } catch { /* ignore */ } } }),
    h('span', { class: 'muted small' }, '(이 브라우저에 저장됩니다)')));
  const acts = h('div', { class: 'review-actions' });
  // In static mode every transition button stays visible (so the flow is still readable) but disabled, with the reason as tooltip + text.
  const lock = IS_STATIC ? { disabled: true, title: STATIC_MSG } : { disabled: state.reviewBusy };
  if (st === 'needs_review') acts.append(btn('▶ 검토 시작', 'btn-primary', () => doReview(ev.event_id, 'in_review'), { ...lock, id: 'btn-start-review' }));
  if (st === 'in_review') {
    acts.append(btn('✓ 처리 완료…', 'btn-success', () => { state.form.open = true; renderDetail(); }, { ...lock, id: 'btn-resolve' }));
    acts.append(btn('↩ 다시 확인 필요로', '', () => doReview(ev.event_id, 'needs_review'), { ...lock, id: 'btn-back' }));
  }
  if (st === 'resolved') acts.append(btn('↻ 재검토', '', () => doReview(ev.event_id, 'in_review'), { ...lock, id: 'btn-reopen' }));
  acts.append(IS_STATIC ? h('span', { class: 'err-text', id: 'review-static-note' }, `⚠ ${STATIC_MSG} (저장된 실행 결과를 읽기 전용으로 재생 중)`)
    : h('span', { class: 'muted small' }, st === 'needs_review' ? '검토를 시작하면 "검토 중"으로 바뀝니다.'
      : st === 'in_review' ? '처리 결과를 골라 완료하거나, 다시 확인 필요로 되돌립니다.' : '처리 완료된 사건입니다. 필요하면 재검토할 수 있습니다.'));
  out.push(acts);
  if (state.form.open && st === 'in_review') out.push(buildResolveForm(ev));
  if (state.reviewError) out.push(h('p', { class: 'err-text' }, `⚠ ${state.reviewError}`));
  const log = full ? ev.review_log || [] : null;
  out.push(h('h3', { style: 'margin-top:12px' }, '처리 이력'));
  if (!log) out.push(h('p', { class: 'muted small' }, state.detailError ? `이력을 불러오지 못했습니다 (${state.detailError})` : '이력을 불러오는 중…'));
  else if (!log.length) out.push(h('p', { class: 'muted small' }, '아직 처리 이력이 없습니다.'));
  else {
    out.push(h('ul', { class: 'log' }, log.slice().reverse().map((l) => h('li', null,
      h('span', { class: 'when' }, fmtDateTime(l.ts)),
      h('span', null, h('span', { class: `chip st-${l.from_state}` }, L.review[l.from_state] || l.from_state), h('span', { class: 'arrow' }, '→'),
        h('span', { class: `chip st-${l.to_state}` }, L.review[l.to_state] || l.to_state),
        l.resolution ? [' ', h('b', null, L.resolution[l.resolution] || l.resolution)] : null,
        ' · ', h('span', { class: 'muted' }, l.actor || '—'), l.note ? [' · ', l.note] : null)))));
  }
  return h('div', { class: 'tab-body' }, ...out);
}
function buildResolveForm(ev) {
  const sel = h('select', { id: 'res-select', onchange: (e) => { state.form.resolution = e.target.value; } },
    Object.entries(L.resolution).map(([k, v]) => h('option', { value: k, selected: k === state.form.resolution }, v)));
  const ta = h('textarea', { id: 'res-note', placeholder: '처리 메모 (선택)', maxlength: '2000', oninput: (e) => { state.form.note = e.target.value; } });
  ta.value = state.form.note;
  return h('div', { class: 'form', id: 'resolve-form' },
    h('label', { for: 'res-select' }, '처리 결과'), sel,
    h('label', { for: 'res-note' }, '메모'), ta,
    h('div', { class: 'form-actions' },
      btn('처리 완료 저장', 'btn-success', () => doReview(ev.event_id, 'resolved', { resolution: state.form.resolution, note: state.form.note }), { id: 'btn-resolve-save', disabled: state.reviewBusy }),
      btn('취소', '', () => { state.form.open = false; renderDetail(); })));
}
function buildEng(ev, full) {
  const run = ev.run || {}; const x = ev.explanation; const lat = ev.latency || {};
  return h('details', { class: 'eng', open: state.engOpen, ontoggle: (e) => { state.engOpen = e.target.open; } },
    h('summary', null, '상세 정보 (엔지니어용)'),
    h('div', { class: 'body' },
      h('div', { class: 'grid2' },
        h('div', null, h('h3', null, '실행 정보'), h('dl', { class: 'fields mono' },
          dt('model'), dd(`${run.model_name || '—'} @ ${run.model_version || '—'}`), dt('backend'), dd(run.backend || '—'),
          dt('tracker'), dd(run.tracker || '—'), dt('config_hash'), dd(run.config_hash || '—'), dt('run_id'), dd(run.run_id || '—'),
          dt('edge_version'), dd(run.edge_version || '—'), dt('input'), dd(run.input_width ? `${run.input_width}×${run.input_height}` : '—'),
          dt('frame'), dd(ev.frame_width ? `${ev.frame_width}×${ev.frame_height}` : '—'), dt('source'), dd(ev.source || '—'), dt('ingest_source'), dd(ev.ingest_source || '—'))),
        h('div', null, h('h3', null, '지연 / 해석'), h('dl', { class: 'fields mono' },
          dt('latency.ingest_ms'), dd(fmtMs(lat.ingest_ms)), dt('latency.first_display_ms'), dd(fmtMs(lat.first_display_ms)),
          ...(x ? [dt('provider / model'), dd(`${x.provider} / ${x.model}`), dt('prompt / schema'), dd(`${x.prompt_version} / ${x.schema_version}`),
            dt('latency_ms'), dd(fmtMs(x.latency_ms)), dt('attempts'), dd(String(x.attempts)), dt('usage'), dd(JSON.stringify(x.usage || {})),
            dt('event_revision'), dd(String(x.event_revision)), dt('error'), dd(x.error || '—')] : [dt('explanation'), dd('없음')])))),
      h('h3', { style: 'margin-top:10px' }, 'VLM 원문 응답'),
      h('pre', { class: 'raw' }, full ? (x && x.raw_text) || '(원문 없음)' : '(불러오는 중…)'),
      h('p', { class: 'note mono' }, `event_id ${ev.event_id} · revision ${ev.revision} · `,
        h('a', { href: rawEventUrl(ev.event_id), target: '_blank', rel: 'noopener' }, 'raw JSON 열기'),
        full && ev.evidence_files ? ` · 근거 파일 ${ev.evidence_files.length}개` : '')));
}

// ---- audit drawer
function toggleAudit(open) {
  state.auditOpen = open == null ? !state.auditOpen : open;
  $('#audit-drawer').classList.toggle('hidden', !state.auditOpen);
  $('#audit-toggle').setAttribute('aria-pressed', String(state.auditOpen));
  if (state.auditOpen) { state.auditSig = ''; refreshAudit(); }
}
async function refreshAudit() {
  try {
    const r = await API.audit(); const items = r.items || [];
    const sig = JSON.stringify(items.map((i) => i.id));
    if (sig !== state.auditSig) { state.audit = items; state.auditSig = sig; renderAudit(); }
  } catch (e) { $('#audit-body').replaceChildren(h('p', { class: 'list-msg' }, `처리 기록을 불러오지 못했습니다 (${e.message})`)); }
}
function renderAudit() {
  const body = $('#audit-body');
  $('#audit-count').textContent = `${state.audit.length}건`;
  if (!state.audit.length) { body.replaceChildren(h('p', { class: 'list-msg' }, '아직 처리 기록이 없습니다.')); return; }
  body.replaceChildren(...state.audit.map((l) => {
    const ev = state.eventsById.get(l.event_id);
    const who = ev ? `${ev.zone_name || ev.zone_id} · ${ev.camera_name || ev.camera_id} · ${fmtClock(ev.entry_ts)}` : `사건 ${l.event_id.slice(0, 8)}…`;
    return h('div', { class: 'audit-row', role: 'button', tabindex: '0', title: '클릭하면 해당 사건을 엽니다',
      onclick: () => { selectEvent(l.event_id, { seek: true, play: false, scroll: true }); setTab('review'); } },
      h('span', { class: 'when' }, fmtDateTime(l.ts)),
      h('span', null, h('span', { class: `chip st-${l.from_state}` }, L.review[l.from_state] || l.from_state), h('span', { class: 'arrow' }, ' → '),
        h('span', { class: `chip st-${l.to_state}` }, L.review[l.to_state] || l.to_state),
        l.resolution ? [' ', h('b', null, L.resolution[l.resolution] || l.resolution)] : null,
        h('div', { class: 'small muted' }, `${who} · ${l.actor || '—'}${l.note ? ` · ${l.note}` : ''}`)));
  }));
}
function updateExportLinks() {
  if (IS_STATIC) { $('#export-csv').classList.add('hidden'); $('#export-json').href = `${STATIC_BASE}/export.json`; return; }
  const p = new URLSearchParams();
  if (state.filters.camera) p.set('camera_id', state.filters.camera);
  if (state.filters.state) p.set('state', state.filters.state);
  const q = p.toString() ? `&${p.toString()}` : '';
  $('#export-csv').href = `/api/export?format=csv${q}`;
  $('#export-json').href = `/api/export?format=json${q}`;
}

// ---------------------------------------------------------------- 7. actions
function selectEvent(id, opts = {}) {
  if (!id) return;
  const changed = id !== state.selectedId;
  state.selectedId = id; state.form.open = false; state.reviewError = null; state.detailError = null;
  if (changed) { state.detail = null; state.detailSig = null; state.frameRole = 'entry'; }
  if (opts.seek) state.pendingSeek = { id, play: !!opts.play };
  const cam = timelineCamera();
  if (cam !== state.timeline.camera) state.timeline = { ...state.timeline, camera: cam, items: [], sig: '' };
  renderList(); renderSelHeader(); renderStage(); renderDetail();
  refreshTimeline(); renderTimeline();
  if (opts.scroll) { const row = $('#event-list').querySelector(`[data-id="${CSS.escape(id)}"]`); if (row) row.scrollIntoView({ block: 'nearest' }); }
  loadDetail(id);
}
async function loadDetail(id) {
  if (state.detailLoading === id) return;
  state.detailLoading = id;
  try {
    const d = await API.event(id);
    if (state.selectedId !== id) return;
    if (!d) { state.detailError = IS_STATIC ? `상세 파일이 없습니다 (${STATIC_BASE}/events/${id}.json)` : '사건을 찾을 수 없습니다'; renderDetail(); return; }
    state.detail = d; state.detailSig = sigOf(d); state.detailError = null;
    renderSelHeader(); renderStage(); renderDetail();
  } catch (e) {
    if (state.selectedId === id) { state.detailError = e.message; renderDetail(); }
  } finally { if (state.detailLoading === id) state.detailLoading = null; }
}
function setTab(tab) { state.tab = tab; renderDetail(); }
function setAnnot(v) { state.annot = !!v; const ev = currentEvent(); if (ev) syncFrames(ev); renderStage(); }
function setFrameRole(r) { state.frameRole = r; const ev = currentEvent(); if (ev) syncFrames(ev); }
function setVideoMode(m) {
  if (state.videoMode === m) return;
  state.videoMode = m;
  if (state.selectedId) state.pendingSeek = { id: state.selectedId, play: true };
  renderStage(); renderSelHeader(); renderTimeline();
}
function setFilter(key, value) {
  state.filters[key] = value;
  const map = { camera: '#f-camera', state: '#f-state', kind: '#f-kind', q: '#f-q' };
  if ($(map[key]).value !== value) $(map[key]).value = value;
  renderList(); renderCameras(); updateExportLinks(); refreshTimeline();
}
function toggleCameraFilter(cameraId) { setFilter('camera', state.filters.camera === cameraId ? '' : cameraId); }

async function doReview(id, toState, extra = {}) {
  if (IS_STATIC) { toast(STATIC_MSG, 'err'); return; }
  if (state.reviewBusy) return;
  state.reviewError = null; state.reviewBusy = true; renderDetail();
  try {
    const body = { state: toState, actor: (state.actor || '').trim() || '담당자', note: extra.note || '', resolution: toState === 'resolved' ? extra.resolution || 'confirmed' : null };
    const updated = await API.review(id, body);
    const ev = state.eventsById.get(id);
    if (ev) {
      const c = state.counts;
      if (ev.review.state !== updated.review.state && c) { c[ev.review.state] = Math.max(0, num(c[ev.review.state]) - 1); c[updated.review.state] = num(c[updated.review.state]) + 1; }
      Object.assign(ev, updated);
    }
    if (state.detail && state.detail.event_id === id) { Object.assign(state.detail, updated); state.detailSig = null; }
    state.form.open = false; state.form.note = '';
    toast(`처리 상태를 "${L.review[toState]}"(으)로 변경했습니다`, 'ok');
  } catch (e) {
    state.reviewError = e.status === 409 ? `허용되지 않는 상태 변경입니다 (${e.message})` : `저장 실패: ${e.message}`;
    toast(state.reviewError, 'err');
  } finally { state.reviewBusy = false; }
  renderTopbar(); renderList(); renderSelHeader(); renderDetail();
  await loadDetail(id);
  refreshTimeline(); if (state.auditOpen) { state.auditSig = ''; refreshAudit(); }
}
async function requestExplain(id) {
  if (IS_STATIC) { toast(STATIC_MSG, 'err'); return; }
  if (state.explainBusy) return;
  state.explainBusy = true; renderDetail();
  try {
    const r = await API.explain(id);
    toast(r.queued ? '다시 해석을 요청했습니다. 결과는 목록과 VLM 탭에 자동 반영됩니다.' : '이미 같은 해석 작업이 대기 중입니다.', 'ok');
  } catch (e) {
    toast(`해석 요청 실패: ${e.message}`, 'err');
  } finally { state.explainBusy = false; }
  renderDetail();
  if (state.selectedId === id) { state.detailSig = null; loadDetail(id); }
}
async function sendCommand(c, type) {
  if (IS_STATIC) { toast(STATIC_MSG, 'err'); return; }
  const key = c.camera_id; const label = type === 'START_CAMERA' ? '분석 시작' : '정지';
  state.commands[key] = { phase: 'wait', message: `${label} 명령 전송 중…` }; renderCameras();
  try {
    const r = await API.command(c.device_id, { type, camera_id: c.camera_id, source: '', issued_by: 'console' });
    const id = r.command.command_id;
    state.commands[key] = { phase: 'wait', id, message: `${label} 명령 전송됨 (구독 장치 ${r.subscribers}대) · 결과 대기 중…` }; renderCameras();
    const t0 = Date.now();
    while (Date.now() - t0 < 30000) {
      await sleep(1000);
      const cs = await API.commandStatus(id);
      if (cs.completed_at) {
        state.commands[key] = { phase: cs.result_ok ? 'ok' : 'err', id, message: `${cs.result_ok ? '✓ 완료' : '✗ 실패'}: ${cs.result_message || '(메시지 없음)'} · ${fmtClock(cs.completed_at)}` };
        renderCameras(); toast(`${c.name}: ${label} ${cs.result_ok ? '완료' : '실패'}`, cs.result_ok ? 'ok' : 'err');
        return;
      }
      if (cs.delivered_at && state.commands[key].message.indexOf('받았습니다') < 0) { state.commands[key] = { phase: 'wait', id, message: '장치가 명령을 받았습니다 · 실행 중…' }; renderCameras(); }
    }
    state.commands[key] = { phase: 'err', id, message: '30초 동안 장치 응답이 없습니다. 장치 연결 상태를 확인하세요.' }; renderCameras();
  } catch (e) {
    const msg = e.status === 409 ? '재생 모드에서는 장치 명령을 보낼 수 없습니다' : e.message;
    state.commands[key] = { phase: 'err', message: `✗ ${label} 명령 실패: ${msg}` }; renderCameras();
    toast(`명령 실패: ${msg}`, 'err');
  }
}

// ---------------------------------------------------------------- 8. keyboard + init
function onKey(e) {
  const t = e.target;
  if (t && (t.tagName === 'INPUT' || t.tagName === 'TEXTAREA' || t.tagName === 'SELECT' || t.isContentEditable)) return;
  if (e.key === 'ArrowDown' || e.key === 'ArrowUp') {
    const vis = filteredEvents(); if (!vis.length) return;
    let i = vis.findIndex((x) => x.event_id === state.selectedId);
    i = e.key === 'ArrowDown' ? Math.min(vis.length - 1, i + 1) : Math.max(0, i < 0 ? 0 : i - 1);
    selectEvent(vis[i].event_id, { seek: true, play: true, scroll: true });
    e.preventDefault();
  } else if (e.key === 'r' || e.key === 'R') {
    setAnnot(!state.annot);
  } else if (e.key === 'Escape' && state.auditOpen) {
    toggleAudit(false);
  }
}
function init() {
  if (IS_STATIC) document.title += ' — 정적 재생';
  try { state.actor = localStorage.getItem('zw.actor') || '담당자'; } catch { /* storage unavailable */ }
  $('#f-camera').addEventListener('change', (e) => setFilter('camera', e.target.value));
  $('#f-state').addEventListener('change', (e) => setFilter('state', e.target.value));
  $('#f-kind').addEventListener('change', (e) => setFilter('kind', e.target.value));
  $('#f-q').addEventListener('input', (e) => setFilter('q', e.target.value));
  $('#retry').addEventListener('click', () => { $('#banner-detail').textContent = '(재시도 중…)'; poll(); });
  $('#audit-toggle').addEventListener('click', () => toggleAudit());
  $('#audit-close').addEventListener('click', () => toggleAudit(false));
  video.addEventListener('timeupdate', () => { updateSpanLabel(); updatePlayhead(); if (state.stageMode === 'clip') drawOverlay(); });
  video.addEventListener('seeked', () => { updateSpanLabel(); updatePlayhead(); if (state.stageMode === 'clip') drawOverlay(); });
  video.addEventListener('loadedmetadata', () => { drawOverlay(); renderTimeline(); updatePlayhead(); updateSpanLabel(); });
  video.addEventListener('error', () => { if (video.dataset.src) toast('영상을 불러오지 못했습니다 (서버의 영상 파일을 확인하세요)', 'err'); });
  frameImg.addEventListener('load', drawOverlay);
  if (window.ResizeObserver) new ResizeObserver(() => drawOverlay()).observe($('#stage'));
  else window.addEventListener('resize', drawOverlay);
  document.addEventListener('keydown', onKey);
  poll();
}
init();
