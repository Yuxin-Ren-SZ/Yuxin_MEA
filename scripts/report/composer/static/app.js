/* Figure composer — canvas logic.
 *
 * The composition object mirrors figure_spec.json exactly, so "what the page
 * holds" and "what gets exported" are the same structure. Every GridStack event
 * writes back into it; every render reads from it. There is no second source of
 * truth to drift.
 *
 * Grid contract: a panel's {x, y, w, h} are the same integers here and in the
 * exported GridSpec. GridStack is configured with 12 columns to match.
 */
'use strict';

const COLS = 12;
const STORE_KEY = 'composer.autosave.v1';

const state = {
  boot: null,          // /api/bootstrap payload
  comp: null,          // the live composition (figure_spec.json shape)
  grids: new Map(),    // figureId -> GridStack
  selectedFigures: new Set(),
  selectedPanels: new Set(),
  dirty: false,
  zoom: 0.82,
};

/* ------------------------------------------------------------------ helpers */
const $ = (sel) => document.querySelector(sel);
const el = (tag, cls, text) => {
  const n = document.createElement(tag);
  if (cls) n.className = cls;
  if (text !== undefined) n.textContent = text;
  return n;
};

async function api(path, body) {
  const res = await fetch(path, body === undefined ? {} : {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify(body),
  });
  const data = await res.json().catch(() => ({error: res.statusText}));
  if (!res.ok) throw new Error(data.error || res.statusText);
  return data;
}

function banner(msg, kind) {
  const b = $('#banner');
  if (!msg) { b.hidden = true; return; }
  b.textContent = msg;
  b.className = 'banner' + (kind ? ' ' + kind : '');
  b.hidden = false;
}

function markDirty(v = true) {
  state.dirty = v;
  $('#dirty').hidden = !v;
  if (v) autosave();
}

function autosave() {
  try { localStorage.setItem(STORE_KEY, JSON.stringify(state.comp)); }
  catch (e) { /* quota — autosave is a convenience, not a guarantee */ }
}

const panelSpec = (id) => state.boot.panels.find((p) => p.id === id);

/* The wire format keeps groups under `globals`; the UI reads them constantly, so
 * expose a non-enumerable alias. Non-enumerable matters — it must not end up
 * duplicated at the top level when the composition is serialised. Every path
 * that replaces the composition wholesale (load, split, merge) goes through
 * here so the alias is never lost. */
function adopt(comp) {
  Object.defineProperty(comp, 'groups', {
    get() { return this.globals.groups; },
    set(v) { this.globals.groups = v; },
    configurable: true,
    enumerable: false,
  });
  state.comp = comp;
  return comp;
}

function figureOf(uid) {
  return state.comp.figures.find((f) => f.panels.some((p) => p.uid === uid));
}

function nextUid() {
  const used = new Set(state.comp.figures.flatMap((f) => f.panels.map((p) => p.uid)));
  let i = 1;
  while (used.has('p' + i)) i++;
  return 'p' + i;
}

function nextFigureId() {
  const used = new Set(state.comp.figures.map((f) => f.id));
  let i = 1;
  while (used.has('F' + i)) i++;
  return 'F' + i;
}

/* Reading-order letters, recomputed after every move so labels stay meaningful. */
function autoLabels(fig) {
  const letters = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ';
  const sorted = [...fig.panels].sort((a, b) => (a.y - b.y) || (a.x - b.x));
  const out = {};
  sorted.forEach((p, i) => { out[p.uid] = p.label || (letters[i] || 'P' + (i + 1)); });
  return out;
}

/* ------------------------------------------------------------------ previews */
const previewInFlight = new Map();

async function loadPreview(fig, p, tileBody) {
  const key = [p.panel, JSON.stringify(p.params || {}), p.w, p.h,
               state.comp.groups.join(',')].join('|');
  tileBody.innerHTML = '';
  const spinner = el('div', 'spinner', 'rendering…');
  tileBody.appendChild(spinner);

  let job = previewInFlight.get(key);
  if (!job) {
    job = api('/api/preview', {
      panel: p.panel, params: p.params || {},
      groups: state.comp.groups, w: p.w, h: p.h,
    });
    previewInFlight.set(key, job);
    job.finally(() => previewInFlight.delete(key));
  }

  let rec;
  try { rec = await job; }
  catch (e) {
    tileBody.innerHTML = '';
    tileBody.appendChild(el('div', 'msg', String(e.message || e)));
    return;
  }

  // the tile may have been resized or deleted while we waited
  if (!tileBody.isConnected) return;
  tileBody.innerHTML = '';
  const img = el('img');
  img.src = rec.url + '?k=' + rec.key;
  img.alt = p.panel;
  tileBody.appendChild(img);
  tileBody.appendChild(el('div', 'tile-cell', `${p.w}×${p.h}`));

  const tile = tileBody.closest('.panel-tile');
  tile.classList.toggle('warn', rec.status === 'no-data');
  tile.classList.toggle('err', rec.status === 'error' || rec.status === 'unknown');
  if (rec.warning) tile.title = rec.warning;
}

/* -------------------------------------------------------------- canvas build */
function buildCanvas() {
  const canvas = $('#canvas');
  canvas.innerHTML = '';
  state.grids.clear();
  state.comp.figures.forEach(buildFigureCard);
  updateToolbar();
}

function buildFigureCard(fig) {
  const card = el('div', 'figure-card');
  card.dataset.figure = fig.id;
  // Page width on screen tracks the real figure width, so relative sizes read true.
  card.style.width = (fig.width_in * 96 * state.zoom) + 'px';
  if (state.selectedFigures.has(fig.id)) card.classList.add('selected');

  /* --- header ---------------------------------------------------------- */
  const head = el('div', 'figure-head');
  head.appendChild(el('span', 'id', fig.id));

  const title = el('input', 'title');
  title.value = fig.title;
  title.placeholder = 'figure title';
  title.addEventListener('input', () => { fig.title = title.value; markDirty(); });
  head.appendChild(title);

  const rows = el('input', 'num');
  rows.type = 'number'; rows.min = '1'; rows.max = '40'; rows.value = fig.rows;
  rows.title = 'grid rows';
  rows.addEventListener('change', () => {
    fig.rows = Math.max(1, parseInt(rows.value, 10) || 1);
    markDirty(); buildCanvas();
  });
  head.appendChild(el('span', 'meta', 'rows'));
  head.appendChild(rows);

  const width = el('input', 'num');
  width.type = 'number'; width.min = '2'; width.max = '20'; width.step = '0.1';
  width.value = fig.width_in;
  width.title = 'figure width (inches)';
  width.addEventListener('change', () => {
    fig.width_in = Math.max(2, parseFloat(width.value) || 9);
    markDirty(); buildCanvas();
  });
  head.appendChild(el('span', 'meta', 'in'));
  head.appendChild(width);

  const preview = el('button', 'btn small', 'Preview');
  preview.title = 'render the whole figure exactly as it will export';
  preview.addEventListener('click', (e) => { e.stopPropagation(); previewFigure(fig); });
  head.appendChild(preview);

  const del = el('button', 'btn small', 'Delete');
  del.addEventListener('click', (e) => {
    e.stopPropagation();
    if (!confirm(`Delete figure ${fig.id} and its ${fig.panels.length} panel(s)?`)) return;
    state.comp.figures = state.comp.figures.filter((f) => f !== fig);
    state.selectedFigures.delete(fig.id);
    markDirty(); buildCanvas();
  });
  head.appendChild(del);
  card.appendChild(head);

  card.addEventListener('click', (e) => {
    if (e.target.closest('.panel-tile')) return;
    if (!e.shiftKey) state.selectedFigures.clear();
    state.selectedFigures.has(fig.id)
      ? state.selectedFigures.delete(fig.id)
      : state.selectedFigures.add(fig.id);
    buildCanvas(); renderFigureInspector();
  });

  /* --- grid ------------------------------------------------------------ */
  const gridEl = el('div', 'grid-stack');
  card.appendChild(gridEl);
  $('#canvas').appendChild(card);

  const cellH = fig.width_in * 96 * state.zoom / COLS * 0.62;
  const grid = GridStack.init({
    column: COLS,
    cellHeight: cellH,
    margin: 3,
    float: true,
    minRow: fig.rows,
    acceptWidgets: true,
    dragOut: true,
    resizable: {handles: 'se, sw, ne, nw, e, w, s, n'},
    handle: '.tile-head',
  }, gridEl);
  state.grids.set(fig.id, grid);

  const labels = autoLabels(fig);
  fig.panels.forEach((p) => grid.addWidget(makeTile(fig, p, labels[p.uid])));

  grid.on('change', (ev, items) => onGridChange(fig, items));
  grid.on('added', (ev, items) => onGridAdded(fig, items));
  grid.on('removed', (ev, items) => onGridRemoved(fig, items));
}

function makeTile(fig, p, label) {
  const spec = panelSpec(p.panel);
  const wrap = el('div', 'grid-stack-item');
  wrap.setAttribute('gs-x', p.x); wrap.setAttribute('gs-y', p.y);
  wrap.setAttribute('gs-w', p.w); wrap.setAttribute('gs-h', p.h);
  wrap.setAttribute('gs-id', p.uid);

  const content = el('div', 'grid-stack-item-content');
  const tile = el('div', 'panel-tile');
  if (state.selectedPanels.has(p.uid)) tile.classList.add('selected');

  const head = el('div', 'tile-head');
  head.appendChild(el('span', 'label', label));
  head.appendChild(el('span', 'name', spec ? spec.title : p.panel));

  const dup = el('button', null, '⧉');
  dup.title = 'duplicate';
  dup.addEventListener('click', (e) => {
    e.stopPropagation();
    const copy = JSON.parse(JSON.stringify(p));
    copy.uid = nextUid();
    copy.y = p.y + p.h;
    copy.label = '';
    fig.panels.push(copy);
    fig.rows = Math.max(fig.rows, copy.y + copy.h);
    markDirty(); buildCanvas();
  });
  head.appendChild(dup);

  const rm = el('button', null, '×');
  rm.title = 'remove';
  rm.addEventListener('click', (e) => {
    e.stopPropagation();
    fig.panels = fig.panels.filter((q) => q.uid !== p.uid);
    state.selectedPanels.delete(p.uid);
    markDirty(); buildCanvas();
  });
  head.appendChild(rm);
  tile.appendChild(head);

  const body = el('div', 'tile-body');
  tile.appendChild(body);

  tile.addEventListener('click', (e) => {
    e.stopPropagation();
    if (!e.shiftKey) state.selectedPanels.clear();
    state.selectedPanels.has(p.uid)
      ? state.selectedPanels.delete(p.uid)
      : state.selectedPanels.add(p.uid);
    document.querySelectorAll('.panel-tile').forEach((t) => t.classList.remove('selected'));
    state.selectedPanels.forEach((uid) => {
      const w = document.querySelector(`[gs-id="${uid}"] .panel-tile`);
      if (w) w.classList.add('selected');
    });
    renderPanelInspector();
    updateToolbar();
  });

  content.appendChild(tile);
  wrap.appendChild(content);
  loadPreview(fig, p, body);
  return wrap;
}

/* --------------------------------------------------------- grid → state sync */
function onGridChange(fig, items) {
  let resized = false;
  (items || []).forEach((it) => {
    const p = fig.panels.find((q) => q.uid === it.id);
    if (!p) return;
    if (p.w !== it.w || p.h !== it.h) resized = true;
    Object.assign(p, {x: it.x, y: it.y, w: it.w, h: it.h});
  });
  const needed = Math.max(...fig.panels.map((p) => p.y + p.h), 1);
  if (needed > fig.rows) fig.rows = needed;
  markDirty();
  relabel(fig);
  if (resized) {
    // aspect changed -> the cached thumbnail is for a different cell
    (items || []).forEach((it) => {
      const p = fig.panels.find((q) => q.uid === it.id);
      const body = document.querySelector(`[gs-id="${it.id}"] .tile-body`);
      if (p && body) loadPreview(fig, p, body);
    });
  }
}

function onGridAdded(fig, items) {
  (items || []).forEach((it) => {
    if (fig.panels.some((p) => p.uid === it.id)) return;
    const fromPalette = it.el && it.el.dataset.panelId;
    const source = fromPalette ? null : findPanelAnywhere(it.id);
    const p = source
      ? {...source.panel, x: it.x, y: it.y, w: it.w, h: it.h}
      : {uid: nextUid(), panel: it.el.dataset.panelId, label: '',
         x: it.x, y: it.y, w: it.w, h: it.h, params: {}};
    if (source) source.figure.panels = source.figure.panels.filter((q) => q.uid !== it.id);
    fig.panels.push(p);
    markDirty();
  });
  buildCanvas();
}

function onGridRemoved(fig, items) {
  // handled by 'added' on the receiving grid; nothing to do for a pure move
}

function findPanelAnywhere(uid) {
  for (const figure of state.comp.figures) {
    const panel = figure.panels.find((p) => p.uid === uid);
    if (panel) return {figure, panel};
  }
  return null;
}

function relabel(fig) {
  const labels = autoLabels(fig);
  Object.entries(labels).forEach(([uid, text]) => {
    const n = document.querySelector(`[gs-id="${uid}"] .tile-head .label`);
    if (n) n.textContent = text;
  });
}

/* ------------------------------------------------------------------ palette */
function buildPalette(filter = '') {
  const host = $('#palette');
  host.innerHTML = '';
  const q = filter.trim().toLowerCase();

  Object.entries(state.boot.sections).forEach(([section, ids]) => {
    const specs = ids.map(panelSpec).filter((s) => !q ||
      s.title.toLowerCase().includes(q) || s.id.toLowerCase().includes(q) ||
      (s.description || '').toLowerCase().includes(q));
    if (!specs.length) return;

    host.appendChild(el('div', 'palette-section', section));
    specs.forEach((s) => {
      const item = el('div', 'palette-item');
      item.dataset.panelId = s.id;
      item.appendChild(el('div', 't', s.title));
      if (s.description) item.appendChild(el('div', 'd', s.description));
      item.setAttribute('gs-w', s.default_w);
      item.setAttribute('gs-h', s.default_h);
      host.appendChild(item);

      // thumbnail, lazily — the palette must paint before 24 renders finish
      requestIdleCallback(() => paletteThumb(item, s));
    });
  });

  GridStack.setupDragIn('.palette-item', {appendTo: 'body', helper: 'clone'});
}

async function paletteThumb(item, spec) {
  try {
    const rec = await api('/api/preview', {
      panel: spec.id, params: {}, groups: state.comp.groups,
      w: spec.default_w, h: spec.default_h,
    });
    if (rec.status === 'error') return;
    const img = el('img', 'thumb');
    img.src = rec.url + '?k=' + rec.key;
    item.appendChild(img);
  } catch (e) { /* a palette thumbnail is optional */ }
}

const requestIdleCallback = window.requestIdleCallback ||
  ((fn) => setTimeout(fn, 60));

/* ------------------------------------------------------------------- groups */
function buildGroups() {
  const host = $('#groups');
  host.innerHTML = '';
  const cov = state.boot.coverage;
  const families = Object.keys(state.boot.families);

  state.boot.groups.forEach((g) => {
    const per = cov[g] || {};
    const missing = families.filter((f) => (per[f] || 0) === 0);
    const label = el('label');

    const cb = el('input');
    cb.type = 'checkbox';
    cb.checked = state.comp.groups.includes(g);
    cb.disabled = g === 'Control';
    cb.addEventListener('change', () => {
      state.comp.groups = state.boot.groups.filter((x) =>
        x === 'Control' || (x === g ? cb.checked : state.comp.groups.includes(x)));
      markDirty();
      buildCanvas();
      buildPalette($('#palette-search').value);
    });
    label.appendChild(cb);

    const sw = el('span', 'swatch');
    sw.style.background = GROUP_COLORS[g] || '#808080';
    label.appendChild(sw);
    label.appendChild(el('span', null, g));

    // The node/criticality/directed metrics were only computed for the focus
    // arms; without this the forests would just render empty rows for NPH etc.
    const cov_n = el('span', 'cov');
    if (missing.length) {
      cov_n.textContent = `no ${missing.join('/')}`;
      cov_n.className = 'cov partial';
      label.title = `${per._wells || 0} wells; no data for: ${missing.join(', ')}`;
    } else {
      cov_n.textContent = `${per._wells || 0} wells`;
    }
    label.appendChild(cov_n);
    host.appendChild(label);
  });
}

const GROUP_COLORS = {
  Control: '#000000', IVH_Early: '#E69F00', IVH_Late: '#D55E00',
  NPH: '#009E73', AraC: '#CC79A7', H2O2_10uM: '#56B4E9', H2O2_20uM: '#0072B2',
};

/* ---------------------------------------------------------------- inspectors */
function renderFigureInspector() {
  const host = $('#figure-inspector');
  host.innerHTML = '';
  const ids = [...state.selectedFigures];
  if (ids.length !== 1) {
    host.appendChild(el('p', 'empty',
      ids.length ? `${ids.length} figures selected` : 'select a figure'));
    return;
  }
  const fig = state.comp.figures.find((f) => f.id === ids[0]);
  if (!fig) return;

  host.appendChild(row('Suptitle', checkbox(fig.show_suptitle, (v) => {
    fig.show_suptitle = v; markDirty();
  })));
  host.appendChild(row('Panels', el('span', null, String(fig.panels.length))));
  host.appendChild(row('Size (in)', el('span', null,
    `${fig.width_in} × ${(fig.rows * state.comp.globals.row_height_in).toFixed(2)}`)));
}

function renderPanelInspector() {
  const host = $('#panel-inspector');
  host.innerHTML = '';
  const uids = [...state.selectedPanels];
  if (uids.length !== 1) {
    host.appendChild(el('p', 'empty',
      uids.length ? `${uids.length} panels selected` : 'select a panel'));
    return;
  }
  const found = findPanelAnywhere(uids[0]);
  if (!found) return;
  const {figure: fig, panel: p} = found;
  const spec = panelSpec(p.panel);
  if (!spec) { host.appendChild(el('p', 'empty', `unknown panel ${p.panel}`)); return; }

  if (spec.description) host.appendChild(el('p', 'desc', spec.description));

  host.appendChild(row('Label', textInput(p.label, (v) => {
    p.label = v; markDirty(); relabel(fig);
  }, 'auto')));

  const rerender = () => {
    markDirty();
    const body = document.querySelector(`[gs-id="${p.uid}"] .tile-body`);
    if (body) loadPreview(fig, p, body);
  };

  Object.entries(spec.params).forEach(([key, meta]) => {
    const cur = (p.params && key in p.params) ? p.params[key] : meta.default;
    const set = (v) => { p.params = {...(p.params || {}), [key]: v}; rerender(); };
    const lab = meta.label || key;

    if (meta.kind === 'bool') {
      host.appendChild(row(lab, checkbox(cur, set)));
    } else if (meta.kind === 'choice') {
      host.appendChild(row(lab, select(meta.choices, cur, set)));
    } else if (meta.kind === 'int' || meta.kind === 'float') {
      host.appendChild(row(lab, numberInput(cur, meta, set)));
    } else if (meta.kind === 'multichoice') {
      multiChoice(host, lab, meta.choices, cur, set);
    } else {
      host.appendChild(row(lab, textInput(cur, set)));
    }
  });
}

/* small widget helpers ----------------------------------------------------- */
function row(label, control) {
  const r = el('label', 'row');
  r.appendChild(el('span', null, label));
  r.appendChild(control);
  return r;
}
function textInput(value, onChange, placeholder) {
  const i = el('input');
  i.type = 'text'; i.value = value ?? ''; if (placeholder) i.placeholder = placeholder;
  i.addEventListener('change', () => onChange(i.value));
  return i;
}
function numberInput(value, meta, onChange) {
  const i = el('input');
  i.type = 'number'; i.value = value;
  i.step = meta.kind === 'int' ? '1' : 'any';
  if (meta.min !== undefined) i.min = meta.min;
  if (meta.max !== undefined) i.max = meta.max;
  i.addEventListener('change', () =>
    onChange(meta.kind === 'int' ? parseInt(i.value, 10) : parseFloat(i.value)));
  return i;
}
function checkbox(value, onChange) {
  const i = el('input');
  i.type = 'checkbox'; i.checked = !!value;
  i.addEventListener('change', () => onChange(i.checked));
  return i;
}
function select(choices, value, onChange) {
  const s = el('select');
  choices.forEach((c) => {
    const o = el('option', null, String(c));
    o.value = c;
    if (c === value) o.selected = true;
    s.appendChild(o);
  });
  s.addEventListener('change', () => onChange(s.value));
  return s;
}
function multiChoice(host, label, choices, value, onChange) {
  const head = el('div', 'row');
  head.appendChild(el('span', null, label));
  host.appendChild(head);
  const box = el('div', 'multi');
  const cur = new Set(value || []);
  choices.forEach((c) => {
    const l = el('label');
    const i = el('input');
    i.type = 'checkbox'; i.checked = cur.has(c);
    i.addEventListener('change', () => {
      i.checked ? cur.add(c) : cur.delete(c);
      onChange(choices.filter((x) => cur.has(x)));   // keep the declared order
    });
    l.appendChild(i);
    l.appendChild(el('span', null, state.boot.metric_labels[c] || c));
    box.appendChild(l);
  });
  host.appendChild(box);
  return box;
}

/* ------------------------------------------------------------------ toolbar */
function updateToolbar() {
  $('#btn-merge').disabled = state.selectedFigures.size < 2;
  $('#btn-split').disabled = state.selectedPanels.size < 1;
}

/* -------------------------------------------------------------------- modal */
function showModal(title, node) {
  $('#modal-title').textContent = title;
  const body = $('#modal-body');
  body.innerHTML = '';
  body.appendChild(node);
  $('#modal').hidden = false;
}

async function previewFigure(fig) {
  banner('rendering figure…');
  try {
    const res = await api('/api/figure_preview',
      {composition: state.comp, figure: fig.id});
    const img = el('img');
    img.src = res.url + '?t=' + Date.now();
    showModal(`${fig.id} — exactly as it will export`, img);
    banner(null);
  } catch (e) { banner(e.message, 'error'); }
}

/* ------------------------------------------------------------------- actions */
async function doSave() {
  state.comp.name = $('#spec-name').value.trim() || 'composition';
  try {
    const res = await api('/api/spec', state.comp);
    state.boot.specs = res.specs;
    fillSpecList();
    markDirty(false);
    banner(`saved → ${res.saved}`, 'ok');
  } catch (e) { banner(e.message, 'error'); }
}

async function doLoad(name) {
  if (!name) return;
  try {
    const res = await api(`/api/spec/${encodeURIComponent(name)}`);
    adopt(res.composition);
    $('#spec-name').value = state.comp.name;
    syncGlobalsToInputs();
    buildGroups(); buildCanvas(); renderFigureInspector(); renderPanelInspector();
    markDirty(false);
    const notes = [...(res.data_delta || []), ...(res.warnings || [])];
    banner(notes.length
      ? `loaded "${name}" — ${notes.join('; ')}`
      : `loaded "${name}"`, notes.length ? '' : 'ok');
  } catch (e) { banner(e.message, 'error'); }
}

async function doExport() {
  state.comp.name = $('#spec-name').value.trim() || 'composition';
  banner('exporting… (rendering every figure at full DPI)');
  try {
    const m = await api('/api/export', {composition: state.comp});
    markDirty(false);
    const box = el('div');
    box.appendChild(el('p', null, `${m.figures.length} figures → ${m.out_dir}`));
    const pre = el('pre');
    pre.textContent = [
      `combined PDF : ${m.combined_pdf}`,
      `resume spec  : ${m.spec}`,
      `manifest     : ${m.out_dir}/manifest.json`,
      '',
      ...m.figures.map((f) => `${f.id}  ${f.size_in[0]}×${f.size_in[1]} in  ` +
        `${f.panels.length} panels`),
      ...(m.warnings.length ? ['', 'warnings:',
        ...m.warnings.map((w) => `  ${w.figure}/${w.label || '?'} ${w.panel}: ` +
          `${w.warning || w.status}`)] : []),
    ].join('\n');
    box.appendChild(pre);
    showModal('Export complete', box);
    banner(null);
  } catch (e) { banner(e.message, 'error'); }
}

async function doSplit() {
  const uids = [...state.selectedPanels];
  const fig = figureOf(uids[0]);
  if (!fig) return;
  if (uids.some((u) => figureOf(u) !== fig)) {
    banner('select panels from a single figure to split', 'error');
    return;
  }
  try {
    const res = await api('/api/split',
      {composition: state.comp, figure: fig.id, uids});
    adopt(res.composition);
    state.selectedPanels.clear();
    markDirty(); buildCanvas(); renderPanelInspector();
  } catch (e) { banner(e.message, 'error'); }
}

async function doMerge() {
  try {
    const res = await api('/api/merge',
      {composition: state.comp, figures: [...state.selectedFigures]});
    adopt(res.composition);
    state.selectedFigures.clear();
    markDirty(); buildCanvas(); renderFigureInspector();
  } catch (e) { banner(e.message, 'error'); }
}

function addFigure() {
  state.comp.figures.push({
    id: nextFigureId(), title: '', width_in: 9.0, rows: 6,
    show_suptitle: true, panels: [],
  });
  markDirty(); buildCanvas();
}

/* ------------------------------------------------------------------ globals */
function syncGlobalsToInputs() {
  const g = state.comp.globals;
  $('#opt-dpi').value = g.dpi;
  $('#opt-rowh').value = g.row_height_in;
  $('#opt-gx').value = g.gutter_x_in;
  $('#opt-gy').value = g.gutter_y_in;
}

function bindGlobals() {
  const bind = (sel, key, parse) => $(sel).addEventListener('change', () => {
    state.comp.globals[key] = parse($(sel).value);
    markDirty();
    if (key === 'row_height_in') buildCanvas();
    renderFigureInspector();
  });
  bind('#opt-dpi', 'dpi', (v) => parseInt(v, 10));
  bind('#opt-rowh', 'row_height_in', parseFloat);
  bind('#opt-gx', 'gutter_x_in', parseFloat);
  bind('#opt-gy', 'gutter_y_in', parseFloat);
}

function fillSpecList() {
  const sel = $('#spec-list');
  sel.innerHTML = '<option value="">— saved specs —</option>';
  (state.boot.specs || []).forEach((n) => {
    const o = el('option', null, n);
    o.value = n;
    sel.appendChild(o);
  });
}

/* --------------------------------------------------------------------- boot */
async function init() {
  state.boot = await api('/api/bootstrap');

  const saved = localStorage.getItem(STORE_KEY);
  adopt(saved ? JSON.parse(saved) : state.boot.default);

  $('#spec-name').value = state.comp.name || 'report';
  $('#data-badge').textContent =
    `${state.boot.data.tidy_rows} rows · ${state.boot.data.tidy_sha256.slice(0, 8)}`;
  $('#data-badge').title = JSON.stringify(state.boot.data, null, 2);

  syncGlobalsToInputs();
  bindGlobals();
  fillSpecList();
  buildGroups();
  buildPalette();
  buildCanvas();
  renderFigureInspector();
  renderPanelInspector();
  if (saved) banner('restored your unsaved session from this browser');

  $('#palette-search').addEventListener('input', (e) => buildPalette(e.target.value));
  $('#btn-save').addEventListener('click', doSave);
  $('#btn-export').addEventListener('click', doExport);
  $('#btn-split').addEventListener('click', doSplit);
  $('#btn-merge').addEventListener('click', doMerge);
  $('#btn-add-figure').addEventListener('click', addFigure);
  $('#spec-list').addEventListener('change', (e) => doLoad(e.target.value));
  $('#btn-reset').addEventListener('click', () => {
    if (!confirm('Discard this composition and start from the default report?')) return;
    localStorage.removeItem(STORE_KEY);
    location.reload();
  });
  $('#modal-close').addEventListener('click', () => { $('#modal').hidden = true; });
  $('#modal').addEventListener('click', (e) => {
    if (e.target.id === 'modal') $('#modal').hidden = true;
  });
  $('#zoom').addEventListener('input', (e) => {
    state.zoom = parseInt(e.target.value, 10) / 100;
    buildCanvas();
  });
  window.addEventListener('beforeunload', (e) => {
    if (state.dirty) { e.preventDefault(); e.returnValue = ''; }
  });
}

init().catch((e) => banner('failed to start: ' + e.message, 'error'));
