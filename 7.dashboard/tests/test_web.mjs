import test from 'node:test';
import assert from 'node:assert/strict';
import { createRequire } from 'node:module';
import { existsSync, readFileSync } from 'node:fs';
const require = createRequire(import.meta.url);
const appPath = new URL('../web/app.js', import.meta.url);
const app = existsSync(appPath) ? require(appPath.pathname) : {};
const points = [
  { id: 'a', group: 'Brain', line: 'DAOY', cluster: 0, well: 'C03', x: 0, y: 0 },
  { id: 'b', group: 'Brain', line: 'ONS', cluster: 1, well: 'C04', x: 10, y: 20 },
  { id: 'c', group: 'Lung', line: 'A549', cluster: 0, well: 'D03', x: 1000, y: -500 },
];
test('bounds stay within each independently embedded group and handle empty or constant axes', () => {
  assert.equal(typeof app.groupBounds, 'function');
  assert.deepEqual(app.groupBounds(points, 'Brain'), { minX: 0, maxX: 10, minY: 0, maxY: 20 });
  assert.deepEqual(app.groupBounds(points, 'Lung'), { minX: 999.5, maxX: 1000.5, minY: -500.5, maxY: -499.5 });
  assert.equal(app.groupBounds(points, 'missing'), null);
});
test('projection preserves geometry and inverts zoomed panned coordinates', () => {
  assert.equal(typeof app.createProjection, 'function');
  const bounds = app.groupBounds(points, 'Brain');
  const project = app.createProjection(bounds, 800, 600, { zoom: 2.5, panX: 37, panY: -19 });
  for (const point of points.slice(0, 2)) {
    const screen = project.forward(point);
    const original = project.inverse(screen);
    assert.ok(Math.abs(original.x - point.x) < 1e-10);
    assert.ok(Math.abs(original.y - point.y) < 1e-10);
  }
  const center = app.createProjection(bounds, 800, 600).forward({ x: 5, y: 10 });
  assert.deepEqual(center, { x: 400, y: 300 });
});
test('nearest point uses screen-space radius and returns no hit outside it', () => {
  assert.equal(typeof app.nearestPoint, 'function');
  const projection = app.createProjection(app.groupBounds(points, 'Brain'), 800, 600);
  const target = projection.forward(points[1]);
  assert.equal(app.nearestPoint(points.slice(0, 2), { x: target.x + 3, y: target.y }, projection, 9), points[1]);
  assert.equal(app.nearestPoint(points.slice(0, 2), { x: 0, y: 0 }, projection, 9), null);
  assert.equal(app.nearestPoint([], target, projection, 9), null);
});
test('selection, toggles and keyboard traversal never retain a hidden or other-group cell', () => {
  assert.equal(typeof app.createState, 'function');
  assert.equal(typeof app.reduceState, 'function');
  let state = app.createState(points);
  assert.equal(state.group, 'Brain');
  state = app.reduceState(state, { type: 'pin', id: 'a' }, points);
  assert.equal(state.selectedId, 'a');
  state = app.reduceState(state, { type: 'toggle', value: 'DAOY' }, points);
  assert.equal(state.selectedId, null);
  assert.ok(state.hidden.has('DAOY'));
  state = app.reduceState(state, { type: 'step', direction: 1 }, points);
  assert.equal(state.selectedId, 'b');
  state = app.reduceState(state, { type: 'color', value: 'well' }, points);
  assert.equal(state.hidden.size, 0);
  assert.equal(state.selectedId, 'b');
  state = app.reduceState(state, { type: 'step', direction: 1 }, points);
  assert.equal(state.selectedId, 'a');
  state = app.reduceState(state, { type: 'step', direction: -1 }, points);
  assert.equal(state.selectedId, 'b');
  state = app.reduceState(state, { type: 'group', value: 'Lung' }, points);
  assert.equal(state.selectedId, null);
  state = app.reduceState(state, { type: 'pin', id: 'a' }, points);
  assert.equal(state.selectedId, null);
  state = app.reduceState(state, { type: 'pin', id: 'c' }, points);
  state = app.reduceState(state, { type: 'unpin' }, points);
  assert.equal(state.selectedId, null);
  state = app.reduceState(state, { type: 'toggle', value: 'D03' }, points);
  state = app.reduceState(state, { type: 'step', direction: 1 }, points);
  assert.equal(state.selectedId, null);
  assert.equal(app.createState([]).group, '');
});
test('image slots clear stale pixels immediately and ignore late completions', () => {
  assert.equal(typeof app.createImageSlot, 'function');
  const requested = [];
  const document = { createElement: tag => {
    const element = { tagName: tag, textContent: '', alt: '' };
    Object.defineProperty(element, 'src', { set(value) { requested.push(element); element.url = value; } });
    return element;
  }};
  const container = { children: [], replaceChildren(...children) { this.children = children; } };
  const update = app.createImageSlot(document, container);
  update('assets/crops/a.jpg', 'DNA for a');
  const first = requested[0];
  assert.match(container.children[0].textContent, /Loading/);
  first.onload();
  assert.equal(container.children[0], first);
  update('assets/crops/b.jpg', 'DNA for b');
  assert.notEqual(container.children[0], first);
  first.onload();
  assert.notEqual(container.children[0], first);
  requested[1].onerror();
  assert.match(container.children[0].textContent, /Image could not be loaded/);
  update(null, 'DNA for c');
  assert.equal(requested.length, 2);
  assert.match(container.children[0].textContent, /unavailable/i);
  update('https://example.invalid/bad.jpg', 'external');
  update('assets/crops/../../bad.jpg', 'traversal');
  assert.equal(requested.length, 2);
});
test('static template inlines exactly one of each build placeholder without external dependencies', () => {
  const templatePath = new URL('../web/template.html', import.meta.url);
  assert.ok(existsSync(templatePath), 'template.html must exist');
  const template = readFileSync(templatePath, 'utf8');
  for (const token of ['__REPORT_CSS__', '__REPORT_DATA__', '__REPORT_JS__']) {
    assert.equal(template.split(token).length - 1, 1, token);
  }
  assert.match(template, /<script type="application\/json" id="report-data">__REPORT_DATA__<\/script>/);
  assert.match(template, /href="https:\/\/github.com\/WayScience\/pccma"/);
  assert.doesNotMatch(template, /<script[^>]+(?:src=|type="module")/);
  assert.doesNotMatch(template, /<link[^>]+stylesheet/);
  assert.ok(existsSync(new URL('../web/style.css', import.meta.url)));
});
// Minimal DOM double: no HTML parsing or browser packages in the unit suite.
function fixture(report) {
  const imageRequests = [];
  class Element {
    constructor(tag = 'div') { this.tagName = tag.toUpperCase(); this.children = []; this.textContent = ''; this.hidden = false; this.style = {}; this.dataset = {}; this.attrs = {}; this.handlers = {}; this.value = ''; }
    set innerHTML(_) { throw new Error('Unsafe innerHTML assignment'); }
    set src(value) { this.url = value; imageRequests.push(this); }
    setAttribute(name, value) { this.attrs[name] = String(value); }
    getAttribute(name) { return this.attrs[name]; }
    append(...children) { this.children.push(...children); }
    replaceChildren(...children) { this.children = children; }
    addEventListener(type, handler) { (this.handlers[type] ||= []).push(handler); }
    emit(type, event = {}) { for (const handler of this.handlers[type] || []) handler({ target: this, preventDefault() {}, ...event }); }
    getBoundingClientRect() { return { width: 800, height: 600, left: 0, top: 0 }; }
    getContext(type, options) { this.contextType = type; this.contextOptions = options; return new Proxy({}, { get: (target, key) => target[key] || (() => {}), set: (target, key, value) => { target[key] = value; return true; } }); }
    setPointerCapture() {}
    releasePointerCapture() {}
    focus() {}
  }
  const html = readFileSync(new URL('../web/template.html', import.meta.url), 'utf8');
  const elements = new Map([...html.matchAll(/id="([^"]+)"/g)].map(([, id]) => [id, new Element(id === 'plot' ? 'canvas' : 'div')]));
  elements.get('report-data').textContent = typeof report === 'string' ? report : JSON.stringify(report);
  elements.get('channel-select').value = 'all';
  elements.get('image-warning').hidden = true;
  elements.get('error-banner').hidden = true;
  const document = { getElementById: id => elements.get(id), createElement: tag => new Element(tag) };
  const window = { devicePixelRatio: 2, addEventListener() {} };
  return { elements, document, window, imageRequests };
}
const reportFixture = {
  schema_version: 1, plate: 'BR00150695', run_id: 'test-run',
  summary: { warehouse_cells: 3, umap_cells: 3, excluded_umap_rows: 0, cell_lines: 3, cancer_groups: 2, crop_size: 224 },
  sources: { umap: { name: 'umap.parquet', sha256: '123' }, crop_index: { name: 'crops_index.parquet', sha256: '456' }, qc: 'Inherited QC.' },
  images: { status: 'unavailable', message: 'Warehouse unavailable.', channels: ['composite', 'DNA', 'ER', 'RNA', 'AGP', 'Mito'] },
  points: points.map(point => ({ ...point, tissue: 'test tissue', site: 1, cell: 2, nucleus: 3, time: 24, cx: 1.25, cy: 2.5, images: {} })),
};
test('mounted dashboard initializes the first group, genuine pointer hover/pin and unavailable-image state', () => {
  assert.equal(typeof app.mountDashboard, 'function');
  const f = fixture(reportFixture);
  const api = app.mountDashboard(f.document, f.window);
  assert.deepEqual(api.visiblePoints.map(point => point.id), ['a', 'b']);
  assert.equal(f.elements.get('image-warning').hidden, false);
  assert.match(f.elements.get('image-warning').textContent, /unavailable/i);
  assert.match(f.elements.get('plot-count').textContent, /2/);
  const target = api.projection.forward(points[0]);
  const canvas = f.elements.get('plot');
  canvas.emit('pointermove', { clientX: target.x, clientY: target.y, pointerType: 'mouse' });
  assert.equal(api.hoveredPoint.id, 'a');
  assert.equal(f.elements.get('cell-id').textContent, 'a');
  canvas.emit('pointerdown', { clientX: target.x, clientY: target.y, pointerId: 1, button: 0 });
  canvas.emit('pointerup', { clientX: target.x, clientY: target.y, pointerId: 1 });
  assert.equal(api.selectedPoint.id, 'a');
  f.elements.get('legend').children[0].emit('click');
  assert.equal(api.selectedPoint, null);
  assert.deepEqual(api.visiblePoints.map(point => point.id), ['b']);
  assert.equal(f.elements.get('legend').children[0].getAttribute('aria-pressed'), 'false');
  f.elements.get('legend').children[1].emit('click');
  assert.equal(api.visiblePoints.length, 0);
  assert.equal(f.elements.get('empty-state').hidden, false);
  f.elements.get('group-select').value = 'Lung';
  f.elements.get('group-select').emit('change');
  assert.deepEqual(api.visiblePoints.map(point => point.id), ['c']);
  f.elements.get('next-cell').emit('click');
  assert.equal(api.selectedPoint.id, 'c');
  assert.equal(f.imageRequests.length, 0, 'unavailable reports must never request images');
  f.elements.get('unpin-cell').emit('click');
  assert.equal(api.selectedPoint, null);
});
test('mounted dashboard has readable malformed and empty payload states', () => {
  assert.equal(typeof app.mountDashboard, 'function');
  const bad = fixture('{broken');
  app.mountDashboard(bad.document, bad.window);
  assert.equal(bad.elements.get('error-banner').hidden, false);
  assert.match(bad.elements.get('error-banner').textContent, /could not|invalid/i);
  const empty = fixture({ ...reportFixture, points: [] });
  const api = app.mountDashboard(empty.document, empty.window);
  assert.equal(api.visiblePoints.length, 0);
  assert.equal(empty.elements.get('empty-state').hidden, false);
  assert.match(empty.elements.get('empty-state').textContent, /No cells/i);
});
test('bundled report requests only real per-cell paths, renders composite plus all or selected channels, and treats payload text literally', () => {
  const malicious = '<img src=x onerror=alert(1)>';
  const bundled = structuredClone(reportFixture);
  bundled.images.status = 'bundled';
  bundled.points[0].line = malicious;
  bundled.points[0].images = Object.fromEntries(['composite', 'DNA', 'ER', 'RNA', 'AGP', 'Mito'].map(channel => [channel, 'assets/crops/a.' + channel + '.jpg']));
  const f = fixture(bundled);
  const api = app.mountDashboard(f.document, f.window);
  assert.equal(f.imageRequests.length, 0, 'do not preload unselected cells');
  api.hoverPoint('a');
  assert.equal(f.elements.get('inspector-title').textContent, malicious);
  assert.equal(f.elements.get('crop-area').children.length, 6);
  assert.equal(f.imageRequests.length, 6);
  assert.ok(f.imageRequests.every(image => image.url.startsWith('assets/crops/a.')));
  f.elements.get('channel-select').value = 'DNA';
  f.elements.get('channel-select').emit('change');
  assert.equal(f.elements.get('crop-area').children.length, 2);
  assert.equal(f.imageRequests.length, 8);
  assert.equal(f.imageRequests[7].url, 'assets/crops/a.DNA.jpg');
  api.hoverPoint('b');
  assert.equal(f.imageRequests.length, 8, 'missing channel paths must not create requests');
  assert.match(f.elements.get('crop-area').children[0].children[1].children[0].textContent, /unavailable/i);
});
test('wheel zoom anchors the cursor, drag pans without pinning, reset restores geometry, and keyboard selects', () => {
  const f = fixture(reportFixture);
  const api = app.mountDashboard(f.document, f.window);
  const canvas = f.elements.get('plot');
  const original = api.projection.forward(points[0]);
  canvas.emit('wheel', { clientX: original.x, clientY: original.y, deltaY: -100 });
  const anchored = api.projection.forward(points[0]);
  assert.ok(Math.abs(anchored.x - original.x) < 1e-9 && Math.abs(anchored.y - original.y) < 1e-9);
  assert.notEqual(f.elements.get('zoom-label').textContent, '100%');
  canvas.emit('pointerdown', { clientX: 200, clientY: 200, pointerId: 1, button: 0 });
  canvas.emit('pointermove', { clientX: 230, clientY: 250, pointerId: 1 });
  canvas.emit('pointerup', { clientX: 230, clientY: 250, pointerId: 1 });
  const panned = api.projection.forward(points[0]);
  assert.ok(Math.abs(panned.x - anchored.x - 30) < 1e-9);
  assert.ok(Math.abs(panned.y - anchored.y - 50) < 1e-9);
  assert.equal(api.selectedPoint, null);
  f.elements.get('reset-view').emit('click');
  assert.deepEqual(api.projection.forward(points[0]), original);
  canvas.emit('keydown', { key: 'ArrowRight' });
  assert.equal(api.selectedPoint.id, 'a');
  canvas.emit('keydown', { key: 'ArrowRight' });
  assert.equal(api.selectedPoint.id, 'b');
  canvas.emit('keydown', { key: 'Escape' });
  assert.equal(api.selectedPoint, null);
});
test('pointerdown does not scroll the canvas away from the pointer before pinning', () => {
  const f = fixture(reportFixture);
  const api = app.mountDashboard(f.document, f.window);
  const canvas = f.elements.get('plot');
  let focusedWith;
  canvas.focus = options => { focusedWith = options; };
  const target = api.projection.forward(points[0]);
  canvas.emit('pointerdown', { clientX: target.x, clientY: target.y, pointerId: 1, button: 0 });
  assert.deepEqual(focusedWith, { preventScroll: true });
  canvas.emit('pointerup', { clientX: target.x, clientY: target.y, pointerId: 1 });
  assert.equal(api.selectedPoint.id, 'a');
});
test('projection uses the canvas content box, not its bordered wrapper, for exact browser hit positions', () => {
  const f = fixture(reportFixture);
  f.elements.get('plot-wrap').getBoundingClientRect = () => ({ width: 804, height: 604, left: 0, top: 0 });
  const api = app.mountDashboard(f.document, f.window);
  const expected = app.createProjection(app.groupBounds(points, 'Brain'), 800, 600);
  assert.deepEqual(api.projection.forward(points[1]), expected.forward(points[1]));
});
test('CPU-backed canvas avoids the observed Chromium blank-compositor regression after resize', () => {
  const f = fixture(reportFixture);
  app.mountDashboard(f.document, f.window);
  assert.equal(f.elements.get('plot').contextType, '2d');
  assert.deepEqual(f.elements.get('plot').contextOptions, { willReadFrequently: true });
});
test('filtering selects exactly one group and excludes hidden category values', () => {
  assert.equal(typeof app.filterPoints, 'function');
  assert.deepEqual(app.filterPoints(points, 'Brain', 'line', new Set()), points.slice(0, 2));
  assert.deepEqual(app.filterPoints(points, 'Brain', 'line', new Set(['DAOY'])), [points[1]]);
  assert.deepEqual(app.filterPoints(points, 'Brain', 'cluster', new Set(['0'])), [points[1]]);
  assert.deepEqual(app.filterPoints(points, 'missing', 'line', new Set()), []);
});
