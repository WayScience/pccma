/* PCCMA static dashboard. No network or library dependencies. */
(function () {
  'use strict';
  function category(point, colorBy) {
    return String(point[colorBy] ?? 'Unknown');
  }
  function filterPoints(points, group, colorBy, hidden = new Set()) {
    return points.filter(point => point.group === group && !hidden.has(category(point, colorBy)));
  }
  function groupBounds(points, group) {
    const selected = points.filter(point => point.group === group && Number.isFinite(point.x) && Number.isFinite(point.y));
    if (!selected.length) return null;
    let minX = Infinity, maxX = -Infinity, minY = Infinity, maxY = -Infinity;
    for (const point of selected) {
      minX = Math.min(minX, point.x); maxX = Math.max(maxX, point.x);
      minY = Math.min(minY, point.y); maxY = Math.max(maxY, point.y);
    }
    if (minX === maxX) { minX -= 0.5; maxX += 0.5; }
    if (minY === maxY) { minY -= 0.5; maxY += 0.5; }
    return { minX, maxX, minY, maxY };
  }
  function createProjection(bounds, width, height, view = {}) {
    const box = bounds || { minX: -1, maxX: 1, minY: -1, maxY: 1 };
    const scale = Math.max(1, Math.min(width - 100, height - 100)) / Math.max(box.maxX - box.minX, box.maxY - box.minY) * (view.zoom ?? 1);
    const midX = (box.minX + box.maxX) / 2, midY = (box.minY + box.maxY) / 2;
    const originX = width / 2 + (view.panX ?? 0), originY = height / 2 + (view.panY ?? 0);
    return {
      forward: point => ({ x: (point.x - midX) * scale + originX, y: originY - (point.y - midY) * scale }),
      inverse: point => ({ x: (point.x - originX) / scale + midX, y: (originY - point.y) / scale + midY }),
    };
  }
  function nearestPoint(points, position, projection, radius = 10) {
    let best = null, distance = radius * radius;
    for (const point of points) {
      const screen = projection.forward(point);
      const squared = (screen.x - position.x) ** 2 + (screen.y - position.y) ** 2;
      if (squared <= distance) { best = point; distance = squared; }
    }
    return best;
  }
  function createState(points) {
    return { group: points[0]?.group ?? '', colorBy: 'line', hidden: new Set(), selectedId: null, hoveredId: null };
  }
  function reduceState(state, action, points) {
    const next = { ...state, hidden: new Set(state.hidden) };
    if (action.type === 'group') {
      next.group = action.value; next.hidden.clear(); next.selectedId = null; next.hoveredId = null;
    }
    if (action.type === 'color') { next.colorBy = action.value; next.hidden.clear(); }
    if (action.type === 'toggle') {
      const value = String(action.value);
      if (next.hidden.has(value)) next.hidden.delete(value); else next.hidden.add(value);
    }
    const visible = filterPoints(points, next.group, next.colorBy, next.hidden);
    const contains = id => visible.some(point => point.id === id);
    if (!contains(next.selectedId)) next.selectedId = null;
    if (!contains(next.hoveredId)) next.hoveredId = null;
    if (action.type === 'pin' && contains(action.id)) next.selectedId = action.id;
    if (action.type === 'hover') next.hoveredId = contains(action.id) ? action.id : null;
    if (action.type === 'unpin') { next.selectedId = null; next.hoveredId = null; }
    if (action.type === 'step' && visible.length) {
      const current = visible.findIndex(point => point.id === (next.selectedId ?? next.hoveredId));
      const index = current < 0 ? (action.direction < 0 ? visible.length - 1 : 0) : (current + action.direction + visible.length) % visible.length;
      next.selectedId = visible[index].id;
    }
    return next;
  }
  function createImageSlot(document, container) {
    let revision = 0;
    return function update(path, description) {
      const current = ++revision;
      const status = document.createElement('p');
      status.className = 'image-status';
      const safe = typeof path === 'string' && /^assets\/crops\/[A-Za-z0-9_./-]+\.jpg$/.test(path) && !path.split('/').includes('..');
      status.textContent = safe ? 'Loading image…' : 'Image unavailable for this cell.';
      container.replaceChildren(status);
      if (!safe) return;
      const image = document.createElement('img');
      image.alt = description;
      image.decoding = 'async';
      image.onload = () => { if (revision === current) container.replaceChildren(image); };
      image.onerror = () => {
        if (revision === current) {
          status.textContent = 'Image could not be loaded. ' + description;
          status.className = 'image-status image-error';
          container.replaceChildren(status);
        }
      };
      image.src = path;
    };
  }
  function mountDashboard(document, window) {
    const $ = id => document.getElementById(id);
    const element = (tag, text, className) => {
      const node = document.createElement(tag);
      if (text !== undefined) node.textContent = String(text);
      if (className) node.className = className;
      return node;
    };
    const fail = error => {
      $('error-banner').textContent = 'The dashboard could not load: ' + error.message;
      $('error-banner').hidden = false;
      $('empty-state').textContent = 'Invalid report data. Rebuild this report from the validated source files.';
      $('empty-state').hidden = false;
      return null;
    };
    let data;
    try {
      data = JSON.parse($('report-data').textContent);
      if (!data || data.schema_version !== 1 || !Array.isArray(data.points)) throw new Error('Invalid report schema (expected version 1).');
      const ids = new Set();
      for (const point of data.points) {
        if (typeof point.id !== 'string' || typeof point.group !== 'string' || !Number.isFinite(point.x) || !Number.isFinite(point.y) || ids.has(point.id)) throw new Error('Invalid or duplicate cell coordinates.');
        ids.add(point.id);
      }
    } catch (error) { return fail(error); }
    const points = data.points;
    const byId = new Map(points.map(point => [point.id, point]));
    const groups = [...new Set(points.map(point => point.group))];
    const number = value => Number.isFinite(value) ? value.toLocaleString('en-US') : 'Not recorded';
    const channels = ['DNA', 'ER', 'RNA', 'AGP', 'Mito'];
    const bundled = data.images?.status === 'bundled';
    const palette = ['#73b9d7', '#eba16d', '#9dc88c', '#c6a0d8', '#e3859d', '#6ec8ba', '#d5c575', '#95a1e1', '#d7ad91', '#88b7a2', '#c2b6e8', '#d7a8bf', '#8cc8d1', '#b6c079'];
    let state = createState(points), visible = [], projection;
    let view = { zoom: 1, panX: 0, panY: 0 };
    let width = 800, height = 600, colors = new Map(), drag = null, lastInspectorKey = null;
    const canvas = $('plot');
    // Software-backed canvas also avoids a verified Chromium compositor blank
    // after viewport changes; the small point clouds do not need GPU backing.
    const ctx = canvas.getContext('2d', { willReadFrequently: true });
    if (!ctx) return fail(new Error('Canvas rendering is not supported by this browser.'));
    $('plate-label').textContent = 'Plate ' + (data.plate ?? 'not recorded') + ' · exact crop-key joins · published post-QC cells';
    const summaries = [[points.length, 'plotted cells'], [new Set(points.map(point => point.line)).size, 'cell lines'], [groups.length, 'cancer groups'], [data.summary?.warehouse_cells, 'warehouse cells']];
    for (const [count, label] of summaries) {
      const item = element('div', undefined, 'summary-item');
      item.append(element('span', number(count), 'summary-value'), element('span', label, 'summary-label'));
      $('summary').append(item);
    }
    if (!bundled) {
      $('image-warning').hidden = false;
      $('image-warning').textContent = 'CROP IMAGES UNAVAILABLE — diagnostic coordinates-only preview. No real cell crops are bundled or loaded. ' + (data.images?.message ?? 'The extracted-cell image warehouse was not supplied.');
    }
    const groupSelect = $('group-select');
    for (const group of groups) {
      const count = points.filter(point => point.group === group).length;
      const option = element('option', group + ' · ' + number(count) + ' cells');
      option.value = group; groupSelect.append(option);
    }
    groupSelect.value = state.group;
    groupSelect.disabled = groups.length === 0;
    const metadataEntry = (container, key, value) => container.append(element('dt', key), element('dd', value ?? 'Not recorded'));
    const provenance = $('provenance-metadata');
    for (const [key, value] of [
      ['Plate', data.plate], ['Extraction run', data.run_id], ['Warehouse cells', number(data.summary?.warehouse_cells)],
      ['Displayed cells', number(points.length)], ['Excluded UMAP rows', number(data.summary?.excluded_umap_rows)],
      ['UMAP source', data.sources?.umap?.name], ['UMAP SHA-256', data.sources?.umap?.sha256],
      ['Crop index', data.sources?.crop_index?.name], ['Crop-index SHA-256', data.sources?.crop_index?.sha256],
      ['QC membership', data.sources?.qc], ['Images', data.images?.status], ['Image details', data.images?.message],
    ]) metadataEntry(provenance, key, value);

    function renderLegend() {
      const allCategories = [...new Set(points.map(point => category(point, state.colorBy)))].sort((a, b) => a.localeCompare(b, undefined, { numeric: true }));
      colors = new Map(allCategories.map((value, index) => [value, palette[index % palette.length]]));
      const counts = new Map();
      for (const point of points) if (point.group === state.group) {
        const value = category(point, state.colorBy); counts.set(value, (counts.get(value) ?? 0) + 1);
      }
      $('legend').replaceChildren();
      $('legend-title').textContent = { line: 'Cell lines', cluster: 'Clusters', well: 'Wells' }[state.colorBy];
      for (const value of allCategories.filter(value => counts.has(value))) {
        const button = element('button');
        button.type = 'button';
        button.setAttribute('aria-pressed', String(!state.hidden.has(value)));
        button.setAttribute('aria-label', value + ', ' + number(counts.get(value)) + ' cells; toggle visibility');
        const swatch = element('span', undefined, 'swatch'); swatch.style.backgroundColor = colors.get(value); swatch.setAttribute('aria-hidden', 'true');
        button.append(swatch, element('span', value), element('span', number(counts.get(value)), 'legend-count'));
        button.addEventListener('click', () => dispatch({ type: 'toggle', value }));
        $('legend').append(button);
      }
    }
    function draw() {
      projection = createProjection(groupBounds(points, state.group), width, height, view);
      ctx.clearRect(0, 0, width, height);
      ctx.strokeStyle = '#24303b'; ctx.lineWidth = 0.7; ctx.fillStyle = '#748498'; ctx.font = '10px system-ui';
      for (let x = 50; x < width - 20; x += 100) {
        ctx.beginPath(); ctx.moveTo(x, 25); ctx.lineTo(x, height - 38); ctx.stroke();
        ctx.fillText(projection.inverse({ x, y: height / 2 }).x.toFixed(1), x - 10, height - 27);
      }
      for (let y = 50; y < height - 45; y += 100) {
        ctx.beginPath(); ctx.moveTo(37, y); ctx.lineTo(width - 20, y); ctx.stroke();
        ctx.fillText(projection.inverse({ x: width / 2, y }).y.toFixed(1), 7, y - 4);
      }
      ctx.save(); ctx.beginPath(); ctx.rect(36, 20, Math.max(1, width - 55), Math.max(1, height - 60)); ctx.clip();
      ctx.globalAlpha = 0.83;
      for (const point of visible) {
        const screen = projection.forward(point);
        if (screen.x < 30 || screen.x > width || screen.y < 15 || screen.y > height - 35) continue;
        ctx.fillStyle = colors.get(category(point, state.colorBy)) || palette[0];
        ctx.beginPath(); ctx.arc(screen.x, screen.y, 2.6, 0, Math.PI * 2); ctx.fill();
      }
      ctx.globalAlpha = 1;
      for (const [id, color, radius] of [[state.hoveredId, '#86dbce', 6], [state.selectedId, '#ffffff', 8]]) {
        const point = byId.get(id);
        if (!point) continue;
        const screen = projection.forward(point);
        ctx.strokeStyle = color; ctx.lineWidth = 1.8;
        ctx.beginPath(); ctx.arc(screen.x, screen.y, radius, 0, Math.PI * 2); ctx.stroke();
      }
      ctx.restore();
      $('zoom-label').textContent = Math.round(view.zoom * 100) + '%';
      $('zoom-in').disabled = view.zoom >= 20;
      $('zoom-out').disabled = view.zoom <= 0.5;
    }
    function renderInspector(force = false) {
      const point = byId.get(state.selectedId ?? state.hoveredId);
      const key = (point?.id ?? '') + ':' + $('channel-select').value + ':' + Boolean(state.selectedId);
      if (!force && key === lastInspectorKey) return;
      lastInspectorKey = key;
      $('pin-status').textContent = state.selectedId ? 'Pinned' : 'Unpinned';
      $('pin-status').className = state.selectedId ? 'badge pinned' : 'badge';
      $('unpin-cell').disabled = !state.selectedId;
      $('previous-cell').disabled = !visible.length; $('next-cell').disabled = !visible.length;
      $('channel-select').disabled = !bundled || !point;
      $('inspector-title').textContent = point ? point.line ?? 'Cell preview' : 'Point preview';
      $('inspector-hint').textContent = state.selectedId ? 'Pinned cell. Unpin to resume hover previews; use Previous / Next to browse visible cells.' : 'Hover a point to inspect its extracted cell crop. Click or tap to keep it here.';
      $('cell-id').textContent = point?.id ?? 'No cell selected';
      $('cell-metadata').replaceChildren();
      $('crop-area').replaceChildren();
      if (!point) {
        $('crop-area').append(element('p', visible.length ? 'Hover a point or use Next to inspect a cell.' : 'No cells are visible. Show a legend category or choose another group.', 'crop-placeholder'));
        return;
      }
      for (const [label, value] of [
        ['Cancer group', point.group], ['Cell line', point.line], ['Tissue', point.tissue], ['Well / site', point.well + ' / ' + point.site],
        ['Cell / nucleus', point.cell + ' / ' + point.nucleus], ['Timepoint', point.time == null ? null : point.time + ' h'],
        ['Cluster', point.cluster], ['UMAP (1, 2)', point.x.toFixed(3) + ', ' + point.y.toFixed(3)],
        ['Nucleus center (x, y)', point.cx == null || point.cy == null ? null : String(point.cx) + ', ' + String(point.cy)],
      ]) metadataEntry($('cell-metadata'), label, value);
      if (!bundled) {
        $('crop-area').append(element('p', 'Crops unavailable. This diagnostic report contains cell coordinates and metadata only; no real images have been loaded.', 'crop-placeholder'));
        return;
      }
      const chosen = $('channel-select').value;
      for (const channel of ['composite', ...(chosen === 'all' ? channels : channels.filter(channel => channel === chosen))]) {
        const figure = element('figure', undefined, 'crop-figure' + (channel === 'composite' ? ' composite' : ''));
        const caption = element('figcaption', channel === 'composite' ? 'Composite · five-channel overlay' : channel);
        const frame = element('div', undefined, 'image-frame');
        figure.append(caption, frame); $('crop-area').append(figure);
        createImageSlot(document, frame)(point.images?.[channel], channel + ' crop · ' + point.id);
      }
    }
    function refreshFilters() {
      visible = filterPoints(points, state.group, state.colorBy, state.hidden);
      $('group-title').textContent = state.group || 'UMAP';
      const total = points.filter(point => point.group === state.group).length;
      $('plot-count').textContent = number(visible.length) + ' / ' + number(total) + ' group cells · ' + number(points.length) + ' on plate';
      $('empty-state').hidden = visible.length > 0;
      $('empty-state').textContent = !points.length ? 'No cells are present in this report.' : 'No visible cells in this group. Click a legend category to show cells, or choose another group.';
      renderLegend(); renderInspector(true); draw();
    }
    function dispatch(action) {
      state = reduceState(state, action, points);
      if (['group', 'color', 'toggle'].includes(action.type)) refreshFilters();
      else { renderInspector(); draw(); }
    }
    function resize() {
      const rect = canvas.getBoundingClientRect();
      width = Math.max(100, rect.width); height = Math.max(100, rect.height);
      const ratio = window.devicePixelRatio || 1;
      canvas.width = Math.round(width * ratio); canvas.height = Math.round(height * ratio);
      ctx.setTransform(ratio, 0, 0, ratio, 0, 0); draw();
    }
    function position(event) {
      const rect = canvas.getBoundingClientRect();
      return { x: (event.clientX - rect.left) * width / rect.width, y: (event.clientY - rect.top) * height / rect.height };
    }
    function zoom(factor, anchor = { x: width / 2, y: height / 2 }) {
      const nextZoom = Math.max(0.5, Math.min(20, view.zoom * factor));
      const ratio = nextZoom / view.zoom;
      view.panX = anchor.x - width / 2 - (anchor.x - width / 2 - view.panX) * ratio;
      view.panY = anchor.y - height / 2 - (anchor.y - height / 2 - view.panY) * ratio;
      view.zoom = nextZoom; draw();
    }
    function reset() { view = { zoom: 1, panX: 0, panY: 0 }; draw(); }
    groupSelect.addEventListener('change', () => { reset(); dispatch({ type: 'group', value: groupSelect.value }); });
    $('color-select').addEventListener('change', () => dispatch({ type: 'color', value: $('color-select').value }));
    $('channel-select').addEventListener('change', () => renderInspector(true));
    $('previous-cell').addEventListener('click', () => dispatch({ type: 'step', direction: -1 }));
    $('next-cell').addEventListener('click', () => dispatch({ type: 'step', direction: 1 }));
    $('unpin-cell').addEventListener('click', () => dispatch({ type: 'unpin' }));
    $('zoom-in').addEventListener('click', () => zoom(1.3));
    $('zoom-out').addEventListener('click', () => zoom(1 / 1.3));
    $('reset-view').addEventListener('click', reset);
    canvas.addEventListener('wheel', event => { event.preventDefault(); zoom(Math.exp(-Math.max(-100, Math.min(100, event.deltaY)) * 0.002), position(event)); }, { passive: false });
    canvas.addEventListener('pointerdown', event => {
      if (event.button !== undefined && event.button !== 0) return;
      const at = position(event);
      drag = { id: event.pointerId, start: at, panX: view.panX, panY: view.panY, moved: false };
      canvas.setPointerCapture(event.pointerId); canvas.focus({ preventScroll: true });
    });
    canvas.addEventListener('pointermove', event => {
      const at = position(event);
      if (drag) {
        if (Math.hypot(at.x - drag.start.x, at.y - drag.start.y) > 4) drag.moved = true;
        if (drag.moved) { view.panX = drag.panX + at.x - drag.start.x; view.panY = drag.panY + at.y - drag.start.y; canvas.style.cursor = 'grabbing'; draw(); }
      } else if (event.pointerType !== 'touch') {
        const point = nearestPoint(visible, at, projection, 10);
        if ((point?.id ?? null) !== state.hoveredId) dispatch({ type: 'hover', id: point?.id });
        canvas.style.cursor = point ? 'pointer' : 'crosshair';
      }
    });
    canvas.addEventListener('pointerup', event => {
      if (!drag || drag.id !== event.pointerId) return;
      if (!drag.moved) {
        const point = nearestPoint(visible, position(event), projection, event.pointerType === 'touch' ? 18 : 10);
        if (point) dispatch({ type: 'pin', id: point.id });
      }
      drag = null; canvas.style.cursor = 'crosshair'; canvas.releasePointerCapture(event.pointerId);
    });
    canvas.addEventListener('pointercancel', () => { drag = null; canvas.style.cursor = 'crosshair'; });
    canvas.addEventListener('pointerleave', () => { if (!drag) dispatch({ type: 'hover', id: null }); });
    canvas.addEventListener('keydown', event => {
      if (['ArrowRight', 'ArrowDown', 'ArrowLeft', 'ArrowUp'].includes(event.key)) { event.preventDefault(); dispatch({ type: 'step', direction: ['ArrowLeft', 'ArrowUp'].includes(event.key) ? -1 : 1 }); }
      if (event.key === 'Escape') { event.preventDefault(); dispatch({ type: 'unpin' }); }
      if (event.key === '+' || event.key === '=') { event.preventDefault(); zoom(1.3); }
      if (event.key === '-') { event.preventDefault(); zoom(1 / 1.3); }
      if (event.key === '0') reset();
    });
    window.addEventListener('resize', resize);
    if (window.ResizeObserver) new window.ResizeObserver(resize).observe($('plot-wrap'));
    const api = {
      get visiblePoints() { return visible.slice(); },
      get projection() { return projection; },
      get selectedPoint() { return byId.get(state.selectedId) ?? null; },
      get hoveredPoint() { return byId.get(state.hoveredId) ?? null; },
      selectPoint(point) { dispatch({ type: 'pin', id: typeof point === 'string' ? point : point?.id }); },
      hoverPoint(point) { dispatch({ type: 'hover', id: typeof point === 'string' ? point : point?.id }); },
    };
    window.dashboard = api;
    refreshFilters(); resize();
    return api;
  }
  const helpers = { category, filterPoints, groupBounds, createProjection, nearestPoint, createState, reduceState, createImageSlot, mountDashboard };
  if (typeof module !== 'undefined' && module.exports) module.exports = helpers;
  if (typeof document !== 'undefined' && typeof window !== 'undefined') mountDashboard(document, window);
})();
