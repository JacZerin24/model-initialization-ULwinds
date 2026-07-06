const state = {
  payload: null,
  modelKey: null,
  observationLayer: null,
  windLayer: null,
  heightLayer: null,
};

const map = L.map('map', {
  center: [20, 0],
  zoom: 2,
  minZoom: 2,
  maxZoom: 7,
  worldCopyJump: true,
  preferCanvas: true,
});
map.createPane('windPane');
map.getPane('windPane').style.zIndex = 250;
map.createPane('heightPane');
map.getPane('heightPane').style.zIndex = 360;
map.createPane('raobPane');
map.getPane('raobPane').style.zIndex = 460;

L.tileLayer('https://{s}.basemaps.cartocdn.com/light_all/{z}/{x}/{y}{r}.png', {
  attribution: '&copy; OpenStreetMap contributors &copy; CARTO',
  subdomains: 'abcd',
  maxZoom: 19,
}).addTo(map);

const WIND_STOPS = [
  [0, '#f7fbff'], [30, '#c6dbef'], [50, '#7fcdbb'], [70, '#41b6c4'],
  [90, '#1d91c0'], [110, '#225ea8'], [130, '#54278f'], [150, '#7a0177'],
];

function fmt(value, suffix = '', decimals = 1) {
  if (value === null || value === undefined || Number.isNaN(Number(value))) return '—';
  return `${Number(value).toFixed(decimals)}${suffix}`;
}

function hexRgb(hex) {
  const value = parseInt(hex.slice(1), 16);
  return [(value >> 16) & 255, (value >> 8) & 255, value & 255];
}

function windRgba(speed, alpha = 0.68) {
  if (speed === null || speed === undefined || !Number.isFinite(Number(speed))) return null;
  const value = Number(speed);
  let lower = WIND_STOPS[0];
  let upper = WIND_STOPS[WIND_STOPS.length - 1];
  for (let i = 1; i < WIND_STOPS.length; i += 1) {
    if (value <= WIND_STOPS[i][0]) {
      lower = WIND_STOPS[i - 1];
      upper = WIND_STOPS[i];
      break;
    }
  }
  const span = Math.max(upper[0] - lower[0], 1);
  const fraction = Math.max(0, Math.min(1, (value - lower[0]) / span));
  const a = hexRgb(lower[1]);
  const b = hexRgb(upper[1]);
  const rgb = a.map((channel, index) => Math.round(channel + fraction * (b[index] - channel)));
  return `rgba(${rgb[0]},${rgb[1]},${rgb[2]},${alpha})`;
}

function errorColor(error) {
  if (error < 5) return '#2c7bb6';
  if (error < 10) return '#73add1';
  if (error < 20) return '#f9d057';
  if (error < 30) return '#f28e2b';
  return '#c62828';
}

const WindGridLayer = L.GridLayer.extend({
  initialize(analysis, options = {}) {
    L.setOptions(this, options);
    this.analysis = analysis;
    this.latitudes = analysis.latitudes;
    this.longitudes = analysis.longitudes;
    this.values = analysis.wind_speed_kt;
    this.latStep = this.latitudes[1] - this.latitudes[0];
    this.lonStep = this.longitudes[1] - this.longitudes[0];
  },

  createTile(coords) {
    const tile = L.DomUtil.create('canvas', 'wind-grid-tile');
    const size = this.getTileSize();
    tile.width = size.x;
    tile.height = size.y;
    requestAnimationFrame(() => this.drawTile(tile, coords));
    return tile;
  },

  sample(lat, lon) {
    const latMin = this.latitudes[0];
    const latMax = this.latitudes[this.latitudes.length - 1];
    if (lat < latMin || lat > latMax) return null;
    const wrappedLon = ((lon + 180) % 360 + 360) % 360 - 180;
    const y = (lat - latMin) / this.latStep;
    const x = (wrappedLon - this.longitudes[0]) / this.lonStep;
    const y0 = Math.max(0, Math.min(this.latitudes.length - 1, Math.floor(y)));
    const y1 = Math.max(0, Math.min(this.latitudes.length - 1, y0 + 1));
    const x0 = ((Math.floor(x) % this.longitudes.length) + this.longitudes.length) % this.longitudes.length;
    const x1 = (x0 + 1) % this.longitudes.length;
    const fy = Math.max(0, Math.min(1, y - Math.floor(y)));
    const fx = Math.max(0, Math.min(1, x - Math.floor(x)));
    const raw = [this.values[y0][x0], this.values[y0][x1], this.values[y1][x0], this.values[y1][x1]];
    if (raw.some(value => value === null || value === undefined)) return null;
    const [q00, q10, q01, q11] = raw.map(Number);
    if (![q00, q10, q01, q11].every(Number.isFinite)) return null;
    return q00 * (1 - fx) * (1 - fy) + q10 * fx * (1 - fy) + q01 * (1 - fx) * fy + q11 * fx * fy;
  },

  drawTile(tile, coords) {
    if (!this._map) return;
    const context = tile.getContext('2d');
    const size = this.getTileSize();
    const origin = coords.scaleBy(size);
    const block = coords.z >= 5 ? 2 : 4;
    for (let y = 0; y < size.y; y += block) {
      for (let x = 0; x < size.x; x += block) {
        const latLng = this._map.unproject(L.point(origin.x + x + block / 2, origin.y + y + block / 2), coords.z);
        const color = windRgba(this.sample(latLng.lat, latLng.lng));
        if (!color) continue;
        context.fillStyle = color;
        context.fillRect(x, y, block, block);
      }
    }
  },
});

function stationPopup(station, label) {
  const biasSign = Number(station.speed_error_kt) > 0 ? '+' : '';
  const heightRows = station.obs_height_m !== null && station.obs_height_m !== undefined
    ? `<span>Observed height</span><strong>${fmt(station.obs_height_m, ' m', 0)}</strong>
       <span>Model height</span><strong>${fmt(station.model_height_m, ' m', 0)}</strong>
       <span>Height error</span><strong>${Number(station.height_error_m) > 0 ? '+' : ''}${fmt(station.height_error_m, ' m', 0)}</strong>`
    : '';
  return `
    <h3 class="popup-title">${station.name || station.station}</h3>
    <div class="popup-grid">
      <span>Station</span><strong>${station.station}</strong>
      <span>Model</span><strong>${label}</strong>
      <span>Observed wind</span><strong>${fmt(station.obs_speed_kt, ' kt')} @ ${fmt(station.obs_direction_deg, '°', 0)}</strong>
      <span>Model wind</span><strong>${fmt(station.model_speed_kt, ' kt')} @ ${fmt(station.model_direction_deg, '°', 0)}</strong>
      <span>Speed bias</span><strong>${biasSign}${fmt(station.speed_error_kt, ' kt')}</strong>
      <span>Absolute error</span><strong>${fmt(station.abs_speed_error_kt, ' kt')}</strong>
      <span>Vector error</span><strong>${fmt(station.vector_error_kt, ' kt')}</strong>
      <span>Direction error</span><strong>${fmt(station.direction_error_deg, '°', 0)}</strong>
      ${heightRows}
      <span>Station metadata</span><strong>${station.metadata_source || '—'}</strong>
      <span>Observation time</span><strong>${station.observation_time ? new Date(station.observation_time).toISOString().replace('.000Z', 'Z') : '—'}</strong>
    </div>`;
}

function renderTabs() {
  const tabs = document.getElementById('model-tabs');
  tabs.innerHTML = '';
  Object.entries(state.payload.models).forEach(([key, model]) => {
    const button = document.createElement('button');
    button.className = 'model-tab';
    button.type = 'button';
    button.role = 'tab';
    button.textContent = model.label;
    button.disabled = model.status !== 'ok';
    button.setAttribute('aria-selected', key === state.modelKey ? 'true' : 'false');
    button.title = model.status === 'ok' ? model.source : model.error || 'Model unavailable';
    button.addEventListener('click', () => selectModel(key));
    tabs.appendChild(button);
  });
}

function renderMetrics(model) {
  const m = model.metrics || {};
  document.getElementById('metric-n').textContent = m.n ?? '—';
  document.getElementById('metric-mae').textContent = fmt(m.mae_kt, ' kt');
  document.getElementById('metric-bias').textContent = fmt(m.bias_kt, ' kt');
  document.getElementById('metric-vector-rmse').textContent = fmt(m.vector_rmse_kt, ' kt');
  document.getElementById('metric-height-mae').textContent = fmt(m.height_mae_m, ' m', 0);
}

function clearAnalysisLayers() {
  if (state.windLayer) state.windLayer.remove();
  if (state.heightLayer) state.heightLayer.remove();
  state.windLayer = null;
  state.heightLayer = null;
}

function renderAnalysis(model) {
  clearAnalysisLayers();
  if (!model.analysis) return;

  state.windLayer = new WindGridLayer(model.analysis, {
    pane: 'windPane',
    tileSize: 256,
    updateWhenZooming: false,
    keepBuffer: 2,
  });
  if (document.getElementById('wind-toggle').checked) state.windLayer.addTo(map);

  state.heightLayer = L.layerGroup();
  for (const contour of model.analysis.height_contours || []) {
    const major = Number(contour.level_dam) % 24 === 0;
    L.polyline(contour.coordinates.map(([lon, lat]) => [lat, lon]), {
      pane: 'heightPane',
      color: '#111827',
      weight: major ? 1.8 : 1.05,
      opacity: major ? 0.9 : 0.7,
      interactive: false,
      smoothFactor: 1.2,
    }).addTo(state.heightLayer);
  }
  for (const label of model.analysis.height_labels || []) {
    const icon = L.divIcon({
      className: 'height-label',
      html: `${label.level_dam}`,
      iconSize: [36, 16],
      iconAnchor: [18, 8],
    });
    L.marker([label.lat, label.lon], { icon, pane: 'heightPane', interactive: false, keyboard: false }).addTo(state.heightLayer);
  }
  if (document.getElementById('height-toggle').checked) state.heightLayer.addTo(map);
}

function renderStations(model) {
  if (state.observationLayer) state.observationLayer.remove();
  state.observationLayer = L.layerGroup();
  for (const station of model.stations || []) {
    const error = Number(station.abs_speed_error_kt);
    const radius = Math.max(4, Math.min(12, 4 + error / 4));
    const marker = L.circleMarker([station.latitude, station.longitude], {
      pane: 'raobPane', radius, color: '#172b3a', weight: 0.75,
      fillColor: errorColor(error), fillOpacity: 0.92,
    });
    const signed = Number(station.speed_error_kt);
    const sign = signed > 0 ? '+' : '';
    marker.bindTooltip(`${station.station}: ${fmt(error, ' kt absolute error')} (${sign}${fmt(signed, ' kt bias')})`, { direction: 'top' });
    marker.bindPopup(stationPopup(station, model.label));
    marker.addTo(state.observationLayer);
  }
  state.observationLayer.addTo(map);
}

function selectModel(key) {
  const model = state.payload.models[key];
  if (!model || model.status !== 'ok') return;
  state.modelKey = key;
  renderTabs();
  renderMetrics(model);
  renderAnalysis(model);
  renderStations(model);
}

function setMetadata(payload) {
  const cycle = new Date(payload.cycle);
  const generated = new Date(payload.generated_at);
  document.getElementById('cycle-label').textContent = cycle.toLocaleString([], {
    timeZone: 'UTC', year: 'numeric', month: 'short', day: 'numeric', hour: '2-digit', hour12: false,
  }).replace('24:', '00:') + ' UTC';
  document.getElementById('generated-label').textContent = `Updated ${generated.toLocaleString()}`;
  document.getElementById('coverage-label').textContent = payload.observation_summary
    ? `${payload.observation_summary.station_count} worldwide 300-hPa RAOBs retrieved`
    : '';

  const notice = document.getElementById('notice');
  if (payload.demo) {
    notice.hidden = false;
    notice.textContent = 'Demonstration data are displayed. Run the live-data GitHub Action to replace them.';
  } else {
    const errors = Object.values(payload.models).filter(model => model.status !== 'ok');
    if (errors.length) {
      notice.hidden = false;
      notice.textContent = `${errors.length} model source${errors.length === 1 ? '' : 's'} failed during the latest update. Available models remain viewable.`;
    }
  }
}

async function loadData() {
  try {
    const response = await fetch(`data/latest.json?v=${Date.now()}`);
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    state.payload = await response.json();
    const firstAvailable = Object.entries(state.payload.models).find(([, model]) => model.status === 'ok');
    if (!firstAvailable) throw new Error('No model data are available');
    state.modelKey = firstAvailable[0];
    setMetadata(state.payload);
    renderTabs();
    selectModel(state.modelKey);
  } catch (error) {
    const notice = document.getElementById('notice');
    notice.hidden = false;
    notice.textContent = `The verification data could not be loaded: ${error.message}`;
    console.error(error);
  }
}

document.getElementById('wind-toggle').addEventListener('change', event => {
  if (!state.windLayer) return;
  if (event.target.checked) state.windLayer.addTo(map); else state.windLayer.remove();
});
document.getElementById('height-toggle').addEventListener('change', event => {
  if (!state.heightLayer) return;
  if (event.target.checked) state.heightLayer.addTo(map); else state.heightLayer.remove();
});

loadData();
