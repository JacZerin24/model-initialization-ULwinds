const state = {
  payload: null,
  modelKey: null,
  observationLayer: null,
  vectorLayer: null,
};

const map = L.map('map', {
  center: [20, 0],
  zoom: 2,
  minZoom: 2,
  maxZoom: 7,
  worldCopyJump: true,
  preferCanvas: true,
});

L.tileLayer('https://{s}.basemaps.cartocdn.com/light_all/{z}/{x}/{y}{r}.png', {
  attribution: '&copy; OpenStreetMap contributors &copy; CARTO',
  subdomains: 'abcd',
  maxZoom: 19,
}).addTo(map);

function fmt(value, suffix = '') {
  if (value === null || value === undefined || Number.isNaN(Number(value))) return '—';
  return `${Number(value).toFixed(1)}${suffix}`;
}

function errorColor(error) {
  if (error < 5) return '#2c7bb6';
  if (error < 10) return '#73add1';
  if (error < 20) return '#f9d057';
  if (error < 30) return '#f28e2b';
  return '#c62828';
}

function windColor(speed) {
  if (speed < 50) return '#536d7a';
  if (speed < 90) return '#168aad';
  if (speed < 130) return '#7b2cbf';
  return '#c1121f';
}

function stationPopup(station, label) {
  const biasSign = Number(station.speed_error_kt) > 0 ? '+' : '';
  return `
    <h3 class="popup-title">${station.name || station.station}</h3>
    <div class="popup-grid">
      <span>Station</span><strong>${station.station}</strong>
      <span>Model</span><strong>${label}</strong>
      <span>Observed wind</span><strong>${fmt(station.obs_speed_kt, ' kt')} @ ${fmt(station.obs_direction_deg, '°')}</strong>
      <span>Model wind</span><strong>${fmt(station.model_speed_kt, ' kt')} @ ${fmt(station.model_direction_deg, '°')}</strong>
      <span>Speed bias</span><strong>${biasSign}${fmt(station.speed_error_kt, ' kt')}</strong>
      <span>Absolute error</span><strong>${fmt(station.abs_speed_error_kt, ' kt')}</strong>
      <span>Vector error</span><strong>${fmt(station.vector_error_kt, ' kt')}</strong>
      <span>Direction error</span><strong>${fmt(station.direction_error_deg, '°')}</strong>
      <span>300-hPa value</span><strong>${station.vertical_method || 'reported'}</strong>
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
    button.dataset.model = key;
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
}

function renderVectors(model) {
  if (state.vectorLayer) state.vectorLayer.remove();
  state.vectorLayer = L.layerGroup();
  for (const vector of model.vectors || []) {
    const rotation = Number(vector.direction_to_deg) - 90;
    const icon = L.divIcon({
      className: 'wind-icon',
      html: `<span class="wind-arrow" style="color:${windColor(vector.speed_kt)};transform:rotate(${rotation}deg)">➤</span>`,
      iconSize: [22, 22],
      iconAnchor: [11, 11],
    });
    L.marker([vector.lat, vector.lon], { icon, interactive: false, keyboard: false }).addTo(state.vectorLayer);
  }
  if (document.getElementById('vector-toggle').checked) state.vectorLayer.addTo(map);
}

function renderStations(model) {
  if (state.observationLayer) state.observationLayer.remove();
  state.observationLayer = L.layerGroup();
  for (const station of model.stations || []) {
    const error = Number(station.abs_speed_error_kt);
    const radius = Math.max(4, Math.min(12, 4 + error / 4));
    const marker = L.circleMarker([station.latitude, station.longitude], {
      radius,
      color: '#172b3a',
      weight: 0.75,
      fillColor: errorColor(error),
      fillOpacity: 0.9,
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
  renderVectors(model);
  renderStations(model);
}

function setMetadata(payload) {
  const cycle = new Date(payload.cycle);
  const generated = new Date(payload.generated_at);
  document.getElementById('cycle-label').textContent = cycle.toLocaleString([], {
    timeZone: 'UTC', year: 'numeric', month: 'short', day: 'numeric', hour: '2-digit', hour12: false,
  }).replace('24:', '00:') + ' UTC';
  document.getElementById('generated-label').textContent = `Updated ${generated.toLocaleString()}`;

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

document.getElementById('vector-toggle').addEventListener('change', event => {
  if (!state.vectorLayer) return;
  if (event.target.checked) state.vectorLayer.addTo(map);
  else state.vectorLayer.remove();
});

loadData();
