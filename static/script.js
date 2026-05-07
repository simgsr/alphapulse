const tickerInput   = document.getElementById('tickerInput');
const predictBtn    = document.getElementById('predictBtn');
const resultSection = document.getElementById('resultContainer');
const loader        = document.getElementById('loader');
const errorBanner   = document.getElementById('errorBanner');
const errorText     = document.getElementById('errorText');

const tickerDisplay = document.getElementById('tickerDisplay');
const priceDisplay  = document.getElementById('priceDisplay');
const signalBadge   = document.getElementById('signalBadge');
const edgeDisplay   = document.getElementById('edgeDisplay');
const conf3Pct      = document.getElementById('conf3Pct');
const conf5Pct      = document.getElementById('conf5Pct');
const conf3Bar      = document.getElementById('conf3Bar');
const conf5Bar      = document.getElementById('conf5Bar');
const lastUpdated   = document.getElementById('lastUpdated');
const chartLegend   = document.getElementById('chartLegend');
const scanSection   = document.getElementById('scanSection');
const scanBtn       = document.getElementById('scanBtn');
const scanLoader    = document.getElementById('scanLoader');
const scanResults   = document.getElementById('scanResults');

let probChart = null;

const SIGNAL_CLASSES = {
    '1':  'badge--up1',
    '0':  'badge--neutral',
    '-1': 'badge--down1',
};

const CHART_CONFIG = [
    { key: '-1', label: 'DOWN >3%', color: '#ef4444' },
    { key: '0',  label: 'STABLE',   color: '#fbbf24' },
    { key: '1',  label: 'UP >3%',   color: '#4ade80' },
];

// ── Watchlist state ──────────────────────────────────────────────────────────

let watchlist = JSON.parse(localStorage.getItem('watchlist') || '[]');

function saveWatchlist() {
    localStorage.setItem('watchlist', JSON.stringify(watchlist));
}

function addToWatchlist(ticker) {
    ticker = ticker.toUpperCase().trim();
    if (!ticker) return false;
    if (watchlist.includes(ticker)) return false;
    if (watchlist.length >= 50) {
        showError('Watchlist is limited to 50 tickers. Remove some before adding more.');
        return false;
    }
    watchlist.push(ticker);
    saveWatchlist();
    renderWatchlistChips();
    return true;
}

function removeFromWatchlist(ticker) {
    watchlist = watchlist.filter(t => t !== ticker);
    saveWatchlist();
    renderWatchlistChips();
}

function renderWatchlistChips() {
    const container = document.getElementById('watchlistChips');
    if (watchlist.length === 0) {
        container.innerHTML = '<p class="watchlist-empty">No tickers added yet.</p>';
    } else {
        container.innerHTML = watchlist.map(t => `
            <span class="chip">
                ${t}
                <button class="chip__remove" onclick="removeFromWatchlist('${t}')" aria-label="Remove ${t}">×</button>
            </span>
        `).join('');
        scanSection.classList.remove('hidden');
    }
    document.getElementById('scanWatchlistBtn').classList.toggle('hidden', watchlist.length === 0);
}

// ── CSV import ───────────────────────────────────────────────────────────────

document.getElementById('csvInput').addEventListener('change', function (e) {
    const file = e.target.files[0];
    if (!file) return;
    const reader = new FileReader();
    reader.onload = function (ev) {
        const lines = ev.target.result.replace(/^﻿/, '').trim().split(/\r\n|\r|\n/);
        if (!lines.length) return;

        const headerKeywords = ['ticker', 'symbol', 'code', 'stock', 'instrument'];
        const firstLower = lines[0].toLowerCase();
        let startRow = 0;
        let tickerCol = 0;

        if (headerKeywords.some(k => firstLower.includes(k))) {
            startRow = 1;
            const headers = lines[0].split(',').map(h => h.trim().toLowerCase().replace(/['"]/g, ''));
            const idx = headers.findIndex(h => headerKeywords.some(k => h.includes(k)));
            if (idx >= 0) tickerCol = idx;
        }

        lines.slice(startRow).forEach(line => {
            const cols = line.split(',');
            const raw = cols[tickerCol]?.trim().replace(/['"]/g, '');
            if (raw) addToWatchlist(raw);
        });
    };
    reader.readAsText(file);
    this.value = '';
});

// ── Watchlist scan ───────────────────────────────────────────────────────────

async function scanWatchlist() {
    if (watchlist.length === 0) return;

    const btn        = document.getElementById('scanWatchlistBtn');
    const wLoader    = document.getElementById('watchlistScanLoader');
    const loaderText = document.getElementById('watchlistScanLoaderText');
    const wResults   = document.getElementById('watchlistResults');

    btn.disabled = true;
    wLoader.classList.remove('hidden');
    wResults.innerHTML = '';

    const total = watchlist.length;
    let done = 0;
    loaderText.textContent = `Scanning 0 / ${total}…`;

    try {
        const promises = watchlist.map(ticker =>
            fetch(`/predict/${encodeURIComponent(ticker)}`)
                .then(r => r.ok ? r.json() : null)
                .catch(() => null)
                .then(result => {
                    done++;
                    loaderText.textContent = `Scanning ${done} / ${total}…`;
                    return result;
                })
        );
        const settled = await Promise.all(promises);
        const valid   = settled.filter(Boolean);
        const sorted  = valid.sort((a, b) => b.edge_ratio - a.edge_ratio);
        renderWatchlistResults(sorted);
    } finally {
        wLoader.classList.add('hidden');
        btn.disabled = false;
    }
}

function renderWatchlistResults(picks) {
    const container = document.getElementById('watchlistResults');
    if (!picks.length) {
        container.innerHTML = '<p class="watchlist-empty">No data found for any ticker in your list.</p>';
        return;
    }
    container.innerHTML = picks.map((p, i) => {
        const edgeClass = p.edge_ratio >= 2 ? '' : 'pick-edge--low';
        const conf3 = Math.round(p.confidence_up_3pct * 100);
        return `
        <div class="pick-card" onclick="fillAndAnalyze('${p.ticker}')">
            <div class="pick-left">
                <span class="pick-ticker">#${i + 1} ${p.ticker}</span>
                <span class="pick-price">$${Number(p.current_price).toFixed(2)} · ${p.signal}</span>
            </div>
            <div class="pick-right">
                <span class="pick-edge ${edgeClass}">${p.edge_ratio.toFixed(2)}×</span>
                <span class="pick-conf">UP&gt;3% conf: ${conf3}%</span>
            </div>
        </div>`;
    }).join('');
}

// ── Watchlist event wiring ───────────────────────────────────────────────────

document.getElementById('watchlistAddBtn').addEventListener('click', () => {
    const input = document.getElementById('watchlistInput');
    if (addToWatchlist(input.value)) input.value = '';
});

document.getElementById('watchlistInput').addEventListener('keydown', e => {
    if (e.key === 'Enter') {
        const input = document.getElementById('watchlistInput');
        if (addToWatchlist(input.value)) input.value = '';
    }
});

document.getElementById('scanWatchlistBtn').addEventListener('click', scanWatchlist);

// ── Error helpers ────────────────────────────────────────────────────────────

function showError(msg) {
    errorText.textContent = msg;
    errorBanner.classList.remove('hidden');
}

function clearError() {
    errorBanner.classList.add('hidden');
}

// ── Ticker analysis ──────────────────────────────────────────────────────────

async function performAnalysis() {
    const ticker = tickerInput.value.trim();
    if (!ticker) return;

    clearError();
    resultSection.classList.add('hidden');
    loader.classList.remove('hidden');
    predictBtn.disabled = true;

    try {
        const res = await fetch(`/predict/${encodeURIComponent(ticker)}`);
        if (!res.ok) {
            const err = await res.json().catch(() => ({ detail: 'Analysis failed' }));
            throw new Error(err.detail || 'Analysis failed');
        }
        const data = await res.json();
        updateUI(data);
    } catch (e) {
        showError(e.message);
    } finally {
        loader.classList.add('hidden');
        predictBtn.disabled = false;
    }
}

function updateUI(data) {
    tickerDisplay.textContent = data.ticker;
    priceDisplay.textContent  = '$' + Number(data.current_price).toLocaleString(undefined, {
        minimumFractionDigits: 2, maximumFractionDigits: 2
    });

    signalBadge.textContent  = data.signal;
    signalBadge.className    = 'badge';
    const predKey = String(data.prediction);
    if (SIGNAL_CLASSES[predKey]) signalBadge.classList.add(SIGNAL_CLASSES[predKey]);

    const edge = data.edge_ratio;
    edgeDisplay.className = 'edge-display';
    if (edge >= 2) {
        edgeDisplay.textContent = `Edge ${edge.toFixed(2)}× bullish`;
        edgeDisplay.classList.add('edge-display--bull');
    } else if (edge < 1) {
        edgeDisplay.textContent = `Edge ${edge.toFixed(2)}× bearish`;
        edgeDisplay.classList.add('edge-display--bear');
    } else {
        edgeDisplay.textContent = `Edge ${edge.toFixed(2)}× neutral`;
    }

    const pct3    = Math.round(data.confidence_up_3pct * 100);
    const pctDown = Math.round(data.confidence_down_3pct * 100);
    conf3Pct.textContent = pct3 + '%';
    conf5Pct.textContent = pctDown + '%';
    requestAnimationFrame(() => {
        conf3Bar.style.width = pct3 + '%';
        conf5Bar.style.width = pctDown + '%';
    });

    renderChart(data.probabilities);

    lastUpdated.textContent = 'Updated: ' + data.last_updated;

    resultSection.classList.remove('hidden');
    scanSection.classList.remove('hidden');
}

// ── Watchlist scan (Top Picks) ────────────────────────────────────────────────

async function performScan() {
    if (watchlist.length === 0) {
        scanResults.innerHTML = '<p style="color:var(--text-muted);font-size:13px">Add tickers to your watchlist first.</p>';
        return;
    }

    const loaderText = document.getElementById('scanLoaderText');
    scanLoader.classList.remove('hidden');
    scanResults.innerHTML = '';
    scanBtn.disabled = true;

    const total = watchlist.length;
    let done = 0;
    loaderText.textContent = `Scanning 0 / ${total}…`;

    try {
        const promises = watchlist.map(ticker =>
            fetch(`/predict/${encodeURIComponent(ticker)}`)
                .then(r => r.ok ? r.json() : null)
                .catch(() => null)
                .then(result => {
                    done++;
                    loaderText.textContent = `Scanning ${done} / ${total}…`;
                    return result;
                })
        );
        const settled = await Promise.all(promises);
        const valid   = settled.filter(Boolean);
        const sorted  = valid.sort((a, b) => b.edge_ratio - a.edge_ratio);
        renderScanResults(sorted);
    } catch (e) {
        scanResults.innerHTML = `<p style="color:var(--down-strong);font-size:13px">${e.message}</p>`;
    } finally {
        scanLoader.classList.add('hidden');
        scanBtn.disabled = false;
    }
}

function renderScanResults(picks) {
    if (!picks.length) {
        scanResults.innerHTML = '<p style="color:var(--text-muted);font-size:13px">No results.</p>';
        return;
    }
    scanResults.innerHTML = picks.map((p, i) => {
        const edgeClass = p.edge_ratio >= 2 ? '' : 'pick-edge--low';
        const conf3 = Math.round(p.confidence_up_3pct * 100);
        return `
        <div class="pick-card" onclick="fillAndAnalyze('${p.ticker}')">
            <div class="pick-left">
                <span class="pick-ticker">#${i + 1} ${p.ticker}</span>
                <span class="pick-price">HK$${Number(p.current_price).toFixed(2)} · ${p.signal}</span>
            </div>
            <div class="pick-right">
                <span class="pick-edge ${edgeClass}">${p.edge_ratio.toFixed(2)}×</span>
                <span class="pick-conf">UP&gt;3% conf: ${conf3}%</span>
            </div>
        </div>`;
    }).join('');
}

function fillAndAnalyze(ticker) {
    tickerInput.value = ticker;
    performAnalysis();
    window.scrollTo({ top: 0, behavior: 'smooth' });
}

// ── Chart ────────────────────────────────────────────────────────────────────

function renderChart(probs) {
    const ctx = document.getElementById('probChart').getContext('2d');
    if (probChart) probChart.destroy();

    const values = CHART_CONFIG.map(c => probs[c.key] ?? 0);
    const colors = CHART_CONFIG.map(c => c.color);

    probChart = new Chart(ctx, {
        type: 'doughnut',
        data: {
            labels: CHART_CONFIG.map(c => c.label),
            datasets: [{
                data: values,
                backgroundColor: colors.map(c => c + '33'),
                borderColor: colors,
                borderWidth: 2,
                hoverOffset: 6,
            }]
        },
        options: {
            responsive: false,
            plugins: {
                legend: { display: false },
                tooltip: {
                    callbacks: {
                        label: ctx => ` ${(ctx.raw * 100).toFixed(1)}%`
                    }
                }
            },
            cutout: '70%',
        }
    });

    chartLegend.innerHTML = CHART_CONFIG.map((c, i) => `
        <span class="legend-item">
            <span class="legend-dot" style="background:${c.color}"></span>
            ${c.label} <strong>${(values[i] * 100).toFixed(1)}%</strong>
        </span>
    `).join('');
}

// ── Init ─────────────────────────────────────────────────────────────────────

predictBtn.addEventListener('click', performAnalysis);
tickerInput.addEventListener('keydown', e => { if (e.key === 'Enter') performAnalysis(); });
scanBtn.addEventListener('click', performScan);

renderWatchlistChips();
