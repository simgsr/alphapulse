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

function showError(msg) {
    errorText.textContent = msg;
    errorBanner.classList.remove('hidden');
}

function clearError() {
    errorBanner.classList.add('hidden');
}

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

    const pct3 = Math.round(data.confidence_up_3pct * 100);
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

async function performScan() {
    scanLoader.classList.remove('hidden');
    scanResults.innerHTML = '';
    scanBtn.disabled = true;

    try {
        const res = await fetch('/scan');
        if (!res.ok) throw new Error('Scan failed');
        const picks = await res.json();
        renderScanResults(picks);
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

predictBtn.addEventListener('click', performAnalysis);
tickerInput.addEventListener('keydown', e => { if (e.key === 'Enter') performAnalysis(); });
scanBtn.addEventListener('click', performScan);
