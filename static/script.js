const tickerInput = document.getElementById('tickerInput');
const predictBtn = document.getElementById('predictBtn');
const resultContainer = document.getElementById('resultContainer');
const loader = document.getElementById('loader');

const tickerDisplay = document.getElementById('tickerDisplay');
const priceDisplay = document.getElementById('priceDisplay');
const signalBadge = document.getElementById('signalBadge');
const confidenceValue = document.getElementById('confidenceValue');

let probChart = null;

async function performAnalysis() {
    const ticker = tickerInput.value.trim();
    if (!ticker) return;

    // UI Reset
    resultContainer.classList.add('hidden');
    loader.classList.remove('hidden');

    try {
        const response = await fetch(`/predict/${ticker}`);
        if (!response.ok) throw new Error('Analysis failed');
        
        const data = await response.json();
        updateUI(data);
    } catch (error) {
        alert("Error: " + error.message);
        loader.classList.add('hidden');
    }
}

function updateUI(data) {
    loader.classList.add('hidden');
    resultContainer.classList.remove('hidden');

    tickerDisplay.textContent = data.ticker;
    priceDisplay.textContent = `$${data.current_price.toLocaleString(undefined, {minimumFractionDigits: 2})}`;
    
    // Signal Styling
    signalBadge.textContent = data.signal;
    signalBadge.className = 'badge';
    if (data.prediction === 1) signalBadge.classList.add('up');
    else if (data.prediction === -1) signalBadge.classList.add('down');
    else signalBadge.classList.add('neutral');

    // Confidence (Highest probability)
    const probs = data.probabilities;
    const maxProb = Math.max(...Object.values(probs));
    confidenceValue.textContent = `${(maxProb * 100).toFixed(1)}%`;

    renderChart(probs);
}

function renderChart(probs) {
    const ctx = document.getElementById('probChart').getContext('2d');
    
    if (probChart) {
        probChart.destroy();
    }

    const labels = ['Down (>3%)', 'Neutral', 'Up (>3%)'];
    const values = [probs['-1'], probs['0'], probs['1']];
    const colors = ['#ff3366', '#ffa500', '#00ff88'];

    probChart = new Chart(ctx, {
        type: 'doughnut',
        data: {
            labels: labels,
            datasets: [{
                data: values,
                backgroundColor: colors.map(c => c + '33'),
                borderColor: colors,
                borderWidth: 2,
                hoverOffset: 4
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: { display: false }
            },
            cutout: '70%'
        }
    });
}

predictBtn.addEventListener('click', performAnalysis);
tickerInput.addEventListener('keypress', (e) => {
    if (e.key === 'Enter') performAnalysis();
});
