document.getElementById('analyzeBtn').addEventListener('click', async () => {
    const btn = document.getElementById('analyzeBtn');
    const loader = document.getElementById('loader');
    const results = document.getElementById('results');
    const errorDiv = document.getElementById('error');

    // Reset UI
    btn.disabled = true;
    loader.style.display = 'block';
    results.style.display = 'none';
    errorDiv.style.display = 'none';

    try {
        // 1. Get current tab
        const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });

        if (!tab.url.includes("youtube.com/watch")) {
            throw new Error("Please open a YouTube video page.");
        }

        // 2. Ask content script for comments
        const response = await chrome.tabs.sendMessage(tab.id, { action: "getComments" });

        if (!response || !response.comments || response.comments.length === 0) {
            throw new Error("No comments found. Scroll down to load some comments first!");
        }

        console.log(`Analyzing ${response.comments.length} comments...`);

        // 3. Send to background for API call
        const analysis = await chrome.runtime.sendMessage({
            action: "analyzeSentiment",
            comments: response.comments
        });

        if (!analysis.success) {
            throw new Error(analysis.error || "Analysis failed.");
        }

        // 4. Process Results
        const predictions = analysis.data.predictions; // ["joy", "sadness", ...]
        displayResults(predictions);

    } catch (err) {
        if (err.message.includes("Receiving end does not exist") || err.message.includes("Could not establish connection")) {
            errorDiv.textContent = "Please REFRESH this YouTube page and try again!";
        } else {
            errorDiv.textContent = err.message;
        }
        errorDiv.style.display = 'block';
    } finally {
        btn.disabled = false;
        loader.style.display = 'none';
    }
});

let emotionChart = null;

function displayResults(predictions) {
    const counts = {};
    predictions.forEach(p => counts[p] = (counts[p] || 0) + 1);

    const labels = Object.keys(counts);
    const data = labels.map(l => counts[l]);
    const backgroundColor = labels.map(l => getColor(l));

    // Initialize/Update Chart
    const ctx = document.getElementById('emotionChart').getContext('2d');

    if (emotionChart) {
        emotionChart.destroy();
    }

    emotionChart = new Chart(ctx, {
        type: 'pie',
        data: {
            labels: labels.map(l => l.toUpperCase()),
            datasets: [{
                data: data,
                backgroundColor: backgroundColor,
                borderWidth: 1
            }]
        },
        options: {
            responsive: false,
            plugins: {
                legend: {
                    position: 'bottom',
                    labels: { boxWidth: 12, font: { size: 10 } }
                }
            }
        }
    });

    // Find dominant
    const dominant = Object.keys(counts).reduce((a, b) => counts[a] > counts[b] ? a : b);

    // Simple "Positive" metric
    const positiveLabels = ['joy', 'love', 'optimism', 'admiration', 'excitement', 'gratitude', 'pride', 'relief'];
    const positiveCount = predictions.filter(p => positiveLabels.includes(p)).length;
    const positivePct = Math.round((positiveCount / predictions.length) * 100);

    document.getElementById('dominant').textContent = dominant.toUpperCase();
    document.getElementById('dominant').style.color = getColor(dominant);

    document.getElementById('positive_pct').textContent = `${positivePct}%`;
    document.getElementById('count').textContent = predictions.length;

    document.getElementById('results').style.display = 'block';
}

function getColor(emotion) {
    const colors = {
        joy: '#38A169',      // Green
        sadness: '#3182CE',  // Blue
        anger: '#E53E3E',    // Red
        fear: '#805AD5',     // Purple
        surprise: '#D69E2E', // Yellow
        disgust: '#DD6B20',  // Orange
        neutral: '#718096'   // Gray
    };
    return colors[emotion] || '#333';
}
