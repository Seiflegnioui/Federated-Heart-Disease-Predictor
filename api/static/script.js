// Example Data
const highRiskData = {
    "age": 58.0, "sex": 0, "cp": 0, "trestbps": 100.0, "chol": 248.0,
    "fbs": 0, "restecg": 0, "thalach": 122.0, "exang": 0, "oldpeak": 1.0,
    "slope": 1, "ca": 0, "thal": 2
};

const lowRiskData = {
    "age": 52.0, "sex": 1, "cp": 0, "trestbps": 125.0, "chol": 212.0,
    "fbs": 0, "restecg": 1, "thalach": 168.0, "exang": 0, "oldpeak": 1.0,
    "slope": 2, "ca": 2, "thal": 3
};

// Populate form fields
function populateForm(data) {
    for (const [key, value] of Object.entries(data)) {
        const element = document.getElementById(key);
        if (element) {
            element.value = value;
            // Add a brief glow effect
            element.parentElement.style.transition = "all 0.3s";
            element.style.borderColor = "#0ea5e9";
            setTimeout(() => { element.style.borderColor = ""; }, 500);
        }
    }
}

document.getElementById('btn-high-risk').addEventListener('click', () => populateForm(highRiskData));
document.getElementById('btn-low-risk').addEventListener('click', () => populateForm(lowRiskData));

// Icons for result
const iconSick = `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M22 12h-4l-3 9L9 3l-3 9H2"></path></svg>`;
const iconHealthy = `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M12 21.35l-1.45-1.32C5.4 15.36 2 12.28 2 8.5 2 5.42 4.42 3 7.5 3c1.74 0 3.41.81 4.5 2.09C13.09 3.81 14.76 3 16.5 3 19.58 3 22 5.42 22 8.5c0 3.78-3.4 6.86-8.55 11.54L12 21.35z"></path></svg>`;

// Form Submission
document.getElementById('prediction-form').addEventListener('submit', async (e) => {
    e.preventDefault();
    
    const btn = document.getElementById('predict-btn');
    const btnText = btn.querySelector('span');
    const originalText = btnText.innerText;
    
    const overlay = document.getElementById('result-overlay');
    const statusEl = document.getElementById('prediction-status');
    const circleProgress = document.getElementById('circle-progress');
    const probText = document.getElementById('prob-text');
    const resultIcon = document.getElementById('result-icon');

    // Loading State
    btn.disabled = true;
    btnText.innerText = "Analyzing...";

    const formData = new FormData(e.target);
    const data = {};
    for (let [key, value] of formData.entries()) {
        data[key] = parseFloat(value);
    }

    try {
        await new Promise(r => setTimeout(r, 800)); // Smooth UX delay

        const response = await fetch('/predict', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(data)
        });

        if (!response.ok) throw new Error(`API error: ${response.status}`);
        const result = await response.json();
        
        // Setup Result Overlay
        overlay.classList.remove('hidden');
        // Small delay to allow display:flex to apply before opacity transition
        setTimeout(() => overlay.classList.add('active'), 10);
        
        const probPct = (result.probability * 100).toFixed(1);
        
        // Reset circle animation
        circleProgress.setAttribute('stroke-dasharray', `0, 100`);
        
        if (result.prediction === 1) {
            statusEl.innerText = "CRITICAL RISK";
            statusEl.className = "prediction-status status-sick";
            resultIcon.innerHTML = iconSick;
            resultIcon.style.color = "#f43f5e";
            circleProgress.className.baseVal = "circle status-sick-stroke";
        } else {
            statusEl.innerText = "NORMAL / HEALTHY";
            statusEl.className = "prediction-status status-healthy";
            resultIcon.innerHTML = iconHealthy;
            resultIcon.style.color = "#10b981";
            circleProgress.className.baseVal = "circle status-healthy-stroke";
        }

        // Animate Circle & Text
        setTimeout(() => {
            circleProgress.setAttribute('stroke-dasharray', `${probPct}, 100`);
            animateValue(probText, 0, parseFloat(probPct), 1500);
        }, 300);

    } catch (error) {
        console.error(error);
        alert("An error occurred while calling the API.");
    } finally {
        btn.disabled = false;
        btnText.innerText = originalText;
    }
});

// Close Overlay
document.getElementById('close-result').addEventListener('click', () => {
    const overlay = document.getElementById('result-overlay');
    overlay.classList.remove('active');
    setTimeout(() => overlay.classList.add('hidden'), 400);
});

// Number counter animation
function animateValue(obj, start, end, duration) {
    let startTimestamp = null;
    const step = (timestamp) => {
        if (!startTimestamp) startTimestamp = timestamp;
        const progress = Math.min((timestamp - startTimestamp) / duration, 1);
        // easeOutQuart
        const easeProgress = 1 - Math.pow(1 - progress, 4);
        obj.innerHTML = (start + easeProgress * (end - start)).toFixed(1) + "%";
        if (progress < 1) {
            window.requestAnimationFrame(step);
        }
    };
    window.requestAnimationFrame(step);
}
