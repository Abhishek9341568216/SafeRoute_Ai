/* SafeRoute AI - Prediction Form Interactive Handler */

document.addEventListener('DOMContentLoaded', () => {
    const form = document.getElementById('predictionForm');
    const resultBox = document.getElementById('predictionResult');
    const spinner = document.getElementById('loadingSpinner');

    if (form) {
        form.addEventListener('submit', async (e) => {
            e.preventDefault();

            if (spinner) spinner.classList.remove('d-none');
            if (resultBox) resultBox.classList.add('d-none');

            const formData = new FormData(form);
            const payload = {};
            formData.forEach((value, key) => {
                payload[key] = value;
            });

            try {
                const response = await fetch('/api/predict', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify(payload)
                });

                const data = await response.json();
                if (spinner) spinner.classList.add('d-none');

                if (data.status === 'success') {
                    document.getElementById('resLocation').innerText = data.location;
                    document.getElementById('resTraffic').innerText = formatNumber(data.traffic_prediction) + " vehicles/hr";
                    document.getElementById('resTrafficLevel').innerText = data.traffic_level;

                    const probElement = document.getElementById('resAccidentProb');
                    if (data.accident_probability !== null) {
                        probElement.innerText = (data.accident_probability * 100).toFixed(1) + "%";
                    } else {
                        probElement.innerText = "N/A";
                    }

                    const riskBadge = document.getElementById('resRiskLevel');
                    riskBadge.innerText = data.risk_level;
                    riskBadge.className = 'risk-badge risk-' + data.risk_level.split(' ')[0];

                    const recList = document.getElementById('resRecommendations');
                    recList.innerHTML = '';
                    data.recommendations.forEach(rec => {
                        const li = document.createElement('li');
                        li.innerText = rec;
                        recList.appendChild(li);
                    });

                    if (resultBox) resultBox.classList.remove('d-none');
                    resultBox.scrollIntoView({ behavior: 'smooth' });
                } else {
                    alert("Prediction Error: " + (data.message || "Unknown error occurred."));
                }
            } catch (err) {
                if (spinner) spinner.classList.add('d-none');
                alert("Failed to connect to prediction server: " + err.message);
            }
        });
    }
});
