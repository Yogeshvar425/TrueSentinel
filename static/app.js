const analyzeBtn = document.getElementById('analyze_btn');
const urlInput = document.getElementById('youtube_url');
const statusText = document.getElementById('status_text');
const dashboard = document.getElementById('results_dashboard');
const btnText = document.querySelector('.btn-text');
const spinner = document.getElementById('loading_spinner');

const emojiMap = {
    "joy": "😄", "anger": "😠", "sadness": "😢", 
    "disgust": "🤢", "fear": "😨", "surprise": "😲", "neutral": "😐",
    "positive": "👍", "negative": "👎"
};

analyzeBtn.addEventListener('click', async () => {
    const url = urlInput.value.trim();
    if(!url) return alert("Please enter a YouTube URL!");

    // UI Reactivity
    btnText.classList.add('default-hide');
    spinner.classList.remove('default-hide');
    statusText.classList.remove('default-hide');
    dashboard.classList.add('default-hide');
    analyzeBtn.disabled = true;

    try {
        const response = await fetch('/api/analyze', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ url })
        });

        const data = await response.json();

        if(!response.ok) {
            throw new Error(data.detail || "Error analyzing video");
        }

        renderDashboard(data);

    } catch (err) {
        alert(err.message);
    } finally {
        btnText.classList.remove('default-hide');
        spinner.classList.add('default-hide');
        statusText.classList.add('default-hide');
        analyzeBtn.disabled = false;
    }
});

function renderDashboard(data) {
    // Reveal Dashboard
    dashboard.classList.remove('default-hide');

    // 1. Meta Stats
    document.getElementById('stat_analyzed').innerText = data.total_analyzed;
    document.getElementById('stat_spam').innerText = data.spam_filtered;
    
    // Model Metrics
    if(data.metrics) {
        document.getElementById('log_accuracy').innerText = data.metrics.logistic_accuracy;
        document.getElementById('svm_accuracy').innerText = data.metrics.svm_accuracy;
    }

    // 2. Topics
    const topicsUl = document.getElementById('topics_list');
    topicsUl.innerHTML = '';
    if(data.topics.length === 0) {
        topicsUl.innerHTML = '<li>No clear topics extracted.</li>';
    } else {
        data.topics.forEach((topic, idx) => {
            const li = document.createElement('li');
            li.innerText = `Topic ${idx+1}: [ ${topic} ]`;
            topicsUl.appendChild(li);
        });
    }

    // 3. Emotion Bars
    const barsContainer = document.getElementById('emotion_bars');
    barsContainer.innerHTML = '';
    data.emotions.forEach(emo => {
        const row = document.createElement('div');
        row.className = 'bar-row';
        
        const emoji = emojiMap[emo.name] || '•';
        
        row.innerHTML = `
            <div class="bar-label">${emoji} ${emo.name}</div>
            <div class="bar-track">
                <div class="bar-fill fill-${emo.name}" style="width: 0%"></div>
            </div>
            <div class="bar-pct">${emo.percentage.toFixed(1)}%</div>
        `;
        barsContainer.appendChild(row);

        // Animate Growth smoothly
        setTimeout(() => {
            row.querySelector('.bar-fill').style.width = `${emo.percentage}%`;
        }, 100);
    });

    // 4. Raw Table
    const tbody = document.getElementById('comments_tbody');
    tbody.innerHTML = '';
    data.detailed_comments.forEach(log => {
        const tr = document.createElement('tr');
        const emoji = emojiMap[log.emotion] || '•';
        
        tr.innerHTML = `
            <td>${emoji} <span style="text-transform: capitalize">${log.emotion}</span></td>
            <td class="td-score">${log.confidence}</td>
            <td class="td-comment">"${log.comment}"</td>
        `;
        tbody.appendChild(tr);
    });
}
