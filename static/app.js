/**
 * TrueSentinel — Frontend Application v2.0
 * ==========================================
 * Pure JS: donut chart, animated counters, progress state machine,
 * toast notifications, comment filtering, and impact stats.
 * Zero external dependencies.
 */

// ─── DOM References ────────────────────────────────────────────
const $ = (sel) => document.querySelector(sel);
const $$ = (sel) => document.querySelectorAll(sel);

const analyzeBtn = $('#analyze-btn');
const urlInput = $('#youtube-url');
const dashboard = $('#dashboard');
const progressSteps = $('#progress-steps');
const toastContainer = $('#toast-container');
const commentsFeed = $('#comments-feed');

// State
let allComments = [];
let currentFilter = 'all';

// ─── Initialize: Fetch cumulative stats on load ────────────────
(async function loadImpactStats() {
    try {
        const res = await fetch('/api/health');
        if (!res.ok) return;
        const data = await res.json();

        const stats = data.cumulative_stats || {};
        animateCounter($('#impact-analyses'), stats.total_analyses || 0);
        animateCounter($('#impact-comments'), stats.total_comments_processed || 0);
        animateCounter($('#impact-spam'), stats.total_spam_blocked || 0);
    } catch {
        // Server might not be ready yet — silent fail
    }
})();

// ─── Event Listeners ───────────────────────────────────────────
analyzeBtn.addEventListener('click', runAnalysis);
urlInput.addEventListener('keydown', (e) => {
    if (e.key === 'Enter') runAnalysis();
});

// Comment filter buttons
$$('.filter-btn').forEach(btn => {
    btn.addEventListener('click', () => {
        $$('.filter-btn').forEach(b => b.classList.remove('active'));
        btn.classList.add('active');
        currentFilter = btn.dataset.filter;
        renderComments(allComments, currentFilter);
    });
});

// ─── Main Analysis Flow ────────────────────────────────────────
async function runAnalysis() {
    const url = urlInput.value.trim();
    if (!url) {
        showToast('Please paste a YouTube URL first.', 'error');
        urlInput.focus();
        return;
    }

    // UI: loading state
    analyzeBtn.classList.add('btn-loading');
    analyzeBtn.disabled = true;
    dashboard.classList.remove('visible');
    showProgress();

    try {
        // Step 1: Fetching
        setStep('fetch');
        await sleep(300);

        const response = await fetch('/api/analyze', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ url })
        });

        // Step 2: Filtering
        setStep('filter');
        await sleep(200);

        const data = await response.json();

        if (!response.ok) {
            throw new Error(data.detail || 'Analysis failed');
        }

        // Step 3: Vectorizing
        setStep('vectorize');
        await sleep(200);

        // Step 4: Predicting
        setStep('predict');
        await sleep(300);

        // All done
        completeSteps();
        await sleep(400);

        renderDashboard(data);
        showToast(`Analyzed ${data.total_analyzed} comments in ${data.inference_time_ms}ms`, 'success');

    } catch (err) {
        showToast(err.message, 'error');
        hideProgress();
    } finally {
        analyzeBtn.classList.remove('btn-loading');
        analyzeBtn.disabled = false;
    }
}

// ─── Dashboard Renderer ────────────────────────────────────────
function renderDashboard(data) {
    dashboard.classList.add('visible');

    // Stagger card reveal animations
    const cards = dashboard.querySelectorAll('.card');
    cards.forEach((card, i) => {
        card.classList.remove('reveal');
        setTimeout(() => card.classList.add('reveal'), 100 + i * 120);
    });

    // ── Stats ──
    animateCounter($('#stat-analyzed'), data.total_analyzed);
    animateCounter($('#stat-spam'), data.spam_filtered);
    $('#stat-inference-time').textContent = `${data.inference_time_ms}ms`;

    // ── Update impact bar ──
    if (data.cumulative_stats) {
        animateCounter($('#impact-analyses'), data.cumulative_stats.total_analyses);
        animateCounter($('#impact-comments'), data.cumulative_stats.total_comments_processed);
        animateCounter($('#impact-spam'), data.cumulative_stats.total_spam_blocked);
    }

    // ── Donut Chart ──
    renderDonut(data.emotions);

    // ── Training Metrics ──
    if (data.training_metrics) {
        const lr = data.training_metrics.logistic_regression;
        const svm = data.training_metrics.svm;

        $('#lr-accuracy').textContent = `${lr.accuracy}%`;
        $('#lr-f1').textContent = `${lr.f1_score}%`;
        $('#lr-precision').textContent = `${lr.precision}%`;
        $('#lr-recall').textContent = `${lr.recall}%`;

        $('#svm-accuracy').textContent = `${svm.accuracy}%`;
        $('#svm-f1').textContent = `${svm.f1_score}%`;
        $('#svm-precision').textContent = `${svm.precision}%`;
        $('#svm-recall').textContent = `${svm.recall}%`;

        const ds = data.training_metrics.dataset;
        const trainedDate = new Date(data.training_metrics.trained_at).toLocaleDateString('en-US', {
            year: 'numeric', month: 'short', day: 'numeric'
        });
        $('#metrics-footer').innerHTML = `
            <span>Trained on ${ds.labeled_samples} labeled samples from ${ds.source_videos} videos</span>
            <span>Last trained: ${trainedDate}</span>
        `;
    } else {
        $('#metrics-footer').innerHTML = '<span style="color: var(--accent-amber);">⚠ No metrics.json found — run train.py to generate real metrics</span>';
    }

    // ── Topics ──
    const topicsGrid = $('#topics-grid');
    topicsGrid.innerHTML = '';
    if (data.topics && data.topics.length > 0) {
        data.topics.forEach((topic, i) => {
            const chip = document.createElement('div');
            chip.className = 'topic-chip';
            chip.innerHTML = `<span class="topic-number">${i + 1}</span><span>${topic}</span>`;
            topicsGrid.appendChild(chip);
        });
    } else {
        topicsGrid.innerHTML = '<span style="color: var(--text-muted); font-size: 0.85rem;">Not enough varied text to extract clear topics.</span>';
    }

    // ── Comments ──
    allComments = data.detailed_comments || [];
    currentFilter = 'all';
    $$('.filter-btn').forEach(b => b.classList.remove('active'));
    $('.filter-btn[data-filter="all"]').classList.add('active');
    renderComments(allComments, 'all');
}

// ─── Donut Chart (Pure Canvas) ─────────────────────────────────
function renderDonut(emotions) {
    const canvas = $('#donut-canvas');
    const ctx = canvas.getContext('2d');

    const dpr = window.devicePixelRatio || 1;
    const size = 220;
    canvas.width = size * dpr;
    canvas.height = size * dpr;
    canvas.style.width = size + 'px';
    canvas.style.height = size + 'px';
    ctx.scale(dpr, dpr);

    const cx = size / 2;
    const cy = size / 2;
    const outerR = 95;
    const innerR = 62;

    const colorMap = {
        positive: '#10b981',
        negative: '#ef4444',
        neutral: '#94a3b8'
    };

    const total = emotions.reduce((s, e) => s + e.count, 0);
    if (total === 0) return;

    // Find dominant sentiment
    const dominant = emotions[0];
    $('#donut-dominant-pct').textContent = `${dominant.percentage.toFixed(0)}%`;
    $('#donut-dominant-label').textContent = dominant.name;

    // Animate drawing
    const targetAngles = emotions.map(e => (e.count / total) * Math.PI * 2);
    let progress = 0;

    function drawFrame() {
        progress = Math.min(progress + 0.03, 1);
        const eased = easeOutCubic(progress);

        ctx.clearRect(0, 0, size, size);

        let startAngle = -Math.PI / 2;
        emotions.forEach((emo, i) => {
            const sweep = targetAngles[i] * eased;
            const color = colorMap[emo.name] || '#8b5cf6';

            ctx.beginPath();
            ctx.arc(cx, cy, outerR, startAngle, startAngle + sweep);
            ctx.arc(cx, cy, innerR, startAngle + sweep, startAngle, true);
            ctx.closePath();
            ctx.fillStyle = color;
            ctx.fill();

            // Subtle gap between segments
            if (emotions.length > 1) {
                ctx.beginPath();
                ctx.arc(cx, cy, outerR, startAngle + sweep - 0.01, startAngle + sweep + 0.01);
                ctx.arc(cx, cy, innerR, startAngle + sweep + 0.01, startAngle + sweep - 0.01, true);
                ctx.closePath();
                ctx.fillStyle = '#06080f';
                ctx.fill();
            }

            startAngle += sweep;
        });

        if (progress < 1) {
            requestAnimationFrame(drawFrame);
        }
    }

    drawFrame();

    // Legend
    const legend = $('#donut-legend');
    legend.innerHTML = '';
    emotions.forEach(emo => {
        const color = colorMap[emo.name] || '#8b5cf6';
        const item = document.createElement('div');
        item.className = 'legend-item';
        item.innerHTML = `
            <span class="legend-dot" style="background: ${color}"></span>
            <span>${capitalize(emo.name)}</span>
            <span class="legend-value">${emo.percentage.toFixed(1)}%</span>
        `;
        legend.appendChild(item);
    });
}

// ─── Comment Rendering ─────────────────────────────────────────
function renderComments(comments, filter) {
    commentsFeed.innerHTML = '';

    const filtered = filter === 'all'
        ? comments
        : comments.filter(c => c.emotion === filter);

    if (filtered.length === 0) {
        commentsFeed.innerHTML = '<div style="text-align:center; color: var(--text-muted); padding: 2rem; font-size: 0.85rem;">No comments match this filter.</div>';
        return;
    }

    filtered.forEach((c, i) => {
        const card = document.createElement('div');
        card.className = `comment-card ${c.emotion}`;
        card.style.animationDelay = `${Math.min(i * 30, 500)}ms`;

        const emoji = c.emotion === 'positive' ? '👍' : '👎';
        const confidence = (c.confidence * 100).toFixed(0);

        card.innerHTML = `
            <span class="comment-emoji">${emoji}</span>
            <div class="comment-body">
                <p class="comment-text">"${escapeHtml(c.comment)}"</p>
                <div class="comment-meta">
                    <span class="comment-badge ${c.emotion}">${c.emotion}</span>
                    <span class="comment-confidence">${confidence}% confidence</span>
                </div>
            </div>
        `;
        commentsFeed.appendChild(card);
    });
}

// ─── Progress Steps ────────────────────────────────────────────
const stepIds = ['step-fetch', 'step-filter', 'step-vectorize', 'step-predict'];
const steps = stepIds.map(id => $(`#${id}`));

function showProgress() {
    progressSteps.classList.add('visible');
    steps.forEach(s => { s.classList.remove('active', 'done'); });
}

function hideProgress() {
    progressSteps.classList.remove('visible');
}

function setStep(name) {
    const idx = stepIds.indexOf(`step-${name}`);
    steps.forEach((s, i) => {
        s.classList.remove('active');
        if (i < idx) s.classList.add('done');
        if (i === idx) s.classList.add('active');
    });
}

function completeSteps() {
    steps.forEach(s => {
        s.classList.remove('active');
        s.classList.add('done');
    });
}

// ─── Toast Notifications ───────────────────────────────────────
function showToast(message, type = 'info') {
    const toast = document.createElement('div');
    toast.className = `toast ${type}`;
    toast.textContent = message;
    toastContainer.appendChild(toast);

    setTimeout(() => {
        toast.classList.add('removing');
        setTimeout(() => toast.remove(), 300);
    }, 4000);
}

// ─── Animated Counter ──────────────────────────────────────────
function animateCounter(el, target) {
    if (!el) return;
    target = parseInt(target) || 0;
    const start = parseInt(el.textContent) || 0;
    const diff = target - start;
    if (diff === 0) { el.textContent = formatNumber(target); return; }

    const duration = 800;
    const startTime = performance.now();

    function tick(now) {
        const elapsed = now - startTime;
        const progress = Math.min(elapsed / duration, 1);
        const eased = easeOutCubic(progress);
        el.textContent = formatNumber(Math.round(start + diff * eased));
        if (progress < 1) requestAnimationFrame(tick);
    }

    requestAnimationFrame(tick);
}

// ─── Utilities ─────────────────────────────────────────────────
function sleep(ms) {
    return new Promise(r => setTimeout(r, ms));
}

function easeOutCubic(t) {
    return 1 - Math.pow(1 - t, 3);
}

function capitalize(str) {
    return str.charAt(0).toUpperCase() + str.slice(1);
}

function formatNumber(n) {
    if (n >= 10000) return (n / 1000).toFixed(1) + 'k';
    return n.toLocaleString();
}

function escapeHtml(str) {
    const div = document.createElement('div');
    div.textContent = str;
    return div.innerHTML;
}
