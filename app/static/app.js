/**
 * VoiceRAG — Frontend Application Logic
 * Handles mic recording, text queries, and response rendering.
 */

(function () {
    'use strict';

    // ── DOM Elements ──────────────────────────────────────────
    const queryInput = document.getElementById('query-input');
    const sendBtn = document.getElementById('send-btn');
    const micBtn = document.getElementById('mic-btn');
    const micHint = document.getElementById('mic-hint');
    const strategySelect = document.getElementById('strategy-select');
    const topkInput = document.getElementById('topk-input');
    const responseSection = document.getElementById('response-section');
    const answerText = document.getElementById('answer-text');
    const languageBadge = document.getElementById('language-badge');
    const latencyGrid = document.getElementById('latency-grid');
    const latencyTotal = document.getElementById('latency-total');
    const passagesList = document.getElementById('passages-list');
    const guardrailsList = document.getElementById('guardrails-list');
    const loadingOverlay = document.getElementById('loading-overlay');
    const headerStatus = document.getElementById('header-status');

    // ── State ─────────────────────────────────────────────────
    let isRecording = false;
    let mediaRecorder = null;
    let audioChunks = [];

    // ── Initialize ────────────────────────────────────────────
    async function init() {
        await checkHealth();
        bindEvents();
    }

    async function checkHealth() {
        const statusDot = headerStatus.querySelector('.status-dot');
        const statusText = headerStatus.querySelector('.status-text');

        try {
            const resp = await fetch('/api/health');
            const data = await resp.json();

            if (data.status === 'healthy') {
                statusDot.className = 'status-dot connected';
                statusText.textContent = `Ready · ${data.index_size.toLocaleString()} chunks`;
            } else {
                statusDot.className = 'status-dot';
                statusText.textContent = 'Degraded';
            }
        } catch {
            statusDot.className = 'status-dot error';
            statusText.textContent = 'Offline';
        }
    }

    function bindEvents() {
        sendBtn.addEventListener('click', handleTextQuery);
        queryInput.addEventListener('keydown', (e) => {
            if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault();
                handleTextQuery();
            }
        });

        micBtn.addEventListener('click', toggleRecording);
    }

    // ── Text Query ────────────────────────────────────────────
    async function handleTextQuery() {
        const query = queryInput.value.trim();
        if (!query) return;

        showLoading();

        try {
            const resp = await fetch('/api/query/text', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    query,
                    strategy: strategySelect.value,
                    top_k: parseInt(topkInput.value) || 5,
                }),
            });

            const data = await resp.json();
            renderResponse(data);
        } catch (err) {
            renderError('Failed to process query: ' + err.message);
        } finally {
            hideLoading();
        }
    }

    // ── Voice Recording ───────────────────────────────────────
    async function toggleRecording() {
        if (isRecording) {
            stopRecording();
        } else {
            await startRecording();
        }
    }

    async function startRecording() {
        try {
            const stream = await navigator.mediaDevices.getUserMedia({
                audio: {
                    sampleRate: 16000,
                    channelCount: 1,
                    echoCancellation: true,
                    noiseSuppression: true,
                },
            });

            audioChunks = [];

            // Prefer webm/opus, fallback to whatever is available
            const mimeType = MediaRecorder.isTypeSupported('audio/webm;codecs=opus')
                ? 'audio/webm;codecs=opus'
                : 'audio/webm';

            mediaRecorder = new MediaRecorder(stream, { mimeType });

            mediaRecorder.ondataavailable = (e) => {
                if (e.data.size > 0) audioChunks.push(e.data);
            };

            mediaRecorder.onstop = async () => {
                stream.getTracks().forEach((t) => t.stop());
                const blob = new Blob(audioChunks, { type: mimeType });
                await sendVoiceQuery(blob);
            };

            mediaRecorder.start(100); // collect data every 100ms
            isRecording = true;
            micBtn.classList.add('recording');
            micHint.textContent = 'Recording... Click to stop';
        } catch (err) {
            console.error('Mic access denied:', err);
            micHint.textContent = 'Microphone access denied';
        }
    }

    function stopRecording() {
        if (mediaRecorder && mediaRecorder.state !== 'inactive') {
            mediaRecorder.stop();
        }
        isRecording = false;
        micBtn.classList.remove('recording');
        micHint.textContent = 'Click mic to start recording';
    }

    async function sendVoiceQuery(audioBlob) {
        showLoading();

        try {
            const formData = new FormData();
            formData.append('audio', audioBlob, 'recording.webm');
            formData.append('strategy', strategySelect.value);
            formData.append('top_k', topkInput.value || '5');

            const resp = await fetch('/api/query/voice', {
                method: 'POST',
                body: formData,
            });

            const data = await resp.json();
            renderResponse(data);
        } catch (err) {
            renderError('Voice query failed: ' + err.message);
        } finally {
            hideLoading();
        }
    }

    // ── Render Response ───────────────────────────────────────
    function renderResponse(data) {
        responseSection.style.display = '';

        // Answer
        answerText.textContent = data.answer || 'No answer available';

        // Language badge
        if (data.language) {
            languageBadge.textContent = data.language;
            languageBadge.style.display = '';
        } else {
            languageBadge.style.display = 'none';
        }

        // Latency
        renderLatency(data);

        // Passages
        renderPassages(data);

        // Guardrails
        renderGuardrails(data);

        // Scroll to response
        responseSection.scrollIntoView({ behavior: 'smooth', block: 'start' });
    }

    function renderLatency(data) {
        latencyGrid.innerHTML = '';

        const stages = data.stage_latencies || {};
        const stageLabels = {
            stt: 'STT (Sarvam)',
            embedding: 'Embedding',
            retrieval: 'Retrieval',
            generation: 'Generation',
            guardrail_pre: 'Pre-Guard',
            guardrail_post: 'Post-Guard',
        };

        // Add STT if present
        if (data.stt_latency_ms != null) {
            addLatencyItem('STT (separate)', data.stt_latency_ms);
        }

        for (const [key, label] of Object.entries(stageLabels)) {
            if (key === 'stt') continue; // handled above
            if (stages[key] != null) {
                addLatencyItem(label, stages[key]);
            }
        }

        // Pipeline total
        const pipelineMs = data.pipeline_latency_ms || 0;
        const overBudget = pipelineMs > 200;
        latencyTotal.className = 'latency-total' + (overBudget ? ' over-budget' : '');
        latencyTotal.innerHTML = `
            <span>Pipeline Total</span>
            <span style="font-weight: 700; color: ${overBudget ? 'var(--accent-red)' : 'var(--accent-green)'}">
                ${pipelineMs.toFixed(1)}ms ${overBudget ? '⚠️' : '✅'} ${overBudget ? '(>200ms)' : '(<200ms)'}
            </span>
        `;
    }

    function addLatencyItem(label, ms) {
        const speedClass = ms < 50 ? 'fast' : ms < 150 ? 'medium' : 'slow';
        const displayVal = ms < 0.05 ? '<0.1ms' : `${ms.toFixed(1)}ms`;
        const el = document.createElement('div');
        el.className = 'latency-item';
        el.innerHTML = `
            <div class="latency-label">${label}</div>
            <div class="latency-value ${speedClass}">${displayVal}</div>
        `;
        latencyGrid.appendChild(el);
    }

    function renderPassages(data) {
        passagesList.innerHTML = '';

        const chunks = data.retrieval_result?.chunks || [];
        if (chunks.length === 0) {
            passagesList.innerHTML = '<p style="color: var(--text-muted); font-size: 0.85rem;">No passages retrieved</p>';
            return;
        }

        chunks.forEach((item, idx) => {
            const text = item.expanded_text || item.chunk?.text || '';
            const score = item.score || 0;
            const strategy = item.chunk?.strategy || 'unknown';

            const el = document.createElement('div');
            el.className = 'passage-item';
            el.innerHTML = `
                <div class="passage-header">
                    <span class="passage-strategy">${strategy} · Passage ${idx + 1}</span>
                    <span class="passage-score">Score: ${score.toFixed(3)}</span>
                </div>
                <div class="passage-text">${escapeHtml(text.slice(0, 400))}${text.length > 400 ? '…' : ''}</div>
            `;
            passagesList.appendChild(el);
        });
    }

    function renderGuardrails(data) {
        guardrailsList.innerHTML = '';

        const checks = data.guardrail_results || [];
        if (checks.length === 0) {
            guardrailsList.innerHTML = '<span class="guardrail-badge passed">✓ No checks triggered</span>';
            return;
        }

        checks.forEach((check) => {
            const el = document.createElement('span');
            el.className = `guardrail-badge ${check.passed ? 'passed' : 'failed'}`;
            const icon = check.passed ? '✓' : '✗';
            const type = (check.guardrail_type || 'check').replace(/_/g, ' ');
            el.textContent = `${icon} ${type}`;
            if (check.reason) {
                el.title = check.reason;
            }
            guardrailsList.appendChild(el);
        });
    }

    function renderError(message) {
        responseSection.style.display = '';
        answerText.textContent = message;
        answerText.style.color = 'var(--accent-red)';
        latencyGrid.innerHTML = '';
        latencyTotal.innerHTML = '';
        passagesList.innerHTML = '';
        guardrailsList.innerHTML = '';

        // Reset color after display
        setTimeout(() => {
            answerText.style.color = '';
        }, 5000);
    }

    // ── Helpers ────────────────────────────────────────────────
    function showLoading() {
        loadingOverlay.style.display = '';
        responseSection.style.display = 'none';
    }

    function hideLoading() {
        loadingOverlay.style.display = 'none';
    }

    function escapeHtml(text) {
        const div = document.createElement('div');
        div.textContent = text;
        return div.innerHTML;
    }

    // ── Boot ──────────────────────────────────────────────────
    document.addEventListener('DOMContentLoaded', init);
})();
