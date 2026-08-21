/**
 * INDICVOICE — Voice-Enabled Multilingual RAG Assistant
 * Production-ready Vanilla JavaScript Application
 */

(function () {
    'use strict';

    // ── DOM Elements ──────────────────────────────────────────────────────────
    const micHeroBtn = document.getElementById('mic-hero-btn');
    const composerMicBtn = document.getElementById('composer-mic-btn');
    const sendBtn = document.getElementById('send-btn');
    const composerInput = document.getElementById('composer-input');
    const conversationArea = document.getElementById('conversation-area');
    const voiceActionTitle = document.getElementById('voice-action-title');
    const voiceStatusLabel = document.getElementById('voice-status-label');
    const micWrapper = document.querySelector('.mic-wrapper');

    // ── State Variables ───────────────────────────────────────────────────────
    let isListening = false;
    let recognition = null;
    let mediaRecorder = null;
    let audioChunks = [];
    let audioStream = null;
    let isProcessing = false;

    // ── Initial Setup ─────────────────────────────────────────────────────────
    document.addEventListener('DOMContentLoaded', () => {
        initSpeechRecognition();
        bindEvents();
        autoResizeTextarea();
        initVideoLoop();
    });

    // ── Speech Recognition Initialization ─────────────────────────────────────
    function initSpeechRecognition() {
        const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;

        if (!SpeechRecognition) {
            console.warn('Browser SpeechRecognition API not supported. Falling back to MediaRecorder.');
            return;
        }

        try {
            recognition = new SpeechRecognition();
            recognition.continuous = false;
            recognition.interimResults = true;

            recognition.onstart = () => {
                setVoiceState('LISTENING');
            };

            recognition.onresult = (event) => {
                let interimTranscript = '';
                let finalTranscript = '';

                for (let i = event.resultIndex; i < event.results.length; ++i) {
                    if (event.results[i].isFinal) {
                        finalTranscript += event.results[i][0].transcript;
                    } else {
                        interimTranscript += event.results[i][0].transcript;
                    }
                }

                const currentText = finalTranscript || interimTranscript;
                if (currentText) {
                    composerInput.value = currentText;
                    autoResizeTextarea();
                }
            };

            recognition.onend = () => {
                if (isListening) {
                    stopListening(true);
                }
            };

            recognition.onerror = (event) => {
                console.error('Speech recognition error:', event.error);
                if (event.error !== 'no-speech') {
                    setVoiceState('ERROR');
                }
                stopListening(false);
            };
        } catch (e) {
            console.error('Failed to initialize SpeechRecognition:', e);
        }
    }

    // ── Bind Event Listeners ──────────────────────────────────────────────────
    function bindEvents() {
        // Mic hero button toggle
        if (micHeroBtn) micHeroBtn.addEventListener('click', toggleVoiceInteraction);
        if (composerMicBtn) composerMicBtn.addEventListener('click', toggleVoiceInteraction);

        // Send text query
        sendBtn.addEventListener('click', handleTextSubmit);

        // Composer enter key
        composerInput.addEventListener('keydown', (e) => {
            if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault();
                handleTextSubmit();
            }
        });

        // Input height auto-resize
        composerInput.addEventListener('input', autoResizeTextarea);

        // Side Navigation Drawer (Sandwich Bar)
        const drawerToggle = document.getElementById('drawer-toggle');
        const drawerCloseBtn = document.getElementById('drawer-close-btn');
        const drawerOverlay = document.getElementById('drawer-overlay');
        const sidenavDrawer = document.getElementById('sidenav-drawer');

        function openDrawer() {
            if (sidenavDrawer) sidenavDrawer.classList.add('open');
            if (drawerOverlay) drawerOverlay.classList.add('open');
        }

        function closeDrawer() {
            if (sidenavDrawer) sidenavDrawer.classList.remove('open');
            if (drawerOverlay) drawerOverlay.classList.remove('open');
        }

        if (drawerToggle) drawerToggle.addEventListener('click', openDrawer);
        if (drawerCloseBtn) drawerCloseBtn.addEventListener('click', closeDrawer);
        if (drawerOverlay) drawerOverlay.addEventListener('click', closeDrawer);
    }

    // ── Voice Toggle Logic ────────────────────────────────────────────────────
    async function toggleVoiceInteraction() {
        if (isProcessing) return;

        if (isListening) {
            stopListening(true);
        } else {
            await startListening();
        }
    }

    async function startListening() {
        audioChunks = [];
        isListening = true;
        setVoiceState('LISTENING');

        // 1. Try Browser Speech Recognition for real-time transcript
        if (recognition) {
            try {
                recognition.start();
            } catch (e) {
                console.warn('SpeechRecognition start error:', e);
            }
        }

        // 2. Start MediaRecorder for audio payload to Sarvam STT backend
        try {
            audioStream = await navigator.mediaDevices.getUserMedia({
                audio: {
                    sampleRate: 16000,
                    channelCount: 1,
                    echoCancellation: true,
                    noiseSuppression: true,
                },
            });

            const mimeType = MediaRecorder.isTypeSupported('audio/webm;codecs=opus')
                ? 'audio/webm;codecs=opus'
                : 'audio/webm';

            mediaRecorder = new MediaRecorder(audioStream, { mimeType });

            mediaRecorder.ondataavailable = (e) => {
                if (e.data && e.data.size > 0) {
                    audioChunks.push(e.data);
                }
            };

            mediaRecorder.start(100);
        } catch (err) {
            console.warn('Microphone stream access unavailable:', err);
            if (!recognition) {
                setVoiceState('ERROR', 'MICROPHONE ACCESS DENIED');
                voiceStatusLabel.textContent = 'Voice input unavailable — type your question below.';
                isListening = false;
            }
        }
    }

    function stopListening(processInput = true) {
        isListening = false;

        if (recognition) {
            try {
                recognition.stop();
            } catch (e) {
                // Ignore
            }
        }

        if (mediaRecorder && mediaRecorder.state !== 'inactive') {
            mediaRecorder.onstop = async () => {
                if (audioStream) {
                    audioStream.getTracks().forEach((track) => track.stop());
                    audioStream = null;
                }

                if (processInput) {
                    const audioBlob = new Blob(audioChunks, { type: 'audio/webm' });
                    const transcriptText = composerInput.value.trim();

                    if (audioBlob.size > 1000) {
                        await processVoiceQuery(audioBlob, transcriptText);
                    } else if (transcriptText) {
                        await processTextQuery(transcriptText);
                    } else {
                        setVoiceState('IDLE');
                    }
                } else {
                    setVoiceState('IDLE');
                }
            };
            mediaRecorder.stop();
        } else {
            const transcriptText = composerInput.value.trim();
            if (processInput && transcriptText) {
                processTextQuery(transcriptText);
            } else {
                setVoiceState('IDLE');
            }
        }
    }

    // ── Background Video Sequential Loop ─────────────────────────────────────────
    let currentVideoIndex = 0;

    function initVideoLoop() {
        const bgVideos = document.querySelectorAll('.bg-video');
        bgVideos.forEach((vid, i) => {
            vid.addEventListener('ended', () => {
                const nextIndex = (i + 1) % bgVideos.length;
                playVideoAtIndex(nextIndex);
            });
        });
        playVideoAtIndex(0);
    }

    function playVideoAtIndex(index) {
        currentVideoIndex = index;
        const bgVideos = document.querySelectorAll('.bg-video');
        bgVideos.forEach((vid, i) => {
            if (i === index) {
                vid.classList.add('active', 'opacity-100');
                vid.classList.remove('opacity-0');
                try {
                    vid.currentTime = 0;
                    vid.play().catch(e => console.warn('Video play error:', e));
                } catch (e) {
                    console.warn(e);
                }
            } else {
                vid.classList.remove('active', 'opacity-100');
                vid.classList.add('opacity-0');
            }
        });
    }

    function setActiveVideo(index) {
        playVideoAtIndex(index);
    }

    // ── Voice State UI Management ──────────────────────────────────────────────
    function setVoiceState(state, customLabel = null) {
        switch (state) {
            case 'IDLE':
                micHeroBtn.classList.remove('listening');
                micWrapper.classList.remove('listening');
                voiceActionTitle.textContent = 'TAP TO SPEAK';
                voiceStatusLabel.textContent = customLabel || 'READY TO LISTEN';
                setActiveVideo(0);
                break;

            case 'LISTENING':
                micHeroBtn.classList.add('listening');
                micWrapper.classList.add('listening');
                voiceActionTitle.textContent = 'TAP TO STOP';
                voiceStatusLabel.textContent = customLabel || 'LISTENING…';
                setActiveVideo(1);
                break;

            case 'PROCESSING':
                micHeroBtn.classList.remove('listening');
                micWrapper.classList.remove('listening');
                voiceActionTitle.textContent = 'PROCESSING';
                voiceStatusLabel.textContent = customLabel || 'RETRIEVING KNOWLEDGE…';
                setActiveVideo(2);
                break;

            case 'ANSWERED':
                micHeroBtn.classList.remove('listening');
                micWrapper.classList.remove('listening');
                voiceActionTitle.textContent = 'TAP TO SPEAK';
                voiceStatusLabel.textContent = customLabel || 'ANSWER READY';
                setActiveVideo(3);
                break;

            case 'ERROR':
                micHeroBtn.classList.remove('listening');
                micWrapper.classList.remove('listening');
                voiceActionTitle.textContent = 'TAP TO SPEAK';
                voiceStatusLabel.textContent = customLabel || 'TRY AGAIN';
                setActiveVideo(0);
                break;
        }
    }

    // ── Centralized API Handler ───────────────────────────────────────────────
    async function askAssistant(queryOrAudio, inputMode = 'text') {
        if (inputMode === 'voice') {
            const formData = new FormData();
            formData.append('audio', queryOrAudio, 'voice_query.webm');
            formData.append('strategy', 'fixed');
            formData.append('top_k', '5');

            // Primary endpoint /api/v1/query/voice with fallback to /api/query/voice
            let res = await fetch('/api/v1/query/voice', {
                method: 'POST',
                body: formData,
            });

            if (!res.ok) {
                res = await fetch('/api/query/voice', {
                    method: 'POST',
                    body: formData,
                });
            }

            if (!res.ok) {
                throw new Error(`Voice query API failed with status ${res.status}`);
            }

            return await res.json();
        } else {
            const payload = JSON.stringify({
                query: queryOrAudio,
                strategy: 'fixed',
                top_k: 5,
            });

            let res = await fetch('/api/v1/query', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: payload,
            });

            if (!res.ok) {
                res = await fetch('/api/query/text', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: payload,
                });
            }

            if (!res.ok) {
                throw new Error(`Text query API failed with status ${res.status}`);
            }

            return await res.json();
        }
    }

    // ── Query Handlers ────────────────────────────────────────────────────────
    async function handleTextSubmit() {
        const query = composerInput.value.trim();
        if (!query || isProcessing) return;

        composerInput.value = '';
        autoResizeTextarea();
        await processTextQuery(query);
    }

    async function processTextQuery(query) {
        isProcessing = true;
        setVoiceState('PROCESSING', 'RETRIEVING KNOWLEDGE…');

        const turnId = appendUserMessage(query);

        try {
            setTimeout(() => {
                if (isProcessing) voiceStatusLabel.textContent = 'SYNTHESIZING ANSWER…';
            }, 600);

            const rawData = await askAssistant(query, 'text');
            const mapped = mapBackendResponse(rawData);

            renderAssistantResponse(turnId, mapped);
            setVoiceState('ANSWERED');
        } catch (err) {
            console.error('Text query processing error:', err);
            renderAssistantError(turnId, 'Something went wrong. Please try again.');
            setVoiceState('ERROR');
        } finally {
            isProcessing = false;
        }
    }

    async function processVoiceQuery(audioBlob, fallbackQueryText) {
        isProcessing = true;
        setVoiceState('PROCESSING', 'TRANSCRIBING VOICE…');

        // Place temporary turn
        const initialLabel = fallbackQueryText || 'Voice Input';
        const turnId = appendUserMessage(initialLabel);

        try {
            setTimeout(() => {
                if (isProcessing) voiceStatusLabel.textContent = 'RETRIEVING KNOWLEDGE…';
            }, 500);

            let rawData;
            try {
                rawData = await askAssistant(audioBlob, 'voice');
            } catch (vErr) {
                console.warn('Voice endpoint failed, attempting fallback to text query:', vErr);
                if (fallbackQueryText) {
                    rawData = await askAssistant(fallbackQueryText, 'text');
                } else {
                    throw vErr;
                }
            }

            const mapped = mapBackendResponse(rawData);

            // Update user message if transcript returned
            if (mapped.sttTranscript) {
                updateUserMessage(turnId, mapped.sttTranscript);
            }

            renderAssistantResponse(turnId, mapped);
            setVoiceState('ANSWERED');
        } catch (err) {
            console.error('Voice query processing error:', err);
            renderAssistantError(turnId, 'Something went wrong. Please try again.');
            setVoiceState('ERROR');
        } finally {
            isProcessing = false;
        }
    }

    // ── Response Mapper ───────────────────────────────────────────────────────
    function mapBackendResponse(data) {
        const answer = data.answer || 'No answer generated.';
        const status = data.status || 'success';
        const fallback = data.fallback_triggered || (data.generation_result && data.generation_result.fallback_triggered);

        let mode = 'rag';
        if (answer.toLowerCase().includes('speech recognition failed') || answer.toLowerCase().includes('voice input error')) {
            mode = 'voice_error';
        } else if (status === 'filtered' || answer.includes('INSUFFICIENT EVIDENCE') || answer.includes('Off-topic') || answer.includes('unsafe')) {
            mode = 'refusal';
        } else if (fallback || !data.retrieval_result || !data.retrieval_result.chunks || data.retrieval_result.chunks.length === 0) {
            mode = 'groq';
        }

        const latencies = {
            pipeline: data.pipeline_latency_ms || 0,
            stt: data.stt_latency_ms || (data.stt_result ? data.stt_result.latency_ms : 0),
            stages: data.stage_latencies || {},
        };

        const sttTranscript = data.stt_result ? data.stt_result.transcript : null;
        const language = data.language || (data.stt_result ? data.stt_result.language : '');

        return {
            answer,
            mode,
            language,
            latencies,
            sttTranscript,
            raw: data,
        };
    }

    // ── Chat History Store ───────────────────────────────────────────────────
    const historyEntries = [];

    function addHistoryEntry(turnId, userText, mode) {
        if (!userText || userText.trim() === '' || userText === 'Voice Input') return;
        const existingIndex = historyEntries.findIndex(e => e.id === turnId);
        const entryObj = {
            id: turnId,
            text: userText,
            mode: mode || 'rag',
            time: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }),
        };
        if (existingIndex >= 0) {
            historyEntries[existingIndex] = entryObj;
        } else {
            historyEntries.unshift(entryObj);
        }
        renderHistoryDrawer();
    }

    function renderHistoryDrawer() {
        const historyListEl = document.getElementById('drawer-history-list');
        if (!historyListEl) return;

        if (historyEntries.length === 0) {
            historyListEl.innerHTML = `
                <div class="history-empty">
                    <i class="fa-solid fa-comments"></i>
                    <p>No past conversations yet.<br />Ask a question or speak to start!</p>
                </div>
            `;
            return;
        }

        historyListEl.innerHTML = historyEntries.map(entry => {
            let modeBadge = 'RAG';
            if (entry.mode === 'groq') modeBadge = 'GROQ AI';
            else if (entry.mode === 'voice_error') modeBadge = 'ERROR';
            else if (entry.mode === 'refusal') modeBadge = 'REFUSAL';

            return `
                <div class="history-item" data-turn-id="${entry.id}">
                    <div class="history-item-title">${escapeHtml(entry.text)}</div>
                    <div class="history-item-meta">
                        <span>${modeBadge}</span>
                        <span>${entry.time}</span>
                    </div>
                </div>
            `;
        }).join('');

        historyListEl.querySelectorAll('.history-item').forEach(item => {
            item.addEventListener('click', () => {
                const turnId = item.getAttribute('data-turn-id');
                const targetTurn = document.getElementById(turnId);
                if (targetTurn) {
                    targetTurn.scrollIntoView({ behavior: 'smooth', block: 'center' });
                    const drawerOverlay = document.getElementById('drawer-overlay');
                    const sidenavDrawer = document.getElementById('sidenav-drawer');
                    if (sidenavDrawer) sidenavDrawer.classList.remove('open');
                    if (drawerOverlay) drawerOverlay.classList.remove('open');
                }
            });
        });
    }

    // ── UI Rendering Helpers ──────────────────────────────────────────────────
    function appendUserMessage(text) {
        const turnId = 'turn-' + Date.now();
        const turnEl = document.createElement('div');
        turnEl.className = 'turn';
        turnEl.id = turnId;

        turnEl.innerHTML = `
            <div class="user-msg">
                <div class="user-label">YOU</div>
                <div class="user-text">${escapeHtml(text)}</div>
            </div>
            <div class="assistant-placeholder"></div>
        `;

        conversationArea.appendChild(turnEl);
        scrollToBottom();
        addHistoryEntry(turnId, text, 'rag');
        return turnId;
    }

    function updateUserMessage(turnId, newText) {
        const turnEl = document.getElementById(turnId);
        if (turnEl) {
            const userTextEl = turnEl.querySelector('.user-text');
            if (userTextEl) {
                userTextEl.textContent = newText;
                addHistoryEntry(turnId, newText, 'rag');
            }
        }
    }

    function renderAssistantResponse(turnId, mapped) {
        const turnEl = document.getElementById(turnId);
        if (!turnEl) return;

        const userTextEl = turnEl.querySelector('.user-text');
        if (userTextEl) {
            addHistoryEntry(turnId, userTextEl.textContent, mapped.mode);
        }

        const placeholder = turnEl.querySelector('.assistant-placeholder');
        if (!placeholder) return;

        // Determine Badge & Main Latency Label
        let badgeHtml = '';
        let latencyLabel = '';

        if (mapped.mode === 'voice_error') {
            badgeHtml = `<span class="badge badge-refusal"><i class="fa-solid fa-triangle-exclamation"></i> VOICE INPUT ERROR</span>`;
            latencyLabel = mapped.latencies.stt > 0 ? `STT: ${Math.round(mapped.latencies.stt)} ms` : '';
        } else if (mapped.mode === 'rag') {
            badgeHtml = `<span class="badge badge-rag"><i class="fa-solid fa-circle-check"></i> GROUNDED RAG</span>`;
            latencyLabel = `RAG LATENCY: ${Math.round(mapped.latencies.pipeline)} ms`;
        } else if (mapped.mode === 'groq') {
            badgeHtml = `<span class="badge badge-groq"><i class="fa-solid fa-bolt"></i> GROQ AI ANSWER</span>`;
            const retMs = mapped.latencies.stages.retrieval ? Math.round(mapped.latencies.stages.retrieval) : Math.round(mapped.latencies.pipeline);
            latencyLabel = `RETRIEVAL: ${retMs} ms`;
        } else {
            badgeHtml = `<span class="badge badge-refusal"><i class="fa-solid fa-triangle-exclamation"></i> INSUFFICIENT EVIDENCE</span>`;
            latencyLabel = `LATENCY: ${Math.round(mapped.latencies.pipeline)} ms`;
        }

        if (mapped.latencies.stt > 0) {
            const totalE2E = Math.round(mapped.latencies.stt + mapped.latencies.pipeline);
            latencyLabel = `VOICE E2E: ${totalE2E} ms`;
        }

        // Build Pipeline Details Breakdown
        const stageItems = [];
        if (mapped.latencies.stt > 0) {
            stageItems.push({ name: 'STT (Sarvam)', val: `${Math.round(mapped.latencies.stt)} ms` });
        }

        const stages = mapped.latencies.stages;
        for (const [sName, sVal] of Object.entries(stages)) {
            if (sVal > 0) {
                const displayName = sName.charAt(0).toUpperCase() + sName.slice(1).replace(/_/g, ' ');
                stageItems.push({ name: displayName, val: `${Math.round(sVal)} ms` });
            }
        }

        if (mapped.latencies.stt > 0 && mapped.latencies.pipeline > 0) {
            const totalE2E = Math.round(mapped.latencies.stt + mapped.latencies.pipeline);
            stageItems.push({ name: 'Voice E2E', val: `${totalE2E} ms` });
        }

        let pipelineDetailsHtml = '';
        if (stageItems.length > 0) {
            const itemsListHtml = stageItems
                .map((item) => `<div class="pipeline-item"><span class="stage-name">${item.name}</span><span class="stage-val">${item.val}</span></div>`)
                .join('');

            pipelineDetailsHtml = `
                <button class="pipeline-toggle" onclick="togglePipelineDetails(this)">
                    VIEW PIPELINE DETAILS <i class="fa-solid fa-chevron-down"></i>
                </button>
                <div class="pipeline-details">
                    <div class="pipeline-header">PIPELINE LATENCY</div>
                    ${itemsListHtml}
                </div>
            `;
        }

        const cardHtml = `
            <div class="assistant-card">
                <div class="card-meta-row">
                    ${badgeHtml}
                    <span class="latency-summary">${latencyLabel}</span>
                </div>
                <div class="answer-text">${escapeHtml(mapped.answer)}</div>
                ${pipelineDetailsHtml}
            </div>
        `;

        placeholder.outerHTML = cardHtml;
        scrollToBottom();
    }

    function renderAssistantError(turnId, message) {
        const turnEl = document.getElementById(turnId);
        if (!turnEl) return;

        const placeholder = turnEl.querySelector('.assistant-placeholder');
        if (!placeholder) return;

        placeholder.outerHTML = `
            <div class="assistant-card">
                <div class="card-meta-row">
                    <span class="badge badge-refusal"><i class="fa-solid fa-triangle-exclamation"></i> ERROR</span>
                </div>
                <div class="answer-text">${escapeHtml(message)}</div>
            </div>
        `;
        scrollToBottom();
    }

    // ── Pipeline Accordion Toggle ─────────────────────────────────────────────
    window.togglePipelineDetails = function (btnEl) {
        const detailsEl = btnEl.nextElementSibling;
        if (!detailsEl) return;

        const isExpanded = detailsEl.classList.contains('expanded');
        if (isExpanded) {
            detailsEl.classList.remove('expanded');
            btnEl.setAttribute('aria-expanded', 'false');
            btnEl.innerHTML = `VIEW PIPELINE DETAILS <i class="fa-solid fa-chevron-down"></i>`;
        } else {
            detailsEl.classList.add('expanded');
            btnEl.setAttribute('aria-expanded', 'true');
            btnEl.innerHTML = `HIDE PIPELINE DETAILS <i class="fa-solid fa-chevron-up"></i>`;
        }
    };

    // ── Helper Utilities ──────────────────────────────────────────────────────
    function autoResizeTextarea() {
        composerInput.style.height = 'auto';
        composerInput.style.height = Math.min(composerInput.scrollHeight, 120) + 'px';
    }

    function scrollToBottom() {
        requestAnimationFrame(() => {
            conversationArea.scrollTop = conversationArea.scrollHeight;
        });
    }

    function escapeHtml(text) {
        if (!text) return '';
        const div = document.createElement('div');
        div.textContent = text;
        return div.innerHTML;
    }
})();
