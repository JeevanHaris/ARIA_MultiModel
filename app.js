/* ══════════════════════════════════════════════════════════════
   ARIA v2.0 — AI Voice Desktop Assistant · app.js
   Multi-Model Architecture: Gateway + Router + Orchestrator
══════════════════════════════════════════════════════════════ */

// ─── Config ──────────────────────────────────────────────────
const BACKEND_URL      = 'http://localhost:5000/api/chat';
const ORCHESTRATE_URL  = 'http://localhost:5000/api/orchestrate';
const MODELS_URL       = 'http://localhost:5000/api/models';
const UPLOAD_DOC_URL   = 'http://localhost:5000/api/upload-doc';
const DOC_QUERY_URL    = 'http://localhost:5000/api/doc-query';
const STT_URL          = 'http://localhost:5000/api/stt';

// ─── State ───────────────────────────────────────────────────
const state = {
  history: [],
  memories: [],
  msgCount: 0,
  isLoading: false,
  isListening: false,
  currentTool: null,
  activeTimers: [],
  calcExpr: '',
  calcDisplay: '0',
  calcJustResult: false,
  availableModels: [],
  // ─── Document Q&A state ───
  doc: {
    id: null,
    filename: null,
    charCount: 0,
    history: [],   // Q&A turns for this document session
    isLoading: false,
  },
  settings: {
    model: 'llama3.2',
    maxTokens: 1024,
    assistantName: 'ARIA',
    systemPrompt: 'You are ARIA (Adaptive Reasoning Intelligence Assistant), a helpful and concise AI desktop assistant. When acknowledging voice commands like opening websites or apps, keep your reply to 1 short sentence. For questions and conversations, be clear and helpful. Use markdown for code or lists.',
    accent: '#00e5ff',
    fontSize: '15px',
    voiceEnabled: true,
    voiceRate: 1,
    voicePitch: 1,
    voiceName: '',
  },
  ttsMuted: true,  // starts muted — browsers block autoplay until user gesture
  deepThink: false, // when true, sends to /api/orchestrate instead of /api/chat
};

// ─── WEBSITE MAP ─────────────────────────────────────────────
const SITE_MAP = {
  'youtube':       'https://youtube.com',
  'google':        'https://google.com',
  'gmail':         'https://mail.google.com',
  'github':        'https://github.com',
  'netflix':       'https://netflix.com',
  'spotify':       'https://open.spotify.com',
  'wikipedia':     'https://wikipedia.org',
  'twitter':       'https://twitter.com',
  'x':             'https://x.com',
  'whatsapp':      'https://web.whatsapp.com',
  'whatsapp web':  'https://web.whatsapp.com',
  'chatgpt':       'https://chat.openai.com',
  'maps':          'https://maps.google.com',
  'google maps':   'https://maps.google.com',
  'amazon':        'https://amazon.in',
  'reddit':        'https://reddit.com',
  'linkedin':      'https://linkedin.com',
  'facebook':      'https://facebook.com',
  'instagram':     'https://instagram.com',
  'stackoverflow': 'https://stackoverflow.com',
  'stack overflow':'https://stackoverflow.com',
  'claude':        'https://claude.ai',
  'news':          'https://news.google.com',
  'google news':   'https://news.google.com',
  'drive':         'https://drive.google.com',
  'google drive':  'https://drive.google.com',
  'docs':          'https://docs.google.com',
  'sheets':        'https://sheets.google.com',
  'calendar':      'https://calendar.google.com',
  'meet':          'https://meet.google.com',
  'zoom':          'https://zoom.us',
  'canva':         'https://canva.com',
  'figma':         'https://figma.com',
  'codepen':       'https://codepen.io',
  'replit':        'https://replit.com',
  'vercel':        'https://vercel.com',
  'notion':        'https://notion.so',
};

// ─── Load Models from Backend ─────────────────────────────────
async function loadModels() {
  try {
    const res = await fetch(MODELS_URL);
    if (res.ok) {
      const data = await res.json();
      state.availableModels = data.models || [];
      populateModelDropdown();
    }
  } catch (err) {
    console.log('Could not fetch models from backend:', err.message);
  }
}

function populateModelDropdown() {
  const select = document.getElementById('model-select');
  if (!select || state.availableModels.length === 0) return;
  select.innerHTML = '';
  state.availableModels.forEach(m => {
    const opt = document.createElement('option');
    opt.value = m.id;
    opt.textContent = m.name;
    opt.title = m.description;
    select.appendChild(opt);
  });
  select.value = state.settings.model;
  if (!select.value && state.availableModels.length > 0) {
    select.value = state.availableModels[0].id;
    state.settings.model = select.value;
  }
}

// ─── VOICE COMMAND ROUTER ────────────────────────────────────
async function runVoiceCommand(rawText) {
  const text = rawText.trim();
  if (!text) return;

  removeWelcome();
  appendMessage('user', `🎤 ${text}`);
  state.history.push({ role: 'user', content: text });
  updateMsgCount();

  const lower = text.toLowerCase();

  // ── 1. OPEN WEBSITE ──
  const openMatch = lower.match(/^(?:open|go to|launch|visit|take me to)\s+(.+)$/);
  if (openMatch) {
    const target = openMatch[1].trim();
    const url = SITE_MAP[target];
    if (url) {
      window.open(url, '_blank');
      const reply = `Opening ${capitalize(target)} for you! 🚀`;
      respond(reply);
      flashAction(`✓ Opened ${capitalize(target)}`);
      return;
    }
  }

  // ── 2. BUILT-IN APPS ──
  if (/open calculator|launch calculator|calculator/i.test(lower) && !/search/i.test(lower)) {
    document.getElementById('calc-modal').style.display = 'flex';
    respond('Calculator is open! 🧮');
    flashAction('✓ Calculator opened');
    return;
  }

  if (/open notepad|notepad|open notes|take a note/i.test(lower)) {
    document.getElementById('notepad-modal').style.display = 'flex';
    respond('Notepad is open! Start typing your notes. 📝');
    flashAction('✓ Notepad opened');
    return;
  }

  // ── 3. SEARCH ──
  const ytSearchMatch = lower.match(/search (?:on )?youtube (?:for\s+)?(.+)/);
  if (ytSearchMatch) {
    const q = ytSearchMatch[1].trim();
    window.open(`https://www.youtube.com/results?search_query=${encodeURIComponent(q)}`, '_blank');
    respond(`Searching YouTube for "${q}" 🎬`);
    flashAction(`✓ YouTube: ${q}`);
    return;
  }

  const googleSearchMatch = lower.match(/^(?:search|google|find|look up|search for)\s+(.+)$/);
  if (googleSearchMatch) {
    const q = googleSearchMatch[1].replace(/^for\s+/i, '').trim();
    window.open(`https://www.google.com/search?q=${encodeURIComponent(q)}`, '_blank');
    respond(`Searching Google for "${q}" 🔍`);
    flashAction(`✓ Searching: ${q}`);
    return;
  }

  // ── 4. TIME & DATE ──
  if (/what(?:'s| is) the (?:current )?time|what time is it/i.test(lower)) {
    const t = new Date().toLocaleTimeString('en-US', { hour: '2-digit', minute: '2-digit', second: '2-digit' });
    respond(`The current time is **${t}** 🕐`);
    return;
  }

  if (/what(?:'s| is)(?: today'?s?)? (?:the )?date|today(?:'s)? date/i.test(lower)) {
    const d = new Date().toLocaleDateString('en-US', { weekday:'long', year:'numeric', month:'long', day:'numeric' });
    respond(`Today is **${d}** 📅`);
    return;
  }

  if (/what(?:'s| is)(?: the)? (?:current )?(?:year|month|day)/i.test(lower)) {
    const now = new Date();
    respond(`It's **${now.toLocaleDateString('en-US',{weekday:'long'})}**, **${now.toLocaleString('default',{month:'long'})} ${now.getDate()}, ${now.getFullYear()}** 📅`);
    return;
  }

  // ── 5. TIMER ──
  const timerMatch = lower.match(/set (?:a )?timer (?:for\s+)?(\d+)\s*(second|seconds|sec|minute|minutes|min|hour|hours|hr)/i);
  if (timerMatch) {
    const amount = parseInt(timerMatch[1]);
    const unit = timerMatch[2].toLowerCase();
    let seconds = amount;
    if (/min/.test(unit)) seconds = amount * 60;
    if (/hour|hr/.test(unit)) seconds = amount * 3600;
    const label = `${amount} ${unit}`;
    setTimer(seconds, label);
    respond(`Timer set for **${label}**! I'll notify you when it's done. ⏱`);
    flashAction(`⏱ Timer: ${label}`);
    return;
  }

  // ── 6. FUN COMMANDS ──
  if (/flip (?:a )?coin/i.test(lower)) {
    respond(`You got ${Math.random() < 0.5 ? '**Heads** 🪙' : '**Tails** 🪙'}!`);
    return;
  }

  if (/roll (?:a )?(?:dice|die)|roll dice/i.test(lower)) {
    const n = Math.floor(Math.random() * 6) + 1;
    respond(`You rolled a **${n}** ${ ['⚀','⚁','⚂','⚃','⚄','⚅'][n-1]}`);
    return;
  }

  if (/random number|pick a number/i.test(lower)) {
    respond(`Your random number is **${Math.floor(Math.random() * 100) + 1}** 🎲`);
    return;
  }

  if (/motivate me|give me motivation|inspire me/i.test(lower)) {
    const quotes = [
      "The secret of getting ahead is getting started. — Mark Twain",
      "It always seems impossible until it's done. — Nelson Mandela",
      "Don't watch the clock; do what it does. Keep going. — Sam Levenson",
      "Believe you can and you're halfway there. — Theodore Roosevelt",
      "Success is not final, failure is not fatal. — Winston Churchill",
    ];
    respond(`💪 *"${quotes[Math.floor(Math.random() * quotes.length)]}"*`);
    return;
  }

  // ── 7. ASSISTANT CONTROL ──
  if (/stop (?:speaking|talking)|shut up|be quiet|silence/i.test(lower)) {
    window.speechSynthesis?.cancel();
    const reply = 'I stopped speaking. 🔇';
    appendMessage('assistant', reply);
    state.history.push({ role: 'assistant', content: reply });
    flashAction('✓ Stopped speaking');
    return;
  }

  if (/clear (?:the )?chat|new conversation|start over/i.test(lower)) {
    clearChat();
    const reply = 'Chat cleared! Ready for a new conversation. 🗑️';
    appendMessage('assistant', reply);
    speakText(reply);
    return;
  }

  if (/go to settings|open settings/i.test(lower)) {
    switchTab('settings', document.querySelector('[data-tab="settings"]'));
    respond('Switched to Settings! ⚙️');
    return;
  }

  if (/go to tools|open tools/i.test(lower)) {
    switchTab('tools', document.querySelector('[data-tab="tools"]'));
    respond('Switched to Tools! 🛠');
    return;
  }

  if (/go to voice|open voice/i.test(lower)) {
    switchTab('voice', document.querySelector('[data-tab="voice"]'));
    respond('Switched to Voice Commands! 🎙️');
    return;
  }

  if (/read (?:the )?last (?:message|reply)/i.test(lower)) {
    const msgs = document.querySelectorAll('.message.assistant .msg-bubble');
    if (msgs.length > 0) {
      speakText(msgs[msgs.length - 1].innerText);
      appendMessage('assistant', '🔊 Reading the last message…');
    } else {
      respond('No assistant messages yet!');
    }
    return;
  }

  if (/what can you do|help|list commands|show commands/i.test(lower)) {
    respond(`Here's what I can do:\n\n**🌐 Open websites** — "Open YouTube", "Open Gmail", "Open GitHub"…\n**🧮 Calculator** — "Open calculator"\n**📝 Notepad** — "Open Notepad"\n**🔍 Search** — "Search for Python tutorials", "Search YouTube for lofi"\n**⏱ Timers** — "Set a timer for 5 minutes"\n**🕐 Time & Date** — "What time is it?", "What is today's date?"\n**🎲 Fun** — "Flip a coin", "Roll a dice", "Motivate me"\n**💬 Chat** — Ask me anything! Powered by Groq open-source AI.\n**🎛 Control** — "Stop speaking", "Clear chat", "Go to settings"`);
    return;
  }

  // ── 8. WEATHER ──
  const weatherMatch = lower.match(/(?:what(?:'s| is) the )?weather(?: (?:in|for|at) (.+))?/i);
  if (weatherMatch) {
    const place = weatherMatch[1]?.trim() || 'my location';
    window.open(`https://www.google.com/search?q=weather+${encodeURIComponent(place)}`, '_blank');
    respond(`Opening weather for **${place}** in Google 🌦️`);
    flashAction(`✓ Weather: ${place}`);
    return;
  }

  // ── 9. FALL THROUGH → Groq AI ──
  await sendToAIWithTyping(text);
}

// ─── SEND TO AI (Gateway + Router) ────────────────────
async function sendToAIWithTyping(text) {
  const typingId = showTyping();
  setLoading(true);
  try {
    if (state.deepThink) {
      // Orchestrator mode: multi-step plan-execute-verify
      const result = await callOrchestrator(text);
      removeTyping(typingId);
      appendOrchestratorResult(result);
      const reply = result.synthesis || 'No result generated.';
      state.history.push({ role: 'assistant', content: reply });
      speakText(stripMarkdown(reply));
    } else {
      // Standard mode: single-shot via Gateway + Router
      const data = await callAI(state.history);
      removeTyping(typingId);
      const reply = data.content || data.text || JSON.stringify(data);
      const routing = data.routing || null;
      appendMessage('assistant', reply, routing);
      state.history.push({ role: 'assistant', content: reply });
      speakText(stripMarkdown(reply));
      autoExtractMemory(text, reply);
    }
  } catch (err) {
    removeTyping(typingId);
    appendMessage('assistant', `⚠ ${err.message}`);
    speakText('Sorry, I encountered an error.');
  } finally {
    setLoading(false);
  }
}

function respond(text) {
  appendMessage('assistant', text);
  state.history.push({ role: 'assistant', content: text });
  speakText(stripMarkdown(text));
}

// ─── CHAT: SEND FROM TEXT INPUT ───────────────────────
async function sendMessage() {
  const input = document.getElementById('user-input');
  const text = input.value.trim();
  if (!text || state.isLoading) return;

  removeWelcome();
  appendMessage('user', text);
  state.history.push({ role: 'user', content: text });
  input.value = '';
  autoResize(input);
  updateMsgCount();

  await sendToAIWithTyping(text);
}

function sendQuick(text) { document.getElementById('user-input').value = text; sendMessage(); }

function handleKey(e) {
  if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); sendMessage(); }
}

function autoResize(el) {
  el.style.height = 'auto';
  el.style.height = Math.min(el.scrollHeight, 160) + 'px';
}

// ─── SPEECH RECOGNITION (faster-whisper via /api/stt) ───
let mediaRecorder = null;
let audioChunks = [];
let whisperStream = null;
let whisperAudioCtx = null;

async function startWhisperRecording() {
  try {
    const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
    whisperStream = stream;
    mediaRecorder = new MediaRecorder(stream);
    audioChunks = [];

    mediaRecorder.ondataavailable = (e) => {
      if (e.data.size > 0) audioChunks.push(e.data);
    };

    mediaRecorder.onstop = async () => {
      const audioBlob = new Blob(audioChunks, { type: 'audio/webm' });
      await sendToWhisper(audioBlob);
      cleanupRecording();
    };

    mediaRecorder.start();

    // Start listening UI
    state.isListening = true;
    document.getElementById('listen-overlay').classList.add('active');
    document.getElementById('listen-transcript').textContent = 'Listening…';
    document.getElementById('listen-label').textContent = 'Listening…';
    document.getElementById('chat-mic-btn').classList.add('listening');
    document.getElementById('big-mic')?.classList.add('listening');
    if (document.getElementById('big-mic-label'))
      document.getElementById('big-mic-label').textContent = 'Listening…';
    document.getElementById('api-status').textContent = '🎙️ Listening';
    document.getElementById('status-dot').style.background = 'var(--accent)';
    document.getElementById('status-dot').style.boxShadow = '0 0 8px var(--accent)';

    // Auto-stop on silence (1.5s of silence threshold)
    startSilenceDetection(stream, 1500);

  } catch (err) {
    console.error('[ARIA] Mic access error:', err);
    appendMessage('assistant', `⚠ Mic error: ${err.message}. Please allow microphone access.`);
    stopListeningUI();
  }
}

function startSilenceDetection(stream, silenceThreshold) {
  try {
    const audioContext = new AudioContext();
    whisperAudioCtx = audioContext;
    const analyser = audioContext.createAnalyser();
    const source = audioContext.createMediaStreamSource(stream);
    source.connect(analyser);
    analyser.fftSize = 512;

    const dataArray = new Uint8Array(analyser.frequencyBinCount);
    let silenceStart = null;

    function checkSilence() {
      if (!mediaRecorder || mediaRecorder.state === 'inactive') return;

      analyser.getByteFrequencyData(dataArray);
      const volume = dataArray.reduce((a, b) => a + b, 0) / dataArray.length;

      if (volume < 10) {
        if (!silenceStart) silenceStart = Date.now();
        else if (Date.now() - silenceStart > silenceThreshold) {
          document.getElementById('listen-label').textContent = '✓ Got it!';
          stopWhisperRecording();
          return;
        }
      } else {
        silenceStart = null;
        // Update transcript to show we're hearing speech
        document.getElementById('listen-transcript').textContent = 'Hearing you…';
      }

      requestAnimationFrame(checkSilence);
    }
    checkSilence();
  } catch (e) {
    console.warn('[ARIA] Silence detection unavailable:', e);
  }
}

function stopWhisperRecording() {
  if (mediaRecorder && mediaRecorder.state !== 'inactive') {
    mediaRecorder.stop();
  }
}

function cleanupRecording() {
  // Release mic tracks
  if (whisperStream) {
    whisperStream.getTracks().forEach(track => track.stop());
    whisperStream = null;
  }
  // Close audio context
  if (whisperAudioCtx) {
    whisperAudioCtx.close().catch(() => {});
    whisperAudioCtx = null;
  }
}

async function sendToWhisper(audioBlob) {
  document.getElementById('listen-transcript').textContent = 'Transcribing…';
  document.getElementById('listen-label').textContent = '⚙️ Processing';
  document.getElementById('api-status').textContent = 'Transcribing…';

  const formData = new FormData();
  formData.append('audio', audioBlob, 'recording.webm');

  try {
    const response = await fetch(STT_URL, {
      method: 'POST',
      body: formData
    });

    if (!response.ok) throw new Error(`STT failed: ${response.status}`);

    const data = await response.json();
    stopListeningUI();

    if (data.text && data.text.trim()) {
      runVoiceCommand(data.text.trim());
    } else {
      appendMessage('assistant', '⚠ No speech detected — try again.');
    }
  } catch (err) {
    console.error('[ARIA] STT error:', err);
    stopListeningUI();
    appendMessage('assistant', `⚠ Transcription failed: ${err.message}. Check that server.py is running.`);
  }
}

function toggleListening() {
  if (state.isListening) {
    stopWhisperRecording();
    stopListeningUI();
    cleanupRecording();
    return;
  }
  startWhisperRecording();
}

function stopListening() {
  stopWhisperRecording();
  stopListeningUI();
  cleanupRecording();
}

function stopListeningUI() {
  state.isListening = false;
  document.getElementById('listen-overlay').classList.remove('active');
  document.getElementById('listen-transcript').textContent = 'Say a command or ask me anything…';
  document.getElementById('chat-mic-btn').classList.remove('listening');
  document.getElementById('big-mic')?.classList.remove('listening');
  if (document.getElementById('big-mic-label'))
    document.getElementById('big-mic-label').textContent = 'Tap to Speak';
  document.getElementById('api-status').textContent = 'Ready';
  document.getElementById('status-dot').style.background = '#34d399';
  document.getElementById('status-dot').style.boxShadow = '0 0 6px #34d399';
}

// ─── TEXT-TO-SPEECH ───────────────────────────────────
let voices = [];
function loadVoices() {
  voices = window.speechSynthesis?.getVoices() || [];
  const sel = document.getElementById('voice-select');
  if (!sel) return;
  sel.innerHTML = '';
  if (voices.length === 0) { sel.innerHTML = '<option>No voices found</option>'; return; }
  const engVoices = voices.filter(v => v.lang.startsWith('en'));
  const list = engVoices.length > 0 ? engVoices : voices;
  list.forEach(v => {
    const opt = document.createElement('option');
    opt.value = v.name;
    opt.textContent = `${v.name} (${v.lang})`;
    if (v.name === state.settings.voiceName) opt.selected = true;
    sel.appendChild(opt);
  });
}

window.speechSynthesis?.addEventListener('voiceschanged', loadVoices);
setTimeout(loadVoices, 500);

function speakText(text) {
  if (!state.settings.voiceEnabled) return;
  if (!window.speechSynthesis) return;
  if (state.ttsMuted) {
    // Show hint that TTS is muted
    const hint = document.getElementById('tts-muted-hint');
    if (hint) hint.style.display = 'inline';
    return;
  }
  window.speechSynthesis.cancel();
  const utt = new SpeechSynthesisUtterance(text.slice(0, 500));
  utt.rate  = parseFloat(state.settings.voiceRate)  || 1;
  utt.pitch = parseFloat(state.settings.voicePitch) || 1;
  const voiceName = state.settings.voiceName || document.getElementById('voice-select')?.value;
  const found = voices.find(v => v.name === voiceName);
  if (found) utt.voice = found;
  const ttsEl = document.getElementById('tts-indicator');
  utt.onstart = () => { if (ttsEl) ttsEl.style.display = 'block'; };
  utt.onend = utt.onerror = () => { if (ttsEl) ttsEl.style.display = 'none'; };
  window.speechSynthesis.speak(utt);
}

/** Unmute TTS — called from the "Enable Voice" banner button */
function unmuteTTS() {
  state.ttsMuted = false;
  // Hide the unmute banner
  const banner = document.getElementById('unmute-banner');
  if (banner) banner.classList.add('hidden');
  // Hide the muted hint
  const hint = document.getElementById('tts-muted-hint');
  if (hint) hint.style.display = 'none';
  // Play the welcome greeting now that user has interacted
  speakText(`Hello! I'm ${state.settings.assistantName}. Press Space or click Speak to give a voice command.`);
}

function testVoice() {
  speakText(`Hello! I am ${state.settings.assistantName}, your AI voice assistant. Powered by Groq open-source AI.`);
}

function stripMarkdown(text) {
  return text
    .replace(/```[\s\S]*?```/g, 'code block')
    .replace(/`([^`]+)`/g, '$1')
    .replace(/\*\*([^*]+)\*\*/g, '$1')
    .replace(/\*([^*]+)\*/g, '$1')
    .replace(/#+\s/g, '')
    .replace(/[-*]\s/g, '')
    .replace(/\n+/g, ' ')
    .trim();
}

// ─── TIMER ───────────────────────────────────────────
function setTimer(seconds, label) {
  const t = setTimeout(() => {
    const msg = `Your ${label} timer is done!`;
    document.getElementById('timer-msg').textContent = msg;
    document.getElementById('timer-notif').style.display = 'flex';
    speakText(msg);
    flashAction('⏱ Timer Done!');
  }, seconds * 1000);
  state.activeTimers.push(t);
}

// ─── ACTION FLASH ─────────────────────────────────────
function flashAction(msg) {
  const el = document.getElementById('action-flash');
  el.textContent = msg;
  el.classList.add('show');
  setTimeout(() => el.classList.remove('show'), 2500);
}

// ─── CALCULATOR ───────────────────────────────────────
function calcNum(n) {
  if (state.calcJustResult) { state.calcExpr = ''; state.calcJustResult = false; }
  if (state.calcDisplay === '0' || state.calcDisplay === 'Error') state.calcDisplay = n;
  else state.calcDisplay += n;
  state.calcExpr += n;
  updateCalc();
}

function calcInput(op) {
  state.calcJustResult = false;
  if (state.calcDisplay === 'Error') state.calcDisplay = '0';
  const last = state.calcExpr.slice(-1);
  if (['+','-','*','/'].includes(last)) state.calcExpr = state.calcExpr.slice(0,-1) + op;
  else state.calcExpr += op;
  state.calcDisplay += { '+':'+', '-':'−', '*':'×', '/':'÷' }[op] || op;
  updateCalc();
}

function calcEquals() {
  try {
    const result = Function('"use strict"; return (' + state.calcExpr + ')')();
    const display = parseFloat(result.toFixed(10)).toString();
    document.getElementById('calc-expr').textContent = state.calcExpr + ' =';
    state.calcExpr = display;
    state.calcDisplay = display;
    state.calcJustResult = true;
    document.getElementById('calc-display').textContent = display;
  } catch(e) {
    state.calcDisplay = 'Error'; state.calcExpr = ''; updateCalc();
  }
}

function calcClear() {
  state.calcDisplay = '0'; state.calcExpr = ''; state.calcJustResult = false;
  document.getElementById('calc-expr').textContent = '';
  updateCalc();
}

function calcDot() {
  if (state.calcJustResult) { state.calcDisplay = '0.'; state.calcExpr = '0.'; state.calcJustResult = false; }
  else if (!state.calcDisplay.split(/[+\-×÷]/).pop().includes('.')) {
    state.calcDisplay += '.'; state.calcExpr += '.';
  }
  updateCalc();
}

function calcToggleSign() {
  if (state.calcDisplay !== '0' && state.calcDisplay !== 'Error') {
    if (state.calcDisplay.startsWith('-')) {
      state.calcDisplay = state.calcDisplay.slice(1);
      state.calcExpr = state.calcExpr.slice(1);
    } else {
      state.calcDisplay = '-' + state.calcDisplay;
      state.calcExpr = '-' + state.calcExpr;
    }
    updateCalc();
  }
}

function updateCalc() {
  document.getElementById('calc-display').textContent = state.calcDisplay;
}

// ─── NOTEPAD ──────────────────────────────────────────
function copyNote() {
  navigator.clipboard?.writeText(document.getElementById('notepad-area').value)
    .then(() => flashAction('✓ Copied to clipboard'));
}

function clearNote() { document.getElementById('notepad-area').value = ''; }

// ─── API CALL (Gateway + Router) ──────────────────────
async function callAI(messages) {
  const res = await fetch(BACKEND_URL, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      messages,
      auto_route: true,          // let the Router pick the best model
      max_tokens: state.settings.maxTokens,
      system: buildSystemPrompt(),
    })
  });
  if (!res.ok) {
    const e = await res.json().catch(() => ({}));
    throw new Error(e.error || `Backend error ${res.status}`);
  }
  return await res.json(); // returns { content, model, routing, usage, ... }
}

async function callOrchestrator(text) {
  const res = await fetch(ORCHESTRATE_URL, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      request: text,
      system: buildSystemPrompt(),
    })
  });
  if (!res.ok) {
    const e = await res.json().catch(() => ({}));
    throw new Error(e.error || `Orchestrator error ${res.status}`);
  }
  return await res.json();
}

function buildSystemPrompt() {
  let p = state.settings.systemPrompt;
  if (state.memories.length > 0)
    p += '\n\n## User Memories:\n' + state.memories.map(m => `- ${m}`).join('\n');
  return p;
}

// ─── DEEP THINK TOGGLE ────────────────────────────────
function toggleDeepThink() {
  state.deepThink = !state.deepThink;
  const btn = document.getElementById('deep-think-btn');
  if (btn) {
    btn.classList.toggle('active', state.deepThink);
    btn.title = state.deepThink ? 'Deep Think ON — multi-step orchestration' : 'Deep Think OFF — single model response';
  }
  flashAction(state.deepThink ? '🧠 Deep Think ON' : '⚡ Deep Think OFF');
}

// ─── RENDER MESSAGES ──────────────────────────────────
function appendMessage(role, content, routing) {
  const win = document.getElementById('chat-window');
  const div = document.createElement('div');
  div.className = `message ${role}`;
  const label = role === 'user' ? 'You' : state.settings.assistantName;
  const avatar = role === 'user' ? 'U' : 'AI';

  // Build routing badge for assistant messages
  let routingBadge = '';
  if (role === 'assistant' && routing && routing.model) {
    const taskIcon = { code: '💻', reasoning: '🧠', general: '💬', vision: '👁', user_selected: '🎯' }[routing.task_type] || '🤖';
    routingBadge = `<span class="routing-badge" title="${routing.reason || ''}">${taskIcon} ${routing.model}</span>`;
  }

  div.innerHTML = `
    <div class="msg-avatar">${avatar}</div>
    <div class="msg-body">
      <div class="msg-label">${label} ${routingBadge}</div>
      <div class="msg-bubble">${renderMarkdown(content)}</div>
    </div>`;
  win.appendChild(div);
  win.scrollTop = win.scrollHeight;
}

/** Render an orchestrator result with step-by-step visualization */
function appendOrchestratorResult(result) {
  const win = document.getElementById('chat-window');
  const div = document.createElement('div');
  div.className = 'message assistant orchestrated';

  let stepsHtml = '';
  if (result.steps && result.steps.length > 0) {
    stepsHtml = '<div class="orch-steps">';
    stepsHtml += `<div class="orch-steps-header">🧠 Plan (${result.steps.length} steps · ${result.total_time?.toFixed(1) || '?'}s)</div>`;
    for (const step of result.steps) {
      const taskIcon = { code: '💻', reasoning: '🧠', general: '💬' }[step.task_type] || '🤖';
      const statusIcon = step.error ? '❌' : (step.code_execution && !step.code_execution.success ? '⚠️' : '✅');
      stepsHtml += `
        <div class="orch-step">
          <div class="orch-step-header">
            <span class="orch-step-num">${statusIcon} Step ${step.step}</span>
            <span class="routing-badge">${taskIcon} ${step.model || '?'}</span>
          </div>
          <div class="orch-step-instruction">${escHtml(step.instruction)}</div>
        </div>`;
    }
    stepsHtml += '</div>';
  }

  const synthesis = result.synthesis || result.error || 'No result generated.';

  div.innerHTML = `
    <div class="msg-avatar">AI</div>
    <div class="msg-body">
      <div class="msg-label">${state.settings.assistantName} <span class="routing-badge deep-think-badge">🧠 Deep Think</span></div>
      ${stepsHtml}
      <div class="msg-bubble">${renderMarkdown(synthesis)}</div>
    </div>`;
  win.appendChild(div);
  win.scrollTop = win.scrollHeight;
}

function renderMarkdown(text) {
  return text
    .replace(/```(\w*)\n?([\s\S]*?)```/g, (_,l,c) => `<pre><code>${escHtml(c.trim())}</code></pre>`)
    .replace(/`([^`]+)`/g, (_,c) => `<code>${escHtml(c)}</code>`)
    .replace(/\*\*([^*]+)\*\*/g, '<strong>$1</strong>')
    .replace(/\*([^*]+)\*/g, '<em>$1</em>')
    .replace(/^### (.+)$/gm, '<h3>$1</h3>').replace(/^## (.+)$/gm, '<h2>$1</h2>').replace(/^# (.+)$/gm, '<h1>$1</h1>')
    .replace(/^[\-\*] (.+)$/gm, '<li>$1</li>').replace(/(<li>[\s\S]+?<\/li>)/g, '<ul>$1</ul>')
    .replace(/\n{2,}/g, '</p><p>').replace(/\n/g, '<br>');
}

function escHtml(s) { return s.replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;'); }

function removeWelcome() {
  const w = document.getElementById('chat-welcome');
  if (w) w.remove();
}

// ─── TYPING INDICATOR ─────────────────────────────────
let typingCtr = 0;
function showTyping() {
  const id = `typing-${++typingCtr}`;
  const win = document.getElementById('chat-window');
  const div = document.createElement('div');
  div.id = id; div.className = 'message assistant';
  div.innerHTML = `<div class="msg-avatar">AI</div><div class="msg-body"><div class="msg-label">${state.settings.assistantName}</div><div class="msg-bubble"><div class="typing-dots"><span></span><span></span><span></span></div></div></div>`;
  win.appendChild(div);
  win.scrollTop = win.scrollHeight;
  return id;
}
function removeTyping(id) { document.getElementById(id)?.remove(); }

// ─── LOADING STATE ────────────────────────────────────
function setLoading(v) {
  state.isLoading = v;
  document.getElementById('send-btn').disabled = v;
  document.getElementById('api-status').textContent = v ? 'Thinking…' : 'Ready';
  document.getElementById('api-status').style.color = v ? 'var(--accent)' : '#34d399';
}

function updateMsgCount() { state.msgCount++; document.getElementById('msg-count').textContent = state.msgCount; }

// ─── CLEAR CHAT ───────────────────────────────────────
function clearChat() {
  state.history = []; state.msgCount = 0;
  document.getElementById('msg-count').textContent = 0;
  document.getElementById('chat-window').innerHTML = `
    <div class="chat-welcome" id="chat-welcome">
      <div class="welcome-orb"><div class="orb-ring r1"></div><div class="orb-ring r2"></div><div class="orb-ring r3"></div><div class="orb-core"></div></div>
      <h2>Hello, I'm ARIA</h2>
      <p>Your Adaptive Reasoning Intelligence Assistant.<br>
      Press <kbd>Space</kbd> or click <strong>Speak</strong> to give a voice command.</p>
      <div class="quick-prompts">
        <button class="quick-btn" onclick="runVoiceCommand('open youtube')">📺 Open YouTube</button>
        <button class="quick-btn" onclick="runVoiceCommand('open calculator')">🧮 Calculator</button>
        <button class="quick-btn" onclick="runVoiceCommand('what is the time')">🕐 What time is it?</button>
        <button class="quick-btn" onclick="runVoiceCommand('set a timer for 30 seconds')">⏱ Timer 30s</button>
        <button class="quick-btn" onclick="runVoiceCommand('open notepad')">📝 Notepad</button>
        <button class="quick-btn" onclick="runVoiceCommand('search for weather today')">🔍 Web Search</button>
        <button class="quick-btn" onclick="runVoiceCommand('tell me a joke')">😂 Tell a Joke</button>
        <button class="quick-btn" onclick="runVoiceCommand('flip a coin')">🪙 Flip a Coin</button>
      </div>
    </div>`;
}

// ─── TAB SWITCHING ────────────────────────────────────
function switchTab(id, btn) {
  document.querySelectorAll('.tab-panel').forEach(p => p.classList.remove('active'));
  document.querySelectorAll('.nav-btn').forEach(b => b.classList.remove('active'));
  document.getElementById('tab-' + id).classList.add('active');
  if (btn) btn.classList.add('active');
}

// ─── TOOLS ───────────────────────────────────────────
const TOOL_CONFIG = {
  summarize:  { title:'📋 Summarize Text', placeholder:'Paste the text to summarize…', extra:null, prompt: t=>`Summarize concisely:\n\n${t}` },
  translate:  { title:'🌐 Translate Text',  placeholder:'Paste text to translate…', extra:'Target language (e.g. Spanish, Tamil)', prompt:(t,l)=>`Translate to ${l||'Spanish'}:\n\n${t}` },
  codereview: { title:'🔍 Code Review',     placeholder:'Paste your code…', extra:null, prompt: t=>`Review this code for bugs and best practices:\n\n\`\`\`\n${t}\n\`\`\`` },
  grammar:    { title:'✏️ Grammar Fix',     placeholder:'Paste your text…', extra:null, prompt: t=>`Fix all grammar and spelling. Return corrected version only:\n\n${t}` },
  explain:    { title:'💡 Explain',         placeholder:'Paste a concept or passage…', extra:'Target audience (e.g. beginner, expert)', prompt:(t,a)=>`Explain to a ${a||'general audience'}:\n\n${t}` },
  brainstorm: { title:'⚡ Brainstorm',      placeholder:'Describe the topic…', extra:'Number of ideas (default 10)', prompt:(t,n)=>`Generate ${n||10} creative ideas for:\n\n${t}\n\nNumbered list with brief explanations.` },
};

function runTool(id) {
  state.currentTool = id;
  const cfg = TOOL_CONFIG[id];
  document.getElementById('tool-modal-title').textContent = cfg.title;
  document.getElementById('tool-input').placeholder = cfg.placeholder;
  document.getElementById('tool-input').value = '';
  const ex = document.getElementById('tool-extra');
  if (cfg.extra) { ex.style.display='block'; ex.placeholder=cfg.extra; ex.value=''; } else ex.style.display='none';
  document.getElementById('tool-output').style.display = 'none';
  document.getElementById('tool-output').textContent = '';
  document.getElementById('tool-modal').style.display = 'flex';
}

function closeToolModal() { document.getElementById('tool-modal').style.display='none'; state.currentTool=null; }

async function executeTool() {
  const cfg = TOOL_CONFIG[state.currentTool];
  const text = document.getElementById('tool-input').value.trim();
  const extra = document.getElementById('tool-extra').value.trim();
  if (!text) return;
  const btn = document.querySelector('.run-btn');
  btn.disabled = true; btn.textContent = 'Running…';
  const outEl = document.getElementById('tool-output');
  outEl.style.display = 'block'; outEl.textContent = '…';
  try {
    const result = await callAI([{ role:'user', content: cfg.prompt(text, extra) }]);
    outEl.textContent = result;
  } catch(e) { outEl.textContent = `Error: ${e.message}`; }
  finally { btn.disabled = false; btn.textContent = 'Run Tool'; }
}

document.addEventListener('click', e => { if (e.target.id === 'tool-modal') closeToolModal(); });

// ─── MEMORY ───────────────────────────────────────────
function addMemory(text) {
  const input = document.getElementById('memory-input');
  const val = text || input.value.trim();
  if (!val) return;
  state.memories.push(val);
  if (!text) input.value = '';
  saveMemories(); renderMemories();
}

function removeMemory(i) { state.memories.splice(i,1); saveMemories(); renderMemories(); }
function clearMemory() { state.memories=[]; saveMemories(); renderMemories(); }

function renderMemories() {
  const list = document.getElementById('memory-list');
  if (!state.memories.length) {
    list.innerHTML = `<div class="memory-empty"><svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5" width="40"><ellipse cx="12" cy="5" rx="9" ry="3"/><path d="M21 12c0 1.66-4 3-9 3s-9-1.34-9-3"/><path d="M3 5v14c0 1.66 4 3 9 3s9-1.34 9-3V5"/></svg><p>No memory entries yet.<br>Memories are created from your conversations.</p></div>`;
    return;
  }
  list.innerHTML = state.memories.map((m,i) => `
    <div class="memory-entry">
      <div class="memory-entry-dot"></div>
      <div class="memory-entry-text">${escHtml(m)}</div>
      <button class="memory-entry-del" onclick="removeMemory(${i})">✕</button>
    </div>`).join('');
}

function autoExtractMemory(user, reply) {
  const nameMatch = user.match(/(?:my name is|i am|i'm|call me)\s+([A-Z][a-z]+)/i);
  if (nameMatch) { const m=`User's name is ${nameMatch[1]}`; if (!state.memories.includes(m)) addMemory(m); }
  const jobMatch = user.match(/(?:i(?:'m| am) a(?:n)?|i work as a(?:n)?)\s+([a-z ]+?)(?:\.|,|$)/i);
  if (jobMatch) { const m=`User works as: ${jobMatch[1].trim()}`; if (!state.memories.includes(m)) addMemory(m); }
}

function saveMemories() { try { localStorage.setItem('aria_memories', JSON.stringify(state.memories)); } catch(e){} }
function loadMemories() { try { const s=localStorage.getItem('aria_memories'); if(s){state.memories=JSON.parse(s);renderMemories();} } catch(e){} }

// ─── SETTINGS ─────────────────────────────────────────
function saveSettings() {
  state.settings.model         = document.getElementById('model-select').value;
  state.settings.maxTokens     = parseInt(document.getElementById('max-tokens').value) || 1024;
  state.settings.assistantName = document.getElementById('assistant-name').value || 'ARIA';
  state.settings.systemPrompt  = document.getElementById('system-prompt').value;
  state.settings.voiceEnabled  = document.getElementById('voice-enabled').checked;
  state.settings.voiceRate     = document.getElementById('voice-rate').value;
  state.settings.voicePitch    = document.getElementById('voice-pitch').value;
  state.settings.voiceName     = document.getElementById('voice-select').value;
  try { localStorage.setItem('aria_settings', JSON.stringify(state.settings)); } catch(e){}
}

function loadSettings() {
  try { const s=localStorage.getItem('aria_settings'); if(s) Object.assign(state.settings, JSON.parse(s)); } catch(e){}
  document.getElementById('model-select').value   = state.settings.model;
  document.getElementById('max-tokens').value     = state.settings.maxTokens;
  document.getElementById('assistant-name').value = state.settings.assistantName;
  document.getElementById('system-prompt').value  = state.settings.systemPrompt;
  document.getElementById('voice-enabled').checked= state.settings.voiceEnabled;
  document.getElementById('voice-rate').value     = state.settings.voiceRate;
  document.getElementById('voice-pitch').value    = state.settings.voicePitch;
  document.getElementById('rate-val').textContent = parseFloat(state.settings.voiceRate).toFixed(1);
  document.getElementById('pitch-val').textContent= parseFloat(state.settings.voicePitch).toFixed(1);
  if (state.settings.accent) applyAccent(state.settings.accent);
  if (state.settings.fontSize) {
    document.documentElement.style.setProperty('--font-size', state.settings.fontSize);
    document.getElementById('font-size').value = state.settings.fontSize;
  }
}

function applyAccent(color) {
  document.documentElement.style.setProperty('--accent', color);
  const r=parseInt(color.slice(1,3),16), g=parseInt(color.slice(3,5),16), b=parseInt(color.slice(5,7),16);
  document.documentElement.style.setProperty('--accent-dim',    `rgba(${r},${g},${b},0.12)`);
  document.documentElement.style.setProperty('--accent-glow',   `rgba(${r},${g},${b},0.28)`);
  document.documentElement.style.setProperty('--border-accent', `rgba(${r},${g},${b},0.28)`);
}

function setAccent(color, btn) {
  document.querySelectorAll('.swatch').forEach(s => s.classList.remove('active'));
  btn.classList.add('active');
  state.settings.accent = color;
  applyAccent(color);
  saveSettings();
}

function applyFontSize() {
  const s = document.getElementById('font-size').value;
  state.settings.fontSize = s;
  document.documentElement.style.setProperty('--font-size', s);
  saveSettings();
}

// ─── UTILS ───────────────────────────────────────────
function capitalize(s) { return s.charAt(0).toUpperCase() + s.slice(1); }

// ─── DOCUMENT Q&A ─────────────────────────────────────

/** Show upload zone, hide chat area, reset doc state */
function clearDocSession() {
  state.doc.id        = null;
  state.doc.filename  = null;
  state.doc.charCount = 0;
  state.doc.history   = [];
  state.doc.isLoading = false;

  document.getElementById('doc-upload-zone').style.display   = 'flex';
  document.getElementById('doc-chat-container').style.display = 'none';
  document.getElementById('doc-clear-btn').style.display      = 'none';
  document.getElementById('doc-upload-status').textContent    = '';
  document.getElementById('doc-upload-status').className      = 'doc-upload-status';
  // Reset file input so the same file can be re-selected
  const fi = document.getElementById('doc-file-input');
  if (fi) fi.value = '';
  // Clear chat messages (keep welcome)
  const win = document.getElementById('doc-chat-window');
  if (win) win.innerHTML = `
    <div class="doc-chat-welcome" id="doc-chat-welcome">
      <div class="doc-welcome-icon">
        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5" width="40" height="40">
          <circle cx="12" cy="12" r="10" />
          <path d="M9.09 9a3 3 0 015.83 1c0 2-3 3-3 3" />
          <line x1="12" y1="17" x2="12.01" y2="17" />
        </svg>
      </div>
      <h3>Document loaded!</h3>
      <p>Ask any question about your document below.</p>
      <div class="quick-prompts" style="margin-top:16px">
        <button class="quick-btn" onclick="sendDocQuick('Summarize this document')">&#128203; Summarize</button>
        <button class="quick-btn" onclick="sendDocQuick('What are the key points?')">&#128273; Key Points</button>
        <button class="quick-btn" onclick="sendDocQuick('What is this document about?')">&#10067; What is this?</button>
        <button class="quick-btn" onclick="sendDocQuick('List the main topics covered')">&#128209; Main Topics</button>
      </div>
    </div>`;
}

/** Called when user picks a file via the browse dialog */
function handleDocFileSelect(event) {
  const file = event.target.files[0];
  if (file) uploadDocument(file);
}

/** Drag-and-drop handlers */
function docDragOver(e) {
  e.preventDefault();
  document.getElementById('doc-drop-area').classList.add('dragover');
}
function docDragLeave(e) {
  document.getElementById('doc-drop-area').classList.remove('dragover');
}
function docDrop(e) {
  e.preventDefault();
  document.getElementById('doc-drop-area').classList.remove('dragover');
  const file = e.dataTransfer.files[0];
  if (file) uploadDocument(file);
}

/** Upload file to /api/upload-doc and store the doc_id */
async function uploadDocument(file) {
  const statusEl = document.getElementById('doc-upload-status');
  const dropArea = document.getElementById('doc-drop-area');

  statusEl.className = 'doc-upload-status uploading';
  statusEl.innerHTML = `
    <div class="doc-uploading">
      <div class="doc-spinner"></div>
      <span>Processing <strong>${escHtml(file.name)}</strong>…</span>
    </div>`;
  dropArea.classList.add('uploading');

  const formData = new FormData();
  formData.append('file', file);

  try {
    const res = await fetch(UPLOAD_DOC_URL, { method: 'POST', body: formData });
    const data = await res.json();

    if (!res.ok) {
      statusEl.className = 'doc-upload-status error';
      statusEl.innerHTML = `<div class="doc-error">⚠️ ${escHtml(data.error || 'Upload failed.')}</div>`;
      dropArea.classList.remove('uploading');
      return;
    }

    // Store doc info
    state.doc.id        = data.doc_id;
    state.doc.filename  = data.filename;
    state.doc.charCount = data.char_count;
    state.doc.history   = [];

    // Switch to doc chat UI
    document.getElementById('doc-upload-zone').style.display    = 'none';
    document.getElementById('doc-chat-container').style.display = 'flex';
    document.getElementById('doc-clear-btn').style.display      = 'inline-flex';
    document.getElementById('doc-info-name').textContent        = data.filename;
    document.getElementById('doc-info-chars').textContent       = `${(data.char_count).toLocaleString()} chars`;
    document.getElementById('doc-input').focus();

    flashAction(`✓ Document loaded: ${data.filename}`);

  } catch (err) {
    statusEl.className = 'doc-upload-status error';
    statusEl.innerHTML = `<div class="doc-error">⚠️ Could not reach server: ${escHtml(err.message)}</div>`;
  } finally {
    dropArea.classList.remove('uploading');
  }
}

/** Send a message to /api/doc-query */
async function sendDocMessage() {
  if (!state.doc.id || state.doc.isLoading) return;
  const input = document.getElementById('doc-input');
  const question = input.value.trim();
  if (!question) return;

  // Hide welcome on first message
  const welcome = document.getElementById('doc-chat-welcome');
  if (welcome) welcome.remove();

  // Show user message
  appendDocMessage('user', question);
  state.doc.history.push({ role: 'user', content: question });
  input.value = '';
  autoResize(input);

  // Show typing
  const typingId = showDocTyping();
  state.doc.isLoading = true;
  document.getElementById('doc-send-btn').disabled = true;

  try {
    const res = await fetch(DOC_QUERY_URL, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        doc_id:   state.doc.id,
        question: question,
        model:    state.settings.model,
        history:  state.doc.history.slice(-10), // last 10 turns for context
      }),
    });
    const data = await res.json();
    removeDocTyping(typingId);

    if (!res.ok) {
      appendDocMessage('assistant', `⚠ ${data.error || 'Unknown error.'}`);
    } else {
      const reply = data.content || '';
      appendDocMessage('assistant', reply);
      state.doc.history.push({ role: 'assistant', content: reply });
      speakText(stripMarkdown(reply));
    }
  } catch (err) {
    removeDocTyping(typingId);
    appendDocMessage('assistant', `⚠ Network error: ${err.message}`);
  } finally {
    state.doc.isLoading = false;
    document.getElementById('doc-send-btn').disabled = false;
    document.getElementById('doc-input').focus();
  }
}

/** Quick-prompt helper for the doc tab */
function sendDocQuick(text) {
  document.getElementById('doc-input').value = text;
  sendDocMessage();
}

function handleDocKey(e) {
  if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); sendDocMessage(); }
}

/** Append a message bubble to the doc-chat-window */
function appendDocMessage(role, content) {
  const win = document.getElementById('doc-chat-window');
  const div = document.createElement('div');
  div.className = `message ${role}`;
  const label  = role === 'user' ? 'You' : state.settings.assistantName;
  const avatar = role === 'user' ? 'U' : 'AI';
  div.innerHTML = `
    <div class="msg-avatar">${avatar}</div>
    <div class="msg-body">
      <div class="msg-label">${label}</div>
      <div class="msg-bubble">${renderMarkdown(content)}</div>
    </div>`;
  win.appendChild(div);
  win.scrollTop = win.scrollHeight;
}

let docTypingCtr = 0;
function showDocTyping() {
  const id  = `doc-typing-${++docTypingCtr}`;
  const win = document.getElementById('doc-chat-window');
  const div = document.createElement('div');
  div.id = id; div.className = 'message assistant';
  div.innerHTML = `<div class="msg-avatar">AI</div><div class="msg-body"><div class="msg-label">${state.settings.assistantName}</div><div class="msg-bubble"><div class="typing-dots"><span></span><span></span><span></span></div></div></div>`;
  win.appendChild(div);
  win.scrollTop = win.scrollHeight;
  return id;
}
function removeDocTyping(id) { document.getElementById(id)?.remove(); }

// ─── KEYBOARD SHORTCUT: Space = toggle mic ────────────
document.addEventListener('keydown', e => {
  if (e.target.tagName === 'TEXTAREA' || e.target.tagName === 'INPUT') return;
  if (e.code === 'Space' && !e.ctrlKey && !e.altKey && !e.metaKey) {
    e.preventDefault();
    // Only toggle mic if not on the docs tab
    if (!document.getElementById('tab-docs')?.classList.contains('active')) {
      toggleListening();
    }
  }
  if (e.key === 'Escape') {
    stopListening();
    closeToolModal();
    document.getElementById('calc-modal').style.display = 'none';
    document.getElementById('notepad-modal').style.display = 'none';
  }
});

// ─── INIT ─────────────────────────────────────────────
document.addEventListener('DOMContentLoaded', () => {
  loadSettings();
  loadMemories();
  loadModels();
  loadVoices();
  document.getElementById('user-input').focus();
  // Greeting is deferred to unmuteTTS() — browsers block autoplay without user gesture
});
