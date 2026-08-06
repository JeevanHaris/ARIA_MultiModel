# ARIA — AI Desktop Assistant

A sleek, dark-themed AI assistant powered by **open-source LLMs** via **Groq** (free & ultra-fast inference).

## 📁 Files

```
ai-assistant/
├── index.html     ← Main UI (open in browser)
├── style.css      ← Styling
├── app.js         ← Frontend logic (API calls, chat, tools, memory, settings)
├── server.py      ← Python Flask backend (proxies Groq API)
├── .env           ← Your API key (create this from .env.example)
└── README.md      ← This file
```

---

## 🚀 Quick Start

### 1. Install Python dependencies

```bash
pip install flask flask-cors groq python-dotenv
```

### 2. Get your FREE Groq API key

1. Sign up at [console.groq.com](https://console.groq.com)
2. Create an API key
3. Create a `.env` file:

```
GROQ_API_KEY=gsk_your-key-here
```

### 3. Start the backend

```bash
python server.py
```

You should see:
```
ARIA Backend — Groq (Open Source LLMs)
URL: http://localhost:5000
API Key: ✅ Set
Provider: Groq — Free & Ultra-fast
Model:   llama-3.3-70b-versatile
```

### 4. Open the frontend

Open `index.html` in your browser (double-click it, or use a simple server):

```bash
# Option A: just open the file
open index.html   # macOS
start index.html  # Windows

# Option B: serve locally to avoid CORS on some browsers
python -m http.server 8080
# then visit http://localhost:8080
```

---

## ✨ Features

| Feature | Description |
|---|---|
| **Chat** | Full conversation with open-source LLMs, with markdown rendering |
| **Quick Tools** | Summarize, Translate, Code Review, Grammar Fix, Explain, Brainstorm |
| **Memory** | Auto-extracts and stores context from conversations |
| **Settings** | Switch AI models, customize system prompt, accent colors, font size |
| **Responsive** | Adapts to smaller screen widths |

---

## 🤖 Available Models (via Groq)

| Model | Speed | Best For |
|---|---|---|
| **Llama 3.3 70B** | Fast | General use, best quality |
| **Llama 3.1 8B** | ⚡ Fastest | Quick tasks, drafts |
| **Mixtral 8x7B** | Fast | Code, reasoning |
| **Gemma 2 9B** | Fast | Efficient, well-rounded |
| **DeepSeek R1 70B** | Fast | Math, reasoning, analysis |

All models are **open-source** and run for **free** on Groq's infrastructure.

---

## 🔌 API Endpoints (server.py)

| Method | Path | Description |
|---|---|---|
| GET | `/` | Health check |
| GET | `/api/health` | Health check JSON |
| GET | `/api/models` | List available models |
| POST | `/api/chat` | Chat with AI (full history) |
| POST | `/api/tool` | One-shot tool prompt |

### POST /api/chat body:
```json
{
  "messages": [{"role": "user", "content": "Hello!"}],
  "model": "llama-3.3-70b-versatile",
  "max_tokens": 1024,
  "system": "You are ARIA..."
}
```

### GET /api/models response:
```json
{
  "models": [
    {"id": "llama-3.3-70b-versatile", "name": "Llama 3.3 70B", "description": "Best quality"},
    ...
  ]
}
```

---

## 🎨 Customization

- **AI Model** — choose between Llama, Mixtral, Gemma, DeepSeek in Settings
- **Accent color** — click color swatches in Settings tab
- **System prompt** — fully customizable in Settings
- **Assistant name** — change from ARIA to anything

---

## 🔄 Migration from Claude

This project was originally built to use the Anthropic Claude API. It has been migrated to use **Groq** with **open-source models** for:
- **Free access** — no paid API subscriptions needed
- **Speed** — Groq's LPU chips deliver near-instant responses
- **Open source** — all models are fully open-source (Llama, Mixtral, Gemma, DeepSeek)

---

Built with HTML, CSS, JavaScript + Python (Flask) · Powered by Groq + Open-Source LLMs
