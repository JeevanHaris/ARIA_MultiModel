"""
ARIA v2.0 — AI Voice Desktop Assistant · server.py
Flask proxy server for Ollama (Local LLMs — no internet required)
────────────────────────────────────────────────────
Setup:
    pip install flask flask-cors ollama pypdf python-docx

Prerequisite:
    1. Install Ollama: https://ollama.com/download
    2. Pull the model: ollama pull llama3.2
    3. Make sure the Ollama service is running (it auto-starts after install,
       or run `ollama serve` manually)

Run:
    python server.py

Then open index.html in your browser (or serve via a local server).
The frontend connects to http://localhost:5000/api/chat by default.

No API key, no .env, no internet needed — everything runs on-device.
"""

import os
import sys
import uuid
import automation
import json
import zipfile
import urllib.request
from flask import Flask, request, jsonify
from flask_cors import CORS

# Force UTF-8 output on Windows to avoid UnicodeEncodeError with box-drawing chars
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

try:
    import ollama
except ImportError:
    print("[ERROR] ollama not installed. Run: pip install ollama")
    exit(1)

# ─── App Setup ────────────────────────────────────────
app = Flask(__name__)
CORS(app)  # Allow all origins for local development

DEFAULT_MODEL = os.environ.get("ARIA_MODEL", "llama3.2")

# ─── In-memory document store ──────────────────────────
# { doc_id: { "filename": str, "text": str } }
_doc_store: dict[str, dict] = {}
KEEP_ALIVE = "30m"  # keep model resident in memory between requests

# Models you have pulled locally via `ollama pull <name>`.
# Edit this list to match whatever you've pulled — it's just for the /api/models dropdown.
AVAILABLE_MODELS = [
    {"id": "llama3.2",   "name": "Llama 3.2 3B", "description": "Fast, balanced local model — good default for voice"},
    {"id": "phi3",       "name": "Phi-3 Mini",   "description": "Very lightweight, fastest on CPU-only machines"},
    {"id": "qwen2.5:3b", "name": "Qwen 2.5 3B",  "description": "Small and quick alternative"},
]


def _ollama_available():
    """Check whether the local Ollama service is reachable."""
    try:
        ollama.list()
        return True
    except Exception:
        return False


# ─── Health Check ─────────────────────────────────────
@app.route("/", methods=["GET"])
def index():
    return jsonify({
        "status": "ok",
        "service": "ARIA v2.0 Backend (Local — Ollama)",
        "ollama_running": _ollama_available(),
        "provider": "Ollama — Local, offline LLM",
        "default_model": DEFAULT_MODEL,
    })


@app.route("/api/health", methods=["GET"])
def health():
    return jsonify({
        "status": "healthy" if _ollama_available() else "ollama_unreachable",
        "ollama_running": _ollama_available(),
        "provider": "ollama",
    })


# ─── Models Endpoint ──────────────────────────────────
@app.route("/api/models", methods=["GET"])
def models():
    """Return list of locally available models (curated list + what's actually pulled)."""
    try:
        pulled = ollama.list()
        # ollama.list() returns a Pydantic object; models is a list of model objects
        pulled_names = {m.model for m in pulled.models}
    except Exception:
        pulled_names = set()

    result = []
    for m in AVAILABLE_MODELS:
        entry = dict(m)
        entry["pulled"] = any(m["id"] in name for name in pulled_names) if pulled_names else None
        result.append(entry)

    return jsonify({"models": result})


# ─── Main Chat Endpoint ───────────────────────────────
@app.route("/api/chat", methods=["POST"])
def chat():
    if not _ollama_available():
        return jsonify({"error": "Ollama service is not running. Start it (it usually auto-starts) or run 'ollama serve'."}), 500

    data = request.get_json(silent=True)
    if not data:
        return jsonify({"error": "Invalid JSON body."}), 400

    messages      = data.get("messages", [])
    model         = data.get("model", DEFAULT_MODEL)
    system_prompt = data.get("system", "You are ARIA, a helpful AI voice assistant.")

    if not messages:
        return jsonify({"error": "No messages provided."}), 400

    # Validate messages format
    for msg in messages:
        if "role" not in msg or "content" not in msg:
            return jsonify({"error": "Each message must have 'role' and 'content'."}), 400
        if msg["role"] not in ("user", "assistant"):
            return jsonify({"error": f"Invalid role: {msg['role']}"}), 400

    # ─── Automation Matching Layer ───
    if messages and messages[-1]["role"] == "user":
        latest_user_text = messages[-1]["content"]
        automation_result = automation.match_automation_command(latest_user_text)
        if automation_result:
            return jsonify({
                "content": f"⚙️ {automation_result}",
                "model": "system-automation",
                "usage": {"input_tokens": 0, "output_tokens": 0},
                "stop_reason": "stop"
            })

    try:
        ollama_messages = [{"role": "system", "content": system_prompt}] + messages

        response = ollama.chat(
            model=model,
            messages=ollama_messages,
            keep_alive=KEEP_ALIVE,
        )

        # ollama.chat() returns a Pydantic object — use attribute access
        reply_text = response.message.content

        return jsonify({
            "content": reply_text,
            "model": model,
            "usage": {
                "input_tokens": response.prompt_eval_count or 0,
                "output_tokens": response.eval_count or 0,
            },
            "stop_reason": "stop" if response.done else None,
        })

    except ollama.ResponseError as e:
        if "not found" in str(e).lower():
            return jsonify({"error": f"Model '{model}' not found locally. Run: ollama pull {model}"}), 404
        return jsonify({"error": f"Ollama error: {str(e)}"}), 502
    except Exception as e:
        return jsonify({"error": f"Unexpected error: {str(e)}"}), 500


# ─── Tool Endpoint (one-shot prompt) ──────────────────
@app.route("/api/tool", methods=["POST"])
def tool():
    """Run a one-shot tool prompt without conversation history."""
    if not _ollama_available():
        return jsonify({"error": "Ollama service is not running."}), 500

    data = request.get_json(silent=True) or {}
    prompt = data.get("prompt", "")
    model  = data.get("model", DEFAULT_MODEL)

    if not prompt:
        return jsonify({"error": "No prompt provided."}), 400

    try:
        response = ollama.chat(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            keep_alive=KEEP_ALIVE,
        )
        reply = response.message.content
        return jsonify({"content": reply})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ─── Warm-up (optional but recommended) ───────────────
def warm_up():
    """Pre-load the model into memory so the first real query isn't slow."""
    try:
        print(f"[INFO] Warming up {DEFAULT_MODEL}...")
        ollama.chat(
            model=DEFAULT_MODEL,
            messages=[{"role": "user", "content": "hi"}],
            keep_alive=KEEP_ALIVE,
        )  # warm-up only — response not used
        print("[INFO] Model warmed up and ready.")
    except Exception as e:
        print(f"[WARNING] Warm-up failed: {e}")
        print(f"[WARNING] Make sure you've run: ollama pull {DEFAULT_MODEL}")


# ─── Document Upload ──────────────────────────────────
def _extract_text(filename: str, file_bytes: bytes) -> str:
    """Extract plain text from PDF, DOCX, TXT, or MD file bytes."""
    ext = os.path.splitext(filename)[1].lower()

    if ext == ".pdf":
        try:
            from pypdf import PdfReader
            import io
            reader = PdfReader(io.BytesIO(file_bytes))
            pages = [page.extract_text() or "" for page in reader.pages]
            return "\n".join(pages).strip()
        except ImportError:
            return "[ERROR] pypdf not installed. Run: pip install pypdf"
        except Exception as e:
            return f"[ERROR] Could not read PDF: {e}"

    if ext == ".docx":
        try:
            from docx import Document
            import io
            doc = Document(io.BytesIO(file_bytes))
            return "\n".join(p.text for p in doc.paragraphs if p.text.strip()).strip()
        except ImportError:
            return "[ERROR] python-docx not installed. Run: pip install python-docx"
        except Exception as e:
            return f"[ERROR] Could not read DOCX: {e}"

    if ext in (".txt", ".md", ".markdown", ".rst", ".csv"):
        try:
            return file_bytes.decode("utf-8", errors="replace").strip()
        except Exception as e:
            return f"[ERROR] Could not read file: {e}"

    return f"[ERROR] Unsupported file type: {ext}"


@app.route("/api/upload-doc", methods=["POST"])
def upload_doc():
    """Accept a file upload, extract its text, and store it for querying."""
    if "file" not in request.files:
        return jsonify({"error": "No file part in the request."}), 400

    f = request.files["file"]
    if not f.filename:
        return jsonify({"error": "No file selected."}), 400

    allowed_ext = {".pdf", ".txt", ".md", ".markdown", ".docx", ".rst", ".csv"}
    ext = os.path.splitext(f.filename)[1].lower()
    if ext not in allowed_ext:
        return jsonify({"error": f"Unsupported file type '{ext}'. Allowed: PDF, TXT, MD, DOCX"}), 415

    file_bytes = f.read()
    text = _extract_text(f.filename, file_bytes)

    if text.startswith("[ERROR]"):
        return jsonify({"error": text}), 422

    doc_id = str(uuid.uuid4())
    _doc_store[doc_id] = {"filename": f.filename, "text": text}

    print(f"[INFO] Document stored: {f.filename} ({len(text):,} chars) — id={doc_id}")
    return jsonify({
        "doc_id":     doc_id,
        "filename":   f.filename,
        "char_count": len(text),
        "preview":    text[:300] + ("…" if len(text) > 300 else ""),
    })


# ─── Document Query ───────────────────────────────────
@app.route("/api/doc-query", methods=["POST"])
def doc_query():
    """Answer a user question grounded in the uploaded document text."""
    if not _ollama_available():
        return jsonify({"error": "Ollama service is not running."}), 500

    data = request.get_json(silent=True) or {}
    doc_id   = data.get("doc_id", "")
    question = data.get("question", "").strip()
    model    = data.get("model", DEFAULT_MODEL)
    history  = data.get("history", [])  # optional prior Q&A in this doc session

    if not doc_id or doc_id not in _doc_store:
        return jsonify({"error": "Document not found. Please re-upload your file."}), 404
    if not question:
        return jsonify({"error": "No question provided."}), 400

    doc = _doc_store[doc_id]
    # Truncate to ~12 000 chars to stay inside context window of small models
    MAX_DOC_CHARS = 12_000
    doc_text = doc["text"]
    if len(doc_text) > MAX_DOC_CHARS:
        doc_text = doc_text[:MAX_DOC_CHARS] + "\n\n[...document truncated for context window...]"

    system_prompt = (
        f"You are ARIA, a helpful AI document assistant.\n"
        f"The user has uploaded a document titled '{doc['filename']}'.\n"
        f"Use ONLY the document content below to answer questions. "
        f"If the answer is not in the document, say so clearly.\n\n"
        f"=== DOCUMENT CONTENT ===\n{doc_text}\n=== END OF DOCUMENT ==="
    )

    # Build message list: optional prior turns + current question
    messages = [{"role": "system", "content": system_prompt}]
    for turn in history:
        if turn.get("role") in ("user", "assistant") and turn.get("content"):
            messages.append({"role": turn["role"], "content": turn["content"]})
    messages.append({"role": "user", "content": question})

    try:
        response = ollama.chat(
            model=model,
            messages=messages,
            keep_alive=KEEP_ALIVE,
        )
        reply = response.message.content
        return jsonify({
            "content": reply,
            "model":   model,
            "doc_id":  doc_id,
        })
    except ollama.ResponseError as e:
        if "not found" in str(e).lower():
            return jsonify({"error": f"Model '{model}' not found locally. Run: ollama pull {model}"}), 404
        return jsonify({"error": f"Ollama error: {str(e)}"}), 502
    except Exception as e:
        return jsonify({"error": f"Unexpected error: {str(e)}"}), 500


# ─── Run ──────────────────────────────────────────────
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    ollama_status = "RUNNING" if _ollama_available() else "NOT REACHABLE"
    print(f"""
+--------------------------------------------------+
|   ARIA v2.0 Backend - Ollama (Local, Offline)    |
+--------------------------------------------------+
|  URL:      http://localhost:{port}                  |
|  Ollama:   {ollama_status:<38} |
|  Provider: Ollama - 100% local, no internet      |
|  Model:    {DEFAULT_MODEL:<38} |
+--------------------------------------------------+
|  Endpoints:                                      |
|    GET  /                -> health check         |
|    GET  /api/models      -> list models          |
|    POST /api/chat        -> chat with AI         |
|    POST /api/tool        -> one-shot tool        |
|    POST /api/upload-doc  -> upload document      |
|    POST /api/doc-query   -> query document       |
+--------------------------------------------------+

Open index.html in Chrome/Edge to start ARIA.
""")
    if _ollama_available():
        warm_up()
    else:
        print("[WARNING] Could not reach Ollama. Is it installed and running?")
        print("[WARNING] Install: https://ollama.com/download  |  Then: ollama pull llama3.2")

    app.run(host="0.0.0.0", port=port, debug=True)