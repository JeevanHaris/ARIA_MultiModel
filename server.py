"""
ARIA v2.0 — AI Voice Desktop Assistant · server.py
Flask proxy server for Ollama (Local LLMs — no internet required)
Multi-Model Architecture: Gateway + Router + Orchestrator + Sandbox
────────────────────────────────────────────────────
Setup:
    pip install flask flask-cors ollama pypdf python-docx

Prerequisite:
    1. Install Ollama: https://ollama.com/download
    2. Pull models: ollama pull llama3.2 && ollama pull qwen3:4b
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
import tempfile
import automation
import json
from flask import Flask, request, jsonify
from flask_cors import CORS

# Suppress HuggingFace symlinks warning on Windows (cosmetic only — caching still works)
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
from faster_whisper import WhisperModel

# Force UTF-8 output on Windows to avoid UnicodeEncodeError with box-drawing chars
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

try:
    import ollama
except ImportError:
    print("[ERROR] ollama not installed. Run: pip install ollama")
    exit(1)

# ─── Multi-Model Architecture Imports ─────────────────
from gateway import ModelGateway, GatewayError, ModelNotFoundError
from router import ModelRouter
from orchestrator import Orchestrator
from sandbox import CodeSandbox

# ─── App Setup ────────────────────────────────────────
app = Flask(__name__)
CORS(app)  # Allow all origins for local development

DEFAULT_MODEL = os.environ.get("ARIA_MODEL", "llama3.2")

# ─── Initialize Multi-Model Components ────────────────
gateway      = ModelGateway(default_model=DEFAULT_MODEL, keep_alive="5m")
router       = ModelRouter(default_model=DEFAULT_MODEL)
sandbox      = CodeSandbox(timeout=5)
orchestrator = Orchestrator(gateway=gateway, router=router, sandbox=sandbox)

print("[ARIA] Multi-model architecture initialized:")
print(f"  -> Gateway:      default={DEFAULT_MODEL}, keep_alive=5m")
print(f"  -> Router:       heuristic mode, {len(router.get_routing_table())} task types")
print(f"  -> Sandbox:      timeout=5s, Python only")
print(f"  -> Orchestrator: max_steps={Orchestrator.MAX_STEPS}, max_retries={Orchestrator.MAX_RETRIES}")

# ─── STT: Load faster-whisper once at startup ──────────
# base model, int8 quantized, on CPU so GPU stays free for Ollama
WHISPER_MODEL = WhisperModel(
    "base",
    device="cpu",
    compute_type="int8"
)
print("[ARIA] faster-whisper STT model loaded (base, int8, CPU)")

# ─── In-memory document store ──────────────────────────
# { doc_id: { "filename": str, "text": str } }
_doc_store: dict[str, dict] = {}

# Models you have pulled locally via `ollama pull <name>`.
# Now includes routing task info for the frontend.
AVAILABLE_MODELS = [
    {"id": "llama3.2",    "name": "Llama 3.2 3B",      "description": "Reasoning, planning, general - default",        "tasks": ["reasoning", "general"]},
    {"id": "qwen3:4b",    "name": "Qwen 3 4B",         "description": "Code generation and review specialist",         "tasks": ["code"]},
    {"id": "phi3",        "name": "Phi-3 Mini",         "description": "Fast fallback for simple queries",              "tasks": ["general"]},
    {"id": "llava",       "name": "LLaVA Vision",       "description": "Image understanding and visual analysis",       "tasks": ["vision"]},
]


# ─── Health Check ─────────────────────────────────────
@app.route("/", methods=["GET"])
def index():
    return jsonify({
        "status": "ok",
        "service": "ARIA v2.0 Backend (Multi-Model Architecture)",
        "ollama_running": gateway.is_available(),
        "provider": "Ollama - Local, offline LLM",
        "default_model": DEFAULT_MODEL,
        "architecture": {
            "gateway": True,
            "router": "heuristic",
            "orchestrator": True,
            "sandbox": True,
        },
    })


@app.route("/api/health", methods=["GET"])
def health():
    available = gateway.is_available()
    return jsonify({
        "status": "healthy" if available else "ollama_unreachable",
        "ollama_running": available,
        "provider": "ollama",
        "gateway_stats": gateway.stats(),
    })


# ─── Models Endpoint ──────────────────────────────────
@app.route("/api/models", methods=["GET"])
def models():
    """Return list of locally available models with routing info."""
    pulled_names = set(gateway.list_available())

    result = []
    for m in AVAILABLE_MODELS:
        entry = dict(m)
        entry["pulled"] = any(m["id"] in name for name in pulled_names) if pulled_names else None
        result.append(entry)

    return jsonify({
        "models": result,
        "routing_table": router.get_routing_table(),
    })


# ─── Main Chat Endpoint (Gateway + Router) ────────────
@app.route("/api/chat", methods=["POST"])
def chat():
    if not gateway.is_available():
        return jsonify({"error": "Ollama service is not running. Start it (it usually auto-starts) or run 'ollama serve'."}), 500

    data = request.get_json(silent=True)
    if not data:
        return jsonify({"error": "Invalid JSON body."}), 400

    messages      = data.get("messages", [])
    model         = data.get("model")          # None means "let Router decide"
    auto_route    = data.get("auto_route", True)
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
                "content": f"!! {automation_result}",
                "model": "system-automation",
                "routing": {"model": "system-automation", "task_type": "automation", "confidence": 1.0, "reason": "Matched automation command"},
                "usage": {"input_tokens": 0, "output_tokens": 0},
                "stop_reason": "stop"
            })

    try:
        # ─── Smart Routing ───
        latest_text = messages[-1]["content"] if messages else ""
        has_image = any("image" in str(msg.get("images", "")) for msg in messages)

        if auto_route and not model:
            # Let the Router pick the best model
            decision = router.route(latest_text, has_image=has_image)
        else:
            # User explicitly selected a model - bypass routing
            decision = router.route(latest_text, has_image=has_image,
                                    force_model=model or DEFAULT_MODEL)

        # ─── Call via Gateway ───
        ollama_messages = [{"role": "system", "content": system_prompt}] + messages

        gw_response = gateway.call(
            decision.model,
            ollama_messages,
        )

        return jsonify({
            "content": gw_response.content,
            "model": gw_response.model,
            "routing": decision.to_dict(),
            "usage": {
                "input_tokens": gw_response.tokens_in,
                "output_tokens": gw_response.tokens_out,
            },
            "latency": round(gw_response.latency, 2),
            "stop_reason": "stop" if gw_response.done else None,
        })

    except ModelNotFoundError as e:
        return jsonify({"error": f"Model '{e.model}' not found locally. Run: ollama pull {e.model}"}), 404
    except GatewayError as e:
        return jsonify({"error": str(e)}), 502
    except Exception as e:
        return jsonify({"error": f"Unexpected error: {str(e)}"}), 500


# ─── Tool Endpoint (one-shot via Gateway) ─────────────
@app.route("/api/tool", methods=["POST"])
def tool():
    """Run a one-shot tool prompt, routed through the Gateway."""
    if not gateway.is_available():
        return jsonify({"error": "Ollama service is not running."}), 500

    data = request.get_json(silent=True) or {}
    prompt = data.get("prompt", "")
    model  = data.get("model")

    if not prompt:
        return jsonify({"error": "No prompt provided."}), 400

    try:
        # Route the tool prompt to the best model
        decision = router.route(prompt, force_model=model)

        gw_response = gateway.call(
            decision.model,
            [{"role": "user", "content": prompt}],
        )
        return jsonify({
            "content": gw_response.content,
            "model": gw_response.model,
            "routing": decision.to_dict(),
        })
    except GatewayError as e:
        return jsonify({"error": str(e)}), 502
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ─── Orchestrate Endpoint (Deep Think) ────────────────
@app.route("/api/orchestrate", methods=["POST"])
def orchestrate():
    """Multi-step plan-execute-verify workflow for complex tasks."""
    if not gateway.is_available():
        return jsonify({"error": "Ollama service is not running."}), 500

    data = request.get_json(silent=True) or {}
    user_request  = data.get("request", "").strip()
    system_prompt = data.get("system")

    if not user_request:
        return jsonify({"error": "No request provided."}), 400

    try:
        print(f"[Orchestrator] Starting multi-step workflow: {user_request[:80]}...")
        result = orchestrator.run(user_request, system_prompt=system_prompt)
        return jsonify(result.to_dict())

    except GatewayError as e:
        return jsonify({"error": str(e)}), 502
    except Exception as e:
        return jsonify({"error": f"Orchestrator error: {str(e)}"}), 500


# ─── Routing Info Endpoint ────────────────────────────
@app.route("/api/routing-info", methods=["GET", "POST"])
def routing_info():
    """Return routing table, recent decisions, and gateway stats."""
    info = {
        "routing_table": router.get_routing_table(),
        "recent_decisions": router.get_recent_decisions(10),
        "gateway_stats": gateway.stats(),
        "available_models": gateway.list_available(),
    }

    # If POST, also show what the router would decide for the given text
    if request.method == "POST":
        data = request.get_json(silent=True) or {}
        text = data.get("text", "")
        if text:
            decision = router.route(text)
            info["preview"] = decision.to_dict()

    return jsonify(info)


# ─── Warm-up (via Gateway) ────────────────────────────
def warm_up():
    """Pre-load the default model into VRAM via the Gateway."""
    gateway.warm_up()


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

    print(f"[INFO] Document stored: {f.filename} ({len(text):,} chars) -- id={doc_id}")
    return jsonify({
        "doc_id":     doc_id,
        "filename":   f.filename,
        "char_count": len(text),
        "preview":    text[:300] + ("..." if len(text) > 300 else ""),
    })


# ─── Document Query (via Gateway) ─────────────────────
@app.route("/api/doc-query", methods=["POST"])
def doc_query():
    """Answer a user question grounded in the uploaded document text, routed via Gateway."""
    if not gateway.is_available():
        return jsonify({"error": "Ollama service is not running."}), 500

    data = request.get_json(silent=True) or {}
    doc_id   = data.get("doc_id", "")
    question = data.get("question", "").strip()
    model    = data.get("model")  # None = let Router decide
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

    # Build message list: system + optional prior turns + current question
    messages = [{"role": "system", "content": system_prompt}]
    for turn in history:
        if turn.get("role") in ("user", "assistant") and turn.get("content"):
            messages.append({"role": turn["role"], "content": turn["content"]})
    messages.append({"role": "user", "content": question})

    try:
        # Route doc queries - use user's model if specified, else auto-route
        decision = router.route(question, force_model=model)

        gw_response = gateway.call(decision.model, messages)

        return jsonify({
            "content": gw_response.content,
            "model":   gw_response.model,
            "routing": decision.to_dict(),
            "doc_id":  doc_id,
        })
    except ModelNotFoundError as e:
        return jsonify({"error": f"Model '{e.model}' not found locally. Run: ollama pull {e.model}"}), 404
    except GatewayError as e:
        return jsonify({"error": str(e)}), 502
    except Exception as e:
        return jsonify({"error": f"Unexpected error: {str(e)}"}), 500


# ─── Speech-to-Text (faster-whisper) ──────────────────
@app.route("/api/stt", methods=["POST"])
def transcribe_audio():
    """Accept an audio file upload, transcribe it with faster-whisper, return text."""
    if "audio" not in request.files:
        return jsonify({"error": "No audio file provided."}), 400

    audio_file = request.files["audio"]

    # Save to a temp file - faster-whisper reads from disk/path
    with tempfile.NamedTemporaryFile(suffix=".webm", delete=False) as tmp:
        audio_file.save(tmp.name)
        tmp_path = tmp.name

    try:
        segments, info = WHISPER_MODEL.transcribe(
            tmp_path,
            language="en",
            beam_size=1,        # fastest setting - good enough for command-style speech
            vad_filter=True     # skips silence automatically, improves accuracy
        )
        text = " ".join(segment.text.strip() for segment in segments)
        return jsonify({"text": text.strip(), "language": info.language})
    except Exception as e:
        return jsonify({"error": f"Transcription failed: {str(e)}"}), 500
    finally:
        os.remove(tmp_path)  # clean up temp file regardless of success/failure


# ─── Run ──────────────────────────────────────────────
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    ollama_status = "RUNNING" if gateway.is_available() else "NOT REACHABLE"
    print(f"""
+-----------------------------------------------------------+
|   ARIA v2.0 Backend - Multi-Model Architecture            |
+-----------------------------------------------------------+
|  URL:      http://localhost:{port}                           |
|  Ollama:   {ollama_status:<48}|
|  Provider: Ollama - 100% local, no internet               |
|  Model:    {DEFAULT_MODEL:<48}|
+-----------------------------------------------------------+
|  Architecture:                                            |
|    Gateway      -> unified model interface, VRAM mgmt     |
|    Router       -> heuristic task-based model selection    |
|    Orchestrator -> multi-step plan-execute-verify loop     |
|    Sandbox      -> isolated code execution (5s timeout)    |
+-----------------------------------------------------------+
|  Endpoints:                                               |
|    GET  /                 -> health check                  |
|    GET  /api/health       -> detailed health + stats       |
|    GET  /api/models       -> list models + routing table   |
|    POST /api/chat         -> chat (Gateway + Router)       |
|    POST /api/tool         -> one-shot tool (Gateway)       |
|    POST /api/orchestrate  -> Deep Think (Orchestrator)     |
|    POST /api/routing-info -> routing decisions & stats     |
|    POST /api/upload-doc   -> upload document               |
|    POST /api/doc-query    -> query document (Gateway)      |
|    POST /api/stt          -> speech-to-text                |
+-----------------------------------------------------------+

Open index.html in Chrome/Edge to start ARIA.
""")
    if gateway.is_available():
        warm_up()
    else:
        print("[WARNING] Could not reach Ollama. Is it installed and running?")
        print("[WARNING] Install: https://ollama.com/download  |  Then: ollama pull llama3.2")

    app.run(host="0.0.0.0", port=port, debug=True)