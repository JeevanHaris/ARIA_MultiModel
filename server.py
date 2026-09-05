"""
ARIA v3.0 — Agentic AI Desktop Assistant · server.py
Flask backend for Ollama (Local LLMs — no internet required)
Full Agentic Architecture:
  Gateway → Router → AgentOrchestrator → ToolRegistry → Verify+Replan
────────────────────────────────────────────────────────────────────────
Setup:
    pip install flask flask-cors ollama pypdf python-docx python-pptx openpyxl
    pip install pytesseract Pillow pdf2image sympy
    # Windows OCR: install Tesseract from https://github.com/UB-Mannheim/tesseract/wiki

Prerequisite:
    1. Install Ollama: https://ollama.com/download
    2. Pull models: ollama pull llama3.2 && ollama pull qwen3:4b && ollama pull llava
    3. Make sure the Ollama service is running (it auto-starts after install,
       or run `ollama serve` manually)

Run:
    python server.py

Then open index.html in your browser (or serve via a local server).
The frontend connects to http://localhost:5000 by default.

No API key, no .env, no internet needed — everything runs on-device.
"""

import os
import sys
import uuid
import time
import tempfile
import automation
import json
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import io

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
from orchestrator import Orchestrator, AgentOrchestrator
from sandbox import CodeSandbox
from doc_editor import apply_edit, is_edit_intent
from tool_registry import ToolRegistry
from agent_state import AgentState

# ─── App Setup ────────────────────────────────────────
app = Flask(__name__)
CORS(app)  # Allow all origins for local development

DEFAULT_MODEL = os.environ.get("ARIA_MODEL", "llama3.2")

# ─── Initialize Multi-Model Components ────────────────
gateway      = ModelGateway(default_model=DEFAULT_MODEL, keep_alive="5m")
router       = ModelRouter(default_model=DEFAULT_MODEL)
sandbox      = CodeSandbox(timeout=5)
orchestrator = Orchestrator(gateway=gateway, router=router, sandbox=sandbox)

from memory_store import MemoryStore
memory_store = MemoryStore()

# ─── v3.0: ToolRegistry + AgentOrchestrator ───────────
# doc_store and download_store are defined below; pass by reference so
# the registry always sees the live dict contents.
_doc_store:      dict[str, dict] = {}
_download_store: dict[str, dict] = {}
DOWNLOAD_TTL_SECONDS = 900  # 15 minutes

# Seed _doc_store with persistent memory docs
for m_id, m_doc in memory_store._docs.items():
    _doc_store[m_id] = {
        "filename": m_doc["filename"],
        "text": m_doc["text"],
        "bytes": b"",
    }

tool_registry = ToolRegistry(
    gateway=gateway,
    doc_store=_doc_store,
    sandbox=sandbox,
    download_store=_download_store,
)
agent_orchestrator = AgentOrchestrator(
    gateway=gateway,
    router=router,
    sandbox=sandbox,
    tool_registry=tool_registry,
)

# In-progress agent runs (for polling)
_agent_runs: dict[str, dict] = {}   # run_id → {state, synthesis, done}

print("[ARIA] v3.0 architecture initialized:")
print(f"  -> Gateway:            default={DEFAULT_MODEL}, keep_alive=5m")
print(f"  -> Router:             heuristic mode, {len(router.get_routing_table())} task types")
print(f"  -> Sandbox:            timeout=5s, Python only")
print(f"  -> Orchestrator (v2):  max_steps={Orchestrator.MAX_STEPS}")
print(f"  -> AgentOrchestrator:  max_steps={AgentOrchestrator.MAX_STEPS}, max_replans={AgentOrchestrator.MAX_REPLANS}")
print(f"  -> ToolRegistry:       {len(tool_registry.list_tools())} tools registered")

# ─── STT: Load faster-whisper once at startup ──────────
# base model, int8 quantized, on CPU so GPU stays free for Ollama
WHISPER_MODEL = WhisperModel(
    "base",
    device="cpu",
    compute_type="int8"
)
print("[ARIA] faster-whisper STT model loaded (base, int8, CPU)")

# NOTE: _doc_store, _download_store, and DOWNLOAD_TTL_SECONDS
# are now initialised above (before ToolRegistry) so the registry
# holds a live reference to the same dict objects.

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

        # ─── Memory Document Context (RAG) ───
        memory_context, memory_sources = memory_store.format_prompt_context(latest_text)
        if memory_context:
            system_prompt = (system_prompt or "") + "\n\n" + memory_context

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
            "memory_sources": memory_sources,
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
        memory_context, memory_sources = memory_store.format_prompt_context(user_request)
        if memory_context:
            system_prompt = (system_prompt or "") + "\n\n" + memory_context

        print(f"[Orchestrator] Starting multi-step workflow: {user_request[:80]}...")
        result = orchestrator.run(user_request, system_prompt=system_prompt)
        res_dict = result.to_dict()
        res_dict["memory_sources"] = memory_sources
        return jsonify(res_dict)

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
    """Extract plain text from PDF, DOCX, XLSX, TXT, or MD file bytes."""
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

    if ext in (".xlsx", ".xls"):
        try:
            import pandas as pd
            df = pd.read_excel(io.BytesIO(file_bytes), engine="openpyxl")
            return df.to_csv(index=False).strip()
        except ImportError:
            return "[ERROR] pandas/openpyxl not installed. Run: pip install pandas openpyxl"
        except Exception as e:
            return f"[ERROR] Could not read Excel file: {e}"

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

    allowed_ext = {".pdf", ".txt", ".md", ".markdown", ".docx", ".rst", ".csv", ".xlsx", ".xls"}
    ext = os.path.splitext(f.filename)[1].lower()
    if ext not in allowed_ext:
        return jsonify({"error": f"Unsupported file type '{ext}'. Allowed: PDF, TXT, MD, DOCX, XLSX, CSV"}), 415

    file_bytes = f.read()
    text = _extract_text(f.filename, file_bytes)

    if text.startswith("[ERROR]"):
        return jsonify({"error": text}), 422

    doc_id = str(uuid.uuid4())
    # Store raw bytes too so the editor can re-serialise the file after edits
    _doc_store[doc_id] = {
        "filename": f.filename,
        "text":     text,
        "bytes":    file_bytes,
    }

    print(f"[INFO] Document stored: {f.filename} ({len(text):,} chars) -- id={doc_id}")
    return jsonify({
        "doc_id":     doc_id,
        "filename":   f.filename,
        "char_count": len(text),
        "preview":    text[:300] + ("..." if len(text) > 300 else ""),
    })


# ─── Memory Document Endpoints (Persistent Knowledge Base) ────
@app.route("/api/memory/upload", methods=["POST"])
def memory_upload():
    """Upload a document directly to persistent memory so all chats have access."""
    if "file" not in request.files:
        return jsonify({"error": "No file part in the request."}), 400

    f = request.files["file"]
    if not f.filename:
        return jsonify({"error": "No file selected."}), 400

    allowed_ext = {".pdf", ".txt", ".md", ".markdown", ".docx", ".rst", ".csv"}
    ext = os.path.splitext(f.filename)[1].lower()
    if ext not in allowed_ext:
        return jsonify({"error": f"Unsupported file type '{ext}'. Allowed: PDF, TXT, MD, DOCX, CSV"}), 415

    file_bytes = f.read()
    text = _extract_text(f.filename, file_bytes)

    if text.startswith("[ERROR]"):
        return jsonify({"error": text}), 422

    doc_meta = memory_store.save_document(f.filename, file_bytes, text)

    # Also register in _doc_store so tools and agents have access
    _doc_store[doc_meta["doc_id"]] = {
        "filename": f.filename,
        "text": text,
        "bytes": file_bytes,
    }

    return jsonify({
        "success": True,
        "document": doc_meta,
        "message": f"'{f.filename}' saved to persistent memory."
    })


@app.route("/api/memory/docs", methods=["GET"])
def memory_list_docs():
    """List all documents currently in persistent memory."""
    return jsonify({
        "documents": memory_store.list_documents()
    })


@app.route("/api/memory/docs/<doc_id>", methods=["GET"])
def memory_get_doc(doc_id):
    """Retrieve full details and text preview of a memory document."""
    doc = memory_store.get_document(doc_id)
    if not doc:
        return jsonify({"error": "Document not found."}), 404
    return jsonify({
        "document": doc
    })


@app.route("/api/memory/docs/<doc_id>", methods=["DELETE"])
def memory_delete_doc(doc_id):
    """Remove a document from persistent memory."""
    success = memory_store.delete_document(doc_id)
    _doc_store.pop(doc_id, None)
    if success:
        return jsonify({"success": True, "message": "Document removed from memory."})
    return jsonify({"error": "Document not found."}), 404


@app.route("/api/memory/promote/<doc_id>", methods=["POST"])
def memory_promote_doc(doc_id):
    """Promote a session document from _doc_store to persistent memory."""
    if doc_id not in _doc_store:
        return jsonify({"error": "Document not found in current session."}), 404

    doc = _doc_store[doc_id]
    doc_meta = memory_store.save_document(
        filename=doc["filename"],
        file_bytes=doc.get("bytes", b""),
        text=doc.get("text", "")
    )
    return jsonify({
        "success": True,
        "document": doc_meta,
        "message": f"'{doc['filename']}' added to assistant memory."
    })


@app.route("/api/memory/clear", methods=["POST"])
def memory_clear_docs():
    """Clear all documents from memory store."""
    count = memory_store.clear_documents()
    return jsonify({"success": True, "cleared_count": count})


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


# ─── Document Edit ────────────────────────────────────
@app.route("/api/doc-edit", methods=["POST"])
def doc_edit():
    """
    Apply an AI-guided edit to an uploaded document, update the doc store,
    create a download token, and return a summary + token.
    """
    if not gateway.is_available():
        return jsonify({"error": "Ollama service is not running."}), 500

    data        = request.get_json(silent=True) or {}
    doc_id      = data.get("doc_id", "")
    instruction = data.get("instruction", "").strip()
    model       = data.get("model")

    if not doc_id or doc_id not in _doc_store:
        return jsonify({"error": "Document not found. Please re-upload your file."}), 404
    if not instruction:
        return jsonify({"error": "No instruction provided."}), 400

    doc = _doc_store[doc_id]

    # Build a simple LLM callable that routes through the Gateway
    chosen_model = model or DEFAULT_MODEL
    def llm_fn(prompt: str) -> str:
        resp = gateway.call(chosen_model, [{"role": "user", "content": prompt}])
        return resp.content

    try:
        result = apply_edit(
            filename=doc["filename"],
            file_bytes=doc["bytes"],
            instruction=instruction,
            llm_fn=llm_fn,
        )
    except Exception as e:
        return jsonify({"error": f"Edit failed: {str(e)}"}), 500

    if not result.success:
        return jsonify({"error": result.error or "Edit failed.", "details": result.to_dict()}), 422

    # Update the doc store with the edited content
    _doc_store[doc_id]["bytes"] = result.new_bytes
    _doc_store[doc_id]["text"]  = result.new_text

    # Purge expired download tokens
    now = time.time()
    expired = [t for t, v in _download_store.items() if v["expires"] < now]
    for t in expired:
        del _download_store[t]

    # Create a fresh download token
    token = str(uuid.uuid4())
    _download_store[token] = {
        "doc_id":   doc_id,
        "filename": doc["filename"],
        "bytes":    result.new_bytes,
        "expires":  now + DOWNLOAD_TTL_SECONDS,
    }

    print(f"[INFO] Document edited: {doc['filename']} -- {result.summary}")
    return jsonify({
        "success":        True,
        "summary":        result.summary,
        "changes":        result.changes,
        "download_token": token,
        "filename":       doc["filename"],
        "char_count":     len(result.new_text),
    })


# ─── Document Download ────────────────────────────────
@app.route("/api/doc-download/<token>", methods=["GET"])
def doc_download(token):
    """Stream the edited document file as a download attachment."""
    entry = _download_store.get(token)
    if not entry:
        return jsonify({"error": "Download token not found or expired."}), 404
    if time.time() > entry["expires"]:
        del _download_store[token]
        return jsonify({"error": "Download token has expired. Please re-apply the edit."}), 410

    filename  = entry["filename"]
    file_bytes = entry["bytes"]

    # Determine MIME type
    ext = os.path.splitext(filename)[1].lower()
    mime_map = {
        ".csv":  "text/csv",
        ".xlsx": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        ".xls":  "application/vnd.ms-excel",
        ".docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        ".txt":  "text/plain",
        ".md":   "text/markdown",
        ".rst":  "text/plain",
        ".pdf":  "application/pdf",
    }
    mime = mime_map.get(ext, "application/octet-stream")

    # Use 'edited_' prefix only for doc edits; draft exports already have a proper name
    doc_id = entry.get("doc_id")
    dl_name = ("edited_" + filename) if doc_id else filename

    return send_file(
        io.BytesIO(file_bytes),
        mimetype=mime,
        as_attachment=True,
        download_name=dl_name,
    )


# ─── Image OCR ───────────────────────────────────────
@app.route("/api/ocr-image", methods=["POST"])
def ocr_image_endpoint():
    """
    Accept an image upload, extract text via Tesseract OCR.
    Falls back to LLaVA vision model if OCR yields < 20 characters.

    Accepts multipart form with field 'file'.
    Optional form field 'prompt' for a custom vision question.
    Returns: { text, method, char_count, preview }
    """
    if "file" not in request.files:
        return jsonify({"error": "No file provided."}), 400

    f = request.files["file"]
    if not f.filename:
        return jsonify({"error": "No file selected."}), 400

    allowed_img = {".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".tif", ".gif", ".webp"}
    ext = os.path.splitext(f.filename)[1].lower()
    if ext not in allowed_img:
        return jsonify({"error": f"Unsupported image type '{ext}'. Allowed: PNG, JPG, BMP, TIFF, GIF, WEBP"}), 415

    file_bytes = f.read()
    custom_prompt = request.form.get("prompt", "").strip()

    from multimodal import ocr_image, analyze_image_with_vision

    # Step 1: Try Tesseract OCR
    ocr_result = ocr_image(file_bytes)
    method = "tesseract"
    text = ""

    if ocr_result.success and len(ocr_result.text.strip()) >= 20:
        text = ocr_result.text
    else:
        # Step 2: Fall back to LLaVA vision model
        if gateway.is_available():
            vision_prompt = custom_prompt or (
                "Please read and extract ALL text visible in this image. "
                "List everything you can see written in the image, word for word."
            )
            text = analyze_image_with_vision(file_bytes, gateway, prompt=vision_prompt)
            method = "llava-vision"
        elif ocr_result.error:
            return jsonify({"error": f"OCR failed: {ocr_result.error}"}), 500
        else:
            text = ocr_result.text or "[No text found in image]"

    print(f"[OCR] {f.filename} → method={method}, chars={len(text)}")
    return jsonify({
        "text":       text,
        "method":     method,
        "char_count": len(text),
        "preview":    text[:400] + ("..." if len(text) > 400 else ""),
        "filename":   f.filename,
    })


# ─── Export Draft (Chat Reply → PDF or DOCX) ─────────
@app.route("/api/export-draft", methods=["POST"])
def export_draft():
    """
    Convert a text draft to a downloadable PDF or DOCX file.

    Body: { "content": "...", "format": "pdf"|"docx", "title": "..." }
    Returns: { download_token, filename }
    """
    data    = request.get_json(silent=True) or {}
    content = data.get("content", "").strip()
    fmt     = data.get("format", "pdf").lower()
    title   = data.get("title", "ARIA Draft").strip() or "ARIA Draft"

    if not content:
        return jsonify({"error": "No content provided."}), 400
    if fmt not in ("pdf", "docx"):
        return jsonify({"error": "Format must be 'pdf' or 'docx'."}), 400

    import datetime
    date_str = datetime.date.today().isoformat()

    try:
        if fmt == "docx":
            from doc_generator import generate_docx
            doc_data = {
                "title": title,
                "author": "ARIA",
                "date": date_str,
                "sections": [{"heading": "", "body": content}],
                "footer": f"Generated by ARIA — {date_str}",
            }
            file_bytes = generate_docx(doc_data)
            filename   = f"{title.replace(' ', '_')}.docx"
            mime       = "application/vnd.openxmlformats-officedocument.wordprocessingml.document"

        else:  # pdf
            file_bytes = _generate_pdf(title, content, date_str)
            filename   = f"{title.replace(' ', '_')}.pdf"
            mime       = "application/pdf"

    except Exception as e:
        return jsonify({"error": f"Export failed: {str(e)}"}), 500

    # Store in download_store
    now   = time.time()
    token = str(uuid.uuid4())
    _download_store[token] = {
        "doc_id":   None,
        "filename": filename,
        "bytes":    file_bytes,
        "expires":  now + DOWNLOAD_TTL_SECONDS,
    }

    print(f"[Export] '{title}' → {fmt.upper()} ({len(file_bytes):,} bytes) token={token[:8]}")
    return jsonify({
        "download_token": token,
        "filename":       filename,
        "format":         fmt,
        "byte_size":      len(file_bytes),
    })


def _generate_pdf(title: str, content: str, date_str: str) -> bytes:
    """Generate a clean, styled PDF from plain text content using reportlab."""
    try:
        from reportlab.lib.pagesizes import A4
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        from reportlab.lib.units import cm
        from reportlab.lib import colors
        from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, HRFlowable
        from reportlab.lib.enums import TA_LEFT, TA_CENTER
    except ImportError:
        raise RuntimeError("reportlab not installed. Run: pip install reportlab")

    buf = io.BytesIO()
    doc = SimpleDocTemplate(
        buf,
        pagesize=A4,
        leftMargin=2.5 * cm,
        rightMargin=2.5 * cm,
        topMargin=2.5 * cm,
        bottomMargin=2.5 * cm,
    )

    styles = getSampleStyleSheet()
    accent = colors.HexColor("#00e5ff")
    dark   = colors.HexColor("#0d1117")
    mid    = colors.HexColor("#555577")

    title_style = ParagraphStyle(
        "ARIATitle",
        parent=styles["Title"],
        fontName="Helvetica-Bold",
        fontSize=22,
        textColor=dark,
        spaceAfter=4,
    )
    meta_style = ParagraphStyle(
        "ARIAMeta",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=9,
        textColor=mid,
        spaceAfter=12,
        alignment=TA_LEFT,
    )
    body_style = ParagraphStyle(
        "ARIABody",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=11,
        textColor=dark,
        leading=17,
        spaceAfter=8,
    )
    bullet_style = ParagraphStyle(
        "ARIABullet",
        parent=body_style,
        leftIndent=18,
        spaceAfter=4,
    )

    story = []
    story.append(Paragraph(title, title_style))
    story.append(Paragraph(f"Generated by ARIA &nbsp;·&nbsp; {date_str}", meta_style))
    story.append(HRFlowable(width="100%", thickness=1, color=accent, spaceAfter=14))

    # Parse content lines into paragraphs / bullets
    for line in content.splitlines():
        stripped = line.strip()
        if not stripped:
            story.append(Spacer(1, 6))
            continue
        # Markdown-style headings
        if stripped.startswith("### "):
            story.append(Paragraph(f"<b>{stripped[4:]}</b>", ParagraphStyle(
                "H3", parent=body_style, fontSize=12, spaceAfter=4, spaceBefore=10)))
        elif stripped.startswith("## "):
            story.append(Paragraph(f"<b>{stripped[3:]}</b>", ParagraphStyle(
                "H2", parent=body_style, fontSize=14, spaceAfter=6, spaceBefore=12)))
        elif stripped.startswith("# "):
            story.append(Paragraph(f"<b>{stripped[2:]}</b>", ParagraphStyle(
                "H1", parent=body_style, fontSize=16, spaceAfter=8, spaceBefore=14)))
        elif stripped.startswith(("- ", "* ", "• ")):
            story.append(Paragraph(f"• {stripped[2:]}", bullet_style))
        else:
            # Escape special XML chars for reportlab
            safe = stripped.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
            story.append(Paragraph(safe, body_style))

    story.append(Spacer(1, 24))
    story.append(HRFlowable(width="100%", thickness=0.5, color=mid))
    story.append(Paragraph("ARIA — Agentic AI Desktop Assistant", meta_style))

    doc.build(story)
    return buf.getvalue()


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


# ─── Agent Run — POST /api/agent ──────────────────────
@app.route("/api/agent", methods=["POST"])
def agent_run():
    """
    Full agentic loop: Plan → Execute → Verify → Replan → Synthesize.

    Request body:
        {
          "goal":   "Analyze the inspection report and create an approval note",
          "files":  ["doc_id_1", "doc_id_2"],   # optional: pre-uploaded doc IDs
          "system": "optional system context"
        }

    Response:
        AgentState.to_dict() plus "synthesis" field.
    """
    if not gateway.is_available():
        return jsonify({"error": "Ollama service is not running."}), 500

    data          = request.get_json(silent=True) or {}
    goal          = data.get("goal", "").strip()
    file_ids      = data.get("files", [])   # list of doc_ids already in _doc_store
    system_prompt = data.get("system")

    if not goal:
        return jsonify({"error": "No goal provided."}), 400

    # Build uploaded_files dict from pre-uploaded documents
    uploaded = {}
    for fid in file_ids:
        if fid in _doc_store:
            uploaded[fid] = _doc_store[fid]

    # Also include ALL currently uploaded docs if no specific files requested
    if not file_ids:
        uploaded = dict(_doc_store)

    state = AgentState(goal=goal, uploaded_files=uploaded)

    # Register this run for optional polling
    _agent_runs[state.run_id] = {"state": state, "synthesis": None, "done": False}

    try:
        print(f"[Agent] Starting run {state.run_id[:8]} — goal: {goal[:80]}")
        state, synthesis = agent_orchestrator.run(
            goal, state, system_prompt=system_prompt
        )

        # Expose generated files as download tokens in the main _download_store
        # (already done inside _store_generated via ToolRegistry, but re-sync here)
        result = state.to_dict()
        result["synthesis"] = synthesis

        _agent_runs[state.run_id]["synthesis"] = synthesis
        _agent_runs[state.run_id]["done"]      = True

        return jsonify(result)

    except GatewayError as e:
        return jsonify({"error": str(e)}), 502
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": f"Agent error: {str(e)}"}), 500


# ─── Agent Status Polling ─────────────────────────────
@app.route("/api/agent/status/<run_id>", methods=["GET"])
def agent_status(run_id):
    """Return current status of an agent run (for live progress updates)."""
    run = _agent_runs.get(run_id)
    if not run:
        return jsonify({"error": "Run not found."}), 404
    state: AgentState = run["state"]
    result = state.to_dict()
    result["synthesis"] = run.get("synthesis")
    result["done"]      = run["done"]
    return jsonify(result)


# ─── Agent File Download ──────────────────────────────
@app.route("/api/agent/download/<token>", methods=["GET"])
def agent_download(token):
    """Download a file generated by the agent (DOCX, PPTX, XLSX)."""
    entry = _download_store.get(token)
    if not entry:
        return jsonify({"error": "Token not found or expired."}), 404
    if time.time() > entry["expires"]:
        del _download_store[token]
        return jsonify({"error": "Token expired."}), 410

    filename   = entry["filename"]
    file_bytes = entry["bytes"]
    ext        = os.path.splitext(filename)[1].lower()
    mime_map   = {
        ".docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        ".pptx": "application/vnd.openxmlformats-officedocument.presentationml.presentation",
        ".xlsx": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        ".csv":  "text/csv",
        ".txt":  "text/plain",
    }
    mime = mime_map.get(ext, "application/octet-stream")
    return send_file(
        io.BytesIO(file_bytes),
        mimetype=mime,
        as_attachment=True,
        download_name=filename,
    )


# ─── Tool Registry Info ───────────────────────────────
@app.route("/api/tools", methods=["GET"])
def tools_info():
    """Return list of registered tools and recent tool call log."""
    return jsonify({
        "tools":    tool_registry.list_tools(),
        "call_log": tool_registry.get_log(20),
    })


# ─── Run ──────────────────────────────────────────────
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    ollama_status = "RUNNING" if gateway.is_available() else "NOT REACHABLE"
    print(f"""
+---------------------------------------------------------------+
|   ARIA v3.0 Backend — Full Agentic Architecture               |
+---------------------------------------------------------------+
|  URL:      http://localhost:{port}                               |
|  Ollama:   {ollama_status:<52}|
|  Provider: Ollama - 100% local, no internet                   |
|  Model:    {DEFAULT_MODEL:<52}|
+---------------------------------------------------------------+
|  Architecture:                                                |
|    Gateway           -> unified model interface, VRAM mgmt    |
|    Router            -> heuristic task-based model selection   |
|    ToolRegistry      -> 11 typed tools (OCR, docs, search...) |
|    AgentOrchestrator -> Plan→Execute→Verify→Replan loop       |
|    Sandbox           -> isolated code execution (10s timeout)  |
+---------------------------------------------------------------+
|  Endpoints:                                                   |
|    GET  /                      -> health check                |
|    GET  /api/health            -> detailed health + stats     |
|    GET  /api/models            -> list models + routing table |
|    GET  /api/tools             -> tool registry info          |
|    POST /api/chat              -> chat (Gateway + Router)     |
|    POST /api/tool              -> one-shot tool (Gateway)     |
|    POST /api/orchestrate       -> Deep Think (v2 Orchestrator)|
|    POST /api/agent             -> Full Agent Mode (v3)        |
|    GET  /api/agent/status/<id> -> live agent progress         |
|    GET  /api/agent/download/<t>-> download generated file     |
|    POST /api/routing-info      -> routing decisions & stats   |
|    POST /api/upload-doc        -> upload document             |
|    POST /api/doc-query         -> query document              |
|    POST /api/doc-edit          -> edit document + return file |
|    GET  /api/doc-download/<t>  -> download edited document    |
|    POST /api/stt               -> speech-to-text             |
+---------------------------------------------------------------+

Open index.html in Chrome/Edge to start ARIA v3.0.
""")
    if gateway.is_available():
        warm_up()
    else:
        print("[WARNING] Could not reach Ollama. Is it installed and running?")
        print("[WARNING] Install: https://ollama.com/download  |  Then: ollama pull llama3.2")

    app.run(host="0.0.0.0", port=port, debug=True)