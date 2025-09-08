# app/main.py
import os, time, uuid, json, re
from typing import List, Optional, Dict, Any, Literal

from fastapi import FastAPI, HTTPException, UploadFile, File, Header, Request, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# ===================== ENV =====================
APP_ENV       = os.getenv("APP_ENV", "production").lower()
MODEL_NAME    = os.getenv("MODEL_NAME", "meta-llama/Llama-3.1-8B-Instruct")
GPU_MEM_UTIL  = float(os.getenv("GPU_MEM_UTIL", "0.6"))
MAX_MODEL_LEN = int(os.getenv("MAX_MODEL_LEN", "8192"))
TEMP          = float(os.getenv("TEMP", "0.7"))
TOP_P         = float(os.getenv("TOP_P", "0.95"))
MAX_TOKENS    = int(os.getenv("MAX_TOKENS", "512"))
SHOW_TPS      = os.getenv("SHOW_TPS", "true").lower() == "true"
API_KEY       = os.getenv("API_KEY", "jakarta321")
ADMIN_TOKEN   = os.getenv("ADMIN_TOKEN", "")
HF_TOKEN      = os.getenv("HUGGINGFACE_HUB_TOKEN") or os.getenv("HF_TOKEN")
RAG_MODE      = os.getenv("RAG_MODE", "off").lower()   # off | on | auto

# ===================== State Management (In-Memory for Demo) =====================
# WARNING: NOT FOR PRODUCTION. Wiped on restart. Fails with multiple replicas.
CONVERSATION_STORE: Dict[str, List[Dict[str, Any]]] = {}
# =================================================================================

# ===================== Schemas =====================
class ChatRequest(BaseModel):
    prompt: str

class ChatResponse(BaseModel):
    response: str
    metrics: Optional[Dict[str, Any]] = None

class BenchRequest(BaseModel):
    prompts: List[str]

# OpenAI-compatible
class ChatMessage(BaseModel):
    role: str
    content: Optional[str] = None
    name: Optional[str] = None
    tool_call_id: Optional[str] = None
    # BARU: LangChain terkadang mengirim 'tool_calls' di dalam pesan user/assistant
    tool_calls: Optional[List[Dict[str, Any]]] = None


class ToolFunction(BaseModel):
    name: str
    description: Optional[str] = None
    parameters: Optional[Dict[str, Any]] = None

class ToolSpec(BaseModel):
    type: Literal["function"] = "function"
    function: ToolFunction

class ChatCompletionsRequest(BaseModel):
    model: Optional[str] = None
    messages: List[ChatMessage]
    max_tokens: Optional[int] = None
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    tools: Optional[List[Dict[str, Any]]] = None
    tool_choice: Optional[Any] = None
    # BARU: Menambahkan conversation_id untuk melacak state
    conversation_id: Optional[str] = Field(None, description="ID to track conversation history.")

def _extract_first_json_obj(text: str) -> Optional[Dict[str, Any]]:
    """Ambil JSON object pertama yang balanced dari string."""
    if not text:
        return None
    start = text.find("{")
    if start == -1:
        return None
    depth = 0
    in_str = False
    esc = False
    for i in range(start, len(text)):
        ch = text[i]
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
        else:
            if ch == '"':
                in_str = True
            elif ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    try:
                        return json.loads(text[start:i+1])
                    except Exception:
                        return None
    return None

def _canon(s: str) -> str:
    return re.sub(r"[^a-z0-9]", "", (s or "").lower())

def _map_tool_name_to_allowed(name: str, allowed_tools: Optional[List[Dict[str, Any]]]) -> str:
    """Cocokkan nama tool LLM ke daftar tool yang dikirim klien (n8n)."""
    if not allowed_tools:
        return name
    target = _canon(name)
    best = name
    for t in allowed_tools:
        if t.get("type") != "function": 
            continue
        fn = (t.get("function") or {}).get("name")
        if not fn:
            continue
        if _canon(fn) == target:
            return fn
    # fallback: fuzzy sederhana (startswith)
    for t in allowed_tools:
        if t.get("type") != "function":
            continue
        fn = (t.get("function") or {}).get("name")
        if fn and (_canon(fn).startswith(target) or target.startswith(_canon(fn))):
            return fn
    return name

def _build_tool_calls_from_text(text: str, allowed_tools: Optional[List[Dict[str, Any]]]) -> Optional[List[Dict[str, Any]]]:
    """
    Baca teks model. Jika berisi JSON dengan kunci 'tool_calls', konversi ke schema OpenAI:
    [
      { "id":"call_1", "type":"function", "function":{"name":"...","arguments":"{...}"} }, ...
    ]
    """
    obj = _extract_first_json_obj(text)
    if not isinstance(obj, dict):
        return None
    tc = obj.get("tool_calls")
    if not isinstance(tc, list) or not tc:
        # dukung format alternatif: {"invocations":[{"name":"...","arguments":{...}}]}
        tc = obj.get("invocations")
        if not isinstance(tc, list) or not tc:
            return None

    out = []
    for idx, item in enumerate(tc, start=1):
        # normalisasi bentuk
        if "function" in item and isinstance(item["function"], dict):
            fn_name = item["function"].get("name")
            args = item["function"].get("arguments", {})
        else:
            fn_name = item.get("name")
            args = item.get("arguments", {})
        # pastikan args adalah string JSON sesuai OpenAI schema
        if not isinstance(args, str):
            try:
                args = json.dumps(args or {})
            except Exception:
                args = "{}"

        mapped_name = _map_tool_name_to_allowed(fn_name or "", allowed_tools)
        out.append({
            "id": f"call_{idx}",
            "type": "function",
            "function": {
                "name": mapped_name or (fn_name or ""),
                "arguments": args
            }
        })
    return out or None

# ===================== Tokenizer helpers =====================
tokenizer = None
chat_tmpl_ok = False

def count_tokens(text: str) -> int:
    if not tokenizer:
        return max(1, len(text)//4)
    try:
        return len(tokenizer.encode(text))
    except Exception:
        return max(1, len(text)//4)

try:
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True, token=HF_TOKEN)
    chat_tmpl_ok = bool(getattr(tokenizer, "chat_template", None))
except Exception as e:
    print(f"[Tokenizer] init error: {e} (lanjut tanpa tokenizer)")

def to_prompt_from_messages(messages: List[Dict[str,str]]) -> str:
    if tokenizer and chat_tmpl_ok:
        try:
            return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        except Exception as e:
            print(f"[ChatTemplate] apply failed: {e} (fallback)")
    parts = []
    for m in messages:
        role = m.get("role","user")
        content = m.get("content","")
        # Handle pesan 'tool' yang content-nya mungkin bukan string
        if not isinstance(content, str):
            content = json.dumps(content, ensure_ascii=False)
        parts.append(f"{role.upper()}: {content}")
    parts.append("ASSISTANT:")
    return "\n".join(parts)

# ===================== LLM init (no lazy) =====================
llm = None
sampling_params = None
vllm_last_error: Optional[str] = None

if APP_ENV == "local":
    class MockLLM:
        def generate(self, prompt_template: str, sampling_params: object):
            time.sleep(0.1)
            class _O:
                def __init__(self, t): self.text=t
            class _R:
                def __init__(self, t): self.outputs=[_O(t)]
            return [_R("Ini jawaban mock. (ENV=local)")]
    llm = MockLLM()
else:
    try:
        from vllm import LLM, SamplingParams
        llm = LLM(
            model=MODEL_NAME,
            trust_remote_code=True,
            max_model_len=MAX_MODEL_LEN,
            gpu_memory_utilization=GPU_MEM_UTIL,
        )
        sampling_params = SamplingParams(
            temperature=TEMP,
            top_p=TOP_P,
            max_tokens=MAX_TOKENS,
        )
    except Exception as e:
        vllm_last_error = str(e)
        print(f"[LLM] init error: {e}")

# ===================== RAG (optional, non-strict) =====================
# (Tidak ada perubahan di bagian ini)
retriever = None
try:
    from rag_processor import (
        get_retriever, ensure_vector_db_ready,
        ingest_documents, list_documents, delete_document
    )
    ensure_vector_db_ready()
    retriever = get_retriever()
except Exception as e:
    print(f"[RAG] init warning: {e} (RAG tetap opsional)")

def build_messages_with_optional_context(user_prompt: str):
    context = ""
    if RAG_MODE in ("on", "auto") and retriever is not None:
        try:
            docs = retriever.invoke(user_prompt)
            if docs:
                context = "\n\n".join([d.page_content for d in docs])[:4000]
        except Exception as e:
            print(f"[RAG] retrieve err: {e}")
    system_msg = {
        "role": "system",
        "content": (
            "You are a helpful assistant. If a 'Context' section is provided and relevant, use it. "
            "If context is missing or insufficient, answer using your general knowledge clearly. "
            "If unsure, say you are unsure."
        )
    }
    user_content = user_prompt if not context else f"Context:\n{context}\n\nQuestion:\n{user_prompt}"
    user_msg = {"role":"user","content": user_content}
    return [system_msg, user_msg]

# ===================== Tool-calling helpers =====================
# (Tidak ada perubahan di bagian ini)
def _extract_json_first(s: str) -> Optional[dict]:
    s = (s or "").strip()
    if s.startswith("```"):
        s = re.sub(r"^```[a-zA-Z0-9_+-]*", "", s).strip()
        if s.endswith("```"):
            s = s[:-3]
    try:
        return json.loads(s)
    except Exception:
        pass
    start = s.find("{")
    while start != -1:
        depth = 0
        i = start
        while i < len(s):
            ch = s[i]
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    cand = s[start:i+1]
                    try:
                        return json.loads(cand)
                    except Exception:
                        break
            i += 1
        start = s.find("{", start+1)
    return None

def _tool_system_instructions(tools: List[ToolSpec], tool_choice: Optional[Any]) -> str:
    tgt = None
    if isinstance(tool_choice, dict):
        fn = tool_choice.get("function", {})
        tgt = fn.get("name")
    lines = [
        "TOOLS AVAILABLE:",
        json.dumps({"tools":[t.model_dump() for t in tools]}, ensure_ascii=False),
        "",
        "If tools are provided:",
        '- If you decide to call tool(s), output ONLY a single JSON object of the form:',
        '  {"tool_calls":[{"type":"function","function":{"name":"<tool_name>","arguments":{...}}}, ...]}',
        '- If no tool is needed, output ONLY: {"final":true,"content":"<your answer>"}',
        "Do NOT add backticks. Do NOT add explanations outside JSON.",
    ]
    if tgt:
        lines.append(f'Prefer calling function name "{tgt}" when appropriate.')
    return "\n".join(lines)

def _maybe_wrap_toolcall_response(text: str):
    """Parse teks model → (tool_calls or final content)."""
    data = _extract_json_first(text)
    if isinstance(data, dict):
        # tool_calls path
        if "tool_calls" in data and isinstance(data["tool_calls"], list) and data["tool_calls"]:
            tool_calls = []
            for _tc in data["tool_calls"]:
                if not isinstance(_tc, dict): continue
                if _tc.get("type") != "function": continue
                fn = _tc.get("function", {})
                name = fn.get("name") or ""
                
                # --- PERUBAHAN DI SINI ---
                # Ambil argumen
                args = fn.get("arguments")
                
                # Pastikan argumen adalah dict, bukan string. JANGAN di-dumps.
                if not isinstance(args, dict):
                    args = {} # Jika format salah, default ke objek kosong

                tool_calls.append({
                    "id": f"call_{uuid.uuid4().hex[:12]}",
                    "type": "function",
                    "function": {"name": name, "arguments": args} # 'args' sekarang adalah dict
                })
            if tool_calls:
                return {"tool_calls": tool_calls}
        # final content path
        if data.get("final") is True and isinstance(data.get("content"), str):
            return {"final": data["content"]}
    # fallback → treat as normal answer
    return {"final": text}

# ===================== Auth (Bearer) =====================
# (Tidak ada perubahan di bagian ini)
def verify_bearer(req: Request):
    auth = req.headers.get("authorization", "")
    if not auth.startswith("Bearer "):
        raise HTTPException(401, "Invalid API key")
    token = auth.split(" ", 1)[1].strip()
    if token != API_KEY:
        raise HTTPException(401, "Invalid API key")

# ===================== FastAPI app =====================
app = FastAPI(title="General LLM API (OpenAI-compatible, Tools, optional RAG)", version="1.2.0-stateful")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"]
)

@app.get("/")
def health():
    return {
        "status": "ok",
        "env": APP_ENV,
        "model": MODEL_NAME,
        "rag_mode": RAG_MODE,
        "rag_available": bool(retriever),
        "vllm_ready": llm is not None,
        "vllm_last_error": vllm_last_error,
        "active_conversations": len(CONVERSATION_STORE)
    }

# ---------------- /v1/models ----------------
# (Tidak ada perubahan di bagian ini)
@app.get("/v1/models")
def list_models(_: None = Depends(verify_bearer)):
    return {
        "object": "list",
        "data": [{
            "id": MODEL_NAME, "object": "model", "owned_by": "local", "created": int(time.time()),
        }]
    }

# ---------------- Simple chat ----------------
# (Tidak ada perubahan di bagian ini)
@app.post("/v1/chat", response_model=ChatResponse)
async def chat(req: ChatRequest, _: None = Depends(verify_bearer)):
    if llm is None: raise HTTPException(503, "LLM not available.")
    messages = build_messages_with_optional_context(req.prompt)
    prompt = to_prompt_from_messages(messages)
    t0=time.perf_counter()
    try:
        out = llm.generate(prompt, sampling_params)
        text = out[0].outputs[0].text
    except Exception as e:
        print(f"[LLM] error: {e}")
        raise HTTPException(500, "Inference error.")
    dt = max(1e-6, time.perf_counter()-t0)
    toks_in  = count_tokens(prompt); toks_out = count_tokens(text)
    metrics = {"latency_sec": round(dt,3), "tokens_in": toks_in,
               "tokens_out": toks_out, "tok_per_sec": round(toks_out/dt,2)} if SHOW_TPS else None
    return ChatResponse(response=text, metrics=metrics)

# ---------------- OpenAI-compatible chat/completions (DIROMBAK TOTAL) ----------------
@app.post("/v1/chat/completions")
async def chat_completions(req: ChatCompletionsRequest):
    if llm is None:
        raise HTTPException(503, "LLM not available.")

    # Base system
    base_sys = {
        "role":"system",
        "content":"You are a helpful assistant. Prefer context if provided. If absent, answer with general knowledge."
    }
    messages = [base_sys] + [m.model_dump() for m in req.messages]

    # Optional RAG context (non-strict)
    try:
        last_user = next((m["content"] for m in reversed(messages) if m["role"]=="user"), "")
        if RAG_MODE in ("on","auto") and retriever is not None and last_user:
            docs = retriever.invoke(last_user)
            if docs:
                ctx = "\n\n".join([d.page_content for d in docs])[:4000]
                messages.append({"role":"system","content": f"Context:\n{ctx}"})
    except Exception as e:
        print(f"[RAG] optional ctx error: {e}")

    # Jika tools ada & tool_choice != "none", paksa format JSON tool_calls
    tool_mode = bool(req.tools) and (req.tool_choice != "none")
    if tool_mode:
        tool_lines = []
        for t in (req.tools or []):
            if t.get("type") == "function" and t.get("function"):
                fn = t["function"]
                nm = fn.get("name","")
                desc = fn.get("description","")
                tool_lines.append(f"- {nm}: {desc}")
        tools_block = "\n".join(tool_lines) if tool_lines else "(no tool descriptions)"

        guidance = {
            "role":"system",
            "content":(
                "You may call ONE function if useful. "
                "Return ONLY a JSON object with this shape and NOTHING else:\n"
                "{\"tool_calls\":[{\"type\":\"function\",\"function\":{\"name\":\"<one of allowed>\",\"arguments\":{...}}}]}\n"
                "Allowed tools:\n" + tools_block
            )
        }
        messages = messages + [guidance]

    prompt = to_prompt_from_messages(messages)

    # Sampling params override (opsional)
    sp = sampling_params
    if req.max_tokens or req.temperature or req.top_p:
        from vllm import SamplingParams
        sp = SamplingParams(
            temperature=req.temperature if req.temperature is not None else TEMP,
            top_p=req.top_p if req.top_p is not None else TOP_P,
            max_tokens=req.max_tokens if req.max_tokens is not None else MAX_TOKENS,
        )

    # Generate
    t0=time.perf_counter()
    try:
        out = llm.generate(prompt, sp)
        text = out[0].outputs[0].text
    except Exception as e:
        print(f"[LLM] error: {e}")
        raise HTTPException(500, "Inference error.")

    dt = max(1e-6, time.perf_counter()-t0)
    toks_in  = count_tokens(prompt)
    toks_out = count_tokens(text)

    # === Tool-calls bridging ===
    tool_calls_block = None
    if tool_mode:
        tool_calls_block = _build_tool_calls_from_text(text, req.tools)

    if tool_calls_block:
        # n8n butuh content string (bukan null) saat ada tool_calls
        message_obj = {"role":"assistant","content":"", "tool_calls": tool_calls_block}
        finish = "tool_calls"
    else:
        message_obj = {"role":"assistant","content": text or ""}
        finish = "stop"

    resp = {
        "id": f"chatcmpl-{uuid.uuid4()}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": req.model or MODEL_NAME,
        "choices": [{
            "index": 0,
            "message": message_obj,
            "finish_reason": finish
        }],
        "usage": {
            "prompt_tokens": toks_in,
            "completion_tokens": toks_out,
            "total_tokens": toks_in + toks_out
        }
    }
    if SHOW_TPS:
        resp["metrics"] = {"latency_sec": round(dt,3), "tps_out": round(toks_out/dt,2)}
    return resp

# ---------------- Endpoint baru untuk membersihkan state ----------------
@app.delete("/v1/conversations/{conv_id}")
def delete_conversation(conv_id: str, _: None = Depends(verify_bearer)):
    if conv_id in CONVERSATION_STORE:
        del CONVERSATION_STORE[conv_id]
        return {"status": "ok", "deleted": conv_id}
    else:
        raise HTTPException(404, f"Conversation ID '{conv_id}' not found.")


# ---------------- Sisa endpoint (Bench & RAG) tidak berubah ----------------
@app.post("/v1/bench")
def bench(req: BenchRequest, _: None = Depends(verify_bearer)):
    if llm is None: raise HTTPException(503, "LLM not available.")
    tms=[]
    for p in req.prompts:
        messages = build_messages_with_optional_context(p)
        prompt = to_prompt_from_messages(messages)
        t0=time.perf_counter(); _=llm.generate(prompt, sampling_params)
        tms.append(time.perf_counter()-t0)
    tms_s=sorted(tms); n=len(tms)
    return {
        "count": n,
        "latency": {"avg_sec": sum(tms)/n, "p50_sec": tms_s[n//2], "p90_sec": tms_s[max(0, int(0.9*n)-1)]},
        "model": MODEL_NAME, "max_tokens_param": MAX_TOKENS
    }

@app.get("/v1/docs")
def docs_list(_: None = Depends(verify_bearer)):
    if "list_documents" not in globals(): return {"items": []}
    try: return {"items": list_documents()}
    except Exception as e: raise HTTPException(500, f"Gagal baca daftar dokumen: {e}")

@app.post("/v1/docs/upload")
async def docs_upload(files: List[UploadFile] = File(...), _: None = Depends(verify_bearer)):
    if "ingest_documents" not in globals(): raise HTTPException(501, "RAG not available in this image.")
    try:
        file_tuples=[]
        for f in files:
            b=await f.read(); file_tuples.append((f.name, b))
        res=ingest_documents(file_tuples)
        global retriever; retriever = get_retriever()
        return res
    except Exception as e: print(f"[UPLOAD] error: {e}"); raise HTTPException(500, "Gagal unggah/memproses dokumen")

@app.delete("/v1/docs/{doc_id}")
def docs_delete(doc_id: str, x_admin_token: str = Header(default=""), _: None = Depends(verify_bearer)):
    if not ADMIN_TOKEN or x_admin_token != ADMIN_TOKEN: raise HTTPException(401, "Unauthorized")
    if "delete_document" not in globals(): raise HTTPException(501, "RAG not available in this image.")
    try:
        delete_document(doc_id); return {"deleted": doc_id}
    except Exception as e: raise HTTPException(500, f"Gagal hapus dokumen: {e}")