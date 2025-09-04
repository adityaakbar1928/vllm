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

class ToolFunction(BaseModel):
    name: str
    description: Optional[str] = None
    parameters: Optional[Dict[str, Any]] = None

class ToolSpec(BaseModel):
    # Pydantic v2: gunakan Literal, jangan Field(..., const=True)
    type: Literal["function"] = "function"
    function: ToolFunction

class ChatCompletionsRequest(BaseModel):
    model: Optional[str] = None
    messages: List[ChatMessage]
    max_tokens: Optional[int] = None
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    tools: Optional[List[ToolSpec]] = None
    tool_choice: Optional[Any] = None  # "none" | "auto" | {"type":"function","function":{"name":"..."}}

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
    """Pakai chat_template kalau ada, fallback format aman."""
    if tokenizer and chat_tmpl_ok:
        try:
            return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        except Exception as e:
            print(f"[ChatTemplate] apply failed: {e} (fallback)")
    # fallback sederhana
    parts = []
    for m in messages:
        role = m.get("role","user")
        content = m.get("content","")
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
            class _O:  # minimal output shape
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
    """
    Non-strict:
      - RAG_MODE=off  → tanpa konteks.
      - RAG_MODE=on   → coba ambil konteks; bila kosong, tetap jawab pakai general knowledge.
      - RAG_MODE=auto → ambil konteks; kalau tak relevan, tetap jawab pakai general knowledge.
    """
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
def _extract_json_first(s: str) -> Optional[dict]:
    """Ekstrak objek JSON pertama secara robust (tanpa regex recursive)."""
    s = (s or "").strip()
    # strip code fences
    if s.startswith("```"):
        s = re.sub(r"^```[a-zA-Z0-9_+-]*", "", s).strip()
        if s.endswith("```"):
            s = s[:-3]
    # coba parse full
    try:
        return json.loads(s)
    except Exception:
        pass
    # scan objek {...} pertama yang balanced
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
                args = fn.get("arguments")
                if isinstance(args, (dict, list)):
                    args = json.dumps(args, ensure_ascii=False)
                if not isinstance(args, str):
                    args = "{}"
                tool_calls.append({
                    "id": f"call_{uuid.uuid4().hex[:12]}",
                    "type": "function",
                    "function": {"name": name, "arguments": args}
                })
            if tool_calls:
                return {"tool_calls": tool_calls}
        # final content path
        if data.get("final") is True and isinstance(data.get("content"), str):
            return {"final": data["content"]}
    # fallback → treat as normal answer
    return {"final": text}

# ===================== Auth (Bearer) =====================
def verify_bearer(req: Request):
    auth = req.headers.get("authorization", "")
    if not auth.startswith("Bearer "):
        raise HTTPException(401, "Invalid API key")
    token = auth.split(" ", 1)[1].strip()
    if token != API_KEY:
        raise HTTPException(401, "Invalid API key")

# ===================== FastAPI app =====================
app = FastAPI(title="General LLM API (OpenAI-compatible, Tools, optional RAG)", version="1.1.1")
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
    }

# ---------------- /v1/models ----------------
@app.get("/v1/models")
def list_models(_: None = Depends(verify_bearer)):
    return {
        "object": "list",
        "data": [{
            "id": MODEL_NAME,
            "object": "model",
            "owned_by": "local",
            "created": int(time.time()),
        }]
    }

# ---------------- Simple chat ----------------
@app.post("/v1/chat", response_model=ChatResponse)
async def chat(req: ChatRequest, _: None = Depends(verify_bearer)):
    if llm is None:
        raise HTTPException(503, "LLM not available.")
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
    toks_in  = count_tokens(prompt)
    toks_out = count_tokens(text)
    metrics = {"latency_sec": round(dt,3), "tokens_in": toks_in,
               "tokens_out": toks_out, "tok_per_sec": round(toks_out/dt,2)} if SHOW_TPS else None
    return ChatResponse(response=text, metrics=metrics)

# ---------------- OpenAI-compatible chat/completions ----------------
@app.post("/v1/chat/completions")
async def chat_completions(req: ChatCompletionsRequest, _: None = Depends(verify_bearer)):
    if llm is None:
        raise HTTPException(503, "LLM not available.")

    # Base system (general purpose)
    base_sys = {
        "role":"system",
        "content":"You are a helpful assistant. Prefer provided context; if absent, answer with general knowledge clearly."
    }
    messages = [base_sys] + [m.model_dump() for m in req.messages]

    # Optional RAG enrichment (non-strict)
    try:
        last_user = next((m["content"] for m in reversed(messages) if m.get("role")=="user"), "")
        if RAG_MODE in ("on","auto") and retriever is not None and last_user:
            docs = retriever.invoke(last_user)
            if docs:
                ctx = "\n\n".join([d.page_content for d in docs])[:4000]
                messages.append({"role":"system","content": f"Context:\n{ctx}"})
    except Exception as e:
        print(f"[RAG] optional ctx error: {e}")

    # Tools? → tambahkan instruksi supaya model mengeluarkan JSON tool_calls/final
    if req.tools:
        tool_sys = {"role":"system", "content": _tool_system_instructions(req.tools, req.tool_choice)}
        messages = [tool_sys] + messages

    prompt = to_prompt_from_messages(messages)

    # Sampling override
    sp = sampling_params
    if req.max_tokens or req.temperature or req.top_p:
        from vllm import SamplingParams
        sp = SamplingParams(
            temperature=req.temperature if req.temperature is not None else TEMP,
            top_p=req.top_p if req.top_p is not None else TOP_P,
            max_tokens=req.max_tokens if req.max_tokens is not None else MAX_TOKENS,
        )

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

    # Post-process tools
    tool_calls_block = None
    final_content = None
    if req.tools:
        parsed = _maybe_wrap_toolcall_response(text)
        if "tool_calls" in parsed:
            tool_calls_block = parsed["tool_calls"]
        else:
            final_content = parsed.get("final", text)
    else:
        final_content = text

    resp = {
        "id": f"chatcmpl-{uuid.uuid4()}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": req.model or MODEL_NAME,
        "choices": [{
            "index": 0,
            "message": (
                {"role":"assistant","content": None, "tool_calls": tool_calls_block}
                if tool_calls_block is not None
                else {"role":"assistant","content": final_content}
            ),
            "finish_reason": "tool_calls" if tool_calls_block is not None else "stop"
        }],
        "usage": {
            "prompt_tokens": toks_in,
            "completion_tokens": toks_out,
            "total_tokens": toks_in + toks_out
        },
        "metrics": {"latency_sec": round(dt,3), "tps_out": round(toks_out/dt,2)} if SHOW_TPS else None
    }
    return resp

# ---------------- Bench ----------------
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
        "model": MODEL_NAME,
        "max_tokens_param": MAX_TOKENS
    }

# ---------------- RAG endpoints (opsional) ----------------
@app.get("/v1/docs")
def docs_list(_: None = Depends(verify_bearer)):
    if "list_documents" not in globals():
        return {"items": []}
    try:
        return {"items": list_documents()}
    except Exception as e:
        raise HTTPException(500, f"Gagal baca daftar dokumen: {e}")

@app.post("/v1/docs/upload")
async def docs_upload(files: List[UploadFile] = File(...), _: None = Depends(verify_bearer)):
    if "ingest_documents" not in globals():
        raise HTTPException(501, "RAG not available in this image.")
    try:
        file_tuples=[]
        for f in files:
            b=await f.read()
            file_tuples.append((f.filename, b))
        res=ingest_documents(file_tuples)
        # refresh retriever
        if "get_retriever" in globals():
            global retriever
            retriever = get_retriever()
        return res
    except Exception as e:
        print(f"[UPLOAD] error: {e}")
        raise HTTPException(500, "Gagal unggah/memproses dokumen")

@app.delete("/v1/docs/{doc_id}")
def docs_delete(doc_id: str, x_admin_token: str = Header(default=""), _: None = Depends(verify_bearer)):
    if not ADMIN_TOKEN or x_admin_token != ADMIN_TOKEN:
        raise HTTPException(401, "Unauthorized")
    if "delete_document" not in globals():
        raise HTTPException(501, "RAG not available in this image.")
    try:
        delete_document(doc_id)
        return {"deleted": doc_id}
    except Exception as e:
        raise HTTPException(500, f"Gagal hapus dokumen: {e}")
