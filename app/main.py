import os, time, uuid
from typing import List, Optional, Dict, Any
from fastapi import FastAPI, HTTPException, UploadFile, File, Header, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel

# ================== ENV ==================
APP_ENV       = os.getenv("APP_ENV", "production").lower()
MODEL_NAME    = os.getenv("MODEL_NAME", "meta-llama/Llama-3.1-8B-Instruct")
GPU_MEM_UTIL  = float(os.getenv("GPU_MEM_UTIL", "0.22"))   # kecil biar aman di GPU shared
MAX_MODEL_LEN = int(os.getenv("MAX_MODEL_LEN", "4096"))    # bisa dinaikkan bertahap
TEMP          = float(os.getenv("TEMP", "0.7"))
TOP_P         = float(os.getenv("TOP_P", "0.95"))
MAX_TOKENS    = int(os.getenv("MAX_TOKENS", "1024"))
SHOW_TPS      = os.getenv("SHOW_TPS", "true").lower() == "true"
HF_TOKEN      = os.getenv("HUGGINGFACE_HUB_TOKEN") or os.getenv("HF_TOKEN")
RAG_MODE      = os.getenv("RAG_MODE", "off").lower()       # off|on|auto (non-strict)
API_KEY       = os.getenv("OPENAI_API_KEY") or os.getenv("API_KEY")  # e.g. "jakarta321"

# vLLM tuning untuk footprint GPU
KV_CACHE_DTYPE = os.getenv("KV_CACHE_DTYPE", "fp8")        # fp8|fp16|bf16|auto
SWAP_SPACE_GB  = int(os.getenv("SWAP_SPACE", "0"))         # 0=off; 8..16 kalau mau offload KV ke RAM

# ===== Pydantic =====
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
    content: str

class ChatCompletionsRequest(BaseModel):
    model: Optional[str] = None
    messages: List[ChatMessage]
    max_tokens: Optional[int] = None
    temperature: Optional[float] = None
    top_p: Optional[float] = None

# ===== Auth (OpenAI style) =====
bearer = HTTPBearer(auto_error=False)
def require_api_key(auth: HTTPAuthorizationCredentials = Depends(bearer)):
    if not API_KEY:
        return  # tanpa API key, endpoint tetap terbuka
    if not auth or not auth.credentials or auth.credentials != API_KEY:
        raise HTTPException(401, "Invalid API key")

# ===== Tokenizer (opsional) =====
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

def to_prompt_from_messages(messages: List[Dict[str, str]]) -> str:
    # Pakai chat_template kalau ada
    if tokenizer and chat_tmpl_ok:
        try:
            return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        except Exception as e:
            print(f"[ChatTemplate] apply failed: {e} (fallback)")
    # fallback format sederhana
    parts = []
    for m in messages:
        role = m.get("role", "user").upper()
        content = m.get("content", "")
        parts.append(f"{role}: {content}")
    parts.append("ASSISTANT:")
    return "\n".join(parts)

# ===== RAG (opsional, non-strict) =====
retriever = None
def _rag_noop_list(): return []
def _rag_noop_ingest(_files): return {"saved": [], "reindexed": False}
def _rag_noop_delete(_id): return None
def _rag_noop_get(): return None
def _rag_noop_ensure(): return None

list_documents     = _rag_noop_list
ingest_documents   = _rag_noop_ingest
delete_document    = _rag_noop_delete
get_retriever      = _rag_noop_get
ensure_vector_db_ready = _rag_noop_ensure

try:
    from rag_processor import (
        get_retriever as _get_ret,
        ensure_vector_db_ready as _ensure,
        ingest_documents as _ingest,
        list_documents as _list,
        delete_document as _del,
    )
    ensure_vector_db_ready = _ensure
    ingest_documents       = _ingest
    list_documents         = _list
    delete_document        = _del
    ensure_vector_db_ready()
    retriever = _get_ret()
except Exception as e:
    print(f"[RAG] init warning: {e} (RAG tetap opsional)")

def build_messages_with_optional_context(user_prompt: str) -> List[Dict[str, str]]:
    """Non-strict RAG: kalau ada konteks, pakai; kalau tidak, jawab dg pengetahuan umum."""
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
            "You are a helpful assistant. Prefer using the provided context if relevant. "
            "If context is missing/insufficient, answer from your general knowledge. "
            "If unsure, say you are unsure."
        ),
    }
    user_content = user_prompt if not context else f"Context:\n{context}\n\nQuestion:\n{user_prompt}"
    return [system_msg, {"role": "user", "content": user_content}]

# ===== vLLM INIT — EAGER (NO LAZY) =====
llm = None
sampling_params = None
VLLM_READY = False
VLLM_LAST_ERROR: Optional[str] = None

if APP_ENV == "local":
    class _MockOut:  # mock untuk dev lokal
        def __init__(self, t): self.text = t
    class _MockRes:
        def __init__(self, t): self.outputs = [_MockOut(t)]
    class MockLLM:
        def generate(self, prompt_template: str, sampling_params: object):
            time.sleep(0.1)
            return [_MockRes("Ini jawaban mock. (ENV=local)")]

    llm = MockLLM()
    from types import SimpleNamespace
    sampling_params = SimpleNamespace(temperature=TEMP, top_p=TOP_P, max_tokens=MAX_TOKENS)
    VLLM_READY = True
else:
    try:
        from vllm import LLM, SamplingParams
        llm = LLM(
            model=MODEL_NAME,
            trust_remote_code=True,
            max_model_len=MAX_MODEL_LEN,
            gpu_memory_utilization=GPU_MEM_UTIL,
            kv_cache_dtype=KV_CACHE_DTYPE,   # hemat KV di GPU shared (H100: fp8 ok)
            swap_space=SWAP_SPACE_GB,        # opsional offload KV ke RAM
        )
        sampling_params = SamplingParams(
            temperature=TEMP,
            top_p=TOP_P,
            max_tokens=MAX_TOKENS,
        )
    except Exception as e:
        VLLM_LAST_ERROR = str(e)
        print(f"[LLM] init error: {e}")

# ===== FastAPI App =====
app = FastAPI(title="General LLM API (OpenAI-compatible)", version="1.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"]
)

@app.on_event("startup")
async def _warmup():
    """Warm-up agar compile/graph capture selesai sebelum serve request pertama."""
    global VLLM_READY, VLLM_LAST_ERROR
    if llm is None:
        return
    try:
        _ = llm.generate(
            to_prompt_from_messages([
                {"role":"system","content":"warmup"},
                {"role":"user","content":"hi"},
            ]),
            sampling_params
        )
        VLLM_READY = True
    except Exception as e:
        VLLM_LAST_ERROR = str(e)
        print(f"[Warmup] error: {e}")

# ===== Health =====
@app.get("/")
def health():
    return {
        "status": "ok",
        "env": APP_ENV,
        "model": MODEL_NAME,
        "rag_mode": RAG_MODE,
        "rag_available": bool(retriever),
        "vllm_ready": VLLM_READY,
        "vllm_last_error": VLLM_LAST_ERROR,
    }

# ===== OpenAI-compatible =====
@app.get("/v1/models")
def list_models(_=Depends(require_api_key)):
    return {
        "object": "list",
        "data": [{
            "id": MODEL_NAME,
            "object": "model",
            "owned_by": "local",
            "created": int(time.time()),
        }]
    }

@app.post("/v1/chat/completions")
def chat_completions(req: ChatCompletionsRequest, _=Depends(require_api_key)):
    if llm is None or not VLLM_READY:
        raise HTTPException(503, "LLM not available.")
    # gabungkan pesan user + system non-strict
    messages = [{"role":"system","content":
                 "You are a helpful assistant. Prefer context if present; otherwise use general knowledge."}]
    messages += [{"role": m.role, "content": m.content} for m in req.messages]

    # optional RAG hanya menambah system-context; tidak strict
    try:
        last_user = next((m.content for m in reversed(req.messages) if m.role == "user"), "")
        if RAG_MODE in ("on", "auto") and retriever is not None and last_user:
            docs = retriever.invoke(last_user)
            if docs:
                ctx = "\n\n".join([d.page_content for d in docs])[:4000]
                messages.append({"role":"system","content": f"Context:\n{ctx}"})
    except Exception as e:
        print(f"[RAG] optional ctx error: {e}")

    prompt = to_prompt_from_messages(messages)
    sp = sampling_params
    if req.max_tokens or req.temperature or req.top_p:
        from vllm import SamplingParams
        sp = SamplingParams(
            temperature=req.temperature if req.temperature is not None else TEMP,
            top_p=req.top_p if req.top_p is not None else TOP_P,
            max_tokens=req.max_tokens if req.max_tokens is not None else MAX_TOKENS,
        )

    t0 = time.perf_counter()
    try:
        out = llm.generate(prompt, sp)
        text = out[0].outputs[0].text
    except Exception as e:
        print(f"[LLM] error: {e}")
        raise HTTPException(500, "Inference error.")

    dt = max(1e-6, time.perf_counter() - t0)
    toks_in  = count_tokens(prompt)
    toks_out = count_tokens(text)
    return {
        "id": f"chatcmpl-{uuid.uuid4()}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": req.model or MODEL_NAME,
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": text},
            "finish_reason": "stop"
        }],
        "usage": {
            "prompt_tokens": toks_in,
            "completion_tokens": toks_out,
            "total_tokens": toks_in + toks_out
        },
        "metrics": (
            {"latency_sec": round(dt,3), "tps_out": round(toks_out/dt,2)}
            if SHOW_TPS else None
        )
    }

# ===== Simple chat (tanpa format OpenAI) =====
@app.post("/v1/chat", response_model=ChatResponse)
def chat(req: ChatRequest, _=Depends(require_api_key)):
    if llm is None or not VLLM_READY:
        raise HTTPException(503, "LLM not available.")
    messages = build_messages_with_optional_context(req.prompt)
    prompt = to_prompt_from_messages(messages)
    t0 = time.perf_counter()
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

# ===== Bench =====
@app.post("/v1/bench")
def bench(req: BenchRequest, _=Depends(require_api_key)):
    if llm is None or not VLLM_READY:
        raise HTTPException(503, "LLM not available.")
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

# ===== RAG endpoints (opsional) =====
@app.get("/v1/docs")
def docs_list(_=Depends(require_api_key)):
    try:
        return {"items": list_documents()}
    except Exception as e:
        raise HTTPException(500, f"Gagal baca daftar dokumen: {e}")

@app.post("/v1/docs/upload")
async def docs_upload(files: List[UploadFile] = File(...), _=Depends(require_api_key)):
    try:
        file_tuples=[]
        for f in files:
            b=await f.read()
            file_tuples.append((f.filename, b))
        res=ingest_documents(file_tuples)
        # refresh retriever
        global retriever
        retriever = get_retriever()
        return res
    except Exception as e:
        print(f"[UPLOAD] error: {e}")
        raise HTTPException(500, "Gagal unggah/memproses dokumen")

@app.delete("/v1/docs/{doc_id}")
def docs_delete(doc_id: str, x_admin_token: str = Header(default=""), _=Depends(require_api_key)):
    admin_token = os.getenv("ADMIN_TOKEN", "")
    if not admin_token or x_admin_token != admin_token:
        raise HTTPException(401, "Unauthorized")
    try:
        delete_document(doc_id)
        return {"deleted": doc_id}
    except Exception as e:
        raise HTTPException(500, f"Gagal hapus dokumen: {e}")
