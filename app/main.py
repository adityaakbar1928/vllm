# app/main.py
import os, time, uuid, json, re
from typing import List, Optional, Dict, Any, Literal, Union

from fastapi import FastAPI, HTTPException, UploadFile, File, Header, Request, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# ===================== ENV =====================
APP_ENV       = os.getenv("APP_ENV", "production").lower()
MODEL_NAME    = os.getenv("MODEL_NAME", "meta-llama/Llama-3.1-8B-Instruct")
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
GPU_MEM_UTIL  = float(os.getenv("GPU_MEM_UTIL", "0.6"))
MAX_MODEL_LEN = int(os.getenv("MAX_MODEL_LEN", "8192"))
TEMP          = float(os.getenv("TEMP", "0.1"))  # Lower temperature for better tool calling
TOP_P         = float(os.getenv("TOP_P", "0.95"))
MAX_TOKENS    = int(os.getenv("MAX_TOKENS", "1024"))  # Increased for tool calls
SHOW_TPS      = os.getenv("SHOW_TPS", "true").lower() == "true"
API_KEY       = os.getenv("API_KEY", "jakarta321")
ADMIN_TOKEN   = os.getenv("ADMIN_TOKEN", "")
HF_TOKEN      = os.getenv("HUGGINGFACE_HUB_TOKEN") or os.getenv("HF_TOKEN")
RAG_MODE      = os.getenv("RAG_MODE", "off").lower()   # off | on | auto

# ===================== State Management (In-Memory for Demo) =====================
CONVERSATION_STORE: Dict[str, List[Dict[str, Any]]] = {}

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
    conversation_id: Optional[str] = Field(None, description="ID to track conversation history.")
    
class EmbeddingsRequest(BaseModel):
    input: Union[str, List[str]]
    model: str

class EmbeddingObject(BaseModel):
    object: str = "embedding"
    embedding: List[float]
    index: int

class EmbeddingsResponse(BaseModel):
    object: str = "list"
    data: List[EmbeddingObject]
    model: str

# ===================== Improved Tool Parsing Functions =====================

def extract_json_from_text(text: str) -> Optional[Dict[str, Any]]:
    """
    Robust JSON extraction from model output.
    Handles markdown code blocks and plain JSON.
    """
    if not text:
        return None
    
    # Remove any leading/trailing whitespace
    text = text.strip()
    
    # Try to find JSON in markdown code blocks first
    json_block_pattern = r'```(?:json)?\s*(\{.*?\})\s*```'
    match = re.search(json_block_pattern, text, re.DOTALL | re.IGNORECASE)
    
    if match:
        json_str = match.group(1).strip()
    else:
        # Look for standalone JSON object
        # Find the first { and try to parse a complete JSON object
        start_idx = text.find('{')
        if start_idx == -1:
            return None
            
        # Find matching closing brace
        brace_count = 0
        end_idx = start_idx
        in_string = False
        escape_next = False
        
        for i in range(start_idx, len(text)):
            char = text[i]
            
            if escape_next:
                escape_next = False
                continue
                
            if char == '\\':
                escape_next = True
                continue
                
            if char == '"' and not escape_next:
                in_string = not in_string
                continue
                
            if not in_string:
                if char == '{':
                    brace_count += 1
                elif char == '}':
                    brace_count -= 1
                    if brace_count == 0:
                        end_idx = i
                        break
        
        if brace_count == 0:
            json_str = text[start_idx:end_idx + 1]
        else:
            return None
    
    # Try to parse the extracted JSON
    try:
        return json.loads(json_str)
    except json.JSONDecodeError:
        return None

def build_tool_calls_from_parsed_json(parsed_json: Dict[str, Any]) -> Optional[List[Dict[str, Any]]]:
    """
    Convert parsed JSON to OpenAI tool calls format.
    """
    if not isinstance(parsed_json, dict):
        return None
    
    # Check if it's already in the correct format
    tool_calls_data = parsed_json.get("tool_calls", [])
    
    if not isinstance(tool_calls_data, list) or not tool_calls_data:
        return None
    
    formatted_tool_calls = []
    
    for i, tool_call in enumerate(tool_calls_data):
        if not isinstance(tool_call, dict):
            continue
            
        if tool_call.get("type") != "function":
            continue
            
        function_data = tool_call.get("function", {})
        if not isinstance(function_data, dict):
            continue
            
        function_name = function_data.get("name")
        if not function_name:
            continue
            
        # Handle arguments - ensure it's a JSON string as per OpenAI spec
        arguments = function_data.get("arguments", {})
        if isinstance(arguments, dict):
            arguments_str = json.dumps(arguments)
        elif isinstance(arguments, str):
            # Validate it's proper JSON
            try:
                json.loads(arguments)
                arguments_str = arguments
            except:
                arguments_str = json.dumps({})
        else:
            arguments_str = json.dumps({})
        
        formatted_tool_calls.append({
            "id": tool_call.get("id", f"call_{uuid.uuid4().hex[:8]}"),
            "type": "function",
            "function": {
                "name": function_name,
                "arguments": arguments_str
            }
        })
    
    return formatted_tool_calls if formatted_tool_calls else None

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
    print(f"[Tokenizer] init error: {e} (continuing without tokenizer)")

def to_prompt_from_messages(messages: List[Dict[str,str]]) -> str:
    """
    Convert messages to prompt string, ensuring content is never None.
    """
    # Create safe messages with guaranteed string content
    safe_messages = []
    for m in messages:
        msg_copy = m.copy()
        if msg_copy.get("content") is None:
            msg_copy["content"] = ""
        
        # Ensure content is always a string
        if not isinstance(msg_copy["content"], str):
             msg_copy["content"] = json.dumps(msg_copy["content"], ensure_ascii=False)

        safe_messages.append(msg_copy)

    # Use chat template if available
    if tokenizer and chat_tmpl_ok:
        try:
            return tokenizer.apply_chat_template(
                safe_messages, 
                tokenize=False, 
                add_generation_prompt=True
            )
        except Exception as e:
            print(f"[ChatTemplate] apply failed: {e} (using fallback)")

    # Fallback formatting
    parts = []
    for m in safe_messages:
        role = m.get("role", "user").upper()
        content = m.get("content", "")
        parts.append(f"{role}: {content}")
    parts.append("ASSISTANT:")
    return "\n".join(parts)

# ===================== LLM init =====================
llm = None
sampling_params = None
vllm_last_error: Optional[str] = None
embedding_model = None

if APP_ENV == "local":
    class MockLLM:
        def generate(self, prompt_template: str, sampling_params: object):
            time.sleep(0.1)
            class _O:
                def __init__(self, t): self.text=t
            class _R:
                def __init__(self, t): self.outputs=[_O(t)]
            return [_R("This is a mock response. (ENV=local)")]
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
        from sentence_transformers import SentenceTransformer
        print(f"[Embeddings] Loading model: {EMBEDDING_MODEL_NAME}...")
        embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME, device='cuda')
        print("[Embeddings] Model loaded successfully.")
    except Exception as e:
        vllm_last_error = str(e)
        print(f"[LLM] init error: {e}")

# ===================== RAG (optional) =====================
retriever = None
try:
    from app.rag_processor import (
        get_retriever, ensure_vector_db_ready,
        ingest_documents, list_documents, delete_document
    )
    ensure_vector_db_ready()
    retriever = get_retriever()
except Exception as e:
    print(f"[RAG] init warning: {e} (RAG remains optional)")

# ===================== Improved Tool System Instructions =====================
def create_tool_system_prompt(tools: List[Dict[str, Any]], tool_choice: Optional[Any] = None) -> str:
    """
    Create optimized system prompt for tool calling.
    Specifically optimized for N8N Qdrant retriever compatibility.
    """
    if not tools:
        return ""
    
    # Format tools for the model with N8N-specific parameter naming
    formatted_tools = []
    for tool in tools:
        if tool.get("type") == "function":
            func_info = tool.get("function", {})
            tool_name = func_info.get("name", "")
            
            # Special handling for retriever tool to match N8N expectations
            if tool_name == "retriever":
                formatted_tools.append({
                    "name": "retriever",
                    "description": func_info.get("description", "Retrieve relevant information from the knowledge base"),
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {
                                "type": "string",
                                "description": "The search query to find relevant information"
                            }
                        },
                        "required": ["query"]
                    }
                })
            else:
                formatted_tools.append({
                    "name": func_info.get("name"),
                    "description": func_info.get("description", ""),
                    "parameters": func_info.get("parameters", {})
                })
    
    system_prompt = f"""You have access to the following tools:

{json.dumps(formatted_tools, indent=2)}

IMPORTANT INSTRUCTIONS FOR TOOL USAGE:
1. When you need to use a tool, respond with ONLY a JSON object in this exact format:
{{"tool_calls": [{{"type": "function", "function": {{"name": "tool_name", "arguments": {{"query": "search terms"}}}}}}]}}

2. For the retriever tool, ALWAYS use "query" as the parameter name (not "input").

3. When you don't need to use any tools, respond normally with your answer.

4. Always provide meaningful search terms in the query parameter.

5. Do not add any explanation or text outside the JSON when making tool calls.

6. Make sure your JSON is valid and properly formatted.

Examples:
- To search for information: {{"tool_calls": [{{"type": "function", "function": {{"name": "retriever", "arguments": {{"query": "your search terms"}}}}}}]}}
- For normal chat: Just respond with regular text."""

    # Handle specific tool choice
    if isinstance(tool_choice, dict) and tool_choice.get("type") == "function":
        preferred_tool = tool_choice.get("function", {}).get("name")
        if preferred_tool:
            system_prompt += f"\n\n7. Prefer using the '{preferred_tool}' tool when appropriate."

    return system_prompt

# ===================== Auth =====================
def verify_bearer(req: Request):
    auth = req.headers.get("authorization", "")
    if not auth.startswith("Bearer "):
        raise HTTPException(401, "Invalid API key")
    token = auth.split(" ", 1)[1].strip()
    if token != API_KEY:
        raise HTTPException(401, "Invalid API key")

# ===================== FastAPI app =====================
app = FastAPI(title="General LLM API (OpenAI-compatible, Tools, optional RAG)", version="1.3.0-improved-tools")
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

# ===================== Models endpoint =====================
@app.get("/v1/models")
def list_models(_: None = Depends(verify_bearer)):
    model_data = [
        {
            "id": MODEL_NAME,
            "object": "model",
            "owned_by": "local",
            "created": int(time.time())
        }
    ]
    
    if embedding_model is not None:
        model_data.append({
            "id": EMBEDDING_MODEL_NAME,
            "object": "model",
            "owned_by": "local",
            "created": int(time.time())
        })

    return {"object": "list", "data": model_data}

# ===================== Simple chat =====================
@app.post("/v1/chat", response_model=ChatResponse)
async def chat(req: ChatCompletionsRequest, _: None = Depends(verify_bearer)):
    if llm is None: 
        raise HTTPException(503, "LLM not available.")
    
    messages = [m.model_dump(exclude_none=True) for m in req.messages]
    prompt = to_prompt_from_messages(messages)
    
    t0 = time.perf_counter()
    try:
        out = llm.generate(prompt, sampling_params)
        text = out[0].outputs[0].text
    except Exception as e:
        print(f"[LLM] error: {e}")
        raise HTTPException(500, "Inference error.")
    
    dt = max(1e-6, time.perf_counter()-t0)
    toks_in = count_tokens(prompt)
    toks_out = count_tokens(text)
    metrics = {
        "latency_sec": round(dt,3), 
        "tokens_in": toks_in,
        "tokens_out": toks_out, 
        "tok_per_sec": round(toks_out/dt,2)
    } if SHOW_TPS else None
    
    # Clean output from prompt bleeding
    text = re.split(r"\n(USER|ASSISTANT|SYSTEM):", text, 1)[0].strip()
    return ChatResponse(response=text, metrics=metrics)

# ===================== Main chat/completions endpoint =====================
@app.post("/v1/chat/completions")
async def chat_completions(req: ChatCompletionsRequest, auth: None = Depends(verify_bearer)):
    if llm is None: 
        raise HTTPException(503, "LLM not available.")

    # Prepare messages
    messages = [m.model_dump(exclude_none=True) for m in req.messages]
    
    # Add tool system prompt if tools are provided
    if req.tools:
        tool_system_prompt = create_tool_system_prompt(req.tools, req.tool_choice)
        
        # Insert or update system message
        system_message_found = False
        for i, msg in enumerate(messages):
            if msg.get("role") == "system":
                # Append tool instructions to existing system message
                messages[i]["content"] = f"{msg.get('content', '')}\n\n{tool_system_prompt}".strip()
                system_message_found = True
                break
        
        if not system_message_found:
            # Add new system message at the beginning
            messages.insert(0, {"role": "system", "content": tool_system_prompt})

    # Optional RAG context
    if RAG_MODE in ("on", "auto") and retriever:
        try:
            last_user_msg = next(
                (m["content"] for m in reversed(messages) 
                 if m["role"] == "user" and m.get("content")), 
                ""
            )
            if last_user_msg:
                docs = retriever.invoke(last_user_msg)
                if docs:
                    context = "\n\n".join([d.page_content for d in docs])[:4000]
                    context_msg = {
                        "role": "system", 
                        "content": f"## Relevant Context:\n{context}"
                    }
                    # Insert context before the last user message
                    messages.insert(-1, context_msg)
        except Exception as e:
            print(f"[RAG] context retrieval error: {e}")

    # Convert to prompt
    prompt = to_prompt_from_messages(messages)
    
    # Setup sampling parameters
    sp = sampling_params
    if any([req.max_tokens, req.temperature, req.top_p]):
        from vllm import SamplingParams
        sp = SamplingParams(
            temperature=req.temperature if req.temperature is not None else TEMP,
            top_p=req.top_p if req.top_p is not None else TOP_P,
            max_tokens=req.max_tokens if req.max_tokens is not None else MAX_TOKENS,
        )

    # Generate response
    t0 = time.perf_counter()
    try:
        out = llm.generate(prompt, sp)
        text = out[0].outputs[0].text
    except Exception as e:
        print(f"[LLM] generate error: {e}")
        raise HTTPException(500, f"Inference error: {e}")
    
    dt = max(1e-6, time.perf_counter() - t0)

    # Parse response for tool calls
    message_obj = {"role": "assistant"}
    finish_reason = "stop"
    
    if req.tools:
        # Try to extract tool calls
        parsed_json = extract_json_from_text(text)
        if parsed_json:
            tool_calls = build_tool_calls_from_parsed_json(parsed_json)
            if tool_calls:
                message_obj["tool_calls"] = tool_calls
                message_obj["content"] = None  # OpenAI standard for tool calls
                finish_reason = "tool_calls"
            else:
                # Not a tool call, treat as regular response
                clean_text = re.split(r"\n(USER|ASSISTANT|SYSTEM):", text, 1)[0].strip()
                message_obj["content"] = clean_text or "I apologize, but I couldn't generate a proper response."
        else:
            # No JSON found, treat as regular response
            clean_text = re.split(r"\n(USER|ASSISTANT|SYSTEM):", text, 1)[0].strip()
            message_obj["content"] = clean_text or "I apologize, but I couldn't generate a proper response."
    else:
        # No tools available, regular chat response
        clean_text = re.split(r"\n(USER|ASSISTANT|SYSTEM):", text, 1)[0].strip()
        message_obj["content"] = clean_text or "I apologize, but I couldn't generate a proper response."

    # Prepare response
    toks_in, toks_out = count_tokens(prompt), count_tokens(text)
    resp = {
        "id": f"chatcmpl-{uuid.uuid4().hex}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": req.model or MODEL_NAME,
        "choices": [{
            "index": 0, 
            "message": message_obj, 
            "finish_reason": finish_reason
        }],
        "usage": {
            "prompt_tokens": toks_in, 
            "completion_tokens": toks_out, 
            "total_tokens": toks_in + toks_out
        }
    }
    
    if SHOW_TPS: 
        resp["metrics"] = {
            "latency_sec": round(dt, 3), 
            "tps_out": round(toks_out/dt, 2)
        }
    
    return resp

# ===================== Conversation management =====================
@app.delete("/v1/conversations/{conv_id}")
def delete_conversation(conv_id: str, _: None = Depends(verify_bearer)):
    if conv_id in CONVERSATION_STORE:
        del CONVERSATION_STORE[conv_id]
        return {"status": "ok", "deleted": conv_id}
    else:
        raise HTTPException(404, f"Conversation ID '{conv_id}' not found.")

# ===================== Embeddings endpoint =====================
@app.post("/v1/embeddings", response_model=EmbeddingsResponse)
async def create_embeddings(req: EmbeddingsRequest, auth: None = Depends(verify_bearer)):
    if embedding_model is None:
        raise HTTPException(status_code=503, detail="Embedding model not available.")

    if isinstance(req.input, str):
        inputs = [req.input]
    else:
        inputs = req.input

    try:
        vectors = embedding_model.encode(inputs, normalize_embeddings=True).tolist()
        response_data = [
            EmbeddingObject(embedding=vec, index=i) for i, vec in enumerate(vectors)
        ]
        return EmbeddingsResponse(data=response_data, model=EMBEDDING_MODEL_NAME)
    except Exception as e:
        print(f"[Embeddings] Error: {e}")
        raise HTTPException(status_code=500, detail="Failed to create embeddings.")

# ===================== Remaining endpoints (Bench & RAG) =====================
@app.post("/v1/bench")
def bench(req: BenchRequest, _: None = Depends(verify_bearer)):
    if llm is None: 
        raise HTTPException(503, "LLM not available.")
    
    times = []
    for p in req.prompts:
        messages = [{"role": "user", "content": p}]
        prompt = to_prompt_from_messages(messages)
        t0 = time.perf_counter()
        _ = llm.generate(prompt, sampling_params)
        times.append(time.perf_counter() - t0)
    
    times_sorted = sorted(times)
    n = len(times)
    return {
        "count": n,
        "latency": {
            "avg_sec": sum(times)/n, 
            "p50_sec": times_sorted[n//2], 
            "p90_sec": times_sorted[max(0, int(0.9*n)-1)]
        },
        "model": MODEL_NAME, 
        "max_tokens_param": MAX_TOKENS
    }

@app.get("/v1/docs")
def docs_list(_: None = Depends(verify_bearer)):
    if "list_documents" not in globals(): 
        return {"items": []}
    try: 
        return {"items": list_documents()}
    except Exception as e: 
        raise HTTPException(500, f"Failed to read document list: {e}")

@app.post("/v1/docs/upload")
async def docs_upload(files: List[UploadFile] = File(...), _: None = Depends(verify_bearer)):
    if "ingest_documents" not in globals(): 
        raise HTTPException(501, "RAG not available in this image.")
    try:
        file_tuples = []
        for f in files:
            b = await f.read()
            file_tuples.append((f.name, b))
        res = ingest_documents(file_tuples)
        global retriever
        retriever = get_retriever()
        return res
    except Exception as e: 
        print(f"[UPLOAD] error: {e}")
        raise HTTPException(500, "Failed to upload/process documents")

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
        raise HTTPException(500, f"Failed to delete document: {e}")