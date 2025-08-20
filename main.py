# main.py
import os
import time
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from rag_processor import get_retriever
from langfuse.fastapi import LangfuseObservability
from langfuse import Langfuse

# --- Pydantic Models: Mendefinisikan struktur data untuk API ---
# Ini memastikan data yang masuk ke API kita sesuai format yang diinginkan.
class ChatRequest(BaseModel):
    prompt: str = Field(
        ...,
        title="Prompt",
        description="Prompt atau pertanyaan dari pengguna.",
        example="Ceritakan tentang sejarah kemerdekaan Indonesia"
    )

class ChatResponse(BaseModel):
    response: str = Field(
        ...,
        title="Response",
        description="Jawaban yang dihasilkan oleh LLM.",
        example="Tentu, mari kita bahas sejarah kemerdekaan Indonesia..."
    )

# --- Logika Hybrid: Mock LLM untuk Lokal, Real LLM untuk Produksi ---

# Kita gunakan environment variable 'ENV' untuk menentukan mode.
# Jika tidak ada, default-nya adalah 'local'.
APP_ENV = os.getenv("APP_ENV", "local")

class MockLLM:
    """
    Kelas LLM tiruan yang sudah di-upgrade untuk menampilkan konteks yang diterima.
    """
    def generate(self, prompt_template: str, sampling_params: object):
        time.sleep(1) # Simulasi berpikir

        # --- Logika baru untuk mengekstrak konteks dari prompt ---
        try:
            # Kita cari teks di antara "Konteks:" dan "Pertanyaan:"
            context_part = prompt_template.split("Konteks:\n")[1]
            context_found = context_part.split("\n\nPertanyaan:")[0]
            if not context_found.strip():
                context_found = "(Tidak ada konteks relevan yang ditemukan)"
        except IndexError:
            context_found = "(Format prompt tidak mengandung 'Konteks:')"
        
        # --- Respons baru yang lebih informatif ---
        mock_response_text = (
            f"🤖 **Respons dari MockLLM (Mode Lokal)**\n\n"
            f"Saya diminta menjawab pertanyaan Anda dan menemukan konteks berikut dari database RAG:\n\n"
            f"```\n{context_found}\n```\n\n"
            f"*Di server produksi, LLM asli akan menggunakan informasi di atas untuk merumuskan jawaban yang lengkap.*"
        )
        
        # Struktur output ini dibuat agar SAMA PERSIS dengan struktur output vLLM
        class MockCompletionOutput:
            def __init__(self, text):
                self.text = text
        
        class MockRequestOutput:
            def __init__(self, text):
                self.outputs = [MockCompletionOutput(text)]

        return [MockRequestOutput(mock_response_text)]

# Inisialisasi LLM berdasarkan lingkungan
if APP_ENV == "local":
    print("🚀 Running in LOCAL mode. Menggunakan MockLLM.")
    llm = MockLLM()
    sampling_params = None  # Tidak perlu sampling params untuk mock
else:
    # Kode ini HANYA akan dijalankan di server produksi/Rafay
    print("☁️ Running in PRODUCTION mode. Memuat model vLLM asli...")
    try:
        from vllm import LLM, SamplingParams
        
        # Ganti dengan model 7B/8B pilihan Anda dari Hugging Face
        MODEL_NAME = "meta-llama/Llama-3-8B-Instruct" 
        
        llm = LLM(
            model=MODEL_NAME,
            trust_remote_code=True,
            gpu_memory_utilization=0.90 # Gunakan 90% memori GPU
        )
        sampling_params = SamplingParams(temperature=0.7, top_p=0.95, max_tokens=1024)
        print("✅ Model vLLM berhasil dimuat.")
    except ImportError:
        print("❌ Gagal mengimpor vllm. Pastikan library terinstall di lingkungan produksi.")
        llm = None
    except Exception as e:
        print(f"❌ Terjadi kesalahan saat memuat model vLLM: {e}")
        llm = None

try:
    langfuse = Langfuse(
        public_key=os.environ.get("LANGFUSE_PUBLIC_KEY"),
        secret_key=os.environ.get("LANGFUSE_SECRET_KEY"),
        host=os.environ.get("LANGFUSE_HOST")
    )
    print("✅ Langfuse client berhasil diinisialisasi.")
except Exception as e:
    langfuse = None
    print(f"⚠️ Gagal menginisialisasi Langfuse. Tracing akan dinonaktifkan. Error: {e}")

retriever = get_retriever()

# --- Inisialisasi Aplikasi FastAPI ---
app = FastAPI(
    title="Asisten LLM Open Source",
    description="API untuk inferensi LLM dengan vLLM, RAG, dan Guardrails.",
    version="0.1.0"
)

if langfuse:
    LangfuseObservability(app=app, langfuse_client=langfuse)
    print("🔍 Langfuse Observability middleware aktif.")

# --- API Endpoints ---
@app.get("/", summary="Health Check")
def read_root():
    """Endpoint untuk memeriksa apakah API berjalan."""
    return {"status": "ok", "environment": APP_ENV}

@app.post("/v1/chat", response_model=ChatResponse, summary="Generate Chat Response")
async def generate_chat_response(request: ChatRequest):
    """
    Menerima prompt, mencari konteks relevan dari RAG, lalu menghasilkan respons.
    """
    if llm is None:
        raise HTTPException(status_code=503, detail="LLM service is not available.")
    
    if retriever is None:
         raise HTTPException(status_code=503, detail="Retriever service (RAG) is not available.")

    # --- LANGKAH RAG DITAMBAHKAN DI SINI ---
    # 1. Retrieve: Cari konteks yang relevan berdasarkan prompt pengguna
    try:
        relevant_docs = retriever.invoke(request.prompt)
        
        # Gabungkan isi dokumen yang relevan menjadi satu string konteks
        context = "\n\n".join([doc.page_content for doc in relevant_docs])
        
        print("\n--- Konteks yang Ditemukan ---")
        print(context)
        print("---------------------------\n")

    except Exception as e:
        print(f"Error during context retrieval: {e}")
        context = "" # Jika gagal mencari, konteks dikosongkan

    # 2. Augment: Gabungkan konteks dengan prompt asli
    augmented_prompt = (
        f"Anda adalah asisten AI yang membantu. Jawab pertanyaan berikut berdasarkan konteks yang diberikan.\n\n"
        f"Konteks:\n{context}\n\n"
        f"Pertanyaan: {request.prompt}"
    )

    # Format prompt sesuai dengan template Llama 3
    prompt_template = f"<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n{augmented_prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
    
    # 3. Generate: Kirim prompt yang sudah diperkaya ke LLM
    try:
        outputs = llm.generate(prompt_template, sampling_params)
        response_text = outputs[0].outputs[0].text
        
        return ChatResponse(response=response_text)
    
    except Exception as e:
        print(f"Error during inference: {e}")
        raise HTTPException(status_code=500, detail="Terjadi kesalahan pada server saat inferensi.")