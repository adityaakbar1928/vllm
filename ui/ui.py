import streamlit as st
import requests
import json
import os
from io import BytesIO

st.set_page_config(page_title="Document Q&A Assistant", page_icon="📄", layout="wide")

# --- AWAL PERUBAHAN 1: Definisikan API Key dan Headers ---
# Ambil API Key dari environment variable atau gunakan default
# Untuk keamanan, lebih baik atur ini sebagai environment variable di environment Anda
API_KEY = os.environ.get("API_KEY", "jakarta321") 
BACKEND_URL = os.environ.get("BACKEND_URL", "http://127.0.0.1:8000")

# Siapkan header untuk otentikasi, akan digunakan di semua request
HEADERS = {
    "Authorization": f"Bearer {API_KEY}"
}
# --- AKHIR PERUBAHAN 1 ---


st.title("📄 Document Q&A Assistant")
st.caption("vLLM + RAG + FastAPI + Streamlit | Upload dokumen → indeks → tanya jawab berbasis dokumen.")

tab_chat, tab_docs = st.tabs(["💬 Chat", "📚 Dokumen (RAG)"])

# -------- Chat --------
with tab_chat:
    if "messages" not in st.session_state:
        st.session_state.messages = [
            {"role": "assistant", "content": "Halo! Unggah dokumen di tab **Dokumen (RAG)**, lalu tanya di sini."}
        ]

    for m in st.session_state.messages:
        with st.chat_message(m["role"]):
            st.markdown(m["content"])

    if prompt := st.chat_input("Tulis pertanyaan Anda..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"): st.markdown(prompt)
        with st.chat_message("assistant"):
            placeholder = st.empty()
            placeholder.markdown("Menjawab... ⏳")
            try:
                # --- PERUBAHAN 2: Tambahkan headers ke request ---
                r = requests.post(f"{BACKEND_URL}/v1/chat", json={"prompt": prompt}, timeout=600, headers=HEADERS)
                r.raise_for_status()
                data = r.json()
                txt = data.get("response","")
                metrics = data.get("metrics")
                if metrics:
                    meta = f"\n\n---\nThroughput: **{metrics['tok_per_sec']} tok/s** · tokens_out={metrics['tokens_out']} · latency={metrics['latency_sec']}s"
                    txt = txt + meta
                placeholder.markdown(txt)
                st.session_state.messages.append({"role":"assistant","content":txt})
            except Exception as e:
                placeholder.error(f"Gagal memanggil backend: {e}")

# -------- Dokumen (RAG) --------
with tab_docs:
    st.subheader("Daftar Dokumen")
    col1, col2 = st.columns([1,1])

    with col1:
        if st.button("🔄 Refresh"):
            pass
    with col2:
        st.write(f"Backend: `{BACKEND_URL}`")

    # List docs
    try:
        # --- PERUBAHAN 3: Tambahkan headers ke request ---
        r = requests.get(f"{BACKEND_URL}/v1/docs", timeout=30, headers=HEADERS)
        r.raise_for_status()
        items = r.json().get("items", [])
    except Exception as e:
        st.error(f"Gagal mengambil daftar dokumen: {e}")
        items = []

    if items:
        st.table(items)
    else:
        st.info("Belum ada dokumen.")

    st.write("---")
    st.subheader("⬆️ Upload & Indeks")
    files = st.file_uploader("Pilih file .txt / .md / .pdf / .docx (bisa multi-select)", type=["txt","md","pdf","docx"], accept_multiple_files=True)
    if st.button("Upload & Bangun Index"):
        if not files:
            st.warning("Pilih minimal satu file.")
        else:
            try:
                m = []
                for f in files:
                    m.append(("files", (f.name, f.getvalue())))
                # --- PERUBAHAN 4: Tambahkan headers ke request ---
                # Untuk upload file, headers dilewatkan secara terpisah dari 'files'
                r = requests.post(f"{BACKEND_URL}/v1/docs/upload", files=m, timeout=600, headers=HEADERS)
                r.raise_for_status()
                st.success(f"OK: {r.json()}")
            except Exception as e:
                st.error(f"Gagal upload: {e}")