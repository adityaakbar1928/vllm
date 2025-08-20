# ui.py
import streamlit as st
import requests
import json

# --- Konfigurasi Halaman ---
st.set_page_config(
    page_title="vLLM Chat Testing",
    page_icon="🤖",
    layout="wide"
)

st.title("🤖 Chatbot LLM with RAG")
st.caption("Dibangun dengan vLLM + RAG + FastAPI + Langfuse + SSO + Guardrails")

# --- URL Backend FastAPI ---
# Pastikan server FastAPI Anda berjalan di port 8000
BACKEND_URL = "http://127.0.0.1:8000/v1/chat"

# --- Inisialisasi Riwayat Chat ---
# 'st.session_state' adalah cara Streamlit untuk menyimpan variabel antar interaksi
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Halo! Ada yang bisa saya bantu? Tanyakan apa saja tentang produk kami."}
    ]

# --- Menampilkan Riwayat Chat ---
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# --- Menerima Input dari Pengguna ---
if prompt := st.chat_input("Tulis pertanyaan Anda di sini..."):
    # 1. Tampilkan pesan pengguna di UI
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # 2. Kirim prompt ke backend FastAPI
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        message_placeholder.markdown("Sedang berpikir... ⏳")
        
        try:
            # Buat request ke API backend
            response = requests.post(BACKEND_URL, json={"prompt": prompt}, timeout=120)
            response.raise_for_status()  # Ini akan error jika status code bukan 2xx
            
            # Ambil jawaban dari JSON respons
            full_response = response.json()["response"]
            message_placeholder.markdown(full_response)
            
            # Tambahkan respons dari asisten ke riwayat chat
            st.session_state.messages.append({"role": "assistant", "content": full_response})

        except requests.exceptions.RequestException as e:
            error_message = f"Gagal terhubung ke backend: {e}"
            st.error(error_message)
            st.session_state.messages.append({"role": "assistant", "content": f"Maaf, terjadi kesalahan: {error_message}"})
        except Exception as e:
            error_message = f"Terjadi kesalahan: {e}"
            st.error(error_message)
            st.session_state.messages.append({"role": "assistant", "content": f"Maaf, terjadi kesalahan: {error_message}"})