# rag_processor.py
import os
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

# --- Konfigurasi ---
DATA_PATH = "data/"
VECTORSTORE_PATH = "vectorstore/faiss_index"

def create_vector_db():
    """
    Fungsi untuk membuat dan menyimpan vector database dari dokumen di folder 'data'.
    Fungsi ini hanya perlu dijalankan sekali (atau setiap kali data diperbarui).
    """
    # 1. Load Dokumen
    # Menggunakan TextLoader agar bisa membaca metadata seperti sumber file
    loader = DirectoryLoader(
        DATA_PATH,
        glob="*.txt",
        loader_cls=TextLoader,
        loader_kwargs={"encoding": "utf-8"}
    )
    documents = loader.load()
    if not documents:
        print("Tidak ada dokumen yang ditemukan. Pastikan ada file .txt di folder 'data'.")
        return

    print(f"Berhasil memuat {len(documents)} dokumen.")

    # 2. Split Dokumen
    # Memecah dokumen menjadi potongan-potongan kecil (chunks)
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    texts = text_splitter.split_documents(documents)
    print(f"Dokumen dipecah menjadi {len(texts)} potongan (chunks).")

    # 3. Embed dan Store
    # Menggunakan model embedding open-source yang populer dan ringan
    print("Membuat embeddings... (Mungkin butuh waktu beberapa saat saat pertama kali)")
    embeddings = HuggingFaceEmbeddings(
        model_name='sentence-transformers/all-MiniLM-L6-v2',
        model_kwargs={'device': 'cpu'} # Gunakan CPU untuk embedding
    )

    # Membuat vector store FAISS dari teks dan embeddings
    db = FAISS.from_documents(texts, embeddings)

    # Simpan vector store ke disk lokal
    db.save_local(VECTORSTORE_PATH)
    print(f"Vector store berhasil dibuat dan disimpan di '{VECTORSTORE_PATH}'.")

def get_retriever():
    """
    Fungsi untuk memuat vector database yang sudah ada dan mengembalikannya sebagai retriever.
    """
    if not os.path.exists(VECTORSTORE_PATH):
        print("Vector store tidak ditemukan. Jalankan `create_vector_db()` terlebih dahulu.")
        return None

    # Siapkan model embedding (harus sama dengan yang digunakan saat membuat DB)
    embeddings = HuggingFaceEmbeddings(
        model_name='sentence-transformers/all-MiniLM-L6-v2',
        model_kwargs={'device': 'cpu'}
    )

    # Muat vector store dari disk
    db = FAISS.load_local(VECTORSTORE_PATH, embeddings, allow_dangerous_deserialization=True)

    # Buat retriever yang akan mencari 2 dokumen paling relevan
    return db.as_retriever(search_kwargs={'k': 2})

# --- Main execution block ---
if __name__ == '__main__':
    # Blok ini akan dieksekusi jika Anda menjalankan 'python rag_processor.py'
    print("Mulai proses indexing data untuk RAG...")
    create_vector_db()