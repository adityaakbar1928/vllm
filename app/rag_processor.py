import os, io, hashlib
from datetime import datetime
from typing import List, Tuple

DATA_PATH = "/workspace/data"
VSTORE_DIR = "/workspace/vectorstore"
VECTORSTORE_PATH = os.path.join(VSTORE_DIR, "faiss_index")

def _ensure_dirs():
    os.makedirs(DATA_PATH, exist_ok=True)
    os.makedirs(VSTORE_DIR, exist_ok=True)

def _safe_name(name: str) -> str:
    h = hashlib.sha1(name.encode("utf-8")).hexdigest()[:8]
    base = os.path.basename(name).replace("..","").replace("/","_")
    return f"{h}__{base}"

def _load_all_documents():
    # dependensi berat diimpor hanya saat dipakai
    docs=[]
    try:
        from langchain.text_splitter import RecursiveCharacterTextSplitter
        from langchain_community.document_loaders import TextLoader, DirectoryLoader, PyPDFLoader, Docx2txtLoader
        # txt+md
        for ext in ("*.txt","*.md"):
            dl = DirectoryLoader(DATA_PATH, glob=ext, loader_cls=TextLoader, loader_kwargs={"encoding":"utf-8"})
            docs += dl.load()
        # pdf
        for root, _, files in os.walk(DATA_PATH):
            for f in files:
                if f.lower().endswith(".pdf"):
                    docs += PyPDFLoader(os.path.join(root, f)).load()
        # docx
        for root, _, files in os.walk(DATA_PATH):
            for f in files:
                if f.lower().endswith(".docx"):
                    docs += Docx2txtLoader(os.path.join(root, f)).load()
    except Exception as e:
        print(f"[RAG] loader error: {e}")
    return docs

def create_vector_db():
    _ensure_dirs()
    docs = _load_all_documents()
    if not docs:
        print("[RAG] Tidak ada dokumen untuk diindeks.")
        return False
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    from langchain_community.vectorstores import FAISS
    from langchain_huggingface import HuggingFaceEmbeddings

    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    texts = splitter.split_documents(docs)
    print(f"[RAG] chunks: {len(texts)}")

    embeddings = HuggingFaceEmbeddings(
        model_name=os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"),
        model_kwargs={"device":"cpu"}
    )
    db = FAISS.from_documents(texts, embeddings)
    db.save_local(VECTORSTORE_PATH)
    print(f"[RAG] Vectorstore saved at {VECTORSTORE_PATH}")
    return True

def ensure_vector_db_ready():
    _ensure_dirs()
    if not os.path.exists(VECTORSTORE_PATH):
        print("[RAG] building vectorstore (awal)...")
        create_vector_db()

def get_retriever():
    _ensure_dirs()
    from langchain_huggingface import HuggingFaceEmbeddings
    from langchain_community.vectorstores import FAISS
    if not os.path.exists(VECTORSTORE_PATH):
        return None
    embeddings = HuggingFaceEmbeddings(
        model_name=os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"),
        model_kwargs={"device":"cpu"}
    )
    db = FAISS.load_local(VECTORSTORE_PATH, embeddings, allow_dangerous_deserialization=True)
    return db.as_retriever(search_kwargs={"k": 4})

def ingest_documents(files: List[Tuple[str, bytes]]):
    _ensure_dirs()
    saved=[]
    for fname, b in files:
        if not b: continue
        out = os.path.join(DATA_PATH, _safe_name(fname))
        with open(out, "wb") as f:
            f.write(b)
        saved.append(out)
    ok = create_vector_db()
    return {"saved": [os.path.basename(p) for p in saved], "reindexed": bool(ok)}

def list_documents():
    _ensure_dirs()
    items=[]
    for f in sorted(os.listdir(DATA_PATH)):
        p = os.path.join(DATA_PATH, f)
        if not os.path.isfile(p): continue
        st = os.stat(p)
        items.append({
            "id": f,
            "filename": f,
            "size": st.st_size,
            "uploaded_at": datetime.fromtimestamp(st.st_mtime).isoformat(timespec="seconds"),
        })
    return items

def delete_document(doc_id: str):
    _ensure_dirs()
    p = os.path.join(DATA_PATH, doc_id)
    if os.path.isfile(p):
        os.remove(p)
    else:
        raise FileNotFoundError(doc_id)
    create_vector_db()
