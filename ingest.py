import os
import re
from typing import List, Dict
from settings import settings
from vectorstore import VectorStore
from pypdf import PdfReader
from docx import Document as DocxDocument
import markdown
from ocr_utils import ocr_pdf, ocr_image_file

CHUNK_SIZE_CHARS = 1200
CHUNK_OVERLAP = 200

TEXT_EXTS = {".txt", ".md", ".markdown"}
DOC_EXTS  = {".docx"}
PDF_EXTS  = {".pdf"}
IMG_EXTS  = {".png", ".jpg", ".jpeg"}

def _read_txt(path: str) -> str:
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()

def _read_md(path: str) -> str:
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()

def _read_pdf_text_only(path: str) -> str:
    reader = PdfReader(path)
    texts = []
    for page in reader.pages:
        t = page.extract_text() or ""
        texts.append(t)
    return "\n".join(texts)

def _read_docx(path: str) -> str:
    doc = DocxDocument(path)
    return "\n".join([p.text for p in doc.paragraphs])

def _clean(s: str) -> str:
    s = s.replace("\x00", "")
    s = re.sub(r"[ \t]+", " ", s)
    s = re.sub(r"\n{3,}", "\n\n", s)
    return s.strip()

def _chunk(text: str, chunk_size=CHUNK_SIZE_CHARS, overlap=CHUNK_OVERLAP) -> List[str]:
    text = _clean(text)
    chunks = []
    start = 0
    n = len(text)
    while start < n:
        end = min(start + chunk_size, n)
        chunk = text[start:end]
        if chunk.strip():
            chunks.append(chunk)
        if end == n:
            break
        start = max(0, end - overlap)
    return chunks

def _load_file(path: str) -> str:
    ext = os.path.splitext(path)[1].lower()

    if ext in TEXT_EXTS:
        return _read_txt(path) if ext == ".txt" else _read_md(path)

    if ext in DOC_EXTS:
        return _read_docx(path)

    if ext in PDF_EXTS:
        # try native text first
        text = _read_pdf_text_only(path)
        if text.strip():
            return text
        # fallback to OCR the PDF
        print(f"[ocr] No extractable text via parser, running OCR for PDF: {path}")
        return ocr_pdf(path, lang="eng")

    if ext in IMG_EXTS:
        # OCR images (png/jpg/jpeg)
        print(f"[ocr] OCR image: {path}")
        return ocr_image_file(path, lang="eng")

    return ""

def ingest_folder():
    vs = VectorStore(settings.INDEX_DIR)
    files = []
    for root, _, fnames in os.walk(settings.DATA_DIR):
        for fn in fnames:
            if os.path.splitext(fn)[1].lower() in TEXT_EXTS | DOC_EXTS | PDF_EXTS | IMG_EXTS:
                files.append(os.path.join(root, fn))

    total_chunks = 0
    if not files:
        print(f"No files found in {settings.DATA_DIR}.")
        return

    for fp in files:
        text = _load_file(fp)
        if not text.strip():
            print(f"[skip] No extractable text: {fp}")
            continue

        chunks = _chunk(text)
        if not chunks:
            print(f"[skip] Text too short after cleaning: {fp}")
            continue

        metas: List[Dict] = []
        for i, c in enumerate(chunks):
            metas.append({
                "source": os.path.relpath(fp, settings.DATA_DIR),
                "chunk_id": i,
                "text": c[:1000]
            })

        vs.add_texts(chunks, metas)
        total_chunks += len(chunks)
        print(f"[ok] {fp} -> {len(chunks)} chunks")

    print(f"Ingested {len(files)} files into {settings.INDEX_DIR} with {total_chunks} chunks.")

if __name__ == "__main__":
    ingest_folder()
