from typing import List, Optional
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from openai import OpenAI
from settings import settings
from vectorstore import VectorStore
from fastapi.middleware.cors import CORSMiddleware
import os, sys, subprocess

app = FastAPI(title="Internal Q&A Bot", version="0.3.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://127.0.0.1:5500", "http://localhost:5500",
        "http://127.0.0.1:5173", "http://localhost:5173",
        "http://127.0.0.1:3000", "http://localhost:3000",
        "file://"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

client = OpenAI(api_key=settings.OPENAI_API_KEY)
vstore = VectorStore(settings.INDEX_DIR)

class AskRequest(BaseModel):
    question: str = Field(..., min_length=3)
    top_k: int = Field(ge=1, le=15, default=settings.MAX_CONTEXT_CHUNKS)
    user_email: Optional[str] = None

class AskResponse(BaseModel):
    answer: str
    citations: List[dict]
    contexts: List[dict]

def guard_domain(email: Optional[str]):
    allowed = settings.allowed_domains
    if not allowed or not email:
        return
    domain = email.split("@")[-1].lower()
    if domain not in allowed:
        raise HTTPException(status_code=403, detail="Email domain is not allowed.")

@app.get("/health")
def health():
    stats = vstore.stats()
    return {"ok": True, **stats}

@app.post("/reindex")
def reindex():
    # run "python ingest.py --reset" using current interpreter
    exe = sys.executable or "python"
    subprocess.Popen([exe, "ingest.py", "--reset"])
    return {"status": "started"}

@app.post("/ask", response_model=AskResponse)
def ask(req: AskRequest):
    guard_domain(req.user_email)
    try:
        hits = vstore.search(req.question, k=req.top_k)
    except AssertionError as e:
        raise HTTPException(status_code=400, detail=str(e))

    hits = hits[:req.top_k]  # safety clamp

    contexts = []
    citations = []
    blocks = []
    for score, meta in hits:
        src = meta.get("source", "unknown")
        cid = meta.get("chunk_id", -1)
        preview = meta.get("text", "")
        contexts.append({"source": src, "chunk_id": cid, "score": round(score, 3), "preview": preview})
        citations.append({"source": src, "chunk_id": cid, "score": round(score, 3)})
        blocks.append(f"[{src} | chunk {cid} | score {score:.3f}]\n{preview}")

    system = (
        "You are an internal company assistant. Answer using ONLY the provided context. "
        "If the answer is not in the context, say you don't know and suggest where it might be."
    )
    context = "\n\n".join(blocks) if blocks else "NO MATCHES."

    prompt = (
        f"User question:\n{req.question}\n\n"
        f"Context chunks (previews; do not invent facts):\n{context}\n\n"
        "Instructions: Provide a concise, accurate answer. If unsure, say you don't know. "
        "At the end, list sources as bullet points in the format [source | chunk]."
    )

    chat = client.chat.completions.create(
        model=settings.CHAT_MODEL,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
        ],
        temperature=0.1,
    )
    answer = chat.choices[0].message.content.strip()
    return AskResponse(answer=answer, citations=citations, contexts=contexts)
