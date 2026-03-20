import os
import re
from contextlib import asynccontextmanager
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from groq import Groq
from pydantic import BaseModel
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
DATA_FILE = Path("data/data.txt")

# --- Load data once at startup ---
data_chunks: list[str] = []

def load_data():
    global data_chunks
    if not DATA_FILE.exists():
        print("WARNING: data/data.txt not found")
        return
    text = DATA_FILE.read_text(encoding="utf-8")
    # Split into chunks of ~500 chars, respecting paragraph breaks
    paragraphs = [p.strip() for p in re.split(r"\n{2,}", text) if p.strip()]
    chunks = []
    current = ""
    for para in paragraphs:
        if len(current) + len(para) < 500:
            current += "\n\n" + para if current else para
        else:
            if current:
                chunks.append(current)
            current = para
    if current:
        chunks.append(current)
    data_chunks = chunks
    print(f"Loaded {len(data_chunks)} chunks from data.txt")

@asynccontextmanager
async def lifespan(app: FastAPI):
    load_data()
    yield

# --- Simple keyword search (replaces chromadb + embeddings) ---
def retrieve_context(query: str, top_k: int = 4) -> str:
    if not data_chunks:
        return ""
    query_words = set(re.findall(r"\w+", query.lower()))
    scored = []
    for chunk in data_chunks:
        chunk_words = set(re.findall(r"\w+", chunk.lower()))
        score = len(query_words & chunk_words)
        if score > 0:
            scored.append((score, chunk))
    scored.sort(key=lambda x: x[0], reverse=True)
    top = [c for _, c in scored[:top_k]]
    return "\n\n---\n\n".join(top)

# --- App setup ---
limiter = Limiter(key_func=get_remote_address)
app = FastAPI(lifespan=lifespan)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory="public"), name="static")

class ChatRequest(BaseModel):
    message: str
    history: list[dict] = []

@app.get("/")
async def root():
    return {"status": "ok"}

@app.post("/chat")
@limiter.limit("20/minute")
async def chat(request: Request, body: ChatRequest):
    if not GROQ_API_KEY:
        raise HTTPException(status_code=500, detail="GROQ_API_KEY not set")
    if not body.message.strip():
        raise HTTPException(status_code=400, detail="Empty message")

    context = retrieve_context(body.message)

    system_prompt = (
        "You are a helpful assistant. Answer questions based on the provided context. "
        "If the answer is not in the context, say you don't have that information.\n\n"
    )
    if context:
        system_prompt += f"Context:\n{context}"

    messages = [{"role": "system", "content": system_prompt}]
    # keep last 6 turns to limit token usage
    for turn in body.history[-6:]:
        if turn.get("role") in ("user", "assistant") and turn.get("content"):
            messages.append({"role": turn["role"], "content": turn["content"]})
    messages.append({"role": "user", "content": body.message})

    try:
        client = Groq(api_key=GROQ_API_KEY)
        completion = client.chat.completions.create(
            model="llama3-8b-8192",
            messages=messages,
            max_tokens=512,
            temperature=0.5,
        )
        reply = completion.choices[0].message.content
        return {"reply": reply}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
