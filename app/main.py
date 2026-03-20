import os
from io import BytesIO
from fastapi import FastAPI, UploadFile, File, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
from slowapi import Limiter
from slowapi.util import get_remote_address
from pypdf import PdfReader

from app.models import QueryRequest, QueryResponse, UploadResponse
from app.rag import add_document, generate_answer
from app.security import validate_query

load_dotenv()

limiter = Limiter(key_func=get_remote_address)

app = FastAPI()
app.state.limiter = limiter

# CORS — allow all origins if env var not set
raw_origins = os.getenv("ALLOWED_ORIGINS", "*")
origins = [o.strip() for o in raw_origins.split(",") if o.strip()]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Health check — lets Render confirm the service is alive
@app.get("/health")
async def health():
    return {"status": "ok"}


# Root — prevents 404 on base URL visits
@app.get("/")
async def root():
    return {"message": "Chatbot API is running. POST to /chat or /api/chat."}


# -----------------------------
# File upload
# -----------------------------
@app.post("/upload", response_model=UploadResponse)
async def upload_file(file: UploadFile = File(...)):
    if file.content_type not in ["application/pdf", "text/plain"]:
        raise HTTPException(status_code=400, detail="Only PDF and TXT files are allowed.")

    # Read bytes first, then wrap in BytesIO so we can reuse the buffer
    content = await file.read()

    if file.content_type == "application/pdf":
        pdf = PdfReader(BytesIO(content))  # FIX: was PdfReader(file.file) which was already exhausted
        text = ""
        for page in pdf.pages:
            text += page.extract_text() or ""
    else:
        text = content.decode("utf-8")

    if not text.strip():
        raise HTTPException(status_code=400, detail="Could not extract text from the file.")

    add_document(text, file.filename)

    return {"message": "Document processed successfully."}


# -----------------------------
# RAG chat — main endpoint
# -----------------------------
@app.post("/chat", response_model=QueryResponse)
@limiter.limit(os.getenv("RATE_LIMIT", "5/minute"))
async def rag_chat(request: Request, body: QueryRequest):
    try:
        query = validate_query(body.query)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    answer, sources = generate_answer(query)
    return {"answer": answer, "sources": sources}


# -----------------------------
# /api/chat — alias that also uses RAG (was a dummy echo before)
# -----------------------------
@app.post("/api/chat", response_model=QueryResponse)
@limiter.limit(os.getenv("RATE_LIMIT", "5/minute"))
async def api_chat(request: Request, body: QueryRequest):
    try:
        query = validate_query(body.query)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    answer, sources = generate_answer(query)
    return {"answer": answer, "sources": sources}
