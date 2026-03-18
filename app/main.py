import os
from fastapi import FastAPI, UploadFile, File, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
from slowapi import Limiter
from slowapi.util import get_remote_address
from pypdf import PdfReader

from app.models import QueryRequest, QueryResponse, UploadResponse
from app.rag import add_document, generate_answer
from app.security import validate_query

# Load environment
load_dotenv()

# Rate limiter
limiter = Limiter(key_func=get_remote_address)

# FastAPI app
app = FastAPI()
app.state.limiter = limiter

# CORS
origins = os.getenv("ALLOWED_ORIGINS", "").split(",")
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -----------------------------
# Chat endpoint (replacing Flask)
# -----------------------------
@app.post("/api/chat")
async def chat_endpoint(body: QueryRequest):
    # Example: echo the message
    return {"reply": "Backend received: " + body.query}

# -----------------------------
# File upload
# -----------------------------
@app.post("/upload", response_model=UploadResponse)
async def upload_file(file: UploadFile = File(...)):
    if file.content_type not in ["application/pdf", "text/plain"]:
        raise HTTPException(status_code=400, detail="Only PDF and TXT allowed.")

    content = await file.read()

    if file.content_type == "application/pdf":
        pdf = PdfReader(file.file)
        text = ""
        for page in pdf.pages:
            text += page.extract_text() or ""
    else:
        text = content.decode("utf-8")

    add_document(text, file.filename)

    return {"message": "Document processed successfully."}

# -----------------------------
# Chat RAG endpoint
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
