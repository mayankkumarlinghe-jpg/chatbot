import os
from io import BytesIO
from contextlib import asynccontextmanager

from dotenv import load_dotenv
from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pypdf import PdfReader
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address

from app.models import QueryRequest, QueryResponse, UploadResponse
from app.rag import add_document, generate_answer
from app.security import validate_query

load_dotenv()

limiter = Limiter(key_func=get_remote_address)
DATA_FOLDER = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")


def _load_file(filepath: str, filename: str):
    """Read a .txt or .pdf file and add it to the RAG store."""
    with open(filepath, "rb") as f:
        content = f.read()
    if filename.endswith(".pdf"):
        pdf = PdfReader(BytesIO(content))
        text = "".join(page.extract_text() or "" for page in pdf.pages)
    else:
        text = content.decode("utf-8", errors="ignore")
    if text.strip():
        add_document(text, filename)
        print(f"[Startup] Loaded: {filename}")
    else:
        print(f"[Startup] Skipped (empty): {filename}")


def preload_data_folder():
    if not os.path.exists(DATA_FOLDER):
        print(f"[Startup] No data folder at {DATA_FOLDER}, skipping.")
        return
    files = [f for f in os.listdir(DATA_FOLDER) if f.endswith((".txt", ".pdf"))]
    if not files:
        print("[Startup] No .txt or .pdf files found in data/")
        return
    for filename in files:
        _load_file(os.path.join(DATA_FOLDER, filename), filename)


@asynccontextmanager
async def lifespan(app: FastAPI):
    preload_data_folder()
    yield


app = FastAPI(lifespan=lifespan)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

raw_origins = os.getenv("ALLOWED_ORIGINS", "*")
origins = [o.strip() for o in raw_origins.split(",") if o.strip()]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve widget.js and any static files from /public
public_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "public")
if os.path.exists(public_dir):
    app.mount("/static", StaticFiles(directory=public_dir), name="static")


# -------------------------------------------------
# Routes
# -------------------------------------------------

@app.get("/health")
async def health():
    return {"status": "ok"}


@app.get("/")
async def root():
    return {"message": "Chatbot API is running. POST to /chat or /api/chat."}


@app.post("/upload", response_model=UploadResponse)
async def upload_file(file: UploadFile = File(...)):
    if file.content_type not in ["application/pdf", "text/plain"]:
        raise HTTPException(status_code=400, detail="Only PDF and TXT files are allowed.")
    content = await file.read()
    if file.content_type == "application/pdf":
        pdf = PdfReader(BytesIO(content))
        text = "".join(page.extract_text() or "" for page in pdf.pages)
    else:
        text = content.decode("utf-8", errors="ignore")
    if not text.strip():
        raise HTTPException(status_code=400, detail="Could not extract text from file.")
    add_document(text, file.filename)
    return {"message": "Document processed successfully."}


@app.post("/chat", response_model=QueryResponse)
@limiter.limit(os.getenv("RATE_LIMIT", "20/minute"))
async def rag_chat(request: Request, body: QueryRequest):
    try:
        query = validate_query(body.query)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    answer, sources = generate_answer(query)
    return {"answer": answer, "sources": sources}


@app.post("/api/chat", response_model=QueryResponse)
@limiter.limit(os.getenv("RATE_LIMIT", "20/minute"))
async def api_chat(request: Request, body: QueryRequest):
    try:
        query = validate_query(body.query)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    answer, sources = generate_answer(query)
    return {"answer": answer, "sources": sources}
