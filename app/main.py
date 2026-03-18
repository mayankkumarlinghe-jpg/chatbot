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
from flask import Flask, request, jsonify
from flask_cors import CORS  # ← import

app = Flask(__name__)
CORS(app)  # ← ye line dal do, saare endpoints ke liye CORS enable ho jaayega

@app.route("/api/chat", methods=["POST"])
def chat():
    data = request.json
    print(data)
    return jsonify({"reply": "Backend received: " + data["message"]})

if __name__ == "__main__":
    app.run(debug=True)

load_dotenv()

limiter = Limiter(key_func=get_remote_address)

app = FastAPI()
app.state.limiter = limiter

# CORS
origins = os.getenv("ALLOWED_ORIGINS", "").split(",")

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["POST"],
    allow_headers=["*"],
)

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

@app.post("/chat", response_model=QueryResponse)
@limiter.limit(os.getenv("RATE_LIMIT", "5/minute"))
async def chat(request: Request, body: QueryRequest):
    try:
        query = validate_query(body.query)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    answer, sources = generate_answer(query)

    return {"answer": answer, "sources": sources}
