import os
import uuid
from io import BytesIO
from typing import List

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastembed import TextEmbedding
from openai import OpenAI
from pydantic import BaseModel
from PyPDF2 import PdfReader
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, PointStruct, VectorParams
import uvicorn


load_dotenv()


class QueryRequest(BaseModel):
    question: str
    top_k: int | None = 5


GROQ_API_KEY = os.getenv("GROQ_API_KEY")
QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
QDRANT_COLLECTION = os.getenv("QDRANT_COLLECTION", "docs_chunks")

EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"
EMBEDDING_DIM = 384
CHAT_MODEL = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")

if not GROQ_API_KEY:
    raise RuntimeError("GROQ_API_KEY must be set")

if not QDRANT_URL:
    raise RuntimeError("QDRANT_URL must be set")

if not QDRANT_API_KEY:
    raise RuntimeError("QDRANT_API_KEY must be set")


def init_qdrant_collection() -> QdrantClient:
    client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
    collections = client.get_collections().collections or []
    existing = {collection.name for collection in collections}
    if QDRANT_COLLECTION not in existing:
        client.create_collection(
            collection_name=QDRANT_COLLECTION,
            vectors_config=VectorParams(size=EMBEDDING_DIM, distance=Distance.COSINE),
        )
    return client


qdrant_client = init_qdrant_collection()
embedding_model = TextEmbedding(model_name=EMBEDDING_MODEL)
llm_client = OpenAI(
    api_key=GROQ_API_KEY,
    base_url="https://api.groq.com/openai/v1",
)


def embed_text(text: str) -> List[float]:
    text = text.strip()
    if not text:
        raise ValueError("Cannot embed empty text")
    # FastEmbed returns a generator of numpy arrays
    embedding = next(embedding_model.embed([text]))
    return embedding.tolist()


def generate_answer(context_chunks: List[str], question: str) -> str:
    if not context_chunks:
        return "I don't know."

    context_text = "\n\n".join(
        f"Chunk {index + 1}: {chunk}" for index, chunk in enumerate(context_chunks)
    )

    messages = [
        {
            "role": "system",
            "content": (
                "You are a helpful assistant. Use the provided context to answer the user's question. "
                "The context contains relevant information - extract and synthesize the answer from it. "
                "Only say 'I don't know' if the context truly does not contain any relevant information to answer the question."
            ),
        },
        {
            "role": "user",
            "content": f"Question: {question}\n\nContext:\n{context_text}",
        },
    ]

    completion = llm_client.chat.completions.create(
        model=CHAT_MODEL,
        messages=messages,
        temperature=0.2,
    )

    return completion.choices[0].message.content.strip()


def extract_text_from_pdf(file_content: bytes) -> str:
    """Extract raw text from a PDF file."""
    try:
        pdf_reader = PdfReader(BytesIO(file_content))
        text_parts = []
        for page in pdf_reader.pages:
            extracted = page.extract_text()
            if extracted:
                text_parts.append(extracted)
        return "\n\n".join(text_parts)
    except Exception as exc:  # pragma: no cover - external dependency
        raise ValueError(f"Failed to parse PDF: {exc}") from exc


def chunk_text(text: str, max_words: int = 400) -> List[str]:
    """Turn raw text into ~max_words chunks."""
    words = text.split()
    if not words:
        return []

    chunks: List[str] = []
    current_chunk: List[str] = []

    for word in words:
        current_chunk.append(word)
        if len(current_chunk) >= max_words:
            chunks.append(" ".join(current_chunk).strip())
            current_chunk = []

    if current_chunk:
        chunks.append(" ".join(current_chunk).strip())

    return chunks


app = FastAPI(
    title="Multi-modal RAG Backend",
    description="Approach 1: text embeddings + media payload via Qdrant",
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def ingest_pdf_document(doc_id: str, pdf_bytes: bytes) -> int:
    """
    Ingest a PDF document (provided as bytes) into Qdrant.

    Args:
        doc_id: Identifier for the PDF (usually filename with extension)
        pdf_bytes: Raw PDF bytes
    """
    text = extract_text_from_pdf(pdf_bytes)
    if not text.strip():
        raise ValueError("No text content found in PDF.")

    text_chunks = chunk_text(text)
    if not text_chunks:
        raise ValueError("No valid chunks produced from PDF content.")

    points: List[PointStruct] = []
    for chunk_index, text_chunk in enumerate(text_chunks):
        if len(text_chunk.split()) < 5:
            continue

        try:
            embedding = embed_text(text_chunk)
        except Exception as exc:
            raise ValueError(f"Embedding failed: {exc}") from exc

        payload = {
            "doc_id": doc_id,
            "chunk_index": chunk_index,
            "text_chunk": text_chunk,
            "image_urls": [],
            "video_urls": [],
            "source_doc": doc_id,
        }

        points.append(
            PointStruct(
                id=str(uuid.uuid4()),
                vector=embedding,
                payload=payload,
            )
        )

    if not points:
        raise ValueError("No chunks met the minimum length requirement.")

    try:
        qdrant_client.upsert(collection_name=QDRANT_COLLECTION, points=points)
    except Exception as exc:
        raise ValueError(f"Qdrant upsert failed: {exc}") from exc

    return len(points)


@app.post("/query")
def query_documents(request: QueryRequest):
    question = request.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="Question must not be empty.")

    top_k = request.top_k or 5
    top_k = max(1, min(top_k, 20))

    try:
        query_vector = embed_text(question)
    except Exception as exc:  # pragma: no cover - external service
        raise HTTPException(status_code=500, detail=f"Embedding failed: {exc}") from exc

    try:
        search_result = qdrant_client.search(
            collection_name=QDRANT_COLLECTION,
            query_vector=query_vector,
            limit=top_k,
        )
    except Exception as exc:  # pragma: no cover - external service
        raise HTTPException(status_code=500, detail=f"Qdrant search failed: {exc}") from exc

    context_chunks: List[str] = []
    image_urls: List[str] = []
    video_urls: List[str] = []

    for hit in search_result:
        payload = hit.payload or {}
        text_chunk = payload.get("text_chunk")
        if text_chunk:
            context_chunks.append(text_chunk)
        image_urls.extend(payload.get("image_urls", []))
        video_urls.extend(payload.get("video_urls", []))

    deduped_images = list(dict.fromkeys(url for url in image_urls if url))
    deduped_videos = list(dict.fromkeys(url for url in video_urls if url))

    answer_text = generate_answer(context_chunks, question)

    return {
        "answer_text": answer_text,
        "context_chunks": context_chunks,
        "image_urls": deduped_images,
        "video_urls": deduped_videos,
    }


@app.get("/health")
def health_check():
    return {"status": "ok"}


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
