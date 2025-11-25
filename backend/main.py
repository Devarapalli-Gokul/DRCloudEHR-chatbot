import os
import re
import uuid
from io import BytesIO
from pathlib import Path
from typing import List, Tuple

import fitz  # PyMuPDF
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastembed import TextEmbedding
from openai import OpenAI
from pydantic import BaseModel
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, FieldCondition, Filter, MatchValue, PointStruct, VectorParams
import uvicorn


load_dotenv()

# Namespace UUID for deterministic UUID generation
DOC_NAMESPACE = uuid.UUID('6ba7b810-9dad-11d1-80b4-00c04fd430c8')


def sanitize_doc_id(doc_id: str) -> str:
    """Sanitize doc_id for use in file paths."""
    # Remove or replace unsafe characters
    sanitized = re.sub(r'[<>:"|?*\x00-\x1f]', '_', doc_id)
    # Remove leading/trailing dots and spaces
    sanitized = sanitized.strip('. ')
    # Replace multiple underscores with single
    sanitized = re.sub(r'_+', '_', sanitized)
    # Ensure it's not empty
    if not sanitized:
        sanitized = "unknown_doc"
    return sanitized


def generate_point_id(doc_id: str, page_index: int, chunk_index: int) -> str:
    """
    Generate a deterministic UUID for a point ID based on doc_id, page, and chunk.
    Uses uuid5 to ensure same input always produces same UUID.
    """
    # Create a unique string identifier
    unique_string = f"{doc_id}|{page_index}|{chunk_index}"
    # Generate deterministic UUID
    point_uuid = uuid.uuid5(DOC_NAMESPACE, unique_string)
    return str(point_uuid)


class QueryRequest(BaseModel):
    question: str
    top_k: int | None = 5


class RetrievedChunk(BaseModel):
    text_chunk: str
    image_urls: List[str]
    doc_id: str | None = None
    page: int | None = None


class QueryResponse(BaseModel):
    answer_text: str
    chunks: List[RetrievedChunk]
    # Keep legacy fields for backward compatibility
    context_chunks: List[str] | None = None
    image_urls: List[str] | None = None
    video_urls: List[str] | None = None


GROQ_API_KEY = os.getenv("GROQ_API_KEY")
QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
QDRANT_COLLECTION = os.getenv("QDRANT_COLLECTION", "docs_chunks")

EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"
EMBEDDING_DIM = 384
CHAT_MODEL = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")
BASE_IMAGE_URL = os.getenv("BASE_IMAGE_URL", "/static")

# Static files directory for images
STATIC_DIR = Path(__file__).parent / "static"
STATIC_DIR.mkdir(exist_ok=True)

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


def generate_answer(context_chunks: List[str], question: str, has_images: bool = False, image_info: List[dict] | None = None) -> str:
    if not context_chunks:
        return "I don't know."

    # Clean up chunks - remove common Wikipedia metadata noise
    cleaned_chunks = []
    for chunk in context_chunks:
        cleaned = chunk
        # Remove Wikipedia page footer metadata (dates, URLs, page numbers)
        cleaned = re.sub(r'\d{1,2}/\d{1,2}/\d{4}, \d{1,2}:\d{2}.*?https?://[^\s]+ \d+/\d+$', '', cleaned, flags=re.MULTILINE | re.DOTALL)
        # Remove standalone URLs that are just references
        cleaned = re.sub(r'https?://[^\s]+(?=\s|$)', '', cleaned)
        # Remove Wikipedia reference patterns like [1], [2], etc. when they're standalone
        cleaned = re.sub(r'\[\d+\](?=\s|$)', '', cleaned)
        # Clean up multiple spaces
        cleaned = re.sub(r'\s+', ' ', cleaned).strip()
        # Remove very short chunks that are likely just metadata
        if len(cleaned) > 50:
            cleaned_chunks.append(cleaned)
    
    if not cleaned_chunks:
        cleaned_chunks = context_chunks  # Fallback to original if all cleaned out

    context_text = "\n\n".join(
        f"Context {index + 1}:\n{chunk}" for index, chunk in enumerate(cleaned_chunks)
    )
    
    # Add image information if available
    image_context = ""
    if has_images and image_info is not None:
        image_context = "\n\nRelated images are available from the same context pages. You may reference them naturally in your answer if they're relevant to the question."

    system_prompt = """You are an AI assistant answering user questions based on retrieved context from a knowledge base.

Your goals:
1. Give a clear, concise, and well-structured answer to the user's question.
2. Use the retrieved text chunks as your primary source of truth.
3. If relevant images are available, mention them naturally in your answer.
4. Never mention internal implementation details like "chunks", "vector database", "embeddings", or "retriever".

Answering Rules:
- Base your answer ONLY on information found in the context. If the answer is not fully in the context, say you're unsure or partially unsure. Do NOT invent facts.
- Use a friendly, professional tone with short paragraphs and bullet points where helpful.
- Avoid filler phrases like "Based on the provided context" or "From the chunks".
- Do NOT talk about how you work (no "I am an AI model", no "I can't see images").

Structure your answer:
- Start with a one-sentence summary answer.
- Provide a short explanation section (2-5 bullet points or short paragraphs).
- If images are available and relevant, add a "Related visuals:" section at the end describing them naturally.

Handling images:
- Treat images as supporting visuals, not the main content.
- Never say "I can't see the image", "There are related images below", or "Related image 1, 2, 3".
- If images exist and are relevant, use a section like: "Related visuals: Image 1: [description]. Image 2: [description]."
- Do NOT include raw URLs in the text.

Don't expose internals:
- Do NOT use words like: "chunks", "chunking", "embedding", "vector DB", "FastEmbed", "Qdrant", "index", "retriever".
- Instead use natural language: "the documentation", "the reference material", "the stored content", "your files".

Length: Aim for 3-8 sentences for simple questions. For complex topics, go longer but stay focused and avoid repetition.

Only say 'I don't know' if the context truly does not contain any relevant information to answer the question."""

    messages = [
        {
            "role": "system",
            "content": system_prompt,
        },
        {
            "role": "user",
            "content": f"Question: {question}\n\nContext:\n{context_text}{image_context}",
        },
    ]

    completion = llm_client.chat.completions.create(
        model=CHAT_MODEL,
        messages=messages,
        temperature=0.2,
    )

    return completion.choices[0].message.content.strip()


def extract_images_from_page(doc_id: str, page_index: int, page: fitz.Page) -> List[str]:
    """
    Extract all images from a PDF page and save them to disk.
    Returns a list of image URLs.
    """
    image_urls = []
    image_list = page.get_images(full=True)
    
    if not image_list:
        return image_urls
    
    # Sanitize doc_id for file system safety
    safe_doc_id = sanitize_doc_id(doc_id)
    
    # Create directory for this document's images
    doc_images_dir = STATIC_DIR / "docs" / safe_doc_id
    doc_images_dir.mkdir(parents=True, exist_ok=True)
    
    for img_index, img in enumerate(image_list):
        try:
            xref = img[0]
            base_image = page.parent.extract_image(xref)
            image_bytes = base_image["image"]
            image_ext = base_image["ext"]
            
            # Save image to disk
            image_filename = f"page_{page_index}_img_{img_index}.{image_ext}"
            image_path = doc_images_dir / image_filename
            with open(image_path, "wb") as img_file:
                img_file.write(image_bytes)
            
            # Build URL (use sanitized doc_id in URL)
            image_url = f"{BASE_IMAGE_URL}/docs/{safe_doc_id}/{image_filename}"
            image_urls.append(image_url)
        except Exception as exc:
            # Log but continue with other images
            print(f"Warning: Failed to extract image {img_index} from page {page_index}: {exc}")
            continue
    
    return image_urls


def extract_text_and_images_from_pdf(doc_id: str, pdf_bytes: bytes) -> List[Tuple[int, str, List[str]]]:
    """
    Extract text and images from a PDF, organized by page.
    
    Returns:
        List of tuples: (page_index, page_text, image_urls)
    """
    try:
        pdf_document = fitz.open(stream=pdf_bytes, filetype="pdf")
        pages_data = []
        
        for page_index in range(len(pdf_document)):
            page = pdf_document[page_index]
            
            # Extract text
            page_text = page.get_text()
            
            # Extract images
            image_urls = extract_images_from_page(doc_id, page_index, page)
            
            pages_data.append((page_index, page_text, image_urls))
        
        pdf_document.close()
        return pages_data
    except Exception as exc:
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

# Mount static files directory for serving images
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


def delete_document_chunks(doc_id: str) -> int:
    """
    Delete all existing chunks for a document from Qdrant.
    This ensures clean re-ingestion without orphaned chunks.
    
    Since Qdrant requires indexes for filtering, we scroll through all points
    and filter by payload in Python if the filter approach fails.
    
    Args:
        doc_id: The document ID to delete chunks for
    
    Returns:
        Number of points deleted
    """
    all_point_ids = []
    
    try:
        # Try filtering approach first (requires index)
        next_page_offset = None
        
        while True:
            scroll_result = qdrant_client.scroll(
                collection_name=QDRANT_COLLECTION,
                scroll_filter=Filter(
                    should=[
                        FieldCondition(
                            key="doc_id",
                            match=MatchValue(value=doc_id)
                        ),
                        FieldCondition(
                            key="source_doc",
                            match=MatchValue(value=doc_id)
                        )
                    ]
                ),
                limit=100,
                offset=next_page_offset,
                with_payload=True,  # Need payload to filter in Python fallback
                with_vectors=False,
            )
            
            points, next_page_offset = scroll_result
            
            if not points:
                break
            
            # Filter points by doc_id in payload
            for point in points:
                payload = point.payload or {}
                if payload.get("doc_id") == doc_id or payload.get("source_doc") == doc_id:
                    all_point_ids.append(point.id)
            
            if next_page_offset is None:
                break
    except Exception:
        # Filter approach failed (likely no index), fallback to scrolling all points
        try:
            next_page_offset = None
            
            while True:
                scroll_result = qdrant_client.scroll(
                    collection_name=QDRANT_COLLECTION,
                    limit=100,
                    offset=next_page_offset,
                    with_payload=True,
                    with_vectors=False,
                )
                
                points, next_page_offset = scroll_result
                
                if not points:
                    break
                
                # Filter points by doc_id in payload
                for point in points:
                    payload = point.payload or {}
                    if payload.get("doc_id") == doc_id or payload.get("source_doc") == doc_id:
                        all_point_ids.append(point.id)
                
                if next_page_offset is None:
                    break
        except Exception as exc:
            print(f"Warning: Failed to scroll points for deletion: {exc}")
            return 0
    
    if not all_point_ids:
        return 0
    
    try:
        # Delete by IDs
        qdrant_client.delete(
            collection_name=QDRANT_COLLECTION,
            points_selector=all_point_ids
        )
        return len(all_point_ids)
    except Exception as exc:
        print(f"Warning: Failed to delete points: {exc}")
        return 0


def ingest_pdf_document(doc_id: str, pdf_bytes: bytes, replace_existing: bool = True) -> int:
    """
    Ingest a PDF document (provided as bytes) into Qdrant.
    Processes the PDF page-by-page, extracting text and images,
    then chunks the text per page and associates images with chunks.

    Args:
        doc_id: Identifier for the PDF (usually filename with extension)
        pdf_bytes: Raw PDF bytes
        replace_existing: If True, delete existing chunks for this document before ingesting
    
    Returns:
        Number of chunks ingested
    """
    # Delete existing chunks if requested (ensures clean re-ingestion)
    if replace_existing:
        deleted_count = delete_document_chunks(doc_id)
        if deleted_count > 0:
            print(f"Deleted {deleted_count} existing chunk(s) for {doc_id}")
    
    # Extract pages with text and images
    pages_data = extract_text_and_images_from_pdf(doc_id, pdf_bytes)
    
    if not pages_data:
        raise ValueError("No pages found in PDF.")
    
    all_points: List[PointStruct] = []
    
    # Process each page
    for page_index, page_text, image_urls in pages_data:
        if not page_text.strip():
            # Skip empty pages, but still create a chunk if there are images
            if not image_urls:
                continue
            page_text = ""  # Empty text but has images
        
        # Chunk the page text
        page_chunks = chunk_text(page_text)
        
        # If no chunks but there's text, create at least one chunk
        if not page_chunks and page_text.strip():
            page_chunks = [page_text.strip()]
        
        # If no text chunks but there are images, create one empty chunk to associate images
        if not page_chunks and image_urls:
            page_chunks = [""]
        
        # Create points for each chunk from this page
        for chunk_index, text_chunk in enumerate(page_chunks):
            # Skip very short chunks unless they have images
            if len(text_chunk.split()) < 5 and not image_urls:
                continue

            # If chunk is too short but has images, use minimal text
            if not text_chunk.strip() and image_urls:
                text_chunk = f"[Page {page_index + 1} with {len(image_urls)} image(s)]"
            
            try:
                embedding = embed_text(text_chunk)
            except Exception as exc:
                raise ValueError(f"Embedding failed: {exc}") from exc

            payload = {
                "doc_id": doc_id,
                "page": page_index,
                "chunk_index": chunk_index,
                "text_chunk": text_chunk,
                "image_urls": image_urls,  # All images from this page
                "video_urls": [],
                "source_doc": doc_id,
            }

            # Create a deterministic UUID point ID
            point_id = generate_point_id(doc_id, page_index, chunk_index)
            
            all_points.append(
                PointStruct(
                    id=point_id,
                    vector=embedding,
                    payload=payload,
                )
            )

    if not all_points:
        raise ValueError("No valid chunks produced from PDF content.")

    try:
        qdrant_client.upsert(collection_name=QDRANT_COLLECTION, points=all_points)
    except Exception as exc:
        raise ValueError(f"Qdrant upsert failed: {exc}") from exc

    return len(all_points)


@app.post("/query", response_model=QueryResponse)
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
            with_payload=True,
        )
    except Exception as exc:  # pragma: no cover - external service
        raise HTTPException(status_code=500, detail=f"Qdrant search failed: {exc}") from exc

    retrieved_chunks: List[RetrievedChunk] = []
    context_chunks: List[str] = []
    all_image_urls: List[str] = []
    all_video_urls: List[str] = []

    for hit in search_result:
        payload = hit.payload or {}
        text_chunk = payload.get("text_chunk", "")
        image_urls = payload.get("image_urls", []) or []
        video_urls = payload.get("video_urls", []) or []
        
        if text_chunk:
            context_chunks.append(text_chunk)
        
        if image_urls:
            all_image_urls.extend(image_urls)
        
        if video_urls:
            all_video_urls.extend(video_urls)
        
        retrieved_chunks.append(
            RetrievedChunk(
                text_chunk=text_chunk,
                image_urls=image_urls,
                doc_id=payload.get("doc_id"),
                page=payload.get("page"),
            )
        )

    # Build context string from text chunks only for LLM
    # Check if any chunks have images and collect image info
    has_images = any(chunk.image_urls for chunk in retrieved_chunks)
    image_info = []
    if has_images:
        for chunk in retrieved_chunks:
            if chunk.image_urls:
                image_info.append({
                    "page": chunk.page,
                    "urls": chunk.image_urls,
                })
    
    answer_text = generate_answer(context_chunks, question, has_images=has_images, image_info=image_info)

    # Deduplicate legacy fields
    deduped_images = list(dict.fromkeys(url for url in all_image_urls if url))
    deduped_videos = list(dict.fromkeys(url for url in all_video_urls if url))

    return QueryResponse(
        answer_text=answer_text,
        chunks=retrieved_chunks,
        context_chunks=context_chunks,  # Legacy field
        image_urls=deduped_images,  # Legacy field (flattened)
        video_urls=deduped_videos,  # Legacy field
    )


@app.get("/health")
def health_check():
    return {"status": "ok"}


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
