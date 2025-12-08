import os
import re
import uuid
from io import BytesIO
from pathlib import Path
from typing import List, Tuple, Dict, Literal
from dataclasses import dataclass

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


def generate_point_id(doc_id: str, page_index: int, chunk_index: int | str) -> str:
    """
    Generate a deterministic UUID for a point ID based on doc_id, page, and chunk.
    Uses uuid5 to ensure same input always produces same UUID.
    """
    # Create a unique string identifier
    unique_string = f"{doc_id}|{page_index}|{chunk_index}"
    # Generate deterministic UUID
    point_uuid = uuid.uuid5(DOC_NAMESPACE, unique_string)
    return str(point_uuid)


# Block representation for page layout
@dataclass
class Block:
    """Represents a block of content on a page (heading, paragraph, image, caption, etc.)"""
    doc_id: str
    page: int
    block_index: int
    block_type: Literal["heading", "paragraph", "image", "caption", "list_item"]
    text: str | None = None
    image_url: str | None = None


class QueryRequest(BaseModel):
    question: str
    top_k: int | None = 1


class RetrievedChunk(BaseModel):
    text_chunk: str
    image_urls: List[str]
    doc_id: str | None = None
    page: int | None = None


class SupportingChunk(BaseModel):
    """Metadata for chunks that support the answer"""
    id: str
    kind: str
    doc_id: str | None = None
    page: int | None = None
    text_snippet: str
    similarity_score: float | None = None


class RelatedImage(BaseModel):
    """Metadata for related images with captions"""
    chunk_id: str
    doc_id: str | None = None
    page: int | None = None
    image_urls: List[str]
    caption: str | None = None
    similarity_score: float | None = None


class QueryResponse(BaseModel):
    answer_text: str
    chunks: List[RetrievedChunk]
    # New structured fields
    supporting_chunks: List[SupportingChunk] | None = None
    related_images: List[RelatedImage] | None = None
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

# Retrieval configuration - tuned for precision and accuracy
SIMILARITY_THRESHOLD = float(os.getenv("SIMILARITY_THRESHOLD", "0.35"))  # Text chunks threshold
IMAGE_SIMILARITY_THRESHOLD = float(os.getenv("IMAGE_SIMILARITY_THRESHOLD", "0.45"))  # Image chunks threshold
ANSWER_TOP_N = int(os.getenv("ANSWER_TOP_N", "1"))  # Top 1 chunk for answer generation
MAX_IMAGES = int(os.getenv("MAX_IMAGES", "3"))  # Maximum number of images to return
RETRIEVAL_LIMIT = int(os.getenv("RETRIEVAL_LIMIT", "30"))  # Initial retrieval limit before filtering

# Figure context configuration
MAX_HEADING_DISTANCE = int(os.getenv("MAX_HEADING_DISTANCE", "10"))  # Max blocks between image and heading
K_BEFORE_PARAGRAPHS = int(os.getenv("K_BEFORE_PARAGRAPHS", "3"))  # Paragraphs before image
K_AFTER_PARAGRAPHS = int(os.getenv("K_AFTER_PARAGRAPHS", "3"))  # Paragraphs after image

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
    
    # Add structured image information if available
    image_context = ""
    if has_images and image_info is not None and len(image_info) > 0:
        image_descriptions = []
        for idx, img_info in enumerate(image_info, 1):
            caption = img_info.get("caption", "")
            page = img_info.get("page", "?")
            if caption:
                image_descriptions.append(f"Image {idx}: {caption} (from page {page + 1 if isinstance(page, int) else page} of the documentation)")
            else:
                image_descriptions.append(f"Image {idx}: A diagram from page {page + 1 if isinstance(page, int) else page} of the documentation")
        
        if image_descriptions:
            image_context = "\n\nImage descriptions:\n" + "\n".join(image_descriptions)

    system_prompt = """You are an AI assistant providing precise and accurate answers based on retrieved context from documentation.

CRITICAL PRINCIPLES:
1. PRECISION: Answer only what is explicitly stated in the context. Do not extrapolate or infer beyond what is written.
2. ACCURACY: Every fact you state must be directly supported by the context provided. If information is missing or uncertain, explicitly say so.
3. RELEVANCE: Focus strictly on answering the user's question. Do not include tangential information unless it directly relates.
4. CLARITY: Be direct and concise. Avoid unnecessary words or filler phrases.

Answering Rules:
- Base your answer ONLY on information explicitly found in the context.
- If the context doesn't fully answer the question, state: "Based on the available information, [partial answer]. However, [what is missing/uncertain]."
- If the context contradicts the question or doesn't address it, say: "The available information doesn't directly address [specific aspect of question]."
- DO NOT invent, infer, or assume facts that aren't in the context.
- DO NOT add information from general knowledge unless it's common sense context needed to understand the answer.
- Use a clear, professional tone. Avoid filler phrases like "Based on the provided context" or "According to the documentation."

Structure your answer:
- Start with a direct, precise one-sentence answer to the question.
- Follow with specific details from the context that support this answer.
- Use bullet points or short paragraphs for clarity.
- End with any relevant caveats or limitations if information is incomplete.

Handling images:
- CRITICAL: ONLY mention images if there is an "Image descriptions" section provided below.
- If NO "Image descriptions" section is provided, DO NOT mention images, diagrams, figures, or visuals at all.
- If images ARE provided in the "Image descriptions" section:
  * Only mention them if they directly support answering the question
  * Reference them naturally (e.g., "As shown in the diagram...")
  * Do NOT include raw URLs

Don't expose internals:
- Do NOT use technical terms like: "chunks", "embedding", "vector DB", "retriever", "index".
- Use natural language: "the documentation", "the reference material", "your files".

Length: Be as concise as possible while remaining complete. Typically 2-6 sentences for simple questions. More for complex topics, but stay focused.

Accuracy check: Before answering, verify that every claim you make can be traced back to a specific part of the context. If unsure, explicitly state the uncertainty."""

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
        temperature=0.1,  # Lower temperature for more precise, deterministic answers
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


def extract_text_and_images_from_pdf(doc_id: str, pdf_bytes: bytes) -> Dict[int, List[Block]]:
    """
    Extract text and images from a PDF, organized as blocks per page.
    
    Returns:
        Dictionary mapping page_index -> List[Block] in reading order
    """
    try:
        pdf_document = fitz.open(stream=pdf_bytes, filetype="pdf")
        pages_blocks: Dict[int, List[Block]] = {}
        
        for page_index in range(len(pdf_document)):
            page = pdf_document[page_index]
            blocks: List[Block] = []
            
            # Extract text with layout information
            text_dict = page.get_text("dict")
            text_blocks = text_dict.get("blocks", [])
            
            # Extract images to get their URLs
            image_urls_list = extract_images_from_page(doc_id, page_index, page)
            
            # Process text blocks first
            block_index = 0
            for text_block in text_blocks:
                if "lines" not in text_block:
                    continue
                
                # Collect text from all lines in this block
                block_text_parts = []
                for line in text_block.get("lines", []):
                    for span in line.get("spans", []):
                        block_text_parts.append(span.get("text", ""))
                
                block_text = " ".join(block_text_parts).strip()
                if not block_text:
                    continue
                
                # Determine block type based on heuristics
                block_type: Literal["heading", "paragraph", "caption", "list_item"] = "paragraph"
                
                # Check if it's a caption (starts with "Figure", "Fig.", etc.)
                if re.match(r'^(Figure|Fig\.?)\s+\d+[:.]', block_text, re.IGNORECASE):
                    block_type = "caption"
                # Check if it's a heading (short, bold, or larger font)
                elif len(block_text.split()) <= 10:
                    # Check font size if available
                    font_size = 0
                    for line in text_block.get("lines", []):
                        for span in line.get("spans", []):
                            size = span.get("size", 0)
                            if size > font_size:
                                font_size = size
                    
                    # If significantly larger than typical (assume 12pt is typical)
                    if font_size > 14:
                        block_type = "heading"
                
                blocks.append(Block(
                    doc_id=doc_id,
                    page=page_index,
                    block_index=block_index,
                    block_type=block_type,
                    text=block_text,
                    image_url=None
                ))
                block_index += 1
            
            # Add image blocks at the end (simpler approach)
            # The figure context builder will still find nearby text by scanning backwards
            for image_url in image_urls_list:
                blocks.append(Block(
                    doc_id=doc_id,
                    page=page_index,
                    block_index=block_index,
                    block_type="image",
                    text=None,
                    image_url=image_url
                ))
                block_index += 1
            
            pages_blocks[page_index] = blocks
        
        pdf_document.close()
        return pages_blocks
    except Exception as exc:
        raise ValueError(f"Failed to parse PDF: {exc}") from exc


def build_figure_context_for_image(blocks: List[Block], image_block_index: int) -> Tuple[str, List[int]]:
    """
    Build figure context text for an image by finding heading, caption, and nearby paragraphs.
    
    Returns:
        (figure_context_text, contributing_block_indices)
    """
    if image_block_index >= len(blocks) or blocks[image_block_index].block_type != "image":
        return "", []
    
    contributing_indices = [image_block_index]
    context_parts = []
    
    # Find nearest heading above
    heading_text = None
    heading_index = None
    for i in range(image_block_index - 1, max(-1, image_block_index - MAX_HEADING_DISTANCE - 1), -1):
        if i < 0:
            break
        if blocks[i].block_type == "heading":
            heading_text = blocks[i].text
            heading_index = i
            contributing_indices.append(i)
            break
    
    # Find caption below (next 1-2 blocks)
    caption_text = None
    caption_indices = []
    for i in range(image_block_index + 1, min(len(blocks), image_block_index + 3)):
        if blocks[i].block_type == "caption":
            caption_text = blocks[i].text
            caption_indices.append(i)
            contributing_indices.append(i)
            break
        # Also check if it's a paragraph that looks like a caption
        elif blocks[i].block_type == "paragraph" and blocks[i].text:
            if re.match(r'^(Figure|Fig\.?)\s+\d+[:.]', blocks[i].text, re.IGNORECASE):
                caption_text = blocks[i].text
                caption_indices.append(i)
                contributing_indices.append(i)
                break
    
    # Collect nearby paragraphs
    start = max(0, image_block_index - K_BEFORE_PARAGRAPHS)
    end = min(len(blocks), image_block_index + K_AFTER_PARAGRAPHS + 1)
    
    nearby_paragraphs = []
    for i in range(start, end):
        if i == image_block_index or i in caption_indices:
            continue
        if blocks[i].block_type in ["paragraph", "list_item"] and blocks[i].text:
            nearby_paragraphs.append((i, blocks[i].text))
            if i not in contributing_indices:
                contributing_indices.append(i)
    
    # Build context text
    if heading_text:
        context_parts.append(f"Section: {heading_text}")
    
    if caption_text:
        context_parts.append(f"Caption: {caption_text}")
    
    # Add nearby paragraphs (limit to ~300 words total for better context)
    # Prioritize paragraphs that mention key terms related to the image
    paragraph_texts = []
    word_count = 0
    for _, para_text in nearby_paragraphs:
        words = para_text.split()
        if word_count + len(words) > 300:
            # Truncate if needed
            remaining = 300 - word_count
            if remaining > 0:
                paragraph_texts.append(" ".join(words[:remaining]))
            break
        paragraph_texts.append(para_text)
        word_count += len(words)
    
    if paragraph_texts:
        context_parts.append("\n".join(paragraph_texts))
    
    # Build the base figure context first
    figure_context = "\n\n".join(context_parts).strip()
    
    # If we have very little context, SKIP this figure entirely
    # instead of embedding a generic "unlabeled diagram" sentence.
    if len(figure_context.split()) < 10:
        # Not enough semantic content to be useful for retrieval
        return "", []
    
    return figure_context, sorted(contributing_indices)


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
    Uses block-based parsing to create figure chunks (with images) and text chunks separately.

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
    
    # Extract pages as blocks
    pages_blocks = extract_text_and_images_from_pdf(doc_id, pdf_bytes)
    
    if not pages_blocks:
        raise ValueError("No pages found in PDF.")
    
    all_points: List[PointStruct] = []
    
    # Process each page
    for page_index, blocks in pages_blocks.items():
        if not blocks:
            continue
        
        # First, create figure chunks for each image
        image_chunk_index = 0
        for block in blocks:
            if block.block_type == "image" and block.image_url:
                # Build figure context
                figure_context, contributing_indices = build_figure_context_for_image(blocks, block.block_index)
                
                if not figure_context.strip():
                    print(f"[INGEST] Skipping image on page {page_index}: no context found")
                    continue
                
                try:
                    embedding = embed_text(figure_context)
                except Exception as exc:
                    print(f"Warning: Failed to embed figure context for image on page {page_index}: {exc}")
                    continue
                
                payload = {
                    "doc_id": doc_id,
                    "page": page_index,
                    "chunk_index": image_chunk_index,
                    "kind": "figure",
                    "text_chunk": figure_context,
                    "image_urls": [block.image_url],  # Only this specific image
                    "video_urls": [],
                    "source_doc": doc_id,
                    "block_indices": contributing_indices,
                }
                
                # Generate unique ID for figure chunk
                point_id = generate_point_id(doc_id, page_index, f"fig_{image_chunk_index}")
                
                all_points.append(
                    PointStruct(
                        id=point_id,
                        vector=embedding,
                        payload=payload,
                    )
                )
                image_chunk_index += 1
                print(f"[INGEST] Created figure chunk for page {page_index}, image {image_chunk_index}: {block.image_url}")
        
        # Now create text chunks from text-like blocks
        text_blocks = [b for b in blocks if b.block_type in ["heading", "paragraph", "list_item"] and b.text]
        
        if not text_blocks:
            continue
        
        # Concatenate text blocks into chunks (similar to chunk_text but block-aware)
        current_chunk_blocks: List[Block] = []
        current_word_count = 0
        text_chunk_index = 0
        
        for block in text_blocks:
            words = block.text.split()
            current_chunk_blocks.append(block)
            current_word_count += len(words)
            
            # If we've reached max_words, create a chunk
            if current_word_count >= 400:
                chunk_text = " ".join(b.text for b in current_chunk_blocks if b.text)
                block_indices = [b.block_index for b in current_chunk_blocks]
                
                if len(chunk_text.split()) >= 5:  # Skip very short chunks
                    try:
                        embedding = embed_text(chunk_text)
                    except Exception as exc:
                        print(f"Warning: Failed to embed text chunk on page {page_index}: {exc}")
                        current_chunk_blocks = []
                        current_word_count = 0
                        continue
                    
                    payload = {
                        "doc_id": doc_id,
                        "page": page_index,
                        "chunk_index": text_chunk_index,
                        "kind": "text",
                        "text_chunk": chunk_text,
                        "image_urls": [],  # Text chunks don't automatically get images
                        "video_urls": [],
                        "source_doc": doc_id,
                        "block_indices": block_indices,
                    }
                    
                    point_id = generate_point_id(doc_id, page_index, text_chunk_index)
                    
                    all_points.append(
                        PointStruct(
                            id=point_id,
                            vector=embedding,
                            payload=payload,
                        )
                    )
                    text_chunk_index += 1
                
                current_chunk_blocks = []
                current_word_count = 0
        
        # Handle remaining blocks
        if current_chunk_blocks:
            chunk_text = " ".join(b.text for b in current_chunk_blocks if b.text)
            block_indices = [b.block_index for b in current_chunk_blocks]
            
            if len(chunk_text.split()) >= 5:
                try:
                    embedding = embed_text(chunk_text)
                except Exception as exc:
                    print(f"Warning: Failed to embed final text chunk on page {page_index}: {exc}")
                else:
                    payload = {
                        "doc_id": doc_id,
                        "page": page_index,
                        "chunk_index": text_chunk_index,
                        "kind": "text",
                        "text_chunk": chunk_text,
                        "image_urls": [],
                        "video_urls": [],
                        "source_doc": doc_id,
                        "block_indices": block_indices,
                    }
                    
                    point_id = generate_point_id(doc_id, page_index, text_chunk_index)
                    
                    all_points.append(
                        PointStruct(
                            id=point_id,
                            vector=embedding,
                            payload=payload,
                        )
                    )

    if not all_points:
        raise ValueError("No valid chunks produced from PDF content.")

    # Count chunks by kind
    figure_count = sum(1 for p in all_points if p.payload.get("kind") == "figure")
    text_count = sum(1 for p in all_points if p.payload.get("kind") == "text")
    print(f"[INGEST] Created {len(all_points)} total chunks: {figure_count} figure chunks, {text_count} text chunks")

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

    try:
        query_vector = embed_text(question)
    except Exception as exc:  # pragma: no cover - external service
        raise HTTPException(status_code=500, detail=f"Embedding failed: {exc}") from exc

    # --- TEXT SEARCH ---
    # Try filtered search first, fallback to Python filtering if filter fails
    try:
        text_search_result = qdrant_client.search(
            collection_name=QDRANT_COLLECTION,
            query_vector=query_vector,
            limit=RETRIEVAL_LIMIT,
            with_payload=True,
            query_filter=Filter(
                must=[
                    FieldCondition(
                        key="kind",
                        match=MatchValue(value="text")
                    )
                ]
            ),
        )
    except Exception as exc:
        # Filter might fail if "kind" field isn't indexed - fallback to Python filtering
        print(f"[WARNING] Qdrant filter failed, using Python filtering: {exc}")
        try:
            all_search_result = qdrant_client.search(
                collection_name=QDRANT_COLLECTION,
                query_vector=query_vector,
                limit=RETRIEVAL_LIMIT,
                with_payload=True,
            )
            # Filter in Python
            text_search_result = [
                hit for hit in all_search_result
                if (hit.payload or {}).get("kind") == "text"
            ]
        except Exception as exc2:
            raise HTTPException(status_code=500, detail=f"Qdrant text search failed: {exc2}") from exc2

    text_hits = []
    for hit in text_search_result:
        score = hit.score if hasattr(hit, "score") else 1.0
        if score >= SIMILARITY_THRESHOLD:
            text_hits.append((hit, score))

    # --- FIGURE SEARCH ---
    # Try filtered search first, fallback to Python filtering if filter fails
    try:
        figure_search_result = qdrant_client.search(
            collection_name=QDRANT_COLLECTION,
            query_vector=query_vector,
            limit=RETRIEVAL_LIMIT,
            with_payload=True,
            query_filter=Filter(
                must=[
                    FieldCondition(
                        key="kind",
                        match=MatchValue(value="figure")
                    )
                ]
            ),
        )
    except Exception as exc:
        # Filter might fail if "kind" field isn't indexed - fallback to Python filtering
        print(f"[WARNING] Qdrant filter failed, using Python filtering: {exc}")
        try:
            all_search_result = qdrant_client.search(
                collection_name=QDRANT_COLLECTION,
                query_vector=query_vector,
                limit=RETRIEVAL_LIMIT,
                with_payload=True,
            )
            # Filter in Python
            figure_search_result = [
                hit for hit in all_search_result
                if (hit.payload or {}).get("kind") == "figure"
            ]
        except Exception as exc2:
            raise HTTPException(status_code=500, detail=f"Qdrant figure search failed: {exc2}") from exc2

    figure_hits = []
    for hit in figure_search_result:
        score = hit.score if hasattr(hit, "score") else 1.0
        if score >= IMAGE_SIMILARITY_THRESHOLD:
            figure_hits.append((hit, score))

    # Sort both by similarity descending
    text_hits.sort(key=lambda x: x[1], reverse=True)
    figure_hits.sort(key=lambda x: x[1], reverse=True)

    if not text_hits and not figure_hits:
        # No good matches in the DB → let LLM answer "I don't know"
        answer_text = generate_answer([], question, has_images=False, image_info=None)
        return QueryResponse(
            answer_text=answer_text,
            chunks=[],
            supporting_chunks=[],
            related_images=[],
            context_chunks=[],
            image_urls=[],
            video_urls=[],
        )

    # Use both text & figure scores for OOD
    best_text = text_hits[0][1] if text_hits else 0.0
    best_figure = figure_hits[0][1] if figure_hits else 0.0
    top_similarity = max(best_text, best_figure)
    OUT_OF_DOMAIN_THRESHOLD = 0.40  # slightly lower than before

    if top_similarity < OUT_OF_DOMAIN_THRESHOLD:
        # Docs probably don't cover this query well
        print(f"[DEBUG] OOD query: top_similarity={top_similarity:.3f} < {OUT_OF_DOMAIN_THRESHOLD}")
        answer_text = generate_answer([], question, has_images=False, image_info=None)
        return QueryResponse(
            answer_text=answer_text,
            chunks=[],
            supporting_chunks=[],
            related_images=[],
            context_chunks=[],
            image_urls=[],
            video_urls=[],
        )

    # Select answer-supporting chunks (top N text chunks)
    # Use top_k from request if provided, otherwise fall back to ANSWER_TOP_N
    top_k = request.top_k if request.top_k is not None else ANSWER_TOP_N
    answer_hits = text_hits[:top_k]
    context_chunks: List[str] = []
    supporting_chunks: List[SupportingChunk] = []
    answer_chunk_ids = set()
    answer_doc_ids = set()
    answer_pages = set()

    for hit, score in answer_hits:
        payload = hit.payload or {}
        text_chunk = payload.get("text_chunk", "")
        if not text_chunk:
            continue

        context_chunks.append(text_chunk)
        answer_chunk_ids.add(hit.id)

        doc_id = payload.get("doc_id")
        page = payload.get("page")

        if doc_id:
            answer_doc_ids.add(doc_id)
        if page is not None:
            answer_pages.add(page)

        supporting_chunks.append(
            SupportingChunk(
                id=str(hit.id),
                kind=payload.get("kind", "text"),
                doc_id=doc_id,
                page=page,
                text_snippet=text_chunk[:200] + "..." if len(text_chunk) > 200 else text_chunk,
                similarity_score=score,
            )
        )

    # Build question keywords (optional)
    question_lower = question.lower()
    stop_words = {"what", "is", "are", "the", "a", "an", "and", "or", "but", "for", "with", "about", "from", "give", "me", "show"}
    question_words = set(re.findall(r'\b\w{3,}\b', question_lower))
    question_words = {w for w in question_words if w not in stop_words}

    # Iterate figure hits with simpler thresholds
    image_candidates = []

    for hit, score in figure_hits:
        payload = hit.payload or {}
        kind = payload.get("kind", "figure")

        if kind != "figure":
            continue

        image_urls = payload.get("image_urls", []) or []
        if not image_urls:
            continue

        image_doc_id = payload.get("doc_id")
        image_page = payload.get("page")

        same_doc_as_answer = image_doc_id in answer_doc_ids if answer_doc_ids and image_doc_id else False

        # Adaptive threshold: slightly lower if from same doc as answer
        base_thresh = IMAGE_SIMILARITY_THRESHOLD  # e.g. 0.45
        adaptive_threshold = base_thresh - 0.05 if same_doc_as_answer else base_thresh

        if score < adaptive_threshold:
            # Too weak even with adaptive threshold
            continue

        # (Optional) keyword check – keep as a bonus, not a hard requirement
        context_text = (payload.get("text_chunk") or "").lower()
        context_has_keywords = False
        for kw in question_words:
            if re.search(r'\b' + re.escape(kw) + r'\b', context_text):
                context_has_keywords = True
                break

        # Extract caption
        figure_context_orig = payload.get("text_chunk", "") or ""
        caption = None

        if "Caption:" in figure_context_orig:
            m = re.search(r'Caption:\s*([^\n]+)', figure_context_orig)
            if m:
                caption = m.group(1).strip()

        if not caption:
            sentences = re.split(r'[.!?]\s+', figure_context_orig)
            if sentences and sentences[0].strip():
                caption = sentences[0].strip()
            else:
                caption = figure_context_orig[:100].strip()

        image_candidates.append({
            "chunk_id": str(hit.id),
            "doc_id": image_doc_id,
            "page": image_page,
            "image_urls": image_urls,
            "caption": caption,
            "similarity_score": score,
            "same_doc": same_doc_as_answer,
            "has_keywords": context_has_keywords,
        })

    # Prioritize same-doc & high-score images
    image_candidates.sort(
        key=lambda img: (
            0 if img["same_doc"] else 1,                 # same-doc first
            0 if img["has_keywords"] else 1,             # keyword matches next
            -img["similarity_score"]                     # then score desc
        )
    )

    image_candidates = image_candidates[:MAX_IMAGES]

    # Build related_images and legacy image_urls
    related_images = [
        RelatedImage(
            chunk_id=img["chunk_id"],
            doc_id=img["doc_id"],
            page=img["page"],
            image_urls=img["image_urls"],
            caption=img["caption"],
            similarity_score=img["similarity_score"],
        )
        for img in image_candidates
    ]

    all_image_urls = []
    for img in image_candidates:
        all_image_urls.extend(img["image_urls"])

    deduped_images = list(dict.fromkeys(url for url in all_image_urls if url))

    # Build RetrievedChunk list (text only)
    retrieved_chunks: List[RetrievedChunk] = []
    for hit, _ in answer_hits:
        payload = hit.payload or {}
        retrieved_chunks.append(
            RetrievedChunk(
                text_chunk=payload.get("text_chunk", ""),
                image_urls=[],                      # keep empty in Option 1
                doc_id=payload.get("doc_id"),
                page=payload.get("page"),
            )
        )

    # Call generate_answer with image info (for better answers)
    has_images = len(image_candidates) > 0
    image_info_for_llm = [
        {
            "caption": img["caption"],
            "page": img["page"],
        }
        for img in image_candidates
    ]

    answer_text = generate_answer(
        context_chunks=context_chunks,
        question=question,
        has_images=has_images,
        image_info=image_info_for_llm,
    )

    return QueryResponse(
        answer_text=answer_text,
        chunks=retrieved_chunks,
        supporting_chunks=supporting_chunks,
        related_images=related_images,
        context_chunks=context_chunks,
        image_urls=deduped_images,
        video_urls=[],
    )


@app.get("/health")
def health_check():
    return {"status": "ok"}


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
