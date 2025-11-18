#!/usr/bin/env python3
"""
CLI script to ingest PDF documents from the ./docs/ folder into Qdrant.

Usage:
    python ingest_docs.py --all              # Ingest all PDF files from ./docs/
    python ingest_docs.py --file docs/file.pdf  # Ingest a single PDF
"""

import argparse
import sys
from pathlib import Path

# Add parent directory to path to import from main
sys.path.insert(0, str(Path(__file__).parent))

from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Import after loading env vars
# Note: Importing main.py will initialize qdrant_client, embedding_model, etc.
from main import ingest_pdf_document, QDRANT_COLLECTION


def get_doc_id_from_filename(filepath: str) -> str:
    """Return the filename (with extension) to use as doc_id."""
    return Path(filepath).name


def ingest_file(filepath: str) -> int:
    """
    Ingest a single PDF file.
    
    Args:
        filepath: Path to the PDF file
    
    Returns:
        Number of chunks indexed
    
    Raises:
        FileNotFoundError: If file doesn't exist
        ValueError: If ingestion fails
    """
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"File not found: {filepath}")

    if not filepath.is_file():
        raise ValueError(f"Path is not a file: {filepath}")

    if filepath.suffix.lower() != ".pdf":
        raise ValueError(f"Unsupported file type (expected .pdf): {filepath}")

    try:
        with open(filepath, "rb") as f:
            pdf_bytes = f.read()
    except Exception as exc:
        raise ValueError(f"Failed to read file: {exc}") from exc

    if not pdf_bytes:
        raise ValueError(f"File is empty: {filepath}")

    doc_id = get_doc_id_from_filename(filepath)
    
    try:
        chunks_indexed = ingest_pdf_document(doc_id, pdf_bytes)
        return chunks_indexed
    except Exception as exc:
        raise ValueError(f"Failed to ingest {filepath}: {exc}") from exc


def ingest_all_from_docs_folder(docs_folder: str = "./docs") -> tuple[int, int]:
    """
    Ingest all PDF files from the docs folder.
    
    Args:
        docs_folder: Path to the docs folder (default: ./docs)
    
    Returns:
        Tuple of (number_of_documents, total_chunks_indexed)
    """
    docs_path = Path(docs_folder)
    
    if not docs_path.exists():
        raise FileNotFoundError(f"Docs folder not found: {docs_path}")
    
    if not docs_path.is_dir():
        raise ValueError(f"Path is not a directory: {docs_path}")
    
    pdf_files = list(docs_path.glob("*.pdf"))
    
    if not pdf_files:
        print(f"No PDF files found in {docs_path}")
        return 0, 0
    
    print(f"Found {len(pdf_files)} PDF file(s) in {docs_path}")
    print("-" * 60)
    
    total_chunks = 0
    successful_docs = 0
    failed_docs = []
    
    for pdf_file in sorted(pdf_files):
        try:
            chunks = ingest_file(str(pdf_file))
            total_chunks += chunks
            successful_docs += 1
            print(f"✓ {pdf_file.name}: {chunks} chunk(s)")
        except Exception as exc:
            failed_docs.append((pdf_file.name, str(exc)))
            print(f"✗ {pdf_file.name}: Failed - {exc}")
    
    print("-" * 60)
    
    if failed_docs:
        print(f"\nFailed to ingest {len(failed_docs)} file(s):")
        for filename, error in failed_docs:
            print(f"  - {filename}: {error}")
    
    return successful_docs, total_chunks


def main():
    parser = argparse.ArgumentParser(
        description="Ingest PDF documents from ./docs/ folder into Qdrant",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python ingest_docs.py --all
  python ingest_docs.py --file ./docs/manual.pdf
        """
    )
    
    parser.add_argument(
        "--all",
        action="store_true",
        help="Ingest all PDF files from ./docs/ folder"
    )
    
    parser.add_argument(
        "--file",
        type=str,
        help="Path to a single PDF file to ingest"
    )
    
    parser.add_argument(
        "--docs-folder",
        type=str,
        default="./docs",
        help="Path to docs folder (default: ./docs)"
    )
    
    args = parser.parse_args()
    
    if not args.all and not args.file:
        parser.error("Either --all or --file must be specified")
    
    if args.all and args.file:
        parser.error("Cannot specify both --all and --file")
    
    try:
        if args.all:
            print(f"Ingesting all documents from {args.docs_folder}...")
            print(f"Qdrant collection: {QDRANT_COLLECTION}")
            print("=" * 60)
            num_docs, total_chunks = ingest_all_from_docs_folder(args.docs_folder)
            print(f"\n✓ Ingested {num_docs} document(s), {total_chunks} chunk(s) total.")
            sys.exit(0 if num_docs > 0 else 1)
        
        elif args.file:
            print(f"Ingesting file: {args.file}")
            print(f"Qdrant collection: {QDRANT_COLLECTION}")
            print("-" * 60)
            chunks = ingest_file(args.file)
            doc_id = get_doc_id_from_filename(args.file)
            print(f"✓ Successfully ingested '{doc_id}': {chunks} chunk(s)")
            sys.exit(0)
    
    except KeyboardInterrupt:
        print("\n\nInterrupted by user.")
        sys.exit(1)
    except Exception as exc:
        print(f"\n✗ Error: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()

