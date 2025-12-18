#!/usr/bin/env python3
"""
Script to re-ingest the PDF and test image retrieval accuracy.
"""
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from main import ingest_pdf_document

# Path to the PDF
pdf_path = '../backend/docs/Woodland animal facts.pdf'

if not os.path.exists(pdf_path):
    print(f"Error: PDF not found at {pdf_path}")
    sys.exit(1)

print("=" * 80)
print("RE-INGESTING PDF WITH UPDATED MATCHING LOGIC")
print("=" * 80)
print()

with open(pdf_path, 'rb') as f:
    pdf_bytes = f.read()

try:
    chunk_count = ingest_pdf_document('Woodland animal facts.pdf', pdf_bytes, replace_existing=True)
    print(f"\n✓ Successfully ingested {chunk_count} chunks")
    print("\nNow running image retrieval accuracy test...")
    print("=" * 80)
    print()
except Exception as e:
    print(f"\n✗ Ingestion failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Now run the test
import subprocess
result = subprocess.run([sys.executable, 'test_image_retrieval.py'], cwd=os.path.dirname(__file__))
sys.exit(result.returncode)

