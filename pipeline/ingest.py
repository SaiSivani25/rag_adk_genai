#!/usr/bin/env python3
# pipeline/ingest.py
#
# Run this once to index your PDF documents into Qdrant.
#
# USAGE:
#   python pipeline/ingest.py --folder /path/to/your/pdfs

import os
import uuid
import argparse
from pathlib import Path
from dotenv import load_dotenv

from google import genai
from google.genai import types
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct, VectorParams, Distance
import pypdf

from pathlib import Path
load_dotenv(Path(__file__).resolve().parents[1] / ".env")

PROJECT_ID     = os.environ["GCP_PROJECT_ID"]
LOCATION       = os.environ["GCP_LOCATION"]
QDRANT_URL     = os.environ["QDRANT_URL"]
QDRANT_API_KEY = os.environ["QDRANT_API_KEY"]
COLLECTION     = os.environ.get("QDRANT_COLLECTION", "rag_adk_collection")
VECTOR_DIM     = 768
BATCH_SIZE     = 5

client = genai.Client(vertexai=True, project=PROJECT_ID, location=LOCATION)
qdrant = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)


def ensure_collection():
    existing = [c.name for c in qdrant.get_collections().collections]
    if COLLECTION not in existing:
        qdrant.create_collection(
            collection_name=COLLECTION,
            vectors_config=VectorParams(size=VECTOR_DIM, distance=Distance.COSINE),
        )
        print(f"  Created collection: {COLLECTION}")
    else:
        print(f"  Collection exists: {COLLECTION}")


def extract_pages(pdf_path: Path) -> list[dict]:
    pages = []
    with open(pdf_path, "rb") as f:
        reader = pypdf.PdfReader(f)
        for i, page in enumerate(reader.pages):
            text = (page.extract_text() or "").strip()
            if text:
                pages.append({
                    "page_number": i + 1,
                    "text":        text,
                    "file_name":   pdf_path.name,
                    "file_url":    str(pdf_path.resolve()),
                })
    return pages


def embed(text: str) -> list[float]:
    response = client.models.embed_content(
        model="gemini-embedding-001",
        contents=text,
        config=types.EmbedContentConfig(output_dimensionality=VECTOR_DIM),
    )
    return response.embeddings[0].values


def ingest_pdf(pdf_path: Path):
    print(f"\nIngesting: {pdf_path.name}")

    pages = extract_pages(pdf_path)
    if not pages:
        print(f"  No text extracted — skipping.")
        return
    print(f"  Extracted {len(pages)} pages")

    all_embeddings = []
    for i in range(0, len(pages), BATCH_SIZE):
        batch = pages[i:i + BATCH_SIZE]
        for page in batch:
            all_embeddings.append(embed(page["text"]))
        print(f"  Embedded pages {i+1}–{min(i+BATCH_SIZE, len(pages))}")

    points = [
        PointStruct(
            id      = str(uuid.uuid4()),
            vector  = emb,
            payload = {
                "page_number": p["page_number"],
                "text":        p["text"],
                "file_name":   p["file_name"],
                "file_url":    p["file_url"],
            },
        )
        for p, emb in zip(pages, all_embeddings)
    ]
    qdrant.upsert(collection_name=COLLECTION, points=points)
    print(f"  Stored {len(points)} pages in Qdrant ✓")


def main():
    parser = argparse.ArgumentParser(
        description="Ingest PDF documents into Qdrant for RAG."
    )
    parser.add_argument(
        "--folder",
        required=True,
        help="Path to folder containing PDF files.",
    )
    args = parser.parse_args()

    folder = Path(args.folder)
    if not folder.exists():
        print(f"Folder not found: {folder}")
        return

    pdfs = list(folder.glob("*.pdf"))
    if not pdfs:
        print(f"No PDF files found in: {folder}")
        return

    print(f"Found {len(pdfs)} PDF(s) in {folder}")
    print("Setting up Qdrant collection...")
    ensure_collection()

    for pdf_path in pdfs:
        ingest_pdf(pdf_path)

    print(f"\nDone — {len(pdfs)} document(s) indexed into '{COLLECTION}'")


if __name__ == "__main__":
    main()