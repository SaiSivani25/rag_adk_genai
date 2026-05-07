# rag_agent/tools/search_docs.py

import os
from dotenv import load_dotenv
from google import genai
from google.genai import types
from qdrant_client import QdrantClient
from google.adk.tools import ToolContext

from pathlib import Path
load_dotenv(Path(__file__).resolve().parents[2] / ".env")

PROJECT_ID     = os.environ["GCP_PROJECT_ID"]
LOCATION       = os.environ["GCP_LOCATION"]
QDRANT_URL     = os.environ["QDRANT_URL"]
QDRANT_API_KEY = os.environ["QDRANT_API_KEY"]
COLLECTION     = os.environ.get("QDRANT_COLLECTION", "rag_adk_collection")
TOP_K          = 5

_client = genai.Client(vertexai=True, project=PROJECT_ID, location=LOCATION)
_qdrant = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)


def _embed(text: str) -> list[float]:
    response = _client.models.embed_content(
        model="gemini-embedding-001",
        contents=text,
        config=types.EmbedContentConfig(output_dimensionality=768),
    )
    return response.embeddings[0].values


def search_docs(question: str, tool_context: ToolContext) -> dict:
    """
    Searches all indexed documents using a natural language question.

    Embeds the question using Gemini Embedding, searches Qdrant for the
    most relevant document chunks, and generates a grounded answer using
    Gemini 2.5 Flash. Returns the answer with source page references.

    Always call this for any question about document content.

    Args:
        question:     Natural language question about the documents.
        tool_context: Injected by ADK automatically.

    Returns:
        A dict with answer, source_pages, and a message.
        On failure, returns a dict with an error key.
    """
    try:
        # Step 1 — embed the question
        query_vector = _embed(question)

        # Step 2 — search Qdrant
        results = _qdrant.search(
            collection_name=COLLECTION,
            query_vector=query_vector,
            limit=TOP_K,
            with_payload=True,
        )

        if not results:
            return {
                "answer": (
                    "No relevant content found. "
                    "Please run the ingestion pipeline to index your documents first."
                ),
                "source_pages": [],
            }

        # Step 3 — build context
        context_parts = []
        source_pages  = []

        for r in results:
            p = r.payload
            context_parts.append(
                f"[Page {p['page_number']} — {p['file_name']}]\n{p['text']}"
            )
            source_pages.append({
                "page_number":     p["page_number"],
                "file_name":       p["file_name"],
                "file_url":        p["file_url"],
                "relevance_score": round(r.score, 4),
            })

        context = "\n\n---\n\n".join(context_parts)

        # Step 4 — generate grounded answer
        prompt = (
            "You are a document assistant. "
            "Answer the question using ONLY the context below. "
            "If the answer is not in the context, say so clearly. "
            "Never invent information.\n\n"
            f"CONTEXT:\n{context}\n\n"
            f"QUESTION: {question}\n\n"
            "ANSWER:"
        )

        response = _client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt,
        )

        return {
            "answer":       response.text.strip(),
            "source_pages": source_pages,
            "message":      f"Answer from {len(source_pages)} source page(s).",
        }

    except Exception as e:
        return {"error": f"Search failed: {str(e)}"}