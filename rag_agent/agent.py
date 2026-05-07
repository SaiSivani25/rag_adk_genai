# rag_agent/agent.py

import os
from dotenv import load_dotenv
from google.adk.agents import Agent
from .tools.search_docs import search_docs

load_dotenv()

INSTRUCTION = """
You are a document intelligence assistant.

You have one tool: search_docs(question)

It searches all indexed documents and returns a grounded answer
with source page references.

RULES:
- For ANY question, ALWAYS call search_docs immediately.
- Present the answer clearly, then list the source pages and file names.
- Never make up information — only use what the tool returns.
- If the tool returns no results, tell the user no documents are indexed yet
  and ask them to run the ingestion pipeline first.
- For follow-up questions, call search_docs again with the new question.
"""

root_agent = Agent(
    name        = "rag_document_agent",
    model       = "gemini-2.5-flash",
    instruction = INSTRUCTION,
    description = "Answers questions about indexed documents using RAG on GCP.",
    tools       = [search_docs],
)