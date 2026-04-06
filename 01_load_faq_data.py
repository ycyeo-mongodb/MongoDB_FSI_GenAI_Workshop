#!/usr/bin/env python3
"""
Load FAQ policies from JSON, chunk bilingual text, embed with Voyage AI, and insert into MongoDB Atlas.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from pymongo import MongoClient
from pymongo.errors import PyMongoError
import voyageai

# Paths
REPO_ROOT = Path(__file__).resolve().parent
DATA_FILE = REPO_ROOT / "data" / "faq_policies.json"

DB_NAME = "banking"
COLLECTION_NAME = "faq_chunks"
VOYAGE_MODEL = "voyage-4-large"
EMBED_BATCH_SIZE = 25
CHUNK_MAX_CHARS = 1000
CHUNK_OVERLAP = 150


def require_env(name: str) -> str:
    value = os.environ.get(name, "").strip()
    if not value:
        print(f"ERROR: Missing required environment variable: {name}", file=sys.stderr)
        sys.exit(1)
    return value


def build_combined_text(item: dict) -> str:
    """Combine English and Khmer title + content for embedding."""
    parts = [
        f"[EN] {item.get('title_en', '').strip()}",
        item.get("content_en", "").strip(),
        "",
        f"[KM] {item.get('title_km', '').strip()}",
        item.get("content_km", "").strip(),
    ]
    return "\n".join(parts).strip()


def chunk_text(text: str, max_chars: int = CHUNK_MAX_CHARS, overlap: int = CHUNK_OVERLAP) -> list[str]:
    """Split long text into overlapping segments."""
    text = text.strip()
    if not text:
        return []
    if len(text) <= max_chars:
        return [text]

    segments: list[str] = []
    start = 0
    n = len(text)
    while start < n:
        end = min(start + max_chars, n)
        if end < n:
            break_at = text.rfind("\n", start, end)
            if break_at <= start:
                break_at = text.rfind(" ", start, end)
            if break_at > start:
                end = break_at
        segment = text[start:end].strip()
        if segment:
            segments.append(segment)
        if end >= n:
            break
        start = max(0, end - overlap)
    return segments


def main() -> None:
    load_dotenv(REPO_ROOT / ".env")

    mongodb_uri = require_env("MONGODB_URI")
    voyage_key = require_env("VOYAGE_API_KEY")

    if not DATA_FILE.is_file():
        print(f"ERROR: Data file not found: {DATA_FILE}", file=sys.stderr)
        sys.exit(1)

    print(f"Reading FAQ policies from {DATA_FILE} ...")
    with open(DATA_FILE, encoding="utf-8") as f:
        policies = json.load(f)
    if not isinstance(policies, list):
        print("ERROR: faq_policies.json must be a JSON array.", file=sys.stderr)
        sys.exit(1)

    # Build chunk records (no embeddings yet)
    chunk_docs: list[dict] = []
    for item in policies:
        source_id = item.get("id", "")
        combined = build_combined_text(item)
        segments = chunk_text(combined)
        title_en = (item.get("title_en") or "").strip()
        for chunk_text_val in segments:
            chunk_docs.append(
                {
                    "source_id": source_id,
                    "title": title_en,
                    "content_en": item.get("content_en", ""),
                    "content_km": item.get("content_km", ""),
                    "category": item.get("category", ""),
                    "department": item.get("department", ""),
                    "language": "bilingual",
                    "chunk_text": chunk_text_val,
                }
            )

    print(f"Prepared {len(chunk_docs)} chunk(s) from {len(policies)} FAQ policy/policies.")

    voyage_client = voyageai.Client(api_key=voyage_key)

    # Embed in batches
    all_embeddings: list[list[float]] = []
    total_batches = (len(chunk_docs) + EMBED_BATCH_SIZE - 1) // EMBED_BATCH_SIZE
    for batch_num, i in enumerate(range(0, len(chunk_docs), EMBED_BATCH_SIZE), start=1):
        batch = chunk_docs[i : i + EMBED_BATCH_SIZE]
        texts = [d["chunk_text"] for d in batch]
        print(f"Embedding batch {batch_num}/{total_batches} ({len(texts)} chunk(s)) ...")
        try:
            result = voyage_client.embed(
                texts,
                model=VOYAGE_MODEL,
                input_type="document",
            )
        except Exception as e:
            print(f"ERROR: Voyage AI embed failed: {e}", file=sys.stderr)
            sys.exit(1)
        all_embeddings.extend(result.embeddings)

    if len(all_embeddings) != len(chunk_docs):
        print(
            f"ERROR: Embedding count mismatch ({len(all_embeddings)} vs {len(chunk_docs)}).",
            file=sys.stderr,
        )
        sys.exit(1)

    for doc, emb in zip(chunk_docs, all_embeddings):
        doc["embedding"] = emb

    print("Connecting to MongoDB Atlas ...")
    try:
        mongo = MongoClient(mongodb_uri, serverSelectionTimeoutMS=15000)
        mongo.admin.command("ping")
    except PyMongoError as e:
        print(f"ERROR: Could not connect to MongoDB: {e}", file=sys.stderr)
        sys.exit(1)

    db = mongo[DB_NAME]
    col = db[COLLECTION_NAME]

    print(f"Dropping existing collection '{DB_NAME}.{COLLECTION_NAME}' if it exists ...")
    col.drop()

    print(f"Inserting {len(chunk_docs)} document(s) ...")
    try:
        if chunk_docs:
            col.insert_many(chunk_docs)
    except PyMongoError as e:
        print(f"ERROR: Insert failed: {e}", file=sys.stderr)
        sys.exit(1)

    print()
    print("--- Summary ---")
    print(f"  Database:   {DB_NAME}")
    print(f"  Collection: {COLLECTION_NAME}")
    print(f"  Policies:   {len(policies)}")
    print(f"  Chunks:     {len(chunk_docs)}")
    print(f"  Model:      {VOYAGE_MODEL}")
    print(f"  Dimensions: {len(chunk_docs[0]['embedding']) if chunk_docs else 0}")
    print("Done.")

    mongo.close()


if __name__ == "__main__":
    main()
