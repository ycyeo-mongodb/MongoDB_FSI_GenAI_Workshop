#!/usr/bin/env python3
"""
Create Atlas Vector Search and Atlas Search indexes for banking workshop collections (PyMongo SearchIndexModel).
"""

from __future__ import annotations

import os
import sys

from dotenv import load_dotenv
from pathlib import Path

from pymongo import MongoClient
from pymongo.errors import OperationFailure, PyMongoError
from pymongo.operations import SearchIndexModel

REPO_ROOT = Path(__file__).resolve().parent
DB_NAME = "banking"
FAQ_COLLECTION = "faq_chunks"
KYC_COLLECTION = "kyc_documents"


def require_env(name: str) -> str:
    value = os.environ.get(name, "").strip()
    if not value:
        print(f"ERROR: Missing required environment variable: {name}", file=sys.stderr)
        sys.exit(1)
    return value


def safe_drop_search_index(collection, name: str) -> None:
    """Drop a search index if it exists (ignore if missing)."""
    try:
        collection.drop_search_index(name)
        print(f"  Dropped existing index: {name}")
    except OperationFailure as e:
        msg = str(e).lower()
        if "index not found" in msg or "not found" in msg or e.code in (27, 404):
            return
        raise
    except PyMongoError as e:
        # Ignore missing index; re-raise unexpected errors
        err = str(e).lower()
        if "index not found" in err or "not found" in err:
            return
        raise


def main() -> None:
    load_dotenv(REPO_ROOT / ".env")
    mongodb_uri = require_env("MONGODB_URI")

    print("Connecting to MongoDB Atlas ...")
    try:
        mongo = MongoClient(mongodb_uri, serverSelectionTimeoutMS=15000)
        mongo.admin.command("ping")
    except PyMongoError as e:
        print(f"ERROR: Could not connect to MongoDB: {e}", file=sys.stderr)
        sys.exit(1)

    db = mongo[DB_NAME]
    faq_col = db[FAQ_COLLECTION]
    kyc_col = db[KYC_COLLECTION]

    # --- FAQ vector index ---
    print(f"\n[{FAQ_COLLECTION}] Vector search index 'faq_vector_index' ...")
    safe_drop_search_index(faq_col, "faq_vector_index")
    faq_vector = SearchIndexModel(
        definition={
            "fields": [
                {
                    "type": "vector",
                    "path": "embedding",
                    "numDimensions": 1024,
                    "similarity": "cosine",
                },
                {"type": "filter", "path": "category"},
                {"type": "filter", "path": "language"},
            ]
        },
        name="faq_vector_index",
        type="vectorSearch",
    )
    try:
        faq_col.create_search_index(model=faq_vector)
        print("  Requested creation of 'faq_vector_index' (vectorSearch, 1024d, cosine).")
        print("  Filter paths: category, language")
    except PyMongoError as e:
        print(f"ERROR: Failed to create faq_vector_index: {e}", file=sys.stderr)
        sys.exit(1)

    # --- FAQ text (Atlas Search) index ---
    print(f"\n[{FAQ_COLLECTION}] Atlas Search index 'faq_text_index' ...")
    safe_drop_search_index(faq_col, "faq_text_index")
    faq_text = SearchIndexModel(
        definition={
            "mappings": {
                "dynamic": False,
                "fields": {
                    "chunk_text": {"type": "string"},
                    "title": {"type": "string"},
                    "category": {"type": "stringFacet"},
                },
            }
        },
        name="faq_text_index",
    )
    try:
        faq_col.create_search_index(model=faq_text)
        print("  Requested creation of 'faq_text_index' (chunk_text, title string; category stringFacet).")
    except PyMongoError as e:
        print(f"ERROR: Failed to create faq_text_index: {e}", file=sys.stderr)
        sys.exit(1)

    # --- KYC vector index ---
    print(f"\n[{KYC_COLLECTION}] Vector search index 'kyc_vector_index' ...")
    safe_drop_search_index(kyc_col, "kyc_vector_index")
    kyc_vector = SearchIndexModel(
        definition={
            "fields": [
                {
                    "type": "vector",
                    "path": "description_embedding",
                    "numDimensions": 1024,
                    "similarity": "cosine",
                },
            ]
        },
        name="kyc_vector_index",
        type="vectorSearch",
    )
    try:
        kyc_col.create_search_index(model=kyc_vector)
        print("  Requested creation of 'kyc_vector_index' on description_embedding (1024d, cosine).")
    except PyMongoError as e:
        print(f"ERROR: Failed to create kyc_vector_index: {e}", file=sys.stderr)
        sys.exit(1)

    print("\n--- Note ---")
    print("Atlas builds search indexes asynchronously. Use list_search_indexes() in Atlas UI or")
    print("the driver to confirm each index status is READY before running vector queries.")
    print("\nCurrent search indexes (best-effort list):")
    try:
        for col, label in [(faq_col, FAQ_COLLECTION), (kyc_col, KYC_COLLECTION)]:
            print(f"  {label}:")
            for idx in col.list_search_indexes():
                print(f"    - {idx.get('name', '?')}: {idx.get('type', '?')} status={idx.get('status', '?')}")
    except PyMongoError as e:
        print(f"  (Could not list indexes: {e})")

    mongo.close()
    print("\nDone.")


if __name__ == "__main__":
    main()
