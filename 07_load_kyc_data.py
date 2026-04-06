#!/usr/bin/env python3
"""
Workshop 07: Load KYC documents with Voyage voyage-4-large embeddings (document mode).

Requires Atlas Vector Search index on banking.kyc_documents path description_embedding
(e.g. index name kyc_vector_index).
"""

import json
import os
from pathlib import Path

from dotenv import load_dotenv
from pymongo import MongoClient
from voyageai import Client as VoyageClient

SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = SCRIPT_DIR / "data"

VOYAGE_MODEL = "voyage-4-large"


def main() -> None:
    load_dotenv(SCRIPT_DIR / ".env")
    uri = os.environ.get("MONGODB_URI")
    v_key = os.environ.get("VOYAGE_API_KEY")
    if not uri or not v_key:
        raise SystemExit("Set MONGODB_URI and VOYAGE_API_KEY in .env")

    path = DATA_DIR / "kyc_documents.json"
    if not path.is_file():
        raise SystemExit(f"Missing {path}")

    with path.open(encoding="utf-8") as f:
        docs = json.load(f)

    voyage = VoyageClient(api_key=v_key)
    descriptions = [d.get("description") or "" for d in docs]

    print(f"Embedding {len(descriptions)} KYC descriptions with {VOYAGE_MODEL} (input_type=document) …")
    result = voyage.embed(texts=descriptions, model=VOYAGE_MODEL, input_type="document")
    vectors = result.embeddings

    for doc, vec in zip(docs, vectors, strict=True):
        doc["description_embedding"] = vec
        doc["embedding_model"] = VOYAGE_MODEL

    client = MongoClient(uri)
    db = client["banking"]

    print("Dropping and inserting banking.kyc_documents …")
    db.kyc_documents.drop()
    if docs:
        db.kyc_documents.insert_many(docs)

    n = db.kyc_documents.count_documents({})
    print()
    print("=== Load summary ===")
    print(f"  Documents inserted: {n}")
    print(f"  Embedding field:    description_embedding")
    print(f"  Model:              {VOYAGE_MODEL}")
    print()
    print("Ensure Atlas Vector Search index (e.g. kyc_vector_index) targets path description_embedding.")
    print("Done.")

    client.close()


if __name__ == "__main__":
    main()
