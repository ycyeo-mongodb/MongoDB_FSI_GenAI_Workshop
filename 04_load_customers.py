#!/usr/bin/env python3
"""
Workshop 04: Load customers, accounts, loans, and transactions into MongoDB Atlas.

Builds ObjectId cross-references so that $lookup pipelines work correctly
between collections.  Also adds alias fields (e.g. full_name, monthly_income)
that the FastAPI backend expects.
"""

import json
import os
from pathlib import Path

from bson import ObjectId
from dotenv import load_dotenv
from pymongo import MongoClient

SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = SCRIPT_DIR / "data"


def _load_json(name: str) -> list[dict]:
    path = DATA_DIR / name
    if not path.is_file():
        raise SystemExit(f"Missing data file: {path}")
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def main() -> None:
    load_dotenv(SCRIPT_DIR / ".env")
    uri = os.environ.get("MONGODB_URI")
    if not uri:
        raise SystemExit("MONGODB_URI is not set. Copy .env.example to .env and configure.")

    customers_raw = _load_json("customers.json")
    loans_raw = _load_json("loan_applications.json")
    accounts_raw = _load_json("accounts.json")
    transactions_raw = _load_json("transactions.json")

    client = MongoClient(uri)
    db = client["banking"]

    # ── Drop existing collections ───────────────────────────────
    for name in ("customers", "loan_applications", "accounts", "transactions"):
        db[name].drop()
    print("Dropped collections: customers, loan_applications, accounts, transactions")

    # ── 1. Customers — assign ObjectIds, add alias fields ───────
    cust_id_map: dict[str, ObjectId] = {}  # "CUST-00001" -> ObjectId

    for doc in customers_raw:
        oid = ObjectId()
        cust_id_map[doc["id"]] = oid
        doc["_id"] = oid
        doc["full_name"] = doc["name"]
        doc["monthly_income"] = doc.get("income_usd", 0)

    if customers_raw:
        db.customers.insert_many(customers_raw)

    # ── 2. Accounts — replace string customer_id with ObjectId ──
    acc_id_map: dict[str, ObjectId] = {}  # "ACC-00001" -> ObjectId

    for doc in accounts_raw:
        oid = ObjectId()
        acc_id_map[doc["id"]] = oid
        doc["_id"] = oid
        cust_str = doc.get("customer_id", "")
        doc["customer_id"] = cust_id_map.get(cust_str, cust_str)

    if accounts_raw:
        db.accounts.insert_many(accounts_raw)

    # ── 3. Loan applications — ObjectId refs + alias fields ─────
    for doc in loans_raw:
        oid = ObjectId()
        doc["_id"] = oid
        cust_str = doc.get("customer_id", "")
        doc["customer_id"] = cust_id_map.get(cust_str, cust_str)
        doc["loan_amount"] = doc.get("amount_usd", 0)
        doc["monthly_payment"] = doc.get("monthly_payment_usd", 0)

    if loans_raw:
        db.loan_applications.insert_many(loans_raw)

    # ── 4. Transactions — ObjectId refs for customer + account ──
    for doc in transactions_raw:
        oid = ObjectId()
        doc["_id"] = oid
        cust_str = doc.get("customer_id", "")
        doc["customer_id"] = cust_id_map.get(cust_str, cust_str)
        acc_str = doc.get("account_id", "")
        doc["account_id"] = acc_id_map.get(acc_str, acc_str)

    if transactions_raw:
        db.transactions.insert_many(transactions_raw)

    # ── Summary ─────────────────────────────────────────────────
    print()
    print("=== Load summary ===")
    print(f"  Customers:          {db.customers.count_documents({})}")
    print(f"  Accounts:           {db.accounts.count_documents({})}")
    print(f"  Loan applications:  {db.loan_applications.count_documents({})}")
    print(f"  Transactions:       {db.transactions.count_documents({})}")
    print()
    print("All string IDs (CUST-*, ACC-*, LOAN-*) have been mapped to ObjectId cross-references.")
    print("Done.")

    client.close()


if __name__ == "__main__":
    main()
