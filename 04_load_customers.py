#!/usr/bin/env python3
"""
Workshop 04: Load customers and loan applications into MongoDB Atlas (banking DB).
"""

import json
import os
from pathlib import Path

from dotenv import load_dotenv
from pymongo import MongoClient

SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = SCRIPT_DIR / "data"


def main() -> None:
    load_dotenv(SCRIPT_DIR / ".env")
    uri = os.environ.get("MONGODB_URI")
    if not uri:
        raise SystemExit("MONGODB_URI is not set. Copy .env.example to .env and configure.")

    customers_path = DATA_DIR / "customers.json"
    loans_path = DATA_DIR / "loan_applications.json"
    if not customers_path.is_file() or not loans_path.is_file():
        raise SystemExit(f"Expected data files under {DATA_DIR}")

    with customers_path.open(encoding="utf-8") as f:
        customers = json.load(f)
    with loans_path.open(encoding="utf-8") as f:
        loan_applications = json.load(f)

    client = MongoClient(uri)
    db = client["banking"]

    print("Dropping and recreating collections: banking.customers, banking.loan_applications")
    db.customers.drop()
    db.loan_applications.drop()

    if customers:
        db.customers.insert_many(customers)
    if loan_applications:
        db.loan_applications.insert_many(loan_applications)

    cust_count = db.customers.count_documents({})
    loan_count = db.loan_applications.count_documents({})

    print()
    print("=== Load summary ===")
    print(f"  Customers inserted:        {cust_count}")
    print(f"  Loan applications inserted: {loan_count}")
    print(f"  Data directory:            {DATA_DIR}")
    print()
    print("Done.")

    client.close()


if __name__ == "__main__":
    main()
