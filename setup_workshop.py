#!/usr/bin/env python3
"""
Single-command workshop setup: loads all data, embeds with Voyage AI, and creates Atlas indexes.

Combines: 01_load_faq_data.py, 04_load_customers.py, 06_product_recommendation.py,
          07_load_kyc_data.py, 02_create_indexes.py

Usage:
    python setup_workshop.py
"""

from __future__ import annotations

import io
import json
import os
import random
import sys
import time
from datetime import date, timedelta
from pathlib import Path

from bson import ObjectId
from dotenv import load_dotenv
from pymongo import MongoClient
from pymongo.errors import PyMongoError
from pymongo.operations import SearchIndexModel
import voyageai

try:
    from fpdf import FPDF
except ImportError:
    FPDF = None

SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = SCRIPT_DIR / "data"
DB_NAME = "banking"
VOYAGE_MODEL = "voyage-4-large"
EMBED_BATCH_SIZE = 25
CHUNK_MAX_CHARS = 1000
CHUNK_OVERLAP = 150


# ──────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────

def require_env(name: str) -> str:
    value = os.environ.get(name, "").strip()
    if not value:
        print(f"ERROR: Missing required environment variable: {name}", file=sys.stderr)
        sys.exit(1)
    return value


def load_json(name: str) -> list[dict]:
    path = DATA_DIR / name
    if not path.is_file():
        raise SystemExit(f"Missing data file: {path}")
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def embed_batch(voyage: voyageai.Client, texts: list[str], input_type: str = "document") -> list[list[float]]:
    all_embeddings: list[list[float]] = []
    total = (len(texts) + EMBED_BATCH_SIZE - 1) // EMBED_BATCH_SIZE
    for batch_num, i in enumerate(range(0, len(texts), EMBED_BATCH_SIZE), start=1):
        batch = texts[i : i + EMBED_BATCH_SIZE]
        print(f"    Embedding batch {batch_num}/{total} ({len(batch)} text(s)) ...")
        result = voyage.embed(batch, model=VOYAGE_MODEL, input_type=input_type)
        all_embeddings.extend(result.embeddings)
    return all_embeddings


def index_exists(collection, name: str) -> bool:
    try:
        for idx in collection.list_search_indexes():
            if idx.get("name") == name:
                return True
    except PyMongoError:
        pass
    return False


# ──────────────────────────────────────────────────────────
# Step 1: FAQ chunks + embeddings
# ──────────────────────────────────────────────────────────

def build_combined_text(item: dict) -> str:
    parts = [
        f"[EN] {item.get('title_en', '').strip()}",
        item.get("content_en", "").strip(),
        "",
        f"[KM] {item.get('title_km', '').strip()}",
        item.get("content_km", "").strip(),
    ]
    return "\n".join(parts).strip()


def chunk_text(text: str) -> list[str]:
    text = text.strip()
    if not text:
        return []
    if len(text) <= CHUNK_MAX_CHARS:
        return [text]

    segments: list[str] = []
    start = 0
    n = len(text)
    while start < n:
        end = min(start + CHUNK_MAX_CHARS, n)
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
        start = max(0, end - CHUNK_OVERLAP)
    return segments


def step_load_faq(db, voyage: voyageai.Client) -> None:
    print("\n" + "=" * 60)
    print("STEP 1: Load FAQ data → banking.faq_chunks")
    print("=" * 60)

    policies = load_json("faq_policies.json")
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

    print(f"  {len(policies)} policies → {len(chunk_docs)} chunk(s)")
    embeddings = embed_batch(voyage, [d["chunk_text"] for d in chunk_docs])
    for doc, emb in zip(chunk_docs, embeddings):
        doc["embedding"] = emb

    db.faq_chunks.drop()
    if chunk_docs:
        db.faq_chunks.insert_many(chunk_docs)
    print(f"  ✓ Inserted {db.faq_chunks.count_documents({})} faq_chunks")


# ──────────────────────────────────────────────────────────
# Step 2: Customers, accounts, loans, transactions
# ──────────────────────────────────────────────────────────

def step_load_customers(db) -> None:
    print("\n" + "=" * 60)
    print("STEP 2: Load customers, accounts, loans, transactions")
    print("=" * 60)

    customers_raw = load_json("customers.json")
    loans_raw = load_json("loan_applications.json")
    accounts_raw = load_json("accounts.json")
    transactions_raw = load_json("transactions.json")

    for name in ("customers", "loan_applications", "accounts", "transactions"):
        db[name].drop()

    cust_id_map: dict[str, ObjectId] = {}
    for doc in customers_raw:
        oid = ObjectId()
        cust_id_map[doc["id"]] = oid
        doc["_id"] = oid
        doc["full_name"] = doc["name"]
        doc["monthly_income"] = doc.get("income_usd", 0)
    if customers_raw:
        db.customers.insert_many(customers_raw)

    acc_id_map: dict[str, ObjectId] = {}
    for doc in accounts_raw:
        oid = ObjectId()
        acc_id_map[doc["id"]] = oid
        doc["_id"] = oid
        cust_str = doc.get("customer_id", "")
        doc["customer_id"] = cust_id_map.get(cust_str, cust_str)
    if accounts_raw:
        db.accounts.insert_many(accounts_raw)

    for doc in loans_raw:
        oid = ObjectId()
        doc["_id"] = oid
        cust_str = doc.get("customer_id", "")
        doc["customer_id"] = cust_id_map.get(cust_str, cust_str)
        doc["loan_amount"] = doc.get("amount_usd", 0)
        doc["monthly_payment"] = doc.get("monthly_payment_usd", 0)
    if loans_raw:
        db.loan_applications.insert_many(loans_raw)

    for doc in transactions_raw:
        oid = ObjectId()
        doc["_id"] = oid
        cust_str = doc.get("customer_id", "")
        doc["customer_id"] = cust_id_map.get(cust_str, cust_str)
        acc_str = doc.get("account_id", "")
        doc["account_id"] = acc_id_map.get(acc_str, acc_str)
    if transactions_raw:
        db.transactions.insert_many(transactions_raw)

    print(f"  ✓ customers:         {db.customers.count_documents({})}")
    print(f"  ✓ accounts:          {db.accounts.count_documents({})}")
    print(f"  ✓ loan_applications: {db.loan_applications.count_documents({})}")
    print(f"  ✓ transactions:      {db.transactions.count_documents({})}")


# ──────────────────────────────────────────────────────────
# Step 3: Bank products + embeddings
# ──────────────────────────────────────────────────────────

BANK_PRODUCTS: list[dict[str, str]] = [
    {
        "product_id": "PROD-PERSONAL",
        "name": "Personal Loan",
        "description": (
            "Personal Loan — flexible terms up to $10,000 for salaried and stable-income customers; "
            "fixed monthly installments and optional early repayment."
        ),
    },
    {
        "product_id": "PROD-MICRO",
        "name": "Micro-Business Loan",
        "description": (
            "Micro-Business Loan — for small business owners and market vendors; working capital "
            "with simplified documentation and relationship-manager support."
        ),
    },
    {
        "product_id": "PROD-AGRI",
        "name": "Agricultural Loan",
        "description": (
            "Agricultural Loan — seasonal repayment aligned to harvest cycles; for crop and livestock "
            "producers with flexible grace periods."
        ),
    },
    {
        "product_id": "PROD-HOUSING",
        "name": "Housing Loan",
        "description": (
            "Housing Loan — long-term financing up to 20 years for home purchase or construction; "
            "competitive rates with property collateral."
        ),
    },
    {
        "product_id": "PROD-EDU",
        "name": "Education Loan",
        "description": (
            "Education Loan — tuition and living expenses for students and families; staged "
            "disbursements tied to enrollment."
        ),
    },
    {
        "product_id": "PROD-SECURED",
        "name": "Secured Savings Loan",
        "description": (
            "Secured Savings Loan — borrow against your fixed deposit or savings balance; "
            "lower rates and fast approval while keeping savings intact."
        ),
    },
    {
        "product_id": "PROD-SME-LINE",
        "name": "SME Credit Line",
        "description": (
            "SME Credit Line — revolving credit for small and medium enterprises; draw and repay "
            "as cash flow allows, ideal for inventory and payroll."
        ),
    },
    {
        "product_id": "PROD-EMERGENCY",
        "name": "Emergency Loan",
        "description": (
            "Emergency Loan — small, quick-disbursement loans for urgent needs; shorter terms and "
            "streamlined approval for existing customers."
        ),
    },
]


def step_load_products(db, voyage: voyageai.Client) -> None:
    print("\n" + "=" * 60)
    print("STEP 3: Load bank products → banking.bank_products")
    print("=" * 60)

    db.bank_products.drop()
    texts = [p["description"] for p in BANK_PRODUCTS]
    print(f"  Embedding {len(texts)} product descriptions ...")
    result = voyage.embed(texts=texts, model=VOYAGE_MODEL, input_type="document")
    vectors = result.embeddings

    docs = []
    for prod, vec in zip(BANK_PRODUCTS, vectors):
        docs.append(
            {
                "product_id": prod["product_id"],
                "name": prod["name"],
                "description": prod["description"],
                "embedding": vec,
                "embedding_model": VOYAGE_MODEL,
            }
        )
    if docs:
        db.bank_products.insert_many(docs)
    print(f"  ✓ Inserted {db.bank_products.count_documents({})} bank_products")


# ──────────────────────────────────────────────────────────
# Step 4: KYC documents + embeddings
# ──────────────────────────────────────────────────────────

def step_load_kyc(db, voyage: voyageai.Client) -> None:
    print("\n" + "=" * 60)
    print("STEP 4: Load KYC documents → banking.kyc_documents")
    print("=" * 60)

    docs = load_json("kyc_documents.json")
    descriptions = [d.get("description") or "" for d in docs]
    embeddings = embed_batch(voyage, descriptions)
    for doc, vec in zip(docs, embeddings):
        doc["description_embedding"] = vec
        doc["embedding_model"] = VOYAGE_MODEL

    db.kyc_documents.drop()
    if docs:
        db.kyc_documents.insert_many(docs)
    print(f"  ✓ Inserted {db.kyc_documents.count_documents({})} kyc_documents")


# ──────────────────────────────────────────────────────────
# Step 5: Generate PDF documents (payslips + bank statements)
# ──────────────────────────────────────────────────────────

EMPLOYERS = [
    "AnyCompany Ltd", "Mekong Trading Co", "Phnom Penh Textiles",
    "Golden Rice Exports", "Sunrise Logistics", "Angkor Tech Solutions",
    "Royal Capital Partners", "Lotus Agriculture Co", "Pacific Garments",
    "Indochina Real Estate", "Atlas Global Services", "Riverfront Holdings",
]


def _make_payslip_pdf(customer: dict, pay_date: date) -> bytes:
    pdf = FPDF()
    pdf.add_page()
    pdf.set_auto_page_break(auto=False)

    employer = random.choice(EMPLOYERS)
    name = customer.get("full_name") or customer.get("name", "Unknown")
    monthly_gross = float(customer.get("income_usd") or customer.get("monthly_income") or 0)
    tax = round(monthly_gross * random.uniform(0.05, 0.12), 2)
    social = round(monthly_gross * 0.02, 2)
    net = round(monthly_gross - tax - social, 2)
    emp_type = customer.get("employment_type", "employed")
    cust_id_str = customer.get("id", "")

    pdf.set_font("Helvetica", "B", 16)
    pdf.cell(0, 10, employer, ln=True, align="C")
    pdf.set_font("Helvetica", "", 9)
    pdf.cell(0, 5, "Phnom Penh, Cambodia | Tax ID: " + str(random.randint(100000, 999999)), ln=True, align="C")
    pdf.ln(4)
    pdf.set_draw_color(0, 100, 60)
    pdf.set_line_width(0.5)
    pdf.line(10, pdf.get_y(), 200, pdf.get_y())
    pdf.ln(4)

    pdf.set_font("Helvetica", "B", 13)
    pdf.cell(0, 8, "PAYSLIP", ln=True, align="C")
    pdf.set_font("Helvetica", "", 10)
    pdf.cell(0, 6, f"Pay Period: {pay_date.strftime('%B %Y')}", ln=True, align="C")
    pdf.ln(6)

    pdf.set_font("Helvetica", "B", 10)
    pdf.cell(95, 7, "Employee Details", border="B")
    pdf.cell(95, 7, "Payment Summary", border="B", ln=True)
    pdf.set_font("Helvetica", "", 10)

    pdf.cell(40, 6, "Name:")
    pdf.cell(55, 6, name)
    pdf.cell(50, 6, "Gross Salary (USD):")
    pdf.cell(45, 6, f"${monthly_gross:,.2f}", ln=True)

    pdf.cell(40, 6, "Employee ID:")
    pdf.cell(55, 6, cust_id_str)
    pdf.cell(50, 6, "Income Tax:")
    pdf.cell(45, 6, f"-${tax:,.2f}", ln=True)

    pdf.cell(40, 6, "Position:")
    pdf.cell(55, 6, emp_type.replace("_", " ").title())
    pdf.cell(50, 6, "Social Security:")
    pdf.cell(45, 6, f"-${social:,.2f}", ln=True)

    pdf.cell(40, 6, "Department:")
    pdf.cell(55, 6, random.choice(["Operations", "Finance", "Sales", "Engineering", "Admin"]))
    pdf.set_font("Helvetica", "B", 10)
    pdf.cell(50, 6, "Net Pay (USD):")
    pdf.cell(45, 6, f"${net:,.2f}", ln=True)
    pdf.set_font("Helvetica", "", 10)

    pdf.ln(8)
    pdf.set_font("Helvetica", "I", 8)
    pdf.cell(0, 5, f"This payslip is system-generated by {employer}. For queries contact HR.", ln=True)
    pdf.cell(0, 5, f"Generated: {pay_date.isoformat()}", ln=True)

    text_content = (
        f"PAYSLIP — {employer}\n"
        f"Employee: {name} ({cust_id_str}), Position: {emp_type}\n"
        f"Pay Period: {pay_date.strftime('%B %Y')}\n"
        f"Gross Salary: ${monthly_gross:,.2f}, Tax: ${tax:,.2f}, "
        f"Social Security: ${social:,.2f}, Net Pay: ${net:,.2f}\n"
    )
    return pdf.output(), text_content


def _make_statement_pdf(customer: dict, accounts: list, transactions: list, stmt_date: date) -> bytes:
    pdf = FPDF()
    pdf.add_page()
    pdf.set_auto_page_break(auto=True, margin=15)

    name = customer.get("full_name") or customer.get("name", "Unknown")
    cust_id_str = customer.get("id", "")

    pdf.set_font("Helvetica", "B", 16)
    pdf.cell(0, 10, "Atlas Digital Bank", ln=True, align="C")
    pdf.set_font("Helvetica", "", 9)
    pdf.cell(0, 5, "No. 123 Norodom Blvd, Phnom Penh | Swift: ATLBKHPP", ln=True, align="C")
    pdf.ln(2)
    pdf.set_draw_color(0, 100, 60)
    pdf.set_line_width(0.5)
    pdf.line(10, pdf.get_y(), 200, pdf.get_y())
    pdf.ln(4)

    pdf.set_font("Helvetica", "B", 13)
    pdf.cell(0, 8, "ACCOUNT STATEMENT", ln=True, align="C")
    period_start = stmt_date.replace(day=1)
    pdf.set_font("Helvetica", "", 10)
    pdf.cell(0, 6, f"Statement Period: {period_start.strftime('%d %b %Y')} - {stmt_date.strftime('%d %b %Y')}", ln=True, align="C")
    pdf.ln(4)

    pdf.set_font("Helvetica", "", 10)
    pdf.cell(40, 6, "Account Holder:")
    pdf.cell(0, 6, name, ln=True)
    pdf.cell(40, 6, "Customer ID:")
    pdf.cell(0, 6, cust_id_str, ln=True)
    pdf.ln(4)

    text_lines = [
        f"ACCOUNT STATEMENT — Atlas Digital Bank",
        f"Customer: {name} ({cust_id_str})",
        f"Period: {period_start.isoformat()} to {stmt_date.isoformat()}",
    ]

    total_balance = 0.0
    for acc in accounts[:3]:
        bal = float(acc.get("balance_usd", 0))
        total_balance += bal
        acc_num = acc.get("account_number", "-")
        acc_type = acc.get("account_type", "unknown").title()
        text_lines.append(f"Account: {acc_num} ({acc_type}) Balance: ${bal:,.2f}")

        pdf.set_font("Helvetica", "B", 10)
        pdf.cell(0, 7, f"{acc_type} Account - {acc_num}", border="B", ln=True)
        pdf.set_font("Helvetica", "", 9)
        pdf.cell(50, 6, f"Balance: ${bal:,.2f}")
        pdf.cell(50, 6, f"Currency: {acc.get('currency', 'USD')}")
        pdf.cell(50, 6, f"Status: {acc.get('status', 'active').title()}", ln=True)
        pdf.ln(2)

        acc_id = acc.get("_id") or acc.get("id")
        acc_txns = [t for t in transactions if str(t.get("account_id")) == str(acc_id)][:8]
        if acc_txns:
            pdf.set_font("Helvetica", "B", 8)
            pdf.cell(25, 5, "Date")
            pdf.cell(65, 5, "Description")
            pdf.cell(25, 5, "Type")
            pdf.cell(30, 5, "Amount", ln=True)
            pdf.set_font("Helvetica", "", 8)
            for tx in acc_txns:
                d = str(tx.get("date", ""))[:10]
                desc = str(tx.get("description", ""))[:35]
                ttype = tx.get("type", "")
                amt = float(tx.get("amount", 0))
                sign = "+" if ttype == "credit" else "-"
                pdf.cell(25, 4.5, d)
                pdf.cell(65, 4.5, desc)
                pdf.cell(25, 4.5, ttype)
                pdf.cell(30, 4.5, f"{sign}${amt:,.2f}", ln=True)
                text_lines.append(f"  {d} {desc} {sign}${amt:,.2f}")
        pdf.ln(4)

    text_lines.append(f"Total Balance: ${total_balance:,.2f}")

    pdf.set_font("Helvetica", "B", 10)
    pdf.cell(120, 7, "")
    pdf.cell(70, 7, f"Total Balance: ${total_balance:,.2f}", ln=True, align="R")

    pdf.ln(4)
    pdf.set_font("Helvetica", "I", 8)
    pdf.cell(0, 5, "This statement is system-generated. Please contact your branch for discrepancies.", ln=True)

    return pdf.output(), "\n".join(text_lines)


def step_generate_documents(db) -> None:
    print("\n" + "=" * 60)
    print("STEP 5: Generate PDF documents → banking.documents")
    print("=" * 60)

    if FPDF is None:
        print("  ⚠ fpdf2 not installed — skipping PDF generation.")
        print("    Install with: pip install fpdf2")
        return

    db.documents.drop()
    customers = list(db.customers.find())
    all_accounts = list(db.accounts.find())
    all_transactions = list(db.transactions.find())

    today = date.today()
    docs_to_insert = []
    rng = random.Random(42)

    for cust in customers:
        cust_oid = cust["_id"]
        cust_id_str = cust.get("id", str(cust_oid))
        cust_accounts = [a for a in all_accounts if a.get("customer_id") == cust_oid]
        cust_txns = [t for t in all_transactions if t.get("customer_id") == cust_oid]

        # Payslip (1-2 months back)
        pay_date = today - timedelta(days=rng.randint(5, 35))
        try:
            pdf_bytes, text = _make_payslip_pdf(cust, pay_date)
            docs_to_insert.append({
                "customer_id": cust_oid,
                "customer_id_str": cust_id_str,
                "doc_type": "payslip",
                "filename": f"payslip_{cust_id_str}_{pay_date.strftime('%Y%m')}.pdf",
                "pdf_data": pdf_bytes,
                "text_content": text,
                "generated_date": pay_date.isoformat(),
            })
        except Exception as e:
            print(f"  ⚠ Payslip failed for {cust_id_str}: {e}")

        # Bank statement
        stmt_date = today - timedelta(days=rng.randint(1, 15))
        try:
            pdf_bytes, text = _make_statement_pdf(cust, cust_accounts, cust_txns, stmt_date)
            docs_to_insert.append({
                "customer_id": cust_oid,
                "customer_id_str": cust_id_str,
                "doc_type": "bank_statement",
                "filename": f"statement_{cust_id_str}_{stmt_date.strftime('%Y%m')}.pdf",
                "pdf_data": pdf_bytes,
                "text_content": text,
                "generated_date": stmt_date.isoformat(),
            })
        except Exception as e:
            print(f"  ⚠ Statement failed for {cust_id_str}: {e}")

    if docs_to_insert:
        db.documents.insert_many(docs_to_insert)
    print(f"  ✓ Generated and stored {len(docs_to_insert)} PDF documents ({len(customers)} customers × 2)")


# ──────────────────────────────────────────────────────────
# Step 6: Create Atlas Search / Vector Search indexes
# ──────────────────────────────────────────────────────────

def step_create_indexes(db) -> None:
    print("\n" + "=" * 60)
    print("STEP 6: Create Atlas Vector Search indexes")
    print("=" * 60)

    indexes = [
        (
            db["faq_chunks"],
            "faq_vector_index",
            {
                "fields": [
                    {"type": "vector", "path": "embedding", "numDimensions": 1024, "similarity": "cosine"},
                    {"type": "filter", "path": "category"},
                    {"type": "filter", "path": "language"},
                ]
            },
        ),
        (
            db["kyc_documents"],
            "kyc_vector_index",
            {
                "fields": [
                    {"type": "vector", "path": "description_embedding", "numDimensions": 1024, "similarity": "cosine"},
                ]
            },
        ),
        (
            db["bank_products"],
            "product_vector_index",
            {
                "fields": [
                    {"type": "vector", "path": "embedding", "numDimensions": 1024, "similarity": "cosine"},
                ]
            },
        ),
    ]

    for col, name, definition in indexes:
        if index_exists(col, name):
            print(f"\n  [{col.name}] {name} — already exists, skipping")
        else:
            print(f"\n  [{col.name}] {name} — creating ...")
            col.create_search_index(model=SearchIndexModel(
                definition=definition,
                name=name,
                type="vectorSearch",
            ))
            print(f"    ✓ Requested {name}")


# ──────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────

def main() -> None:
    load_dotenv(SCRIPT_DIR / ".env")

    mongodb_uri = require_env("MONGODB_URI")
    voyage_key = require_env("VOYAGE_API_KEY")

    print("Connecting to MongoDB Atlas ...")
    try:
        mongo = MongoClient(mongodb_uri, serverSelectionTimeoutMS=15000)
        mongo.admin.command("ping")
    except PyMongoError as e:
        print(f"ERROR: Could not connect to MongoDB: {e}", file=sys.stderr)
        sys.exit(1)
    print("  Connected.\n")

    db = mongo[DB_NAME]
    voyage = voyageai.Client(api_key=voyage_key)

    t0 = time.time()

    step_load_faq(db, voyage)
    step_load_customers(db)
    step_load_products(db, voyage)
    step_load_kyc(db, voyage)
    step_generate_documents(db)
    step_create_indexes(db)

    elapsed = time.time() - t0

    print("\n" + "=" * 60)
    print("ALL DONE")
    print("=" * 60)
    print(f"  Database: {DB_NAME}")
    print(f"  Collections loaded: faq_chunks, customers, accounts,")
    print(f"    loan_applications, transactions, bank_products,")
    print(f"    kyc_documents, documents")
    print(f"  PDF documents: ~100 (payslips + bank statements)")
    print(f"  Indexes requested: faq_vector_index, kyc_vector_index,")
    print(f"    product_vector_index  (3 total — M0 free tier limit)")
    print(f"  Total time: {elapsed:.1f}s")
    print()
    print("  ⚠  Atlas builds search indexes asynchronously.")
    print("     Wait until all indexes show ACTIVE in the Atlas UI")
    print("     before testing vector search queries in the app.")

    mongo.close()


if __name__ == "__main__":
    main()
