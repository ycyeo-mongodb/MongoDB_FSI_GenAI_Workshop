"""
Full FastAPI application for the banking GenAI workshop.
"""

from __future__ import annotations

import json
import os
import time
from contextlib import asynccontextmanager
from datetime import date, datetime
from pathlib import Path
from typing import Any, AsyncIterator, List, Optional

import base64

from bson import ObjectId
from bson.errors import InvalidId
from dotenv import load_dotenv
from fastapi import FastAPI, File, HTTPException, Query, Request, UploadFile
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles
import requests
from pymongo import MongoClient
from pymongo.collection import Collection
from pymongo.cursor import Cursor
import voyageai

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"

MONGODB_URI = os.getenv("MONGODB_URI", "")
VOYAGE_API_KEY = os.getenv("VOYAGE_API_KEY", "")
LLM_API_URL = os.getenv("LLM_API_URL", "")

VOYAGE_EMBED_MODEL = "voyage-4-large"

FAQ_VECTOR_INDEX = "faq_vector_index"
KYC_VECTOR_INDEX = "kyc_vector_index"
KYC_EMBEDDING_FIELD = "description_embedding"
PRODUCT_VECTOR_INDEX = "product_vector_index"
EMBEDDING_FIELD = "embedding"


def _serialize(obj: Any) -> Any:
    """Convert BSON-friendly structures for JSON: stringify _id, round floats."""
    if isinstance(obj, dict):
        out: dict[str, Any] = {}
        for k, v in obj.items():
            if k == "_id" and isinstance(v, ObjectId):
                out["_id"] = str(v)
            else:
                out[k] = _serialize(v)
        return out
    if isinstance(obj, list):
        return [_serialize(x) for x in obj]
    if isinstance(obj, ObjectId):
        return str(obj)
    if isinstance(obj, float):
        return round(obj, 4)
    return obj


def _serialize_cursor(cursor: Cursor[Any]) -> List[dict[str, Any]]:
    return [_serialize(doc) for doc in cursor]


def parse_object_id(value: str, name: str = "id") -> ObjectId:
    try:
        return ObjectId(value)
    except (InvalidId, TypeError) as exc:
        raise HTTPException(status_code=400, detail=f"Invalid {name}") from exc


def call_llm(messages: List[dict[str, str]], llm_url: str, temperature: float = 0.2) -> str:
    """Send chat messages to the LLM API Gateway and return the assistant response."""
    payload = {"messages": messages}
    try:
        resp = requests.post(llm_url, json=payload, timeout=55)
        resp.raise_for_status()
    except requests.RequestException as exc:
        raise HTTPException(status_code=502, detail=f"LLM API request failed: {exc}") from exc
    data = resp.json()
    if "error" in data:
        raise HTTPException(status_code=502, detail=f"LLM API error: {data['error']}")
    return (data.get("choices", [{}])[0].get("message", {}).get("content", "")).strip()


def call_llm_debug(messages: List[dict[str, str]], llm_url: str) -> dict[str, Any]:
    """Like call_llm but returns the full payload for pipeline transparency."""
    payload = {"messages": messages}
    t0 = time.time()
    try:
        resp = requests.post(llm_url, json=payload, timeout=55)
        resp.raise_for_status()
    except requests.RequestException as exc:
        raise HTTPException(status_code=502, detail=f"LLM API request failed: {exc}") from exc
    latency_ms = round((time.time() - t0) * 1000)
    data = resp.json()
    if "error" in data:
        raise HTTPException(status_code=502, detail=f"LLM API error: {data['error']}")
    answer = (data.get("choices", [{}])[0].get("message", {}).get("content", "")).strip()
    return {
        "answer": answer,
        "model": data.get("model", ""),
        "usage": data.get("usage", {}),
        "latency_ms": latency_ms,
    }


def get_query_embedding(voyage_client: voyageai.Client, text: str) -> List[float]:
    """Embed a search query with Voyage (query-optimized)."""
    response = voyage_client.embed(
        texts=[text],
        model=VOYAGE_EMBED_MODEL,
        input_type="query",
    )
    if not response.embeddings:
        raise HTTPException(status_code=502, detail="Embedding provider returned no vectors")
    return list(response.embeddings[0])


def compute_risk_score(customer: dict[str, Any], loan: dict[str, Any]) -> tuple[int, str, str, dict[str, Any]]:
    """
    Deterministic risk score from customer + loan fields.
    Returns (score 0-100, risk_level, decision, factors).
    """
    score = 50
    factors: dict[str, Any] = {}

    credit_score = int(customer.get("credit_score") or 0)
    factors["credit_score"] = credit_score
    if credit_score > 700:
        score += 20
    elif credit_score > 600:
        score += 10
    elif credit_score > 0 and credit_score < 400:
        score -= 20

    payment_history = str(customer.get("payment_history") or "").strip().lower()
    factors["payment_history"] = payment_history or None
    ph_map = {"excellent": 15, "good": 10, "fair": 0, "poor": -15}
    if payment_history in ph_map:
        score += ph_map[payment_history]

    monthly_income = float(customer.get("monthly_income") or 0)
    monthly_payment = float(loan.get("monthly_payment") or 0)
    dti = (monthly_payment / monthly_income * 100.0) if monthly_income > 0 else 0.0
    factors["dti"] = round(dti, 2)
    if monthly_income <= 0:
        factors["dti_note"] = "no_income_stated"
    if dti < 30:
        score += 10
    elif dti < 50:
        score += 0
    else:
        score -= 15

    account_age = int(customer.get("account_age_months") or 0)
    factors["account_age"] = account_age
    if account_age > 36:
        score += 5

    employment = str(customer.get("employment_type") or "").strip().lower()
    factors["employment"] = employment or None
    if employment in ("salaried", "government"):
        score += 5

    score = max(0, min(100, score))

    if score >= 65:
        risk_level = "low"
    elif score >= 40:
        risk_level = "medium"
    else:
        risk_level = "high"

    decision = "approved" if risk_level != "high" else "declined"

    return score, risk_level, decision, factors


def _credit_explanation_prompt(
    customer: dict[str, Any],
    loan: dict[str, Any],
    risk_score: int,
    risk_level: str,
    decision: str,
    factors: dict[str, Any],
) -> str:
    return (
        "You explain retail loan underwriting decisions clearly for bank staff.\n"
        "Summarize why this application received the given decision, referencing the factors below.\n"
        "Be concise (3-5 sentences), no markdown headings.\n\n"
        f"Risk score: {risk_score}/100\n"
        f"Risk level: {risk_level}\n"
        f"Decision: {decision}\n"
        f"Factors (JSON): {factors}\n"
        f"Customer (subset): { {k: customer.get(k) for k in ('full_name', 'credit_score', 'monthly_income', 'employment_type', 'payment_history', 'account_age_months')} }\n"
        f"Loan (subset): { {k: loan.get(k) for k in ('loan_amount', 'monthly_payment', 'purpose', 'term_months')} }\n"
    )


def _build_customer_profile_text(customer: dict[str, Any]) -> str:
    parts = [
        f"Name: {customer.get('full_name', '')}",
        f"Monthly income: {customer.get('monthly_income', '')}",
        f"Employment: {customer.get('employment_type', '')}",
        f"Payment history: {customer.get('payment_history', '')}",
        f"Credit score: {customer.get('credit_score', '')}",
        f"Goals / notes: {customer.get('financial_goals', '') or customer.get('notes', '')}",
    ]
    return "\n".join(str(p) for p in parts if p.split(": ", 1)[-1])


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    if not MONGODB_URI:
        raise RuntimeError("MONGODB_URI is not set")
    if not VOYAGE_API_KEY:
        raise RuntimeError("VOYAGE_API_KEY is not set")
    if not LLM_API_URL:
        raise RuntimeError("LLM_API_URL is not set")

    client = MongoClient(MONGODB_URI)
    db = client["banking"]
    faq_chunks: Collection[Any] = db["faq_chunks"]
    customers: Collection[Any] = db["customers"]
    loan_applications: Collection[Any] = db["loan_applications"]
    kyc_documents: Collection[Any] = db["kyc_documents"]
    bank_products: Collection[Any] = db["bank_products"]
    accounts: Collection[Any] = db["accounts"]
    transactions: Collection[Any] = db["transactions"]
    documents: Collection[Any] = db["documents"]

    voyage_client = voyageai.Client(api_key=VOYAGE_API_KEY)

    app.state.mongo_client = client
    app.state.db = db
    app.state.faq_chunks = faq_chunks
    app.state.customers = customers
    app.state.loan_applications = loan_applications
    app.state.kyc_documents = kyc_documents
    app.state.bank_products = bank_products
    app.state.accounts = accounts
    app.state.transactions = transactions
    app.state.documents = documents
    app.state.voyage = voyage_client
    app.state.llm_url = LLM_API_URL

    yield

    client.close()


app = FastAPI(title="GenAI for Financial Services Workshop", lifespan=lifespan)

if STATIC_DIR.is_dir():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.get("/")
async def serve_index() -> FileResponse:
    index_path = STATIC_DIR / "index.html"
    if not index_path.is_file():
        raise HTTPException(status_code=404, detail="index.html not found in static/")
    return FileResponse(index_path)


@app.get("/api/faq")
async def api_faq(
    request: Request,
    q: str = Query(..., min_length=1),
    limit: int = Query(5, ge=1, le=25),
) -> dict[str, Any]:
    voyage_client: voyageai.Client = request.app.state.voyage
    llm_url: str = request.app.state.llm_url
    faq_coll: Collection[Any] = request.app.state.faq_chunks

    t_embed_start = time.time()
    query_vector = get_query_embedding(voyage_client, q)
    embed_ms = round((time.time() - t_embed_start) * 1000)

    pipeline: List[dict[str, Any]] = [
        {
            "$vectorSearch": {
                "index": FAQ_VECTOR_INDEX,
                "path": EMBEDDING_FIELD,
                "queryVector": query_vector,
                "numCandidates": min(200, max(50, limit * 20)),
                "limit": limit,
            }
        },
        {"$addFields": {"score": {"$meta": "vectorSearchScore"}}},
        {"$project": {EMBEDDING_FIELD: 0}},
    ]

    t_search_start = time.time()
    raw_chunks = list(faq_coll.aggregate(pipeline))
    search_ms = round((time.time() - t_search_start) * 1000)

    sources: List[dict[str, Any]] = []
    context_blocks: List[str] = []
    for doc in raw_chunks:
        score = doc.get("score")
        title = doc.get("title", "")
        content_en = doc.get("content_en", "")
        content_km = doc.get("content_km", "")
        category = doc.get("category", "")
        sources.append(
            {
                "title": title,
                "content_en": content_en,
                "content_km": content_km,
                "category": category,
                "score": round(float(score), 6) if isinstance(score, (int, float)) else score,
            }
        )
        block = f"Title: {title}\nCategory: {category}\nEN: {content_en}\nKM: {content_km}"
        context_blocks.append(block)

    context = "\n\n---\n\n".join(context_blocks) if context_blocks else "(no context retrieved)"

    system_prompt = (
        "You are a helpful banking assistant. Answer based ONLY on the "
        "provided context. If the context doesn't contain the answer, say so. Respond in the "
        "same language as the question."
    )
    user_prompt = f"Context:\n{context}\n\nQuestion:\n{q}"

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    llm_result = call_llm_debug(messages, llm_url)

    pipeline_readable = [
        {"$vectorSearch": {"index": FAQ_VECTOR_INDEX, "path": EMBEDDING_FIELD, "queryVector": f"<{len(query_vector)}-dim float array>", "numCandidates": min(200, max(50, limit * 20)), "limit": limit}},
        {"$addFields": {"score": {"$meta": "vectorSearchScore"}}},
        {"$project": {EMBEDDING_FIELD: 0}},
    ]

    return {
        "query": q,
        "answer": llm_result["answer"],
        "sources": sources,
        "debug": {
            "embedding": {
                "model": VOYAGE_EMBED_MODEL,
                "dimensions": len(query_vector),
                "input_type": "query",
                "latency_ms": embed_ms,
            },
            "vector_search": {
                "index": FAQ_VECTOR_INDEX,
                "collection": "banking.faq_chunks",
                "pipeline": pipeline_readable,
                "results_returned": len(raw_chunks),
                "latency_ms": search_ms,
            },
            "llm": {
                "endpoint": llm_url,
                "model": llm_result["model"],
                "system_prompt": system_prompt,
                "user_prompt": user_prompt,
                "usage": llm_result["usage"],
                "latency_ms": llm_result["latency_ms"],
            },
        },
    }


@app.get("/api/credit-score")
async def api_credit_score(
    request: Request,
    customer_id: str = Query(...),
    loan_id: str = Query(...),
) -> dict[str, Any]:
    customers: Collection[Any] = request.app.state.customers
    loans: Collection[Any] = request.app.state.loan_applications
    llm_url: str = request.app.state.llm_url

    cust_oid = parse_object_id(customer_id, "customer_id")
    loan_oid = parse_object_id(loan_id, "loan_id")

    pipeline: List[dict[str, Any]] = [
        {"$match": {"_id": cust_oid}},
        {
            "$lookup": {
                "from": loans.name,
                "let": {"cid": "$_id"},
                "pipeline": [
                    {"$match": {"$expr": {"$and": [{"$eq": ["$customer_id", "$$cid"]}, {"$eq": ["$_id", loan_oid]}]}}},
                    {"$limit": 1},
                ],
                "as": "loan_docs",
            }
        },
        {"$addFields": {"loan": {"$first": "$loan_docs"}}},
        {"$project": {"loan_docs": 0}},
    ]

    rows = list(customers.aggregate(pipeline))
    if not rows:
        raise HTTPException(status_code=404, detail="Customer not found")
    row = rows[0]
    loan_doc = row.get("loan")
    if not loan_doc:
        raise HTTPException(status_code=404, detail="Loan application not found for this customer")

    customer_doc = {k: v for k, v in row.items() if k != "loan"}
    risk_score, risk_level, decision, factors = compute_risk_score(customer_doc, loan_doc)

    docs_coll: Collection[Any] = request.app.state.documents
    supporting_docs = list(docs_coll.find(
        {"customer_id": cust_oid},
        {"pdf_data": 0},
    ))

    doc_context_parts: list[str] = []
    for sd in supporting_docs:
        text = sd.get("text_content", "")
        if text:
            doc_context_parts.append(
                f"[{sd.get('doc_type', 'document').upper()}] {sd.get('filename', '')}\n{text}"
            )
    doc_context = "\n\n".join(doc_context_parts) if doc_context_parts else ""

    expl_user = _credit_explanation_prompt(
        customer_doc, loan_doc, risk_score, risk_level, decision, factors
    )
    if doc_context:
        expl_user += (
            "\n\nSupporting documents on file for this customer:\n"
            + doc_context
            + "\n\nReference specific document evidence (income figures, balances) in your explanation."
        )

    explanation = call_llm(
        [
            {"role": "system", "content": "You write clear, professional credit decision explanations."},
            {"role": "user", "content": expl_user},
        ],
        llm_url,
    )

    return {
        "customer": _serialize(customer_doc),
        "loan": _serialize(loan_doc),
        "risk_score": risk_score,
        "risk_level": risk_level,
        "decision": decision,
        "explanation": explanation,
        "factors": factors,
        "supporting_documents": [
            {
                "_id": str(sd["_id"]),
                "doc_type": sd.get("doc_type"),
                "filename": sd.get("filename"),
                "generated_date": sd.get("generated_date"),
            }
            for sd in supporting_docs
        ],
    }


@app.get("/api/customers")
async def api_customers(request: Request) -> list[dict[str, Any]]:
    coll: Collection[Any] = request.app.state.customers
    return _serialize_cursor(coll.find({}))


@app.get("/api/loan-applications")
async def api_loan_applications(
    request: Request,
    customer_id: Optional[str] = Query(None),
) -> list[dict[str, Any]]:
    coll: Collection[Any] = request.app.state.loan_applications
    query_filter: dict[str, Any] = {}
    if customer_id:
        query_filter["customer_id"] = parse_object_id(customer_id, "customer_id")
    return _serialize_cursor(coll.find(query_filter))


@app.get("/api/kyc-check")
async def api_kyc_check(
    request: Request,
    document_id: str = Query(...),
    threshold: float = Query(0.92, ge=0.0, le=1.0),
) -> dict[str, Any]:
    voyage_client: voyageai.Client = request.app.state.voyage
    kyc_coll: Collection[Any] = request.app.state.kyc_documents

    doc_oid = parse_object_id(document_id, "document_id")
    doc = kyc_coll.find_one({"_id": doc_oid})
    if not doc:
        raise HTTPException(status_code=404, detail="KYC document not found")

    description = str(doc.get("description") or doc.get("summary") or "")
    if not description.strip():
        raise HTTPException(status_code=400, detail="Document has no embeddable description")

    query_vector = get_query_embedding(voyage_client, description)

    pipeline: List[dict[str, Any]] = [
        {
            "$vectorSearch": {
                "index": KYC_VECTOR_INDEX,
                "path": KYC_EMBEDDING_FIELD,
                "queryVector": query_vector,
                "numCandidates": 150,
                "limit": 15,
                "filter": {"_id": {"$ne": doc_oid}},
            }
        },
        {"$addFields": {"similarity": {"$meta": "vectorSearchScore"}}},
        {"$project": {KYC_EMBEDDING_FIELD: 0}},
    ]

    similar = list(kyc_coll.aggregate(pipeline))
    duplicates: List[dict[str, Any]] = []
    for s in similar:
        sim = s.get("similarity")
        sim_f = float(sim) if isinstance(sim, (int, float)) else 0.0
        if sim_f >= threshold:
            duplicates.append(
                {
                    "id": str(s.get("_id")),
                    "customer_id": str(s.get("customer_id")) if s.get("customer_id") else None,
                    "document_type": s.get("document_type"),
                    "similarity": round(sim_f, 6),
                }
            )

    expired = False
    risk_flags: List[str] = []
    expiry = doc.get("expiry_date") or doc.get("expires_at")
    today = date.today()
    if expiry:
        exp_date: Optional[date] = None
        if isinstance(expiry, datetime):
            exp_date = expiry.date()
        elif isinstance(expiry, date):
            exp_date = expiry
        elif isinstance(expiry, str):
            try:
                exp_date = datetime.fromisoformat(expiry.replace("Z", "+00:00")).date()
            except ValueError:
                risk_flags.append("invalid_expiry_format")
        if exp_date and exp_date < today:
            expired = True
            risk_flags.append("document_expired")

    if duplicates:
        risk_flags.append("possible_duplicate_documents")

    doc_out = dict(doc)
    doc_out.pop(KYC_EMBEDDING_FIELD, None)
    doc_out.pop("embedding_model", None)
    doc_out.pop("pdf_data", None)

    return {
        "document": _serialize(doc_out),
        "duplicates": duplicates,
        "expired": expired,
        "risk_flags": risk_flags,
    }


@app.get("/api/kyc-documents")
async def api_kyc_documents(request: Request) -> list[dict[str, Any]]:
    coll: Collection[Any] = request.app.state.kyc_documents
    projection = {KYC_EMBEDDING_FIELD: 0, "embedding_model": 0, "pdf_data": 0}
    return _serialize_cursor(coll.find({}, projection))


@app.get("/api/kyc-documents/{doc_id}/pdf")
async def api_kyc_document_pdf(request: Request, doc_id: str) -> Response:
    """Serve the scanned KYC document PDF."""
    coll: Collection[Any] = request.app.state.kyc_documents
    oid = parse_object_id(doc_id, "doc_id")
    doc = coll.find_one({"_id": oid})
    if not doc or "pdf_data" not in doc:
        raise HTTPException(status_code=404, detail="KYC document PDF not found")
    pdf_bytes = bytes(doc["pdf_data"])
    doc_type = doc.get("document_type", "document")
    doc_num = doc.get("document_number", doc_id)
    filename = f"{doc_type}_{doc_num}.pdf"
    return Response(
        content=pdf_bytes,
        media_type="application/pdf",
        headers={"Content-Disposition": f'inline; filename="{filename}"'},
    )


@app.get("/api/accounts")
async def api_accounts(
    request: Request,
    customer_id: str = Query(...),
) -> list[dict[str, Any]]:
    coll: Collection[Any] = request.app.state.accounts
    cust_oid = parse_object_id(customer_id, "customer_id")
    return _serialize_cursor(coll.find({"customer_id": cust_oid}))


@app.get("/api/transactions")
async def api_transactions(
    request: Request,
    account_id: Optional[str] = Query(None),
    customer_id: Optional[str] = Query(None),
    limit: int = Query(20, ge=1, le=200),
) -> list[dict[str, Any]]:
    coll: Collection[Any] = request.app.state.transactions
    query_filter: dict[str, Any] = {}
    if account_id:
        query_filter["account_id"] = parse_object_id(account_id, "account_id")
    elif customer_id:
        query_filter["customer_id"] = parse_object_id(customer_id, "customer_id")
    cursor = coll.find(query_filter).sort("date", -1).limit(limit)
    return _serialize_cursor(cursor)


@app.get("/api/recommend-products")
async def api_recommend_products(
    request: Request,
    customer_id: str = Query(...),
    limit: int = Query(5, ge=1, le=20),
) -> dict[str, Any]:
    voyage_client: voyageai.Client = request.app.state.voyage
    customers: Collection[Any] = request.app.state.customers
    products: Collection[Any] = request.app.state.bank_products

    cust_oid = parse_object_id(customer_id, "customer_id")
    customer = customers.find_one({"_id": cust_oid})
    if not customer:
        raise HTTPException(status_code=404, detail="Customer not found")

    profile = _build_customer_profile_text(customer)
    query_vector = get_query_embedding(voyage_client, profile)

    pipeline: List[dict[str, Any]] = [
        {
            "$vectorSearch": {
                "index": PRODUCT_VECTOR_INDEX,
                "path": EMBEDDING_FIELD,
                "queryVector": query_vector,
                "numCandidates": min(200, max(50, limit * 20)),
                "limit": limit,
            }
        },
        {"$addFields": {"score": {"$meta": "vectorSearchScore"}}},
        {"$project": {EMBEDDING_FIELD: 0}},
    ]

    recs: List[dict[str, Any]] = []
    for p in products.aggregate(pipeline):
        sc = p.get("score")
        recs.append(
            {
                "name": p.get("name", ""),
                "description": p.get("description", ""),
                "score": round(float(sc), 6) if isinstance(sc, (int, float)) else sc,
            }
        )

    return {"customer_id": customer_id, "recommendations": recs}


# ──────────────────────────────────────────────────────────
# Customer documents (payslips, bank statements)
# ──────────────────────────────────────────────────────────

@app.get("/api/documents")
async def api_documents(
    request: Request,
    customer_id: str = Query(...),
) -> list[dict[str, Any]]:
    """List documents for a customer (metadata only, no binary PDF)."""
    coll: Collection[Any] = request.app.state.documents
    cust_oid = parse_object_id(customer_id, "customer_id")
    docs = coll.find({"customer_id": cust_oid}, {"pdf_data": 0})
    return _serialize_cursor(docs)


@app.get("/api/documents/{doc_id}/pdf")
async def api_document_pdf(request: Request, doc_id: str) -> Response:
    """Download a single PDF document by its _id."""
    coll: Collection[Any] = request.app.state.documents
    oid = parse_object_id(doc_id, "doc_id")
    doc = coll.find_one({"_id": oid})
    if not doc or "pdf_data" not in doc:
        raise HTTPException(status_code=404, detail="Document not found")
    pdf_bytes = bytes(doc["pdf_data"])
    filename = doc.get("filename", f"{doc_id}.pdf")
    return Response(
        content=pdf_bytes,
        media_type="application/pdf",
        headers={"Content-Disposition": f'inline; filename="{filename}"'},
    )


# ──────────────────────────────────────────────────────────
# Debug: CloudWatch Lambda Logs + Bedrock evidence
# ──────────────────────────────────────────────────────────

@app.get("/api/debug/lambda-logs")
async def api_lambda_logs(
    minutes: int = Query(10, ge=1, le=60),
    limit: int = Query(50, ge=1, le=200),
) -> dict[str, Any]:
    """Fetch recent BEDROCK_REQUEST / BEDROCK_RESPONSE logs from CloudWatch."""
    try:
        import boto3
    except ImportError:
        raise HTTPException(status_code=501, detail="boto3 not installed")

    log_group = "/aws/lambda/fsi_workshop"
    start_ms = int((time.time() - minutes * 60) * 1000)

    try:
        cw = boto3.client("logs", region_name="us-east-1")
        resp = cw.filter_log_events(
            logGroupName=log_group,
            startTime=start_ms,
            limit=limit,
            interleaved=True,
        )
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"CloudWatch error: {exc}")

    raw_events = []
    bedrock_events = []
    for ev in resp.get("events", []):
        msg = ev.get("message", "").strip()
        if not msg:
            continue
        ts = ev.get("timestamp")
        if msg.startswith("REPORT"):
            parts = msg.split("\t")
            raw_events.append({"timestamp": ts, "type": "REPORT", "message": msg})
            continue
        if msg.startswith(("START", "END", "INIT_START")):
            continue
        try:
            parsed = json.loads(msg.split("\t")[-1] if "\t" in msg else msg)
            if isinstance(parsed, dict) and parsed.get("event", "").startswith("BEDROCK_"):
                parsed["_timestamp"] = ts
                bedrock_events.append(parsed)
                continue
        except (json.JSONDecodeError, IndexError):
            pass
        raw_events.append({"timestamp": ts, "type": "LOG", "message": msg[:500]})

    return {
        "log_group": log_group,
        "last_minutes": minutes,
        "bedrock_events": bedrock_events,
        "raw_events": raw_events,
    }


# ──────────────────────────────────────────────────────────
# OCR: PDF text extraction → MongoDB
# ──────────────────────────────────────────────────────────

@app.post("/api/loan-application/upload")
async def api_loan_upload(
    request: Request,
    file: UploadFile = File(...),
    customer_id: str = Query(...),
) -> dict[str, Any]:
    """
    Accept a PDF upload, extract text via PyMuPDF (OCR),
    store the extracted data as a new loan support document in MongoDB,
    and return the extracted text + metadata.
    """
    import fitz  # pymupdf

    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are accepted")

    pdf_bytes = await file.read()
    if len(pdf_bytes) > 10 * 1024 * 1024:
        raise HTTPException(status_code=400, detail="File too large (max 10 MB)")

    t0 = time.time()

    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    pages = []
    full_text = ""
    for i, page in enumerate(doc):
        text = page.get_text("text")
        pages.append({"page": i + 1, "text": text, "char_count": len(text)})
        full_text += text + "\n"
    doc.close()

    extraction_ms = round((time.time() - t0) * 1000)

    try:
        cust_oid = ObjectId(customer_id)
    except InvalidId:
        cust_oid = None

    record = {
        "customer_id": cust_oid,
        "filename": file.filename,
        "content_type": file.content_type or "application/pdf",
        "file_size_bytes": len(pdf_bytes),
        "page_count": len(pages),
        "extracted_text": full_text.strip(),
        "pages": pages,
        "extraction_method": "pymupdf",
        "extraction_ms": extraction_ms,
        "uploaded_at": datetime.utcnow(),
        "type": "loan_support_document",
    }

    db = request.app.state.db
    result = db.loan_support_docs.insert_one(record)
    record_id = str(result.inserted_id)

    return _serialize({
        "status": "success",
        "document_id": record_id,
        "filename": file.filename,
        "page_count": len(pages),
        "total_characters": len(full_text.strip()),
        "extraction_ms": extraction_ms,
        "extraction_method": "pymupdf",
        "pages": pages,
        "extracted_text": full_text.strip()[:3000],
        "mongodb_collection": "banking.loan_support_docs",
        "mongodb_document_id": record_id,
    })

