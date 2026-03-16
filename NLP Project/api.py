"""
FastAPI service for Truth-Seeker RAG.
Run with: uvicorn api:app --host 0.0.0.0 --port 8000
"""

import os
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from wiki_loader import load_local_corpus, chunk_text
from retriever import add_documents, retrieve
from reranker import rerank
from claim_conflict_graph import build_claim_conflict_matrix, compute_claim_conflict_penalty
from confidence_calibrator import compute_calibrated_scores, confidence_summary
from generator import generate_answer
from web_retriever import web_retrieve
from evaluator import run_full_evaluation


# -----------------------------------------------
# Startup: index corpus once
# -----------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    pages = load_local_corpus("corpus")
    all_chunks, all_metadata = [], []
    for page in pages:
        for chunk in chunk_text(page["content"]):
            all_chunks.append(chunk)
            all_metadata.append({"source": page["title"]})
    add_documents(all_chunks, all_metadata)
    app.state.total_chunks = len(all_chunks)
    yield


app = FastAPI(
    title="Truth-Seeker RAG API",
    description="Conflict-aware Retrieval-Augmented Generation",
    version="2.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# -----------------------------------------------
# Request / Response Models
# -----------------------------------------------
class QueryRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=500)
    top_k: int = Field(default=5, ge=1, le=20)
    use_web: bool = Field(default=False)
    use_reranker: bool = Field(default=True)


class DocumentScore(BaseModel):
    text: str
    score: float
    source: str = "corpus"


class TransparencyMetrics(BaseModel):
    conflict_detected: bool
    avg_conflict_score: float
    confidence_level: str
    confidence_mean: float
    confidence_interpretation: str


class EvalMetrics(BaseModel):
    overall_score: float
    faithfulness: float
    answer_relevance: float
    context_relevance: float
    hallucination_score: float


class QueryResponse(BaseModel):
    answer: str
    documents: list[DocumentScore]
    transparency: TransparencyMetrics
    evaluation: EvalMetrics | None = None


# -----------------------------------------------
# Endpoints
# -----------------------------------------------
@app.get("/health")
def health():
    return {"status": "ok", "indexed_chunks": getattr(app.state, "total_chunks", 0)}


@app.post("/query", response_model=QueryResponse)
def query_endpoint(req: QueryRequest):
    # 1. Retrieve from corpus
    retrieved = retrieve(req.query, k=req.top_k)
    docs = [doc for doc, _ in retrieved]
    retrieval_scores = [score for _, score in retrieved]
    source_types = ["corpus"] * len(docs)

    # 2. Optional web retrieval
    if req.use_web:
        web_docs = web_retrieve(req.query, max_results=3)
        for wd in web_docs:
            docs.append(wd["content"][:1000])
            retrieval_scores.append(0.5)  # default score for web results
            source_types.append("web")

    # 3. Cross-encoder reranking
    rerank_scores = None
    if req.use_reranker and docs:
        reranked = rerank(req.query, docs, retrieval_scores)
        rerank_scores = [s for _, s in reranked]
        # Re-order docs to match reranked order
        rerank_order = {doc: i for i, (doc, _) in enumerate(reranked)}
        ordered_indices = sorted(range(len(docs)), key=lambda i: rerank_order.get(docs[i], i))
        docs = [docs[i] for i in ordered_indices]
        retrieval_scores = [retrieval_scores[i] for i in ordered_indices]
        source_types = [source_types[i] for i in ordered_indices]

    # 4. Conflict detection
    conflict_matrix = build_claim_conflict_matrix(docs)
    penalties = compute_claim_conflict_penalty(conflict_matrix)

    # 5. Calibrated confidence
    calibrated = compute_calibrated_scores(
        docs, retrieval_scores, penalties,
        rerank_scores=rerank_scores,
        source_types=source_types,
    )
    conf_summary = confidence_summary(calibrated)

    # 6. Generate answer
    top_docs = calibrated[:3]
    answer = generate_answer(req.query, top_docs)

    # 7. Evaluation
    context_texts = [d for d, _ in top_docs]
    eval_result = run_full_evaluation(req.query, answer, context_texts, conflict_matrix)

    # Build response
    avg_conflict = sum(penalties) / len(penalties) if penalties else 0

    doc_scores = [
        DocumentScore(text=doc[:500], score=score, source=source_types[i] if i < len(source_types) else "corpus")
        for i, (doc, score) in enumerate(calibrated)
    ]

    transparency = TransparencyMetrics(
        conflict_detected=avg_conflict > 0.3,
        avg_conflict_score=round(avg_conflict, 3),
        confidence_level=conf_summary["level"],
        confidence_mean=conf_summary["mean"],
        confidence_interpretation=conf_summary["interpretation"],
    )

    evaluation = EvalMetrics(
        overall_score=eval_result["overall_score"],
        faithfulness=eval_result["faithfulness"]["score"],
        answer_relevance=eval_result["answer_relevance"]["score"],
        context_relevance=eval_result["context_relevance"]["score"],
        hallucination_score=eval_result["hallucination"]["hallucination_score"],
    )

    return QueryResponse(
        answer=answer,
        documents=doc_scores,
        transparency=transparency,
        evaluation=evaluation,
    )
