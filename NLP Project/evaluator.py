"""
RAG Benchmark Evaluation Module
Measures: Faithfulness, Answer Relevance, Context Relevance, Conflict Detection Accuracy
"""

import re
import numpy as np
from claim_decomposer import decompose_into_claims
from judge import get_contradiction_prob


# -----------------------------------------------
# 1. Faithfulness — are answer claims grounded in context?
# -----------------------------------------------
def evaluate_faithfulness(answer, context_docs):
    """
    Decompose the answer into claims and check each against context docs.
    A claim is grounded if it does NOT contradict any context doc
    and has low contradiction probability.

    Returns:
        dict with 'score', 'grounded_claims', 'total_claims', 'details'.
    """
    answer_claims = decompose_into_claims(answer)
    if not answer_claims:
        return {"score": 1.0, "grounded_claims": 0, "total_claims": 0, "details": []}

    grounded = 0
    details = []

    for claim in answer_claims:
        max_contradiction = 0.0
        for doc in context_docs:
            prob = get_contradiction_prob(claim, doc)
            max_contradiction = max(max_contradiction, prob)

        is_grounded = max_contradiction < 0.5
        if is_grounded:
            grounded += 1

        details.append({
            "claim": claim[:100],
            "max_contradiction": round(max_contradiction, 3),
            "grounded": is_grounded,
        })

    score = grounded / len(answer_claims) if answer_claims else 1.0
    return {
        "score": round(score, 3),
        "grounded_claims": grounded,
        "total_claims": len(answer_claims),
        "details": details,
    }


# -----------------------------------------------
# 2. Answer Relevance — does the answer address the query?
# -----------------------------------------------
def evaluate_answer_relevance(query, answer):
    """
    Check if the answer is relevant to the query using NLI.
    Low contradiction between query and answer implies relevance.

    Returns:
        dict with 'score' and 'contradiction_prob'.
    """
    contradiction = get_contradiction_prob(query, answer)
    relevance = 1 - contradiction
    return {
        "score": round(relevance, 3),
        "contradiction_prob": round(contradiction, 3),
    }


# -----------------------------------------------
# 3. Context Relevance — are retrieved docs relevant to query?
# -----------------------------------------------
def evaluate_context_relevance(query, context_docs):
    """
    Measure how relevant each retrieved doc is to the query.

    Returns:
        dict with 'score' (avg), 'per_doc' scores.
    """
    scores = []
    per_doc = []

    for doc in context_docs:
        contradiction = get_contradiction_prob(query, doc)
        relevance = 1 - contradiction
        scores.append(relevance)
        per_doc.append(round(relevance, 3))

    avg = float(np.mean(scores)) if scores else 0.0
    return {
        "score": round(avg, 3),
        "per_doc": per_doc,
    }


# -----------------------------------------------
# 4. Conflict Detection Accuracy
# -----------------------------------------------
def evaluate_conflict_detection(conflict_matrix, threshold=0.3):
    """
    Analyze the conflict matrix to report detection statistics.

    Returns:
        dict with detection rate, max conflict, stats.
    """
    n = len(conflict_matrix)
    pairs = []
    detected = 0

    for i in range(n):
        for j in range(i + 1, n):
            score = conflict_matrix[i][j]
            pairs.append(score)
            if score > threshold:
                detected += 1

    total_pairs = len(pairs)
    return {
        "total_pairs": total_pairs,
        "conflicts_detected": detected,
        "detection_rate": round(detected / total_pairs, 3) if total_pairs > 0 else 0,
        "max_conflict": round(max(pairs), 3) if pairs else 0,
        "mean_conflict": round(float(np.mean(pairs)), 3) if pairs else 0,
    }


# -----------------------------------------------
# 5. Hallucination Score — inverse of faithfulness
# -----------------------------------------------
def evaluate_hallucination(answer, context_docs):
    """
    Hallucination = 1 - Faithfulness.
    """
    faith = evaluate_faithfulness(answer, context_docs)
    return {
        "hallucination_score": round(1 - faith["score"], 3),
        "faithfulness_score": faith["score"],
        "ungrounded_claims": faith["total_claims"] - faith["grounded_claims"],
        "total_claims": faith["total_claims"],
    }


# -----------------------------------------------
# Full Evaluation Suite
# -----------------------------------------------
def run_full_evaluation(query, answer, context_docs, conflict_matrix):
    """
    Run all evaluation metrics and return a comprehensive report.
    """
    faithfulness = evaluate_faithfulness(answer, context_docs)
    answer_relevance = evaluate_answer_relevance(query, answer)
    context_relevance = evaluate_context_relevance(query, context_docs)
    conflict_stats = evaluate_conflict_detection(conflict_matrix)
    hallucination = evaluate_hallucination(answer, context_docs)

    overall = (
        0.35 * faithfulness["score"]
        + 0.25 * answer_relevance["score"]
        + 0.20 * context_relevance["score"]
        + 0.20 * (1 - hallucination["hallucination_score"])
    )

    return {
        "overall_score": round(overall, 3),
        "faithfulness": faithfulness,
        "answer_relevance": answer_relevance,
        "context_relevance": context_relevance,
        "conflict_detection": conflict_stats,
        "hallucination": hallucination,
    }
