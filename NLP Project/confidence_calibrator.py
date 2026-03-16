import numpy as np


def calibrate_confidence(retrieval_score, conflict_penalty, rerank_score=None,
                         source_type="corpus"):
    """
    Compute a calibrated confidence score using multiple signals.

    Formula:
        base = retrieval_score * (1 - conflict_penalty)
        If rerank_score available: base = 0.5 * rerank_score + 0.5 * base
        Apply source trust weight and sigmoid calibration.

    Args:
        retrieval_score: Semantic similarity from vector retrieval [0, 1].
        conflict_penalty: Average contradiction penalty [0, 1].
        rerank_score: Cross-encoder reranking score [0, 1] or None.
        source_type: "corpus" (trusted) or "web" (less trusted).

    Returns:
        Calibrated confidence float in [0, 1].
    """
    # Source trust factor
    trust = 0.95 if source_type == "corpus" else 0.75

    base = retrieval_score * (1 - conflict_penalty) * trust

    if rerank_score is not None:
        base = 0.5 * rerank_score * trust + 0.5 * base

    # Sigmoid calibration to avoid extreme values
    calibrated = _sigmoid_calibrate(base)

    return round(calibrated, 4)


def _sigmoid_calibrate(score, temperature=5.0, midpoint=0.5):
    """
    Apply sigmoid calibration to push ambiguous scores toward 0 or 1.
    """
    return 1.0 / (1.0 + np.exp(-temperature * (score - midpoint)))


def compute_calibrated_scores(docs, retrieval_scores, penalties,
                              rerank_scores=None, source_types=None):
    """
    Compute calibrated confidence for a batch of documents.

    Returns:
        List of (doc, calibrated_score) sorted descending by score.
    """
    results = []

    for i, doc in enumerate(docs):
        r_score = retrieval_scores[i] if i < len(retrieval_scores) else 0.5
        penalty = penalties[i] if i < len(penalties) else 0.0
        rr_score = rerank_scores[i] if rerank_scores and i < len(rerank_scores) else None
        src = source_types[i] if source_types and i < len(source_types) else "corpus"

        confidence = calibrate_confidence(r_score, penalty, rr_score, src)
        results.append((doc, confidence))

    results.sort(key=lambda x: x[1], reverse=True)
    return results


def confidence_summary(calibrated_scores):
    """
    Generate interpretive summary of confidence distribution.
    """
    scores = [s for _, s in calibrated_scores]

    if not scores:
        return {"level": "unknown", "mean": 0, "std": 0, "interpretation": "No evidence."}

    mean = float(np.mean(scores))
    std = float(np.std(scores))
    top = scores[0]

    if mean > 0.7 and std < 0.15:
        level = "high"
        interp = "Evidence is consistent and highly confident."
    elif mean > 0.5:
        level = "moderate"
        interp = "Evidence has moderate confidence with some disagreement."
    else:
        level = "low"
        interp = "Evidence is conflicting or weakly supported. Answer may be unreliable."

    return {
        "level": level,
        "mean": round(mean, 3),
        "std": round(std, 3),
        "top_score": round(top, 3),
        "interpretation": interp,
    }
