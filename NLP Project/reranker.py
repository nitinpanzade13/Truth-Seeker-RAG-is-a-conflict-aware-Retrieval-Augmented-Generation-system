from sentence_transformers import CrossEncoder

# Cross-encoder for query-document relevance reranking
_cross_encoder = None


def _get_cross_encoder():
    global _cross_encoder
    if _cross_encoder is None:
        _cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
    return _cross_encoder


def rerank_with_indices(query, documents, scores=None, top_k=None):
    """
    Rerank documents and preserve original indices for stable alignment.

    Returns:
        List of (original_index, document, rerank_score) sorted descending.
    """
    if not documents:
        return []

    cross_encoder = _get_cross_encoder()

    pairs = [[query, doc] for doc in documents]
    ce_scores = cross_encoder.predict(pairs).tolist()

    # Normalize cross-encoder scores to [0, 1]
    min_s = min(ce_scores)
    max_s = max(ce_scores)
    score_range = max_s - min_s if max_s != min_s else 1.0
    ce_normalized = [(s - min_s) / score_range for s in ce_scores]

    # Blend with original retrieval scores if provided (0.6 CE + 0.4 retrieval)
    if scores and len(scores) == len(documents):
        blended = [
            0.6 * ce + 0.4 * orig
            for ce, orig in zip(ce_normalized, scores)
        ]
    else:
        blended = ce_normalized

    ranked = sorted(
        [(idx, doc, blended[idx]) for idx, doc in enumerate(documents)],
        key=lambda x: x[2],
        reverse=True,
    )

    if top_k:
        ranked = ranked[:top_k]

    return ranked


def rerank(query, documents, scores=None, top_k=None):
    """
    Rerank documents using a cross-encoder model for more accurate relevance scoring.

    Args:
        query: The user query string.
        documents: List of document text strings.
        scores: Optional list of original retrieval scores (for blending).
        top_k: Number of top results to return. None returns all.

    Returns:
        List of (document, rerank_score) tuples sorted by relevance descending.
    """
    ranked = rerank_with_indices(query, documents, scores=scores, top_k=top_k)

    return [(doc, score) for _, doc, score in ranked]
