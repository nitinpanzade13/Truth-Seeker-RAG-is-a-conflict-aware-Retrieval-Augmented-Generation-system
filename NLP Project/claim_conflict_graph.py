from judge import get_contradiction_prob
from claim_decomposer import decompose_into_claims
import random


def build_claim_conflict_matrix(docs, max_claims_per_doc=10, max_comparisons_per_pair=100):
    """
    Build conflict matrix at CLAIM level with optimization.
    
    Args:
        docs: List of document strings
        max_claims_per_doc: Limit claims per document to speed up (default 10)
        max_comparisons_per_pair: Max claim pairs to compare per doc pair (default 100)
    """
    doc_claims = []

    for doc in docs:
        claims = decompose_into_claims(doc)
        # Sample claims if too many to speed up computation
        if len(claims) > max_claims_per_doc:
            claims = random.sample(claims, max_claims_per_doc)
        doc_claims.append(claims)

    n = len(docs)
    matrix = [[0.0] * n for _ in range(n)]

    for i in range(n):
        for j in range(i + 1, n):
            claims_i = doc_claims[i]
            claims_j = doc_claims[j]
            
            # Limit total comparisons to avoid timeout
            max_per_side = int((max_comparisons_per_pair ** 0.5) + 1)
            
            sample_i = claims_i if len(claims_i) <= max_per_side else random.sample(claims_i, max_per_side)
            sample_j = claims_j if len(claims_j) <= max_per_side else random.sample(claims_j, max_per_side)
            
            conflicts = []

            for claim1 in sample_i:
                for claim2 in sample_j:
                    prob = get_contradiction_prob(claim1, claim2)
                    conflicts.append(prob)

            if conflicts:
                avg_conflict = sum(conflicts) / len(conflicts)
            else:
                avg_conflict = 0.0

            matrix[i][j] = avg_conflict
            matrix[j][i] = avg_conflict

    return matrix


def compute_claim_conflict_penalty(conflict_matrix):
    penalties = []

    for row in conflict_matrix:
        avg_conflict = sum(row) / (len(row) - 1) if len(row) > 1 else 0
        penalties.append(avg_conflict)

    return penalties