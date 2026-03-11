from judge import get_contradiction_prob
from claim_decomposer import decompose_into_claims

def build_claim_conflict_matrix(docs):
    """
    Build conflict matrix at CLAIM level.
    """

    doc_claims = []

    for doc in docs:
        claims = decompose_into_claims(doc)
        doc_claims.append(claims)

    n = len(docs)
    matrix = [[0]*n for _ in range(n)]

    for i in range(n):
        for j in range(i+1, n):

            conflicts = []

            for claim1 in doc_claims[i]:
                for claim2 in doc_claims[j]:
                    prob = get_contradiction_prob(claim1, claim2)
                    conflicts.append(prob)

            if conflicts:
                avg_conflict = sum(conflicts) / len(conflicts)
            else:
                avg_conflict = 0

            matrix[i][j] = avg_conflict
            matrix[j][i] = avg_conflict

    return matrix


def compute_claim_conflict_penalty(conflict_matrix):
    penalties = []

    for row in conflict_matrix:
        avg_conflict = sum(row) / (len(row) - 1) if len(row) > 1 else 0
        penalties.append(avg_conflict)

    return penalties