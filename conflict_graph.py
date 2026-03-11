from judge import get_contradiction_prob

def build_conflict_matrix(docs):
    n = len(docs)
    matrix = [[0]*n for _ in range(n)]

    for i in range(n):
        for j in range(i+1, n):
            prob = get_contradiction_prob(docs[i], docs[j])
            matrix[i][j] = prob
            matrix[j][i] = prob

    return matrix


def compute_conflict_penalty(conflict_matrix):
    penalties = []

    for row in conflict_matrix:
        # average contradiction with other docs
        avg_conflict = sum(row) / (len(row) - 1) if len(row) > 1 else 0
        penalties.append(avg_conflict)

    return penalties