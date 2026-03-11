from judge import get_contradiction_prob

def conflict_aware_ranking(query, docs):
    results = []

    for doc in docs:
        contradiction = get_contradiction_prob(query, doc)
        confidence = 1 - contradiction

        results.append({
            "doc": doc,
            "contradiction_prob": contradiction,
            "confidence": confidence
        })

    return sorted(results, key=lambda x: x["confidence"], reverse=True)