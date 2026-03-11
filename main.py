from wiki_loader import load_local_corpus, chunk_text
from retriever import add_documents, retrieve
from claim_conflict_graph import build_claim_conflict_matrix, compute_claim_conflict_penalty
from generator import generate_answer

# ---------------------------------------
# STEP 1 — Load Local Corpus
# ---------------------------------------
pages = load_local_corpus("corpus")

all_chunks = []
all_metadata = []

for page in pages:
    chunks = chunk_text(page["content"])

    for chunk in chunks:
        all_chunks.append(chunk)
        all_metadata.append({"source": page["title"]})

print(f"\nTotal chunks created: {len(all_chunks)}")

if not all_chunks:
    raise ValueError("No documents found in corpus folder.")

# ---------------------------------------
# STEP 2 — Add to Vector DB
# ---------------------------------------
add_documents(all_chunks, all_metadata)

# ---------------------------------------
# STEP 3 — User Query
# ---------------------------------------
query = input("\nEnter your question: ")

# Retrieve top documents
retrieved = retrieve(query)

retrieved_docs = [doc for doc, _ in retrieved]
retrieval_scores = [score for _, score in retrieved]

print("\nRetrieved Documents:")
for doc, score in retrieved:
    print(f"\nSimilarity: {round(score, 3)}")
    print(doc[:250], "...")

# ---------------------------------------
# STEP 4 — Claim-Level Conflict Detection
# ---------------------------------------
conflict_matrix = build_claim_conflict_matrix(retrieved_docs)
penalties = compute_claim_conflict_penalty(conflict_matrix)

print("\nConflict Penalties:")
for doc, penalty in zip(retrieved_docs, penalties):
    print(f"{round(penalty, 3)} → {doc[:80]}...")

# ---------------------------------------
# STEP 5 — Final Confidence Scoring
# Formula:
#   Final Score = Retrieval × (1 - Conflict)
# ---------------------------------------
final_scores = []

for doc, retrieval_score, penalty in zip(retrieved_docs, retrieval_scores, penalties):
    confidence = retrieval_score * (1 - penalty)
    final_scores.append((doc, confidence))

# Sort descending
final_scores.sort(key=lambda x: x[1], reverse=True)

print("\nFinal Ranked Chunks:")
for doc, score in final_scores:
    print(f"\nFinal Score: {round(score, 3)}")
    print(doc[:250], "...")

# ---------------------------------------
# STEP 6 — System Transparency Metrics
# ---------------------------------------
average_conflict = sum(penalties) / len(penalties)
top_confidence = final_scores[0][1]
conflict_detected = "Yes" if average_conflict > 0.3 else "No"

print("\nSystem Analysis:")
print(f"Conflict Detected: {conflict_detected}")
print(f"Average Conflict Score: {round(average_conflict, 3)}")
print(f"Answer Confidence Score: {round(top_confidence, 3)}")

# ---------------------------------------
# STEP 7 — Generate Final Answer
# ---------------------------------------
print("\n--- Generating Final Answer ---\n")

answer = generate_answer(query, final_scores[:3])

print("Answer:\n")
print(answer)