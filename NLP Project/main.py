from wiki_loader import load_local_corpus, chunk_text
from retriever import add_documents, retrieve
from reranker import rerank_with_indices
from claim_conflict_graph import build_claim_conflict_matrix, compute_claim_conflict_penalty
from confidence_calibrator import compute_calibrated_scores, confidence_summary
from generator import generate_answer
from web_retriever import web_retrieve
from evaluator import run_full_evaluation

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
retrieved = retrieve(query, k=5)

retrieved_docs = [doc for doc, _ in retrieved]
retrieval_scores = [score for _, score in retrieved]
source_types = ["corpus"] * len(retrieved_docs)

print("\nRetrieved Documents:")
for doc, score in retrieved:
    print(f"\nSimilarity: {round(score, 3)}")
    print(doc[:250], "...")

# ---------------------------------------
# STEP 3.5 — Web Retrieval (Optional)
# ---------------------------------------
use_web = input("\nEnable web retrieval? (y/n): ").strip().lower() == "y"
if use_web:
    web_docs = web_retrieve(query, max_results=3)
    for wd in web_docs:
        retrieved_docs.append(wd["content"][:1000])
        retrieval_scores.append(0.5)
        source_types.append("web")
    print(f"\nAdded {len(web_docs)} web documents.")

# ---------------------------------------
# STEP 3.6 — Cross-Encoder Reranking
# ---------------------------------------
print("\nRunning cross-encoder reranking...")
reranked = rerank_with_indices(query, retrieved_docs, retrieval_scores)

ordered_indices = [idx for idx, _, _ in reranked]
retrieved_docs = [retrieved_docs[i] for i in ordered_indices]
retrieval_scores = [retrieval_scores[i] for i in ordered_indices]
source_types = [source_types[i] for i in ordered_indices]
rerank_scores = [score for _, _, score in reranked]

print("\nReranked Documents:")
for doc, score in zip(retrieved_docs[:5], rerank_scores[:5]):
    print(f"\nRerank Score: {round(score, 3)}")
    print(doc[:250], "...")

# ---------------------------------------
# STEP 4 — Claim-Level Conflict Detection
# ---------------------------------------
print("\nRunning claim-level conflict detection (optimized with sampling)...")
conflict_matrix = build_claim_conflict_matrix(retrieved_docs, max_claims_per_doc=10)
penalties = compute_claim_conflict_penalty(conflict_matrix)

print("\nConflict Penalties:")
for doc, penalty in zip(retrieved_docs, penalties):
    print(f"{round(penalty, 3)} → {doc[:80]}...")

# ---------------------------------------
# STEP 5 — Calibrated Confidence Scoring
# ---------------------------------------
calibrated = compute_calibrated_scores(
    retrieved_docs, retrieval_scores, penalties,
    rerank_scores=rerank_scores,
    source_types=source_types,
)
conf_summary = confidence_summary(calibrated)

print("\nCalibrated Ranked Chunks:")
for doc, score in calibrated:
    print(f"\nCalibrated Score: {round(score, 3)}")
    print(doc[:250], "...")

# ---------------------------------------
# STEP 6 — System Transparency Metrics
# ---------------------------------------
average_conflict = sum(penalties) / len(penalties)
conflict_detected = "Yes" if average_conflict > 0.3 else "No"

print("\nSystem Analysis:")
print(f"Conflict Detected: {conflict_detected}")
print(f"Average Conflict Score: {round(average_conflict, 3)}")
print(f"Confidence Level: {conf_summary['level'].upper()}")
print(f"Confidence Mean: {conf_summary['mean']}")
print(f"Interpretation: {conf_summary['interpretation']}")

# ---------------------------------------
# STEP 7 — Generate Final Answer
# ---------------------------------------
print("\n--- Generating Final Answer ---\n")

top_docs = calibrated[:3]
answer = generate_answer(query, top_docs)

print("Answer:\n")
print(answer)

# ---------------------------------------
# STEP 8 — Evaluation (Optional)
# ---------------------------------------
run_eval = input("\nRun evaluation metrics? (y/n): ").strip().lower() == "y"
if run_eval:
    print("\n--- RAG Evaluation ---\n")
    context_texts = [d for d, _ in top_docs]
    eval_result = run_full_evaluation(query, answer, context_texts, conflict_matrix)

    print(f"Overall Score:      {eval_result['overall_score']}")
    print(f"Faithfulness:       {eval_result['faithfulness']['score']}")
    print(f"Answer Relevance:   {eval_result['answer_relevance']['score']}")
    print(f"Context Relevance:  {eval_result['context_relevance']['score']}")
    print(f"Hallucination:      {eval_result['hallucination']['hallucination_score']}")
    print(f"Conflict Detection: {eval_result['conflict_detection']['detection_rate']} "
          f"({eval_result['conflict_detection']['conflicts_detected']}/"
          f"{eval_result['conflict_detection']['total_pairs']} pairs)")