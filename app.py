import streamlit as st
import matplotlib.pyplot as plt
import numpy as np

from wiki_loader import load_local_corpus, chunk_text
from retriever import add_documents, retrieve
from claim_conflict_graph import build_claim_conflict_matrix, compute_claim_conflict_penalty
from generator import generate_answer

st.set_page_config(page_title="Truth-Seeker RAG", layout="wide")

st.title("🔍 Truth-Seeker RAG")
st.caption("Conflict-Aware Retrieval Augmented Generation System")

# ----------------------------
# Load corpus once
# ----------------------------
@st.cache_resource
def setup_corpus():
    pages = load_local_corpus("corpus")
    all_chunks = []
    all_metadata = []

    for page in pages:
        chunks = chunk_text(page["content"])
        for chunk in chunks:
            all_chunks.append(chunk)
            all_metadata.append({"source": page["title"]})

    add_documents(all_chunks, all_metadata)
    return len(all_chunks)

total_chunks = setup_corpus()

st.sidebar.header("System Info")
st.sidebar.metric("Indexed Chunks", total_chunks)

# ----------------------------
# Query Input
# ----------------------------
query = st.text_input("Enter your question:")

if st.button("Generate Answer") and query:

    with st.spinner("Retrieving relevant documents..."):
        retrieved = retrieve(query)

    retrieved_docs = [doc for doc, _ in retrieved]
    retrieval_scores = [score for _, score in retrieved]

    # Layout: 2 Columns
    col1, col2 = st.columns(2)

    # ----------------------------
    # LEFT: Retrieved Docs
    # ----------------------------
    with col1:
        st.subheader("📚 Retrieved Documents")

        for doc, score in retrieved:
            st.progress(min(score, 1.0))
            st.write(f"Similarity: {round(score,3)}")
            st.write(doc[:250] + "...")
            st.divider()

    # ----------------------------
    # RIGHT: Conflict Analysis
    # ----------------------------
    with col2:
        st.subheader("⚖ Conflict Analysis")

        conflict_matrix = build_claim_conflict_matrix(retrieved_docs)
        penalties = compute_claim_conflict_penalty(conflict_matrix)

        for i, penalty in enumerate(penalties):
            st.progress(min(penalty, 1.0))
            st.write(f"Document {i+1} Penalty: {round(penalty,3)}")

        # Heatmap Visualization
        st.subheader("🔥 Conflict Heatmap")

        fig = plt.figure()
        plt.imshow(np.array(conflict_matrix))
        plt.colorbar()
        plt.title("Conflict Matrix")
        plt.xlabel("Document Index")
        plt.ylabel("Document Index")

        st.pyplot(fig)

    # ----------------------------
    # Final Scoring
    # ----------------------------
    final_scores = []
    for doc, retrieval_score, penalty in zip(retrieved_docs, retrieval_scores, penalties):
        confidence = retrieval_score * (1 - penalty)
        final_scores.append((doc, confidence))

    final_scores.sort(key=lambda x: x[1], reverse=True)

    average_conflict = sum(penalties) / len(penalties)
    conflict_detected = "Yes" if average_conflict > 0.3 else "No"

    # ----------------------------
    # Transparency Section
    # ----------------------------
    st.subheader("🔎 System Transparency")

    colA, colB, colC = st.columns(3)

    colA.metric("Conflict Detected", conflict_detected)
    colB.metric("Avg Conflict Score", round(average_conflict,3))
    colC.metric("Top Confidence Score", round(final_scores[0][1],3))

    # ----------------------------
    # Generate Answer
    # ----------------------------
    with st.spinner("Generating final answer..."):
        answer = generate_answer(query, final_scores[:3])

    st.subheader("🤖 Final Answer")
    st.write(answer)