import streamlit as st
import matplotlib.pyplot as plt
import numpy as np

from wiki_loader import load_local_corpus, chunk_text
from retriever import add_documents, retrieve
from reranker import rerank
from claim_conflict_graph import build_claim_conflict_matrix, compute_claim_conflict_penalty
from confidence_calibrator import compute_calibrated_scores, confidence_summary
from conflict_visualizer import plot_conflict_graph_interactive, get_graph_stats
from generator import generate_answer
from web_retriever import web_retrieve
from evaluator import run_full_evaluation

st.set_page_config(page_title="Truth-Seeker RAG v2", layout="wide")

st.title("🔍 Truth-Seeker RAG v2")
st.caption("Conflict-Aware Retrieval Augmented Generation — Enhanced Pipeline")

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

# ----------------------------
# Sidebar — Settings
# ----------------------------
st.sidebar.header("⚙ System Settings")
st.sidebar.metric("Indexed Chunks", total_chunks)

top_k = st.sidebar.slider("Top-K Documents", 3, 10, 5)
use_web = st.sidebar.checkbox("Enable Web Retrieval", value=False)
use_reranker = st.sidebar.checkbox("Enable Cross-Encoder Reranking", value=True)
run_eval = st.sidebar.checkbox("Run Evaluation Metrics", value=False)
conflict_threshold = st.sidebar.slider("Conflict Threshold", 0.1, 0.6, 0.3, 0.05)

# ----------------------------
# Query Input
# ----------------------------
query = st.text_input("Enter your question:")

if st.button("Generate Answer") and query:

    # ==========================================
    # STEP 1 — Retrieve from corpus
    # ==========================================
    with st.spinner("Retrieving relevant documents..."):
        retrieved = retrieve(query, k=top_k)

    retrieved_docs = [doc for doc, _ in retrieved]
    retrieval_scores = [score for _, score in retrieved]
    source_types = ["corpus"] * len(retrieved_docs)

    # ==========================================
    # STEP 2 — Web retrieval (optional)
    # ==========================================
    if use_web:
        with st.spinner("Fetching web results..."):
            web_docs = web_retrieve(query, max_results=3)

        for wd in web_docs:
            retrieved_docs.append(wd["content"][:1000])
            retrieval_scores.append(0.5)
            source_types.append("web")

        if web_docs:
            st.info(f"🌐 Added {len(web_docs)} web documents to the evidence pool.")
        else:
            st.warning("⚠ Web retrieval returned no results. Only corpus documents will be used.")

    # ==========================================
    # STEP 3 — Cross-encoder reranking (optional)
    # ==========================================
    rerank_scores = None
    if use_reranker:
        with st.spinner("Reranking with cross-encoder..."):
            reranked = rerank(query, retrieved_docs, retrieval_scores)
            rerank_scores = [s for _, s in reranked]

    # Layout: 2 Columns
    col1, col2 = st.columns(2)

    # ----------------------------
    # LEFT: Retrieved Docs
    # ----------------------------
    with col1:
        st.subheader("📚 Retrieved Documents")

        display_order = list(enumerate(zip(retrieved_docs, retrieval_scores)))
        if rerank_scores:
            display_order.sort(key=lambda x: rerank_scores[x[0]], reverse=True)

        for idx, (doc, score) in display_order:
            src_label = f"🌐 Web" if source_types[idx] == "web" else "📄 Corpus"
            rr_label = f" | Rerank: {round(rerank_scores[idx], 3)}" if rerank_scores else ""
            st.progress(min(score, 1.0))
            st.write(f"{src_label} — Similarity: {round(score, 3)}{rr_label}")
            st.write(doc[:250] + "...")
            st.divider()

    # ==========================================
    # STEP 4 — Conflict detection
    # ==========================================
    with col2:
        st.subheader("⚖ Conflict Analysis")

        with st.spinner("Running claim-level conflict detection..."):
            conflict_matrix = build_claim_conflict_matrix(retrieved_docs)
            penalties = compute_claim_conflict_penalty(conflict_matrix)

        for i, penalty in enumerate(penalties):
            src_label = "🌐" if source_types[i] == "web" else "📄"
            st.progress(min(penalty, 1.0))
            st.write(f"{src_label} Document {i+1} Penalty: {round(penalty, 3)}")

        # Heatmap
        st.subheader("🔥 Conflict Heatmap")
        fig_hm = plt.figure()
        plt.imshow(np.array(conflict_matrix), cmap="RdYlGn_r")
        plt.colorbar(label="Contradiction Score")
        plt.title("Pairwise Conflict Matrix")
        plt.xlabel("Document Index")
        plt.ylabel("Document Index")
        st.pyplot(fig_hm)

        # Interactive conflict graph
        st.subheader("🕸 Conflict Graph")
        fig_graph = plot_conflict_graph_interactive(
            conflict_matrix,
            doc_labels=[f"Doc {i+1}" for i in range(len(retrieved_docs))],
            threshold=conflict_threshold,
        )
        st.plotly_chart(fig_graph, use_container_width=True)

        # Graph stats
        stats = get_graph_stats(conflict_matrix, threshold=conflict_threshold)
        stat_cols = st.columns(3)
        stat_cols[0].metric("Conflict Edges", stats["edges"])
        stat_cols[1].metric("Graph Density", round(stats["density"], 3))
        stat_cols[2].metric("Components", stats["connected_components"])

    # ==========================================
    # STEP 5 — Calibrated confidence scoring
    # ==========================================
    calibrated = compute_calibrated_scores(
        retrieved_docs, retrieval_scores, penalties,
        rerank_scores=rerank_scores,
        source_types=source_types,
    )
    conf_summary = confidence_summary(calibrated)

    average_conflict = sum(penalties) / len(penalties) if penalties else 0

    # ----------------------------
    # Transparency Section
    # ----------------------------
    st.subheader("🔎 System Transparency")

    colA, colB, colC, colD = st.columns(4)
    colA.metric("Conflict Detected", "Yes" if average_conflict > conflict_threshold else "No")
    colB.metric("Avg Conflict Score", round(average_conflict, 3))
    colC.metric("Confidence Level", conf_summary["level"].upper())
    colD.metric("Top Confidence", conf_summary["top_score"])

    st.caption(f"💡 {conf_summary['interpretation']}")

    # ==========================================
    # STEP 6 — Generate Answer
    # ==========================================
    top_docs = calibrated[:3]

    with st.spinner("Generating final answer..."):
        answer = generate_answer(query, top_docs)

    st.subheader("🤖 Final Answer")
    st.write(answer)

    # ==========================================
    # STEP 7 — Evaluation Metrics (optional)
    # ==========================================
    if run_eval:
        st.subheader("📊 RAG Evaluation Metrics")

        with st.spinner("Computing evaluation metrics..."):
            context_texts = [d for d, _ in top_docs]
            eval_result = run_full_evaluation(query, answer, context_texts, conflict_matrix)

        eval_cols = st.columns(5)
        eval_cols[0].metric("Overall", eval_result["overall_score"])
        eval_cols[1].metric("Faithfulness", eval_result["faithfulness"]["score"])
        eval_cols[2].metric("Answer Relevance", eval_result["answer_relevance"]["score"])
        eval_cols[3].metric("Context Relevance", eval_result["context_relevance"]["score"])
        eval_cols[4].metric("Hallucination", eval_result["hallucination"]["hallucination_score"])

        # Faithfulness details
        with st.expander("Faithfulness Claim Details"):
            for detail in eval_result["faithfulness"]["details"]:
                icon = "✅" if detail["grounded"] else "❌"
                st.write(f"{icon} `{detail['claim']}` — contradiction: {detail['max_contradiction']}")