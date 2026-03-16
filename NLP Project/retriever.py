import chromadb
from sentence_transformers import SentenceTransformer

# Create embedding model
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

# Create Chroma client
client = chromadb.Client()
collection = client.create_collection("truth_seeker")

def add_documents(docs, metadatas):
    embeddings = embedding_model.encode(docs).tolist()

    collection.add(
        documents=docs,
        embeddings=embeddings,
        metadatas=metadatas,
        ids=[str(i) for i in range(len(docs))]
    )

def retrieve(query, k=3):
    query_embedding = embedding_model.encode([query]).tolist()

    results = collection.query(
        query_embeddings=query_embedding,
        n_results=k
    )

    docs = results["documents"][0]
    distances = results["distances"][0]

    # Convert distance to similarity score
    similarities = [1 - d for d in distances]

    return list(zip(docs, similarities))

