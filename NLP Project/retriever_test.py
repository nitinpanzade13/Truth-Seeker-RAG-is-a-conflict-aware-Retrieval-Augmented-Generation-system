from retriever import add_documents, retrieve

docs = [
    "Stock X increased by 4 percent today.",
    "Stock X crashed due to regulatory concerns.",
    "Investors are optimistic about Stock X."
]

add_documents(docs)

results = retrieve("How is Stock X performing?")

print(results)