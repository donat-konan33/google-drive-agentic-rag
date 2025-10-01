

def format_query_results(question, query_embedding, documents, metadatas, model):
    """Format and print the search results with similarity scores"""
    from sentence_transformers import util

    print(f"Question: {question}\n")

    for i, doc in enumerate(documents):
        # Calculate accurate similarity using sentence-transformers util
        doc_embedding = model.encode([doc])
        similarity = util.cos_sim(query_embedding, doc_embedding)[0][0].item()
        source = metadatas[i].get("document", "Unknown")

        print(f"Result {i+1} (similarity: {similarity:.3f}):")
        print(f"Document: {source}")
        print(f"Content: {doc[:300]}...")
        print()


def query_knowledge_base(question, model, collection, n_results=2):
    """Query the knowledge base with natural language"""
    # Encode the query using our SentenceTransformer model
    query_embedding = model.encode([question])

    results = collection.query(
        query_embeddings=query_embedding.tolist(),
        n_results=n_results,
        include=["documents", "metadatas", "distances"],
    )

    # Extract results and format them
    documents = results["documents"][0]
    metadatas = results["metadatas"][0]

    format_query_results(question, query_embedding, documents, metadatas)
