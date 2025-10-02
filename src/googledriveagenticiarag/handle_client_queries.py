from sentence_transformers import util

def format_query_results(question, query_embedding, documents, metadatas, model):
    """Format and print the search results with similarity scores"""
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

def retrieve_context(collection, model, question, n_results=5):
    """Retrieve relevant context using embeddings"""
    query_embedding = model.encode([question])
    results = collection.query(
        query_embeddings=query_embedding.tolist(),
        n_results=n_results,
        include=["documents", "metadatas", "distances"],
    )

    documents = results["documents"][0]
    context = "\n\n---SECTION---\n\n".join(documents)
    return context, documents


def get_llm_answer(chain, question, context):
    """Generate answer using retrieved context"""
    answer = chain.invoke(
        {
            "context": context[:2000],
            "question": question,
        }
    )
    return answer


def format_response(question, answer, source_chunks):
    """Format the final response with sources"""
    response = f"**Question:** {question}\n\n"
    response += f"**Answer:** {answer}\n\n"
    response += "**Sources:**\n"

    for i, chunk in enumerate(source_chunks[:3], 1):
        preview = chunk[:100].replace("\n", " ") + "..."
        response += f"{i}. {preview}\n"

    return response


def enhanced_query_with_llm(question, n_results=5):
    """Query function combining retrieval with LLM generation"""
    context, documents = retrieve_context(question, n_results)
    answer = get_llm_answer(question, context)
    return format_response(question, answer, documents)


def stream_llm_answer(question, context, chain):
    """Stream LLM answer generation token by token"""
    for chunk in chain.stream({
        "context": context[:2000],
        "question": question,
    }):
        yield getattr(chunk, "content", str(chunk))
