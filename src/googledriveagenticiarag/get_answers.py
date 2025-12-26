from googledriveagenticiarag.llm_generation import LLMClientLC
from googledriveagenticiarag.handle_client_queries import retrieve_context
import chromadb
from googledriveagenticiarag.llm_tools import get_url


# Instantiate chat
chat_instance = LLMClientLC(provider="anthropic", tools=get_url)

# connect to chromadb get Drive_Data_Science_Docs connection
client = chromadb.PersistentClient(path="./chroma_db")
collection = client.get_collection(name="Drive_Data_Science_Docs")

def rag_interface(message, history):
    """Gradio interface reusing existing format_response function"""
    if not message.strip():
        yield {"role": "assistant", "content": "Please enter a question."}
        return

    # Use modular retrieval and streaming
    context, metadatas, documents = retrieve_context(
        message,
        n_results=5,
        collection=collection
    )

    # get doc links
    doc_links = []
    for m in metadatas:
        doc_id = m.get("document_id")
        if doc_id:
            url = get_url(doc_id)
            doc_title = m.get("document_title", "Document")
            doc_links.append(f"- [{doc_title}]({url})")
    links_md = "\n".join(doc_links)
    if links_md:
        links_md = "\n\n**Sources:**\n" + links_md


    # stream the answer from the chat instance
    partial_answer = ""
    for token in chat_instance.chat(context, message, metadatas):
        if isinstance(token, dict):
            if token.get("type") == "text":
                # Nettoyage du texte pour Markdown
                text = token["text"].replace("\n\n", "\n").strip()
                partial_answer += text
                yield {"role": "assistant", "content": partial_answer + links_md}

            elif token.get("type") == "tool_use" and token["name"] == "get_url":
                # Ajouter lien généré par le tool (si besoin)
                doc_id = token["input"]["document_id"]
                url = get_url(doc_id)
                partial_answer += f"\n\n📄 [Source document]({url})"
                yield {"role": "assistant", "content": partial_answer}
        else:
            partial_answer += str(token)
            yield {"role": "assistant", "content": partial_answer + links_md}


def rag_answer(question: str):
    """Function for serving api"""
    if not question.strip():
        return {"answer": "Veuillez entrer une question."}

    context, metadatas, documents = retrieve_context(
        question,
        n_results=5,
        collection=collection
    )

    # construire sources
    doc_links = []
    for m in metadatas:
        doc_id = m.get("document_id")
        if doc_id:
            url = get_url(doc_id)
            doc_title = m.get("document_title", "Document")
            doc_links.append(f"- [{doc_title}]({url})")

    links_md = "\n".join(doc_links)
    if links_md:
        links_md = "\n\n**Sources:**\n" + links_md

    # Génération (pas en streaming)
    partial_answer = ""
    tokens = list(chat_instance.chat(context, question, metadatas))

    for token in tokens:
        if isinstance(token, dict):

            if token.get("type") == "text":
                text = token["text"].replace("\n\n", "\n").strip()
                partial_answer += text

            elif token.get("type") == "tool_use" and token["name"] == "get_url":
                doc_id = token["input"]["document_id"]
                url = get_url(doc_id)
                partial_answer += f"\n\n📄 [Source document]({url})"

        else:
            partial_answer += str(token)

    return {"answer": partial_answer}
