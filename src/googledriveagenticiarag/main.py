
from src.googledriveagenticiarag.llm_generation import LLMClientLC
from src.googledriveagenticiarag.handle_client_queries import retrieve_context
import chromadb
from src.googledriveagenticiarag.llm_tools import get_url
import gradio as gr

# Instantiate chat
tools = get_url
chat_instance = LLMClientLC(provider="anthropic", tools=tools)

def rag_interface(message, history):
    """Gradio interface reusing existing format_response function"""
    if not message.strip():
        yield {"role": "assistant", "content": "Please enter a question."}
        return

    # connect to chromadb get Drive_Data_Science_Docs connection
    client = chromadb.PersistentClient(path="./chroma_db")
    collection = client.get_collection(name="Drive_Data_Science_Docs")
    # Use modular retrieval and streaming
    context, metadatas, documents = retrieve_context(message, n_results=5, collection=collection)

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
# Create Gradio interface with streaming support

demo = gr.ChatInterface(
    fn=rag_interface,
    title="Intelligent Data Science Q&A System",
    description="Ask questions about Data Science, AI, ML concepts and get instant answers with sources from your Google Drive storage.",
    chatbot=gr.Chatbot(height=400),
    textbox=gr.Textbox(
        placeholder="Ask a question like: What is machine learning?",
        label="Your Question",
    ),
    type="messages",
    examples=[
        "Donnes moi des librairies python utiles pour la finance?",
        "Décris moi CNN dans le deep learning",
        "Qu'est ce qu'un apprentissage profond?",
        "Qu'est-ce que le machine learning ?"
    ],

)

if __name__ == "__main__":
    demo.queue().launch(share=True)
