
import gradio as gr
from googledriveagenticiarag.get_answers import rag_interface

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
