
"""LLM generation with prompt template and chain"""
# find out langchain llm ::: https://python.langchain.com/docs/integrations/llms/
# well find out model from langchain mostly OpenLLM and OpenLM

# We need semantic embeddings models
import os
from langchain_google_vertexai import ChatVertexAI
from langchain_anthropic import ChatAnthropic, AnthropicLLM, convert_to_anthropic_tool
from langchain_mistralai import ChatMistralAI
from langchain_core.messages import AIMessage, HumanMessage
from langchain.prompts import PromptTemplate, ChatPromptTemplate

from getpass import getpass
import os

def get_api_key(provider_api_key: str):
    if f"{provider_api_key}" not in os.environ:
        os.environ[f"{provider_api_key}"] = getpass(f"Enter your {provider_api_key}:")
    return os.environ.get(f"{provider_api_key}")


class LLMClientLC:
    def __init__(self, provider: str, tools=None, model: str = None, api_key: str = None):
        """
        Args:
            tools : tool fonction or List of tools functions
        """
        self.provider = provider.lower()
        self.tools = tools or []
        self.chat_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant specialized in Data Science, AI, ML, RAG and corresponding Python programming"),
    ("human", """Based on the provided documentation, answer the question clearly and accurately.
     First, answer the question directly and completely using ONLY the provided context.
    After answering, if metadata is available, suggest the Google Drive link(s)
    using the provided tool `get_url`.

    Documentation:
    {context}

    sources (metadata) : {metadatas}

    Question:
    {question}

    Answer (be specific about syntax, keywords, and provide examples when helpful):""")
])

        if self.provider == "claude" or self.provider == "anthropic":

            self.api_key = api_key or get_api_key("ANTHROPIC_API_KEY")
            if not self.api_key:
                raise ValueError("⚠️ API key for Anthropic missed")
            self.model = model or "claude-3-7-sonnet-20250219"
            # Instantiation
            self.llm = ChatAnthropic(model=self.model, api_key=self.api_key, temperature=0, max_tokens=1024)
            if self.tools:
                file_url_tool = convert_to_anthropic_tool(self.tools)
                file_url_tool["cache_control"] = {"type": "ephemeral"} # Enable caching on the tool
                self.llm = self.llm.bind_tools([file_url_tool])

        elif self.provider == "mistral":
            self.api_key = api_key or get_api_key("MISTRAL_API_KEY")
            if not self.api_key:
                raise ValueError("⚠️ Mistral API KEY missed")
            # "magistral-small-2509" : new open model with reasoning
            self.model = model or "magistral-small-2509" or "mistral-medium-2508"
            # Instantiation
            self.llm = ChatMistralAI(model=self.model, api_key=self.api_key, temperature=0, max_tokens=1024)

        else:
            raise ValueError("⚠️ Unsupported Provider (Choose whether 'anthropic' or 'mistral')")

    def chat(self, context: str, question: str, metadatas: list=None) -> str:
            # we chain model with a prompt template
        meta_str = ""
        if metadatas:
            meta_str = "\n\nAvailable sources (metadata):\n" + "\n".join(
                [f"- {m['document_title']} (id: {m['document_id']})" for m in metadatas]
            )
        chain = self.chat_prompt | self.llm

        # Pass argument to get AIMessage (Prompt + context)
        response = chain.invoke({
            "context": context,
            "question": question,
            "metadatas": meta_str
        })

        # return wanted content
        return response.content[0].text if hasattr(response.content[0], "text") else response.content
