
"""LLM generation with prompt template and chain"""
# find out langchain llm ::: https://python.langchain.com/docs/integrations/llms/
# well find out model from langchain mostly OpenLLM and OpenLM

# We need semantic embeddings models
import os
from langchain_google_vertexai import ChatVertexAI
from langchain_anthropic import ChatAnthropic, AnthropicLLM
from langchain_mistralai import ChatMistralAI
from langchain_core.messages import AIMessage, HumanMessage
from langchain.prompts import PromptTemplate, ChatPromptTemplate

from getpass import getpass
import os


def get_api_key(provider_api_key: str):
    if f"{provider_api_key}" not in os.environ:
        os.environ.get[f"{provider_api_key}"] = getpass(f"Enter your {provider_api_key}:")
    return os.environ.get[f"{provider_api_key}"]





class LLMClientLC:
    def __init__(self, provider: str, model: str = None, api_key: str = None):
        self.provider = provider.lower()
        self.chat_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant specialized in Python programming."),
    ("human", """Based on the provided documentation, answer the question clearly and accurately.

    Documentation:
    {context}

    Question: {question}

    Answer (be specific about syntax, keywords, and provide examples when helpful):""")
])
        if self.provider == "claude" or self.provider == "anthropic":

            self.api_key = api_key or get_api_key("ANTHROPIC_API_KEY")
            if not self.api_key:
                raise ValueError("⚠️ API key for Anthropic missed")
            self.model = model or "claude-3-haiku-20240307"
            # Instantiation
            self.llm = ChatAnthropic(model=self.model, api_key=self.api_key, temperature=0, max_tokens=1024)

        elif self.provider == "mistral":
            self.api_key = api_key or get_api_key("MISTRAL_API_KEY")
            if not self.api_key:
                raise ValueError("⚠️ Mistral API KEY missed")
            self.model = model or "mistral-tiny"
            # Instantiation
            self.llm = ChatMistralAI(model=self.model, api_key=self.api_key, temperature=0, max_tokens=1024)

        else:
            raise ValueError("⚠️ Unsupported Provider (Choose whether 'anthropic' or 'mistral')")

    def chat(self, context: str, question) -> str:
            # On combine prompt + LLM dans une chain
        chain = self.chat_prompt | self.llm

        # On injecte les variables du template
        response = chain.invoke({
            "context": context,
            "question": question
        })

        # response est un AIMessage, donc on récupère son contenu
        return response.content[0].text if hasattr(response.content[0], "text") else response.content


# caches management
