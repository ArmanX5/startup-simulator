from langchain.chat_models import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage, AIMessage
import os

def make_chat_model(temperature: float = 0.6, model: str = "gpt-4o-mini"):
    """
    Create a LangChain Chat model instance. You can change model to "gpt-4o", "gpt-4o-mini",
    or to any OpenAI chat-model name you have access to.
    """
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("Set OPENAI_API_KEY in environment.")
    # For typical LangChain + OpenAI usage:
    return ChatOpenAI(temperature=temperature, model=model, openai_api_key=api_key)

def ask_ai(chat_model, system_prompt: str, user_prompt: str, max_tokens: int = 512):
    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_prompt),
    ]
    resp = chat_model(messages)  # returns an LLMResult-like object with .content on first message
    # LangChain ChatOpenAI returns a ChatResult; map to text
    return resp.content if hasattr(resp, "content") else resp.generations[0][0].text
