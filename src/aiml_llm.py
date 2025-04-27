# src/aiml_llm.py
from langchain_core.language_models import BaseChatModel
from openai import OpenAI

class AIMLLLM(BaseChatModel):
    def __init__(self, api_key, temperature=0.4, max_tokens=500):
        self.client = OpenAI(
            base_url="https://api.aimlapi.com/v1",
            api_key=api_key
        )
        self.temperature = temperature
        self.max_tokens = max_tokens

    def _call(self, messages, **kwargs):
        response = self.client.chat.completions.create(
            model="gpt-4o",  # fixed model inside call
            messages=messages,
            temperature=self.temperature,
            max_tokens=self.max_tokens
        )
        return response.choices[0].message.content

    @property
    def _llm_type(self) -> str:
        return "aiml"
