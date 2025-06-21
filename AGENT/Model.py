import ollama
import re
import logging

# Set up logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class OllamaChatService:
    def __init__(self, model_name="deepseek-r1:8b"):
        self.model_name = model_name

    def clean_response(self, text: str) -> str:
        # Remove all <think>...</think> blocks (including multiline)
        cleaned = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
        # Replace multiple blank lines with just two newlines
        cleaned = re.sub(r"\n\s*\n", "\n\n", cleaned)
        return cleaned.strip()

    def chat(self, message: str) -> str:
        try:
            response = ollama.chat(
                model=self.model_name,
                messages=[{"role": "user", "content": message}]
            )
            if hasattr(response, "message") and hasattr(response.message, "content"):
                raw = response.message.content
                return self.clean_response(raw)
            else:
                raise ValueError(f"Unexpected response format: {type(response)}")
        except Exception as e:
            logger.error(f"Error during Ollama chat: {str(e)}")
            raise

    def chat_from_history(self, messages: list[dict]) -> str:
        try:
            response = ollama.chat(
                model=self.model_name,
                messages=messages
            )
            if hasattr(response, "message") and hasattr(response.message, "content"):
                raw = response.message.content
                return self.clean_response(raw)
            else:
                raise ValueError(f"Unexpected response format: {type(response)}")
        except Exception as e:
            logger.error(f"Error during chat_from_history: {str(e)}")
            raise
