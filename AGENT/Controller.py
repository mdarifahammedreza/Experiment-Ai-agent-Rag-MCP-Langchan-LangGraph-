from typing import TypedDict, List, Union
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage
from Model import OllamaChatService  # Your custom chat handler

class ModelMessage(TypedDict):
    messages: List[Union[HumanMessage, AIMessage]]

class CallModel:
    service = OllamaChatService()

    @staticmethod
    def callModel(payload: ModelMessage) -> dict:
        messages: List[BaseMessage] = payload["messages"]
        ollama_messages = []

        for msg in messages:
            if isinstance(msg, HumanMessage):
                role = "user"
            elif isinstance(msg, AIMessage):
                role = "assistant"
            else:
                raise ValueError(f"Unsupported message type: {type(msg)}")
            ollama_messages.append({
                "role": role,
                "content": msg.content.strip()  # Strip whitespace just in case
            })

        # Call Ollama model
        response_text = CallModel.service.chat_from_history(ollama_messages)

        # Return AIMessage for LangGraph chain compatibility
        return {"messages": [AIMessage(content=response_text.strip())]}
