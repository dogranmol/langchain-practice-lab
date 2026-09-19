import os
from dotenv import load_dotenv
from langchain_nvidia_ai_endpoints import ChatNVIDIA

load_dotenv() 


llm = ChatNVIDIA(
  model="openai/gpt-oss-20b",
  api_key=os.getenv("NVIDIA_CHAT_API_KEY"),
  temperature=1,
  top_p=1,
  max_completion_tokens=4096,
)
