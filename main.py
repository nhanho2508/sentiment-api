from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from langchain.chat_models import ChatOpenAI
from langchain.schema import SystemMessage, HumanMessage
import os

# Load API key từ biến môi trường (hoặc hardcode nếu muốn)
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "sk-xxx")  # <-- thay bằng key của bạn nếu cần

# Khởi tạo ChatOpenAI (endpoint tùy chỉnh như Azure hoặc inference API)
llm = ChatOpenAI(
    openai_api_base="https://models.inference.ai.azure.com",
    openai_api_key=OPENAI_API_KEY,
    model_name="gpt-4o-mini",
    temperature=1,
    max_tokens=4096,
)

# Khởi tạo FastAPI app
app = FastAPI()

# Cho phép mọi origin gọi API (CORS)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Định nghĩa request model
class SentimentRequest(BaseModel):
    text: str

# Route kiểm tra server sống
@app.get("/")
def health_check():
    return {"status": "ok"}

# Route phân tích sentiment
@app.post("/sentiment")
async def analyze_sentiment(req: SentimentRequest):
    # Prompt theo kiểu multi-message (cho LLM hiệu quả hơn)
    messages = [
        SystemMessage(content="You are a sentiment analysis expert."),
        HumanMessage(content=f"What is the sentiment of the following text? Answer with Positive, Negative, or Neutral only:\n\n\"{req.text}\""),
    ]

    # Gọi model
    response = llm.invoke(messages)

    # Trả về kết quả
    return {"sentiment": response.content.strip()}
