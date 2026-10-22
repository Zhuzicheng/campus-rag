from fastapi import FastAPI
from pydantic import BaseModel
from rag import ask

app = FastAPI(title="校园知识库问答助手")

# 请求体：{"question": "..."}
class Question(BaseModel):
    question: str

# 问答接口
@app.post("/ask")
def ask_endpoint(q: Question):
    answer = ask(q.question)
    return {"question": q.question, "answer": answer}

# 健康检查
@app.get("/")
def root():
    return {"message": "campus-rag 服务运行中，POST /ask 可提问"}
