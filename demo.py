import os
from dotenv import load_dotenv

# 从 .env 加载两个 API Key（不写死在代码里，防止泄露到 GitHub）
load_dotenv()

from langchain_community.embeddings import DashScopeEmbeddings
from langchain_openai import ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

# 1. 读入文本
text = open("data/rules.txt", encoding="utf-8").read()

# 2. 切分：每 500 字一块、块间重叠 50 字，避免关键信息被切断
docs = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50).split_text(text)

# 3. 向量化并存入 Chroma（本地向量库）
emb = DashScopeEmbeddings(model="text-embedding-v2")
store = Chroma.from_texts(docs, emb)
retriever = store.as_retriever(search_kwargs={"k": 3})

# 4. 大模型：DeepSeek（兼容 OpenAI 接口）
llm = ChatOpenAI(model="deepseek-chat", temperature=0)

# 5. 提示词：只根据资料回答，资料没有就说不知道
prompt = ChatPromptTemplate.from_template(
    "只根据下面的资料回答问题，资料里没有就说不知道。\n资料：{context}\n问题：{question}"
)

# 6. 把「检索 + 提示词 + 模型 + 输出」串成一条链
chain = {"context": retriever, "question": RunnablePassthrough()} | prompt | llm | StrOutputParser()

# 7. 问一个问题试试
print(chain.invoke("奖学金怎么评定？"))
