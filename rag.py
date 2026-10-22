import os
import re
from dotenv import load_dotenv
load_dotenv()

from pypdf import PdfReader
from langchain_community.embeddings import DashScopeEmbeddings
from langchain_openai import ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

# 清洗脏文本
def clean_text(text):
    text = re.sub(r"[ \t\u3000]+", "", text)
    text = re.sub(r"\n+", "\n", text)
    return text

# 读取 data/pdfs 下所有 PDF
def load_pdfs(folder="data/pdfs"):
    all_text = []
    for fname in sorted(os.listdir(folder)):
        if not fname.endswith(".pdf"):
            continue
        reader = PdfReader(os.path.join(folder, fname))
        for page in reader.pages:
            t = page.extract_text()
            if t:
                all_text.append(clean_text(t))
        print(f"已加载 PDF: {fname}")
    return "\n\n".join(all_text)

# 读取 data/txts 下所有 TXT（官网网页转录的文本数据）
def load_txts(folder="data/txts"):
    all_text = []
    for fname in sorted(os.listdir(folder)):
        if not fname.endswith(".txt"):
            continue
        t = open(os.path.join(folder, fname), encoding="utf-8").read()
        if t:
            all_text.append(clean_text(t))
        print(f"已加载 TXT: {fname}")
    return "\n\n".join(all_text)

# 构建「检索 + 生成」问答链
def build_chain():
    text = load_pdfs() + "\n\n" + load_txts()
    docs = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50).split_text(text)
    emb = DashScopeEmbeddings(model="text-embedding-v2")
    store = Chroma.from_texts(docs, emb)
    retriever = store.as_retriever(search_kwargs={"k": 3})
    llm = ChatOpenAI(model="deepseek-chat", temperature=0)
    prompt = ChatPromptTemplate.from_template(
        "只根据下面的资料回答问题，资料里没有就说不知道。\n资料：{context}\n问题：{question}"
    )
    return {"context": retriever, "question": RunnablePassthrough()} | prompt | llm | StrOutputParser()

# 启动时初始化一次
chain = build_chain()

# 对外统一问答入口
def ask(question):
    return chain.invoke(question)
