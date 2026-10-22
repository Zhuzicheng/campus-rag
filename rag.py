import os
import re
import jieba
from dotenv import load_dotenv
load_dotenv()

from pypdf import PdfReader
from rank_bm25 import BM25Okapi
from langchain_community.embeddings import DashScopeEmbeddings
from langchain_openai import ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.documents import Document

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

# 读取 data/txts 下所有 TXT
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

# 构建「混合检索 + 生成」问答链
def build_chain():
    text = load_pdfs() + "\n\n" + load_txts()
    docs = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50).split_text(text)

    # ① 向量检索（语义）：DashScope embedding + Chroma
    emb = DashScopeEmbeddings(model="text-embedding-v2")
    store = Chroma.from_texts(docs, emb)

    # ② 关键词检索（字面）：BM25 + jieba 分词
    tokenized_docs = [list(jieba.cut(d)) for d in docs]
    bm25 = BM25Okapi(tokenized_docs)

    # ③ 混合检索：向量 + BM25，用 RRF 融合排序
    def hybrid_retrieve(query, k=5):
        # 向量检索 top-k
        vec_docs = store.similarity_search(query, k=k)
        # BM25 检索 top-k
        q_tokens = list(jieba.cut(query))
        scores = bm25.get_scores(q_tokens)
        bm25_idx = sorted(range(len(scores)), key=lambda i: -scores[i])[:k]
        # RRF 融合：每个结果按「排名倒数」加权求和
        fused = {}
        for rank, d in enumerate(vec_docs):
            fused[d.page_content] = fused.get(d.page_content, 0) + 1.0 / (rank + 60)
        for rank, i in enumerate(bm25_idx):
            fused[docs[i]] = fused.get(docs[i], 0) + 1.0 / (rank + 60)
        top = sorted(fused.items(), key=lambda x: -x[1])[:k]
        return [Document(page_content=c) for c, _ in top]

    llm = ChatOpenAI(model="deepseek-chat", temperature=0)
    prompt = ChatPromptTemplate.from_template(
        "只根据下面的资料回答问题，资料里没有就说不知道。\n资料：{context}\n问题：{question}"
    )
    chain = {"context": RunnableLambda(hybrid_retrieve), "question": RunnablePassthrough()} | prompt | llm | StrOutputParser()
    return chain

# 启动时初始化一次
chain = build_chain()

# 对外统一问答入口
def ask(question):
    return chain.invoke(question)
