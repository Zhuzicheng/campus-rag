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

# 清洗 PDF 提取的脏文本：去掉字间空格、合并多余换行
def clean_text(text):
    text = re.sub(r"[ \t\u3000]+", "", text)  # 去掉半角/全角空格、制表符
    text = re.sub(r"\n+", "\n", text)          # 合并连续换行
    return text

# 1. 读取 data/pdfs 下所有 PDF 的文本
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
        print(f"已加载 {fname}")
    return "\n\n".join(all_text)

text = load_pdfs()

# 2. 切分
docs = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50).split_text(text)

# 3. 向量化并存入 Chroma
emb = DashScopeEmbeddings(model="text-embedding-v2")
store = Chroma.from_texts(docs, emb)
retriever = store.as_retriever(search_kwargs={"k": 3})

# 4. 大模型
llm = ChatOpenAI(model="deepseek-chat", temperature=0)

# 5. 提示词
prompt = ChatPromptTemplate.from_template(
    "只根据下面的资料回答问题，资料里没有就说不知道。\n资料：{context}\n问题：{question}"
)

# 6. 串成链
chain = {"context": retriever, "question": RunnablePassthrough()} | prompt | llm | StrOutputParser()

# 7. 问答
print(chain.invoke("国家奖学金的奖励标准是多少？"))
