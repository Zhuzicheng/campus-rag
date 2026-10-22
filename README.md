# campus-rag 校园知识库问答助手

基于 **RAG（检索增强生成）** 的垂直领域问答系统：把山东理工大学的规章制度文档变成「能用自然语言查询」的知识库，回答学分、奖学金、请假流程等校规问题。

> 核心亮点：通过「向量检索 + BM25 关键词检索 + RRF 融合」的混合检索方案，将 20 题评测集准确率从 **80% 提升到 100%**。

## ✨ 功能特性

- 🔍 **混合检索**：向量检索（语义）+ BM25 关键词检索（字面）双路融合，召回更准
- 💬 **自然语言问答**：约束模型仅基于资料作答，资料没有就诚实说「不知道」，不瞎编
- 🖥️ **三种使用方式**：命令行 / FastAPI 接口 / Gradio 网页
- 📊 **真实评测**：自建 20 题评测集，人工判分，量化指标可复现

## 🛠️ 技术栈

| 环节 | 选型 |
|---|---|
| 语言 | Python |
| 编排 | LangChain |
| 生成 | DeepSeek（deepseek-chat，兼容 OpenAI 接口） |
| 向量化 | 阿里云百炼 DashScope（text-embedding-v2） |
| 向量库 | Chroma |
| 关键词检索 | BM25 + jieba 分词 |
| 后端 | FastAPI |
| 界面 | Gradio |

## 🏗️ 项目架构

```
┌─────────────┐   ┌─────────────┐
│  ui.py      │   │  app.py     │   ← 两个入口（Gradio 网页 / FastAPI 接口）
│  (Gradio)   │   │  (FastAPI)  │
└──────┬──────┘   └──────┬──────┘
       └────────┬────────┘
                ▼
        ┌───────────────┐
        │   rag.py      │   ← RAG 核心：加载 → 清洗 → 切分 → 混合检索 → 生成
        └──────┬────────┘
               ▼
   ┌──────────────────────────┐
   │ data/pdfs（PDF）+ data/txts（TXT）│   ← 学校官网公开规章制度
   └──────────────────────────┘
```

## 🚀 快速开始

### 1. 安装依赖

```bash
pip install langchain langchain-community langchain-openai langchain-text-splitters \
    langchain-core chromadb gradio dashscope python-dotenv pypdf fastapi uvicorn jieba rank-bm25
```

### 2. 配置 API Key

在项目根目录创建 `.env` 文件（已被 `.gitignore` 忽略）：

```
OPENAI_API_KEY=你的DeepSeek密钥
OPENAI_BASE_URL=https://api.deepseek.com
DASHSCOPE_API_KEY=你的DashScope密钥
```

### 3. 运行

```bash
# 方式一：Gradio 网页（推荐演示）
python ui.py          # 打开 http://127.0.0.1:7860

# 方式二：FastAPI 接口
python -m uvicorn app:app   # 打开 http://127.0.0.1:8000/docs

# 方式三：命令行
python -c "from rag import ask; print(ask('国家奖学金的奖励标准是多少？'))"
```

## 📊 评测结果

自建 20 题评测集（`eval.py` + `eval_results.json`），覆盖奖学金、奖励办法、学籍管理、请假销假四类校规，人工判分：

| 阶段 | 检索方案 | 准确率 |
|---|---|---|
| W2 | 纯向量检索 | 80%（16/20） |
| W3 | 向量 + BM25 混合检索 | **100%（20/20）** |

评测命令：`python eval.py`

## 📁 核心文件

- `rag.py` —— RAG 核心（加载、清洗、切分、混合检索、生成）
- `app.py` —— FastAPI 后端（`POST /ask` 接口）
- `ui.py` —— Gradio 网页界面
- `eval.py` / `eval_results.json` —— 20 题评测集与结果
- `data/pdfs/` —— 学校官网公开 PDF（奖励办法、奖学金、学籍管理）
- `data/txts/` —— 官网网页转录文本（计算机学院请销假制度）

## 📝 数据来源

数据均为山东理工大学官网公开文件（`sdut.edu.cn`），非虚构：

1. 《山东理工大学本科学生奖励办法》
2. 《山东理工大学本专科生国家奖学金、省政府奖学金管理办法》
3. 《山东理工大学学生学籍管理规定》
4. 《计算机科学与技术学院学生请销假制度》（官网网页转录）

## 项目地址

GitHub：https://github.com/Zhuzicheng/campus-rag
