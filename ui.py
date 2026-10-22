import os
import gradio as gr
from rag import ask

def chat(question):
    q = question.strip()
    if not q:
        return "请输入一个问题"
    return ask(q)

gr.Interface(
    fn=chat,
    inputs=gr.Textbox(label="你的问题", placeholder="例如：国家奖学金的奖励标准是多少？"),
    outputs=gr.Textbox(label="回答"),
    title="校园知识库问答助手",
    description="基于山东理工大学真实规章制度的 RAG 问答（数据：奖学金、学籍、奖励办法、请假销假）"
).launch(server_name="0.0.0.0", server_port=int(os.environ.get("PORT", 7860)))
