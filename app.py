import streamlit as st
from langchain.agents import initialize_agent, Tool
from langchain_openai import ChatOpenAI  # 使用 ChatOpenAI 统一调用智谱
from langchain_community.tools import WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper
from langchain.tools import BaseTool  # 必须导入 BaseTool
import math
import os
from dotenv import load_dotenv

load_dotenv()

# 自定义计算器工具（与 agent.py 保持一致，带类型注解）
class CalculatorTool(BaseTool):
    name: str = "Calculator"
    description: str = "用于执行数学计算，输入应为数学表达式，例如 '2+2'、'sqrt(16)' 或 'sin(pi/2)'。"

    def _run(self, query: str) -> str:
        # 清理输入（与 agent.py 相同）
        raw = query
        query_clean = raw.strip().split('\n')[0]
        if 'Observation' in query_clean:
            query_clean = query_clean.split('Observation')[0].strip()
        if (query_clean.startswith("'") and query_clean.endswith("'")) or \
           (query_clean.startswith('"') and query_clean.endswith('"')):
            query_clean = query_clean[1:-1].strip()
        if not query_clean:
            return "计算错误：表达式为空"
        try:
            import math
            safe_dict = {name: getattr(math, name) for name in dir(math) if not name.startswith('_')}
            safe_dict['abs'] = abs
            result = eval(query_clean, {"__builtins__": {}}, safe_dict)
            return f"计算结果：{result}"
        except SyntaxError as e:
            return f"计算错误：表达式语法有误 ({e})。请检查输入，例如 'sqrt(16)'、'2+2'。"
        except Exception as e:
            return f"计算错误：{str(e)}"

# 维基百科工具
wikipedia = WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper())
tools = [
    Tool(name="Wikipedia", func=wikipedia.run, description="当需要查询人物、事件、概念等知识时，可以使用此工具。输入关键词。"),
    CalculatorTool()
]

# 初始化 LLM（使用环境变量自动配置智谱AI）
# 从环境变量读取
api_key = os.getenv("OPENAI_API_KEY")
api_base = os.getenv("OPENAI_API_BASE", "https://open.bigmodel.cn/api/paas/v4/")

if not api_key:
    st.error("未找到 OPENAI_API_KEY 环境变量，请在 .env 文件或 Streamlit Secrets 中设置")
    st.stop()

llm = ChatOpenAI(
    temperature=0,
    model="glm-4",  # 或 glm-3-turbo
    openai_api_key=api_key,
    openai_api_base=api_base,
    max_tokens=1024
)

# 初始化 Agent
@st.cache_resource
def get_agent():
    return initialize_agent(
        tools,
        llm,
        agent="zero-shot-react-description",
        verbose=True,
        handle_parsing_errors=True,
        max_iterations=5,
        early_stopping_method="generate"
    )

agent = get_agent()

# Streamlit 界面
st.set_page_config(page_title="多功能知识助手", page_icon="🤖")
st.title("🤖 多功能知识助手")
st.write("这是一个能使用计算器和维基百科的AI助手。")

# 初始化对话历史
if "messages" not in st.session_state:
    st.session_state.messages = []

# 显示历史消息
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# 用户输入
if prompt := st.chat_input("请输入你的问题"):
    # 显示用户消息
    with st.chat_message("user"):
        st.markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    # 调用 Agent
    with st.chat_message("assistant"):
        with st.spinner("思考中..."):
            response = agent.run(prompt)
        st.markdown(response)
    st.session_state.messages.append({"role": "assistant", "content": response})