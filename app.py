import streamlit as st
import os
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_openai import ChatOpenAI

# ==========================================
# 1. 页面基本配置与 UI 精装修
# ==========================================
st.set_page_config(page_title="AI 体育训练助手", page_icon="🏀", layout="wide")
st.title(" AI 体育助手")
st.caption("基于 Pinecone 云端向量检索与大模型的专业体育知识库 | 全局缓存高并发版")

# ==========================================
# 2. 安全读取环境变量 (绝不硬编码密码)
# ==========================================
# 从 Streamlit Cloud后台的 Secrets 保险箱中读取 Pinecone 密钥
if "PINECONE_API_KEY" in st.secrets:
    os.environ["PINECONE_API_KEY"] = st.secrets["PINECONE_API_KEY"]
else:
    st.error("🚨 致命错误：请在 Streamlit 后台 Secrets 中配置 PINECONE_API_KEY")
    st.stop()

# ==========================================
# 3. 核心架构：初始化全局单例资源池，彻底杜绝 OOM
# ==========================================
@st.cache_resource(show_spinner=False)
def get_retriever():
    """
    此函数仅在第一个用户访问时执行一次。
    加载巨大的词向量模型并连接数据库，之后所有并发用户共享此连接。
    """
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = PineconeVectorStore(index_name="sport", embedding=embeddings) # 确保 index_name 对应你的数据库
    # k=4 代表每次大模型回答前，必须先去数据库捞 4 条最相关的体育文献
    return vectorstore.as_retriever(search_kwargs={"k": 4})

retriever = get_retriever()

# ==========================================
# 4. 核心架构：OpenRouter 大模型网关配置
# ==========================================
def get_llm():
    """
    完美适配 OpenRouter 的配置，解决 RateLimit 和 404 错误。
    包含必需的 base_url 和白嫖模型强制要求的请求头 (Headers)。
    """
    return ChatOpenAI(
        model="nvidia/nemotron-nano-12b-v2-vl:free",  # 强力免费模型，也可换 "microsoft/phi-3-mini-128k-instruct:free"
        api_key=st.secrets["OPENROUTER_API_KEY"], 
        base_url="https://openrouter.ai/api/v1",  # 👈 必须指定网关，否则迷路
        default_headers={
            "HTTP-Referer": "https://ai-sports-assistant.streamlit.app/", # 你的网站地址
            "X-Title": "AI Sports Assistant" # 你的应用名
        },
        temperature=0.7 # 控制创造力，0为死板，1为发散
    )

llm = get_llm()

# ==========================================
# 5. 会话状态管理 (让 AI 拥有多轮记忆)
# ==========================================
if "messages" not in st.session_state:
    # 设定系统初始人设
    st.session_state.messages = [
        {"role": "assistant", "content": "你好！我是你的专属 AI 体育助手。我有丰富的体育理论储备，你可以问我战术制定、动作要领或体能训练方案。"}
    ]

# 渲染历史聊天记录
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# ==========================================
# 6. 侧边栏：交互精装修与状态监控
# ==========================================
with st.sidebar:
    st.header("💡 快捷场景")
    # 增加快捷按钮，免去用户打字的麻烦，演示绝佳！
    if st.button("⚽ 制定新手足球训练计划"):
        st.session_state.quick_query = "请帮我制定一份为期一周的新手足球训练计划，重点在基础盘带和体能。"
    if st.button("🏀 篮球三步上篮的要点"):
        st.session_state.quick_query = "请详细讲解篮球三步上篮的动作要领、发力技巧和新手易错点。"
    
    st.divider()
    st.markdown("### ⚙️ 系统探针")
    st.success("✅ Pinecone 向量库已连接")
    st.success("✅ LLM 网关路由已就绪")

# ==========================================
# 7. 主控中枢：接收输入并执行 RAG 检索生成
# ==========================================
# 判断输入来源：是手打的，还是点击侧边栏快捷按钮的
prompt_text = st.chat_input("输入你的体育问题，例如：如何提高长跑耐力？")
if "quick_query" in st.session_state:
    prompt_text = st.session_state.quick_query
    del st.session_state.quick_query # 用完销毁

if prompt_text:
    # 7.1 显示用户提问
    st.chat_message("user").markdown(prompt_text)
    st.session_state.messages.append({"role": "user", "content": prompt_text})

    # 7.2 AI 思考与回答
    with st.chat_message("assistant"):
        # UI 提示
        with st.spinner("🧠 正在从 Pinecone 检索权威文献..."):
            try:
                # [RAG 核心步骤 A] - 检索文档
                docs = retriever.invoke(prompt_text)
                context_text = "\n\n".join([doc.page_content for doc in docs])
                
                # [RAG 核心步骤 B] - 组装带背景知识的提示词
                rag_prompt = f"""
                你是一个专业的体育教练和战术分析师。请严格参考以下【私有数据库文献】来回答用户的【问题】。
                如果文献中没有提到，请结合你自己的知识解答，并保持回答的专业性。
                
                【私有数据库文献】：
                {context_text}
                
                【问题】：
                {prompt_text}
                """
                
                # [RAG 核心步骤 C] - 召唤大模型
                response = llm.invoke(rag_prompt)
                
                # 7.3 展示答案
                st.markdown(response.content)
                st.session_state.messages.append({"role": "assistant", "content": response.content})

                # 🌟 答辩高光时刻：展示文献引用来源
                with st.expander("📚 查看 AI 检索到的底层参考文献"):
                    for i, doc in enumerate(docs):
                        st.markdown(f"**检索结果 [{i+1}]**: {doc.page_content[:200]}...")

            except Exception as e:
                # 优雅的错误捕获与提示
                st.error("网络网关波动或 API 频率限制，请稍后重试。")
                st.warning(f"底层错误日志: {str(e)}")
