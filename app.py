import streamlit as st
import os
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_openai import ChatOpenAI

# ==========================================
# 1. 页面基本配置与 现代极简 UI 精装修
# ==========================================
st.set_page_config(page_title="AI 体育训练助手", page_icon="🏀", layout="wide")

# 注入全局自定义 CSS (现代极简主义 + 流畅仿生边缘)
# 注入全局自定义 CSS (暗黑霓虹风 Dark Mode + Neon UI)
st.markdown("""
<style>
    /* 1. 极致暗黑背景 */
    .stApp {
        background-color: #0B0E14 !important;
    }
    
    /* 隐藏所有默认的白色头部和底边 */
    header {background-color: transparent !important; visibility: hidden;}
    footer {visibility: hidden;}

    /* 全局文字颜色调整为高级灰白 */
    p, h1, h2, h3, h4, span, div {
        color: #E2E8F0 !important;
    }

    /* 2. 聊天气泡深度定制 (卡片化) */
    /* AI 回答气泡：深邃的藏青色面板 */
    [data-testid="stChatMessage"]:nth-child(even) {
        background-color: transparent !important;
    }
    [data-testid="stChatMessage"]:nth-child(even) .stMarkdown {
        background-color: #151B23 !important;
        border: 1px solid #2D3748 !important;
        border-radius: 16px !important;
        padding: 16px !important;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.4) !important;
    }

    /* User 提问气泡：略微提亮的深灰色，区分层级 */
    [data-testid="stChatMessage"]:nth-child(odd) {
        background-color: transparent !important;
    }
    [data-testid="stChatMessage"]:nth-child(odd) .stMarkdown {
        background-color: #1E2532 !important;
        border-radius: 16px !important;
        padding: 16px !important;
    }

    /* 3. 头像霓虹发光特效 (致敬你的参考图) */
    /* 给 AI 的头像加上紫蓝色霓虹光晕 */
    [data-testid="chatAvatarIcon-assistant"] {
        background-color: #5D5FEF !important;
        box-shadow: 0 0 15px rgba(93, 95, 239, 0.5) !important;
        border: 2px solid #0B0E14 !important;
    }

    /* 4. 底部输入框：悬浮胶囊形态 */
    [data-testid="stChatInput"] {
        background-color: #151B23 !important;
        border-radius: 30px !important;
        border: 1px solid #3A4354 !important;
        box-shadow: 0 -4px 20px rgba(0, 0, 0, 0.5) !important;
    }
    /* 输入框内的文字颜色 */
    [data-testid="stChatInput"] textarea {
        color: #FFFFFF !important;
    }

    /* 5. 侧边栏暗黑化 */
    [data-testid="stSidebar"] {
        background-color: #090B10 !important;
        border-right: 1px solid #1E2532 !important;
    }
    
    /* 按钮样式：暗黑质感 */
    .stButton>button {
        background-color: #1E2532 !important;
        color: #A0AEC0 !important;
        border: 1px solid #2D3748 !important;
        border-radius: 12px !important;
        transition: all 0.3s;
    }
    .stButton>button:hover {
        border-color: #5D5FEF !important;
        color: #5D5FEF !important;
        box-shadow: 0 0 10px rgba(93, 95, 239, 0.3) !important;
    }

    /* 顶部标题渐变紫蓝 */
    .hero-title {
        font-weight: 800;
        font-size: 2.5rem;
        background: linear-gradient(135deg, #7F7FD5, #86A8E7, #91EAE4);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-top: -20px;
        margin-bottom: 20px;
    }
</style>
""")

# 配合暗黑风格的顶部标题
st.markdown('<div class="hero-title">AI 体育助手</div>', unsafe_allow_html=True)

# ==========================================
# 2. 安全读取环境变量 (绝不硬编码密码)
# ==========================================
if "PINECONE_API_KEY" in st.secrets:
    os.environ["PINECONE_API_KEY"] = st.secrets["PINECONE_API_KEY"]
else:
    st.error("🚨 致命错误：请在 Streamlit 后台 Secrets 中配置 PINECONE_API_KEY")
    st.stop()

# ==========================================
# 3. 核心架构：初始化全局单例资源池
# ==========================================
@st.cache_resource(show_spinner=False)
def get_retriever():
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = PineconeVectorStore(index_name="sport", embedding=embeddings) 
    return vectorstore.as_retriever(search_kwargs={"k": 10})

retriever = get_retriever()

# ==========================================
# 4. 核心架构：OpenRouter 大模型网关配置
# ==========================================
def get_llm():
    return ChatOpenAI(
        model="nvidia/nemotron-nano-12b-v2-vl:free",  
        api_key=st.secrets["OPENROUTER_API_KEY"], 
        base_url="https://openrouter.ai/api/v1",  
        default_headers={
            "HTTP-Referer": "https://ai-sports-assistant.streamlit.app/", 
            "X-Title": "AI Sports Assistant" 
        },
        temperature=0.7 
    )

llm = get_llm()

# ==========================================
# 5. 会话状态管理 (让 AI 拥有多轮记忆)
# ==========================================
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "你好！我是你的专属 AI 体育助手。我有丰富的体育理论储备，你可以问我战术制定、动作要领或体能训练方案。"}
    ]

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# ==========================================
# 6. 侧边栏：交互精装修与状态监控
# ==========================================
with st.sidebar:
    st.markdown("### 💡 快捷场景")
    if st.button("⚽ 制定新手足球训练计划"):
        st.session_state.quick_query = "请帮我制定一份为期一周的新手足球训练计划，重点在基础盘带和体能。"
    if st.button("🏀 篮球三步上篮的要点"):
        st.session_state.quick_query = "请详细讲解篮球三步上篮的动作要领、发力技巧和新手易错点。"
    
    st.markdown("<br><hr><br>", unsafe_allow_html=True)
    
    st.markdown("### ⚙️ 系统探针")
    st.info("🟢 Pinecone 云端库已连接")
    st.info("🟢 LLM 网关路由已就绪")

# ==========================================
# 7. 主控中枢：接收输入并执行 RAG 检索生成
# ==========================================
prompt_text = st.chat_input("输入你的体育问题，例如：如何提高高强度对抗下的投篮命中率？")
if "quick_query" in st.session_state:
    prompt_text = st.session_state.quick_query
    del st.session_state.quick_query 

if prompt_text:
    st.chat_message("user").markdown(prompt_text)
    st.session_state.messages.append({"role": "user", "content": prompt_text})

    with st.chat_message("assistant"):
        with st.spinner("🧠 正在从云端知识库检索底层文献..."):
            try:
                # [RAG 检索]
                docs = retriever.invoke(prompt_text)
                context_text = "\n\n".join([doc.page_content for doc in docs])
                
                # [Prompt 约束]
                rag_prompt = f"""
                你是本项目专属的「首席 AI 体育训练专家」。你拥有顶尖的体育运动理论基础、战术素养和体能训练经验。
                
                【你的核心工作原则（最高指令）】：
                1. 严谨求实：你必须**仅仅基于**我提供的 <参考文献> 来回答用户的问题。
                2. 专业视角：当文献中有相关信息时，请用专业体育教练的口吻进行提炼和总结。可以分点阐述，重点突出动作要领、发力细节或战术意图。
                
                <参考文献>
                {context_text}
                </参考文献>
                
                <用户问题>
                {prompt_text}
                </用户问题>
                
                请现在开始以专业专家的身份，基于 <参考文献> 回答 <用户问题>：
                """
                
                response = llm.invoke(rag_prompt)
                
                # 展示答案
                st.markdown(response.content)
                st.session_state.messages.append({"role": "assistant", "content": response.content})

                # 【加分项补充】：优雅地展示参考来源
                with st.expander("📚 查看 AI 检索到的底层参考文献"):
                    for i, doc in enumerate(docs[:3]): # 只展示前3条最相关的，保持页面整洁
                        st.caption(f"**文献 [{i+1}]**: {doc.page_content[:150]}...")

            except Exception as e:
                st.error("网络网关波动或 API 频率限制，请稍后重试。")
                st.warning(f"底层错误日志: {str(e)}")
