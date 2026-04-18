import streamlit as st
import os
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_openai import ChatOpenAI

# ==========================================
# 1. 页面基本配置与 沉浸式暗黑交互 UI 注入
# ==========================================
st.set_page_config(page_title="AI 体育训练助手", page_icon="🏀", layout="wide")

# 注入高阶 CSS：彻底移除侧边栏样式，并强化主体交互
st.markdown("""
<style>
    /* 1. 核心布局：强制全屏并隐藏侧边栏与默认组件 */
    [data-testid="stSidebar"] { display: none !important; }
    [data-testid="stSidebarNav"] { display: none !important; }
    #MainMenu {visibility: hidden;}
    header {background-color: transparent !important; visibility: hidden;}
    footer {visibility: hidden;}
    
    .stApp {
        background-color: #05070A !important;
        background-image: radial-gradient(circle at 50% 0%, #1A1F2E 0%, #05070A 100%);
    }

    /* 2. 交互式顶部卡片 (取代侧边栏) */
    .hero-container {
        text-align: center;
        padding: 2rem 0 3rem;
    }
    .hero-title {
        font-weight: 900;
        font-size: 3.5rem;
        background: linear-gradient(135deg, #00F0FF, #5D5FEF, #FF00E5);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        letter-spacing: -1px;
    }

    /* 3. 聊天气泡：赛博卡片化设计 */
    [data-testid="stChatMessage"] {
        padding: 1.5rem !important;
        border-radius: 24px !important;
        margin-bottom: 1.5rem !important;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
    }
    
    /* AI 气泡：带蓝色呼吸边框 */
    [data-testid="stChatMessage"]:nth-child(even) {
        background-color: #0D1117 !important;
        border: 1px solid #1A202C !important;
        box-shadow: 0 4px 24px rgba(0, 0, 0, 0.6) !important;
    }
    [data-testid="stChatMessage"]:nth-child(even):hover {
        border-color: #5D5FEF !important;
        box-shadow: 0 0 20px rgba(93, 95, 239, 0.2) !important;
    }

    /* User 气泡：深空紫渐变 */
    [data-testid="stChatMessage"]:nth-child(odd) {
        background: linear-gradient(135deg, #1A1F2E, #11141D) !important;
        border: 1px solid #2D3748 !important;
    }

    /* 4. 底部输入框：浮动胶囊 */
    [data-testid="stChatInput"] {
        background-color: #161B22 !important;
        border: 1px solid #30363D !important;
        border-radius: 40px !important;
        box-shadow: 0 -10px 40px rgba(0,0,0,0.8) !important;
    }
    
    /* 5. 交互组件定制 */
    .stButton>button {
        background: #161B22 !important;
        color: #58A6FF !important;
        border: 1px solid #30363D !important;
        border-radius: 12px !important;
        font-weight: 600 !important;
        transition: all 0.2s;
    }
    .stButton>button:hover {
        background: #1F2937 !important;
        border-color: #58A6FF !important;
        transform: scale(1.02);
        box-shadow: 0 0 15px rgba(88, 166, 255, 0.3);
    }
</style>
""", unsafe_allow_html=True)

# ==========================================
# 2. 核心后端逻辑 (保持原有高性能配置)
# ==========================================
if "PINECONE_API_KEY" in st.secrets:
    os.environ["PINECONE_API_KEY"] = st.secrets["PINECONE_API_KEY"]
else:
    st.error("🚨 密钥缺失")
    st.stop()

@st.cache_resource(show_spinner=False)
def get_resources():
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = PineconeVectorStore(index_name="sport", embedding=embeddings) 
    llm = ChatOpenAI(
        # 👇 就是这里！换回你之前一直稳定使用的 Nvidia 模型
        model="nvidia/nemotron-nano-12b-v2-vl:free", 
        api_key=st.secrets["OPENROUTER_API_KEY"], 
        base_url="https://openrouter.ai/api/v1",
        default_headers={"HTTP-Referer": "http://localhost", "X-Title": "SportsApp"}
    )
    return vectorstore.as_retriever(search_kwargs={"k": 8}), llm

retriever, llm = get_resources()

# ==========================================
# 3. 页面主体渲染
# ==========================================
st.markdown('<div class="hero-container"><h1 class="title">AI 专属教练</h1><p style="color:#6B7280; letter-spacing:3px;">NEON-CORE / HYBRID RAG ENGINE</p></div>', unsafe_allow_html=True)

# 顶部交互式快捷场景 (取代侧边栏)
col1, col2, col3 = st.columns(3)
with col1:
    if st.button("🔥 制定全能体能方案"):
        st.session_state.temp_query = "请根据专业文献，为我制定一份针对篮球运动员的周体能训练计划。"
with col2:
    if st.button("🏀 进阶投篮细节教学"):
        st.session_state.temp_query = "详细解析库里投篮的发力链条，从脚踝到指尖的能量传递细节。"
with col3:
    if st.button("🔬 战术执行力分析"):
        st.session_state.temp_query = "在职业比赛中，如何有效应对全场紧逼防守？请给出三种破防战术。"

# 聊天记录渲染
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "核心引擎已就绪。我是你的 AI 体育专家，准备好突破极限了吗？"}]

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# 接收输入
prompt = st.chat_input("向教练下达指令...")
if "temp_query" in st.session_state:
    prompt = st.session_state.temp_query
    del st.session_state.temp_query

if prompt:
    st.chat_message("user").markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("assistant"):
        with st.spinner("⚡ 正在穿梭向量维度..."):
            try:
                docs = retriever.invoke(prompt)
                context = "\n\n".join([d.page_content for d in docs])
                final_prompt = f"你是一位顶尖教练。请基于以下文献回答：\n{context}\n问题：{prompt}"
                
                response = llm.invoke(final_prompt)
                st.markdown(response.content)
                st.session_state.messages.append({"role": "assistant", "content": response.content})
                
                with st.expander("📚 查阅底层数据流"):
                    for d in docs[:2]: st.caption(f"源片段: {d.page_content[:150]}...")
            except Exception as e:
                st.error(f"接口波动: {str(e)}")
