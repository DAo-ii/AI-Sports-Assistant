import streamlit as st
import os
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_openai import ChatOpenAI

# ==========================================
# 1. 页面基本配置与 暗黑霓虹极简 UI 注入
# ==========================================
st.set_page_config(page_title="AI 体育训练助手", page_icon="🤖", layout="wide")

# 注入全局自定义 CSS (暗黑霓虹风 Dark Mode + Neon UI)
st.markdown("""
<style>
    /* 1. 极致暗黑背景 */
    .stApp {
        background-color: #0B0E14 !important;
    }
    
    /* 隐藏所有默认的白色头部和底边 */
    #MainMenu {visibility: hidden;}
    header {background-color: transparent !important; visibility: hidden;}
    footer {visibility: hidden;}

    /* 全局文字颜色调整为高级灰白 */
    p, h1, h2, h3, h4, span, div, label {
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

    /* User 提问气泡：略微提亮的深灰色，带有一丝紫光 */
    [data-testid="stChatMessage"]:nth-child(odd) {
        background-color: transparent !important;
    }
    [data-testid="stChatMessage"]:nth-child(odd) .stMarkdown {
        background: linear-gradient(135deg, #2D275A, #1E2532) !important;
        border: 1px solid #3B316A !important;
        border-radius: 16px !important;
        padding: 16px !important;
    }

    /* 3. 头像霓虹发光特效 */
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
    
    /* 按钮样式：暗黑质感 + 悬浮霓虹发光 */
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
        transform: translateY(-2px);
    }

    /* 顶部标题渐变紫蓝 */
    .hero-title {
        font-weight: 900;
        font-size: 2.8rem;
        background: linear-gradient(135deg, #7F7FD5, #86A8E7, #91EAE4);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-top: -30px;
        margin-bottom: 5px;
        letter-spacing: 2px;
    }
    .hero-subtitle {
        text-align: center;
        color: #4B5563 !important;
        font-size: 0.9rem;
        margin-bottom: 30px;
        letter-spacing: 4px;
    }
    
    /* 展开组件 (参考文献) 深度暗黑化 */
    .streamlit-expanderHeader {
        background-color: #151B23 !important;
        color: #A0AEC0 !important;
        border-radius: 8px !important;
        border: 1px solid #2D3748 !important;
    }
    .streamlit-expanderContent {
        background-color: #0B0E14 !important;
        border: 1px solid #1E2532 !important;
        border-top: none !important;
        color: #718096 !important;
    }
</style>
""", unsafe_allow_html=True)

# 配合暗黑风格的顶部炫酷标题
st.markdown('<div class="hero-title">AI 专属教练</div>', unsafe_allow_html=True)
st.markdown('<div class="hero-subtitle">NEON-SPORTS ENGINE V1.0</div>', unsafe_allow_html=True)

# ==========================================
# 2. 安全读取环境变量
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
# 4. 核心架构：大模型网关配置
# ==========================================
def get_llm():
    return ChatOpenAI(
        model="qwen/qwen-2.5-7b-instruct:free",  # 推荐使用稳定模型
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
        {"role": "assistant", "content": "系统已激活。我是你的 AI 体育教练，随时准备为您解析战术与训练动作。"}
    ]

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# ==========================================
# 6. 侧边栏：暗黑操控台
# ==========================================
with st.sidebar:
    st.markdown("### ⚡ 战术直达")
    if st.button("⚽ 新手足球训练计划"):
        st.session_state.quick_query = "请帮我制定一份为期一周的新手足球训练计划，重点在基础盘带和体能。"
    if st.button("🏀 篮球三步上篮要点"):
        st.session_state.quick_query = "请详细讲解篮球三步上篮的动作要领、发力技巧和新手易错点。"
    
    st.markdown("<br><hr style='border-color: #1E2532;'><br>", unsafe_allow_html=True)
    
    st.markdown("### 📡 引擎状态")
    st.success("🟢 Pinecone 神经突触已连接")
    st.success("🟢 LLM 逻辑核心已在线")

# ==========================================
# 7. 主控中枢：接收输入并执行 RAG
# ==========================================
prompt_text = st.chat_input("向教练提问...")
if "quick_query" in st.session_state:
    prompt_text = st.session_state.quick_query
    del st.session_state.quick_query 

if prompt_text:
    # 7.1 显示用户提问
    st.chat_message("user").markdown(prompt_text)
    st.session_state.messages.append({"role": "user", "content": prompt_text})

    # 7.2 AI 思考与回答
    with st.chat_message("assistant"):
        with st.spinner("🌌 正在穿越向量星海检索底层文献..."):
            try:
                # 检索
                docs = retriever.invoke(prompt_text)
                context_text = "\n\n".join([doc.page_content for doc in docs])
                
                # 组装强约束 Prompt (防止幻觉)
                rag_prompt = f"""
                你是本项目专属的「首席 AI 体育训练专家」。你拥有顶尖的体育运动理论基础。
                
                【你的核心工作原则】：
                1. 优先查阅：请仔细阅读并优先依据 <参考文献> 中的内容来回答用户问题。
                2. 专业兜底：如果用户的提问在 <参考文献> 中找不到直接答案，**允许你利用自身的专业体育常识进行解答**。
                3. 诚实声明：当你没有使用 <参考文献>，而是依靠自身常识回答时，**必须在回答的最开头加上这句话**：“(注：当前私有知识库暂未收录该细节，以下基于通用专业体育知识为您解答)”。
                
                <参考文献>
                {context_text}
                </参考文献>
                
                <用户问题>
                {prompt_text}
                </用户问题>
                
                请现在开始专业解答：
                """
                
                # 召唤大模型
                response = llm.invoke(rag_prompt)
                
                # 展示答案
                st.markdown(response.content)
                st.session_state.messages.append({"role": "assistant", "content": response.content})

           

            except Exception as e:
                st.error("网络网关波动或 API 频率限制，请稍后重试。")
                st.warning(f"底层错误日志: {str(e)}")
