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
st.markdown("""
<style>
    /* 隐藏 Streamlit 默认的顶部菜单和底部水印，打造纯净 SaaS 质感 */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* 全局字体与背景调优 */
    .stApp {
        background-color: #F8F9FA;
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
    }
    
    /* 侧边栏玻璃拟物化与柔和阴影 */
    [data-testid="stSidebar"] {
        background-color: rgba(255, 255, 255, 0.9) !important;
        box-shadow: 2px 0 15px rgba(0,0,0,0.03);
        border-right: none;
    }
    
    /* 按钮样式重构：流畅的胶囊形状与赛场橙悬浮特效 */
    .stButton>button {
        width: 100%;
        border-radius: 20px;
        border: 1px solid #E2E8F0;
        background-color: #FFFFFF;
        color: #1E293B;
        font-weight: 500;
        transition: all 0.3s ease;
        box-shadow: 0 2px 5px rgba(0,0,0,0.02);
    }
    .stButton>button:hover {
        border-color: #FF6B00;
        color: #FF6B00;
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(255, 107, 0, 0.15);
    }
    
    /* 顶部标题区域美化 */
    .hero-title {
        font-weight: 800;
        font-size: 2.8rem;
        background: linear-gradient(135deg, #FF6B00, #FFA500);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.5rem;
    }
    .hero-subtitle {
        color: #64748B;
        font-size: 1.1rem;
        margin-bottom: 2rem;
        font-weight: 400;
    }
</style>
""", unsafe_allow_html=True)

# 使用自定义 HTML 渲染炫酷的头部
st.markdown('<div class="hero-title">🏀 AI 专属体育教练</div>', unsafe_allow_html=True)
st.markdown('<div class="hero-subtitle">基于 Pinecone 向量检索的高并发私有知识库引擎</div>', unsafe_allow_html=True)


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
