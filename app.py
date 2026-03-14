import streamlit as st
import os
from langchain_pinecone import PineconeVectorStore
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

# === 1. 隐身术：读取云端秘钥 ===
os.environ["PINECONE_API_KEY"] = st.secrets["PINECONE_API_KEY"]
api_key = st.secrets["OPENROUTER_API_KEY"]
base_url = "https://openrouter.ai/api/v1"

# === 2. 页面 UI 初始化 ===
st.title("🏀 专属 AI 体育教练 (云端记忆版)")
st.divider()  # 你看，这行现在紧贴左边了，绝对不会报错！

# === 3. 侧边栏：上传与投喂 ===
with st.sidebar:
    st.header("📚 云端知识库投喂")
    uploaded_file = st.file_uploader("上传体育资料 (PDF)", type="pdf")
    
    if uploaded_file and st.button("🚀 永久写入云端大脑"):
        with st.spinner("正在切分资料并同步至 Pinecone..."):
            with open("temp.pdf", "wb") as f:
                f.write(uploaded_file.getvalue())
                
            loader = PyPDFLoader("temp.pdf")
            docs = loader.load()
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
            splits = text_splitter.split_documents(docs)
            embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
            
            # 写入云端！(确保你建的索引名字叫 sport)
            PineconeVectorStore.from_documents(splits, embeddings, index_name="sport")
            st.success("🎉 资料已永久保存！重启网页也不会失忆！")
            
            if os.path.exists("temp.pdf"):
                os.remove("temp.pdf")

# === 4. 连接云端大脑与大模型 ===
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vectorstore = PineconeVectorStore(index_name="sport", embedding=embeddings)
retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

llm = ChatOpenAI(
    model="nvidia/nemotron-3-super-120b-a12b:free", # 用了免费模型测试
    api_key=api_key,
    base_url=base_url,
    temperature=0.3
)

# ... 下面保留你原本的 Prompt 和对话逻辑 (st.chat_message 等) ...
# ================= 4. 聊天界面与记忆逻辑 =================
# 渲染历史聊天记录（把之前说过的话显示在屏幕上）
# 初始化聊天记录（插上记忆卡）
if "messages" not in st.session_state:
    st.session_state.messages = []
# ==========================================
# 下面是 app.py 的最后一部分：聊天界面与对话逻辑
# ==========================================

# 1. 每次刷新页面时，先把之前的聊天记录打印出来
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# 2. 接收学员（你）的提问
question = st.chat_input("向教练提问...")

# 3. 核心大闸门：只要你发了问题，它才开始运转
if question:
    # 先把你的问题打印到屏幕上，并记入记忆卡
    with st.chat_message("user"):
        st.markdown(question)
    st.session_state.messages.append({"role": "user", "content": question})

    # AI 教练开始表演
    with st.chat_message("assistant"):
        with st.spinner("教练正在翻阅云端战术板..."):
            # A. 从 Pinecone 云端大脑提取和你问题相关的体育资料
            docs = retriever.invoke(question)
            context = "\n".join([doc.page_content for doc in docs])
            
            # B. 组装提示词（让大模型带上资料来回答）
            prompt = f"""你是一个专业的体育教练。请根据以下参考资料回答学员的问题。
            参考资料：\n{context}\n\n
            学员问题：{question}"""
            
            # C. 呼叫大模型去思考
            response = llm.invoke(prompt)
            
            # D. 最关键的一步！让它把肚子里的答案打印到网页上！
            st.markdown(response.content)
            
    # E. 把教练的回答存入记忆卡，方便进行多轮对话
    st.session_state.messages.append({"role": "assistant", "content": response.content})
            
            # ... 把你原本后面的回答逻辑贴在这里（记得也要保持缩进） ...
    if not api_key:
        st.error("请先在左侧输入 API Key！")
        st.stop()
        # 初始化云端大脑记忆卡（既然换了云端，它就永远存在了）
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = vectorstore
    if not st.session_state.vectorstore:
        st.error("请先上传PDF并构建知识库！")
        st.stop()

    # 1. 把用户的问题显示出来并存入记忆
    with st.chat_message("user"):
        st.write(question)
    st.session_state.messages.append({"role": "user", "content": question})

    # 2. 从知识库中检索相关内容
    retriever = st.session_state.vectorstore.as_retriever(search_kwargs={"k": 4})
    docs = retriever.invoke(question)
    context = "\n\n".join(doc.page_content for doc in docs)

    # 3. 把“历史记忆”格式化成字符串
    history_str = ""
    # 只取最近的 4 条对话，防止记忆太长把大模型撑爆
    for msg in st.session_state.messages[-4:-1]: 
        role = "用户" if msg["role"] == "user" else "教练"
        history_str += f"{role}: {msg['content']}\n"

    # 4. 组装终极 Prompt（包含规则、资料、记忆、新问题）
    # 
    final_prompt = f"""你是一位专业的运动训练教练。请严格基于以下【参考资料】回答。
    如果资料中没有，请结合你的专业知识回答，并说明“此部分非资料原文”。
    
    【参考资料】：
    {context}
    
    【历史聊天记录】：
    {history_str}
    
    【用户最新问题】：{question}
    """

    # 5. 调用云端大模型 API
    llm = ChatOpenAI(
        model="nvidia/nemotron-3-super-120b-a12b:free", # 如果用其他厂商，这里改成对应的模型名字
        api_key=api_key,
        base_url=base_url,
        temperature=0.3
    )

    with st.chat_message("assistant"):
        with st.spinner("教练回忆中..."):
            try:
                response = llm.invoke(final_prompt)
                answer = response.content
                st.write(answer)
                # 把 AI 的回答也存入记忆
                st.session_state.messages.append({"role": "assistant", "content": answer})
            except Exception as e:
                st.error(f"API 调用失败，请检查 Key 或网络: {e}")