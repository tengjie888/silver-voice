import streamlit as st
import dashscope
from dashscope.audio.asr import Recognition  # <--- 核心修改：换成这个服务
from dashscope import Generation
import os  # <--- 修复：之前漏了这个库
import json

# =================配置区=================
# 自动适配：优先读 Streamlit Secrets，读不到就用下面的硬编码
# 如果您还没配置 Secrets，请直接修改下面的 sk-xxx 为您的真实 Key
if "DASHSCOPE_API_KEY" in st.secrets:
    API_KEY = st.secrets["DASHSCOPE_API_KEY"]
else:
    API_KEY = "sk-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx" # 👈 请确保这里替换了您的Key

dashscope.api_key = API_KEY
# =======================================

st.set_page_config(page_title="银龄知音·手机版", page_icon="📱", layout="centered")

# CSS 样式：让界面更像 App
st.markdown("""
    <style>
    /* 隐藏 Streamlit 默认菜单 */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    .stDeployButton {display:none;}
    header {visibility: hidden;}

    /* 按钮样式 */
    .stButton>button {
        height: 3.5em; 
        width: 100%; 
        font-size: 22px !important; 
        border-radius: 25px; 
        background-color: #FF5733; 
        color: white;
        box-shadow: 0px 4px 6px rgba(0,0,0,0.1);
        border: none;
    }
    
    /* 聊天气泡 */
    .chat-bubble {
        background: #ffffff; 
        padding: 18px; 
        border-radius: 18px; 
        margin-top: 15px; 
        box-shadow: 0px 2px 5px rgba(0,0,0,0.05);
        font-size: 18px;
    }
    
    .stApp { background-color: #F7F7F7; }
    </style>
""", unsafe_allow_html=True)

st.title("👵 银龄知音")
st.write("请按下方红色按钮说话，说完点击停止")

# 初始化历史记录
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

# --- 核心功能区 ---
audio_value = st.audio_input("点此开始录音")

if audio_value:
    st.info("正在听您说...") # 状态提示优化
    
    try:
        # 1. 保存临时文件
        with open("temp_audio.wav", "wb") as f:
            f.write(audio_value.getvalue())
        
        # 2. 调用阿里云语音识别 (Recognition接口)
        # 这个接口支持直接上传本地文件，特别适合 Streamlit Cloud 环境
        rec_response = Recognition.call(
            model='paraformer-realtime-v1',
            file='temp_audio.wav',
            language_hints=['zh'],
            format='wav'
        )
        
        # 3. 解析语音结果
        user_text = ""
        if rec_response.status_code == 200:
            # 提取识别出的文字
            if 'sentences' in rec_response.output:
                 # 拼接所有句子
                user_text = "".join([s['text'] for s in rec_response.output['sentences']])
            else:
                # 容错：如果结构不同，尝试直接取 text
                user_text = rec_response.output.get('text', '')
        else:
            st.error(f"语音识别出错: {rec_response.message}")

        # 4. 如果识别成功，调用大模型
        if user_text:
            # 存个状态防止刷新丢失
            st.session_state.last_user_text = user_text
            
            # 判断意图 (简单的关键词分类)
            system_prompt = "你是一个温暖的老年人陪伴助手，请简短、亲切地回复。"
            if any(k in user_text for k in ["查", "问", "怎么", "哪里", "医生", "药"]):
                system_prompt = "你是一个生活助手，请直接给出简单的办事建议，不要废话。"
            
            messages = [
                {'role': 'system', 'content': system_prompt},
                {'role': 'user', 'content': user_text}
            ]
            
            # 调用通义千问
            llm_resp = Generation.call(
                api_key=API_KEY,
                model="qwen-turbo",
                messages=messages,
                result_format='message'
            )
            
            if llm_resp.status_code == 200:
                reply = llm_resp.output.choices[0].message.content
                
                # 存入历史 (插到最前面)
                st.session_state.chat_history.insert(0, {"role": "bot", "content": reply})
                st.session_state.chat_history.insert(0, {"role": "user", "content": user_text})
                
                # 强制刷新一下页面显示新消息
                st.rerun()
            else:
                st.error("知音正在思考，请稍后再试")
        elif rec_response.status_code == 200:
            st.warning("好像没听清，请大声一点再说一次~")
            
    except Exception as e:
        st.error(f"发生错误: {e}")

# --- 展示对话流 ---
st.markdown("---")
for chat in st.session_state.chat_history:
    if chat["role"] == "user":
        st.markdown(f"**👴 您说：** {chat['content']}")
    else:
        st.markdown(f"<div class='chat-bubble'>🤖 <b>知音：</b>{chat['content']}</div>", unsafe_allow_html=True)
