import streamlit as st
import dashscope
from dashscope.audio.asr import Transcription  # 使用文件转写接口(更稳)
from dashscope import Generation
import os
import json
import time

# =================配置区=================
if "DASHSCOPE_API_KEY" in st.secrets:
    API_KEY = st.secrets["DASHSCOPE_API_KEY"]
else:
    # 👇 请确保这里填入了您的真实Key
    API_KEY = "sk-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx" 

dashscope.api_key = API_KEY
# =======================================

st.set_page_config(page_title="银龄知音·手机版", page_icon="📱", layout="centered")

# CSS 样式优化
st.markdown("""
    <style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    .stDeployButton {display:none;}
    header {visibility: hidden;}
    
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

if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

# --- 核心功能区 ---
audio_value = st.audio_input("点此开始录音")

if audio_value:
    st.info("正在处理录音，请稍候...")
    
    try:
        # 1. 保存临时文件
        with open("temp_audio.wav", "wb") as f:
            f.write(audio_value.getvalue())
        
        # 2. 获取绝对路径 (修复了之前的 json.os 报错)
        abs_path = os.path.abspath("temp_audio.wav")
        file_url = f"file://{abs_path}"
        
        # 3. 调用文件转写服务 (异步提交)
        # 这个接口兼容性最好，能自动处理文件上传
        task_response = Transcription.async_call(
            model='paraformer-v1',
            file_urls=[file_url],
            language_hints=['zh']
        )
        
        # 4. 等待结果
        transcribe_response = Transcription.wait(task=task_response.task_id, api_key=API_KEY)
        
        if transcribe_response.status_code == 200:
            # 提取文字
            user_text = ""
            if 'results' in transcribe_response.output and transcribe_response.output['results']:
                for sent in transcribe_response.output['results'][0]['sentences']:
                    user_text += sent['text']
            
            if user_text:
                st.success("听清楚啦！")
                st.session_state.last_user_text = user_text
                
                # 5. 调用大模型
                system_prompt = "你是一个温暖的老年人陪伴助手，请简短、亲切地回复。"
                if any(k in user_text for k in ["查", "问", "怎么", "哪里", "医生", "药"]):
                    system_prompt = "你是一个生活助手，请直接给出简单的办事建议。"
                
                messages = [
                    {'role': 'system', 'content': system_prompt},
                    {'role': 'user', 'content': user_text}
                ]
                
                llm_resp = Generation.call(
                    api_key=API_KEY,
                    model="qwen-turbo",
                    messages=messages,
                    result_format='message'
                )
                
                if llm_resp.status_code == 200:
                    reply = llm_resp.output.choices[0].message.content
                    
                    # 存入历史并刷新
                    st.session_state.chat_history.insert(0, {"role": "bot", "content": reply})
                    st.session_state.chat_history.insert(0, {"role": "user", "content": user_text})
                    st.rerun()
                else:
                    st.error("知音正在思考，请稍后再试")
            else:
                st.warning("好像没听到声音，请大声一点~")
        else:
            st.error(f"语音识别服务繁忙: {transcribe_response.message}")
            
    except Exception as e:
        st.error(f"发生错误: {e}")

# --- 展示对话流 ---
st.markdown("---")
for chat in st.session_state.chat_history:
    if chat["role"] == "user":
        st.markdown(f"**👴 您说：** {chat['content']}")
    else:
        st.markdown(f"<div class='chat-bubble'>🤖 <b>知音：</b>{chat['content']}</div>", unsafe_allow_html=True)
