import streamlit as st
import dashscope
from dashscope.audio.asr import Recognition  # 👈 换回了更快的短语音接口
from dashscope import Generation
import os

# =================配置区=================
# 优先读取 Streamlit Secrets，如果没有配置，则读取下面的字符串
if "DASHSCOPE_API_KEY" in st.secrets:
    API_KEY = st.secrets["DASHSCOPE_API_KEY"]
else:
    # ⚠️⚠️⚠️ 请务必在这里填入您的真实 API Key，保留双引号 ⚠️⚠️⚠️
    API_KEY = "sk-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx" 

dashscope.api_key = API_KEY
# =======================================

st.set_page_config(page_title="银龄知音", page_icon="👴", layout="centered")

# CSS 样式优化
st.markdown("""
    <style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    .stButton>button {
        height: 3.5em; 
        width: 100%; 
        font-size: 22px !important; 
        border-radius: 25px; 
        background-color: #FF5733; 
        color: white;
        border: none;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
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

audio_value = st.audio_input("点此开始录音")

if audio_value:
    st.info("正在听...")
    
    try:
        # 1. 保存录音文件
        with open("temp_audio.wav", "wb") as f:
            f.write(audio_value.getvalue())
        
        # 2. 调用阿里云识别（短语音模式，速度快）
        # 使用 Recognition.call 直接发送文件
        rec_response = Recognition.call(
            model='paraformer-realtime-v1',
            file='temp_audio.wav',
            format='wav',
            language_hints=['zh']
        )
        
        # 3. 检查识别结果
        if rec_response.status_code == 200:
            user_text = ""
            # 提取文字内容
            if 'sentences' in rec_response.output:
                user_text = "".join([s['text'] for s in rec_response.output['sentences']])
            else:
                user_text = rec_response.output.get('text', '')
            
            if user_text:
                st.success("听清啦！")
                
                # 4. 调用大模型
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
                    # 存入历史并强制刷新
                    st.session_state.chat_history.insert(0, {"role": "bot", "content": reply})
                    st.session_state.chat_history.insert(0, {"role": "user", "content": user_text})
                    st.rerun()
                else:
                    # 如果大模型报错，打印具体原因
                    st.error(f"大脑思考失败: {llm_resp.code} - {llm_resp.message}")
            else:
                st.warning("好像没听到声音，请大声一点~")
        else:
            # ⚠️ 这里是关键：如果识别失败，打印出阿里云返回的真实错误信息
            st.error(f"耳朵出问题了: {rec_response.code} - {rec_response.message}")
            
    except Exception as e:
        st.error(f"程序内部错误: {e}")

# 显示历史对话
st.markdown("---")
for chat in st.session_state.chat_history:
    if chat["role"] == "user":
        st.markdown(f"**👴 您说：** {chat['content']}")
    else:
        st.markdown(f"<div class='chat-bubble'>🤖 <b>知音：</b>{chat['content']}</div>", unsafe_allow_html=True)
