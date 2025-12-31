import streamlit as st
import dashscope
from dashscope.audio.asr import Transcription
from dashscope import Generation
import os
import json
import time

# =================配置区=================
if "DASHSCOPE_API_KEY" in st.secrets:
    API_KEY = st.secrets["DASHSCOPE_API_KEY"]
else:
    # 👇 请务必在这里填入您的真实 API Key
    API_KEY = "sk-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx" 

dashscope.api_key = API_KEY
# =======================================

st.set_page_config(page_title="银龄知音", page_icon="👴", layout="centered")

# CSS 样式
st.markdown("""
    <style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    .stButton>button {
        height: 3.5em; width: 100%; font-size: 22px !important; 
        border-radius: 25px; background-color: #FF5733; color: white; border: none;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .chat-bubble {
        background: #ffffff; padding: 18px; border-radius: 18px; 
        margin-top: 15px; box-shadow: 0px 2px 5px rgba(0,0,0,0.05); font-size: 18px;
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
    st.info("正在听您说...")
    
    try:
        # 1. 保存临时文件
        with open("temp_audio.wav", "wb") as f:
            f.write(audio_value.getvalue())
            
        # 2. 获取绝对路径
        abs_path = os.path.abspath("temp_audio.wav")
        file_url = f"file://{abs_path}"
        
        # 3. 提交任务
        task_response = Transcription.async_call(
            model='paraformer-v1',
            file_urls=[file_url],
            language_hints=['zh']
        )
        
        if task_response.status_code != 200:
            st.error(f"连接阿里云失败: {task_response.message}")
        else:
            task_id = task_response.output.task_id
            
            # 4. 等待结果
            transcribe_response = Transcription.wait(task=task_id, api_key=API_KEY)
            
            if transcribe_response.status_code == 200:
                # --- 🔥 核心修复：超强鲁棒的文字提取逻辑 ---
                user_text = ""
                results = transcribe_response.output.get('results', [])
                
                if results:
                    first_result = results[0]
                    # 优先找 sentences 列表
                    if 'sentences' in first_result:
                        user_text = "".join([s.get('text', '') for s in first_result['sentences']])
                    # 如果没有 sentences，尝试直接找 text 字段
                    elif 'text' in first_result:
                        user_text = first_result['text']
                
                # 如果 user_text 还是空的，说明真的没听见
                if user_text.strip():
                    st.success("听清啦！")
                    
                    # 5. 调用大模型
                    system_prompt = "你是一个温暖的老年人陪伴助手，请简短、亲切地回复。"
                    if any(k in user_text for k in ["查", "问", "怎么", "哪里", "医生", "药"]):
                        system_prompt = "你是一个生活助手，请直接给出简单的办事建议。"
                    
                    messages = [{'role': 'system', 'content': system_prompt},
                                {'role': 'user', 'content': user_text}]
                    
                    llm_resp = Generation.call(api_key=API_KEY, model="qwen-turbo", messages=messages, result_format='message')
                    
                    if llm_resp.status_code == 200:
                        reply = llm_resp.output.choices[0].message.content
                        st.session_state.chat_history.insert(0, {"role": "bot", "content": reply})
                        st.session_state.chat_history.insert(0, {"role": "user", "content": user_text})
                        st.rerun()
                    else:
                        st.error(f"大脑思考失败: {llm_resp.message}")
                else:
                    st.warning("好像没听到声音，请大声一点~")
            else:
                st.error(f"转写服务出错: {transcribe_response.message}")
                
    except Exception as e:
        # 把错误打印出来，方便看
        st.error(f"程序内部错误: {str(e)}")

# 显示历史
st.markdown("---")
for chat in st.session_state.chat_history:
    if chat["role"] == "user":
        st.markdown(f"**👴 您说：** {chat['content']}")
    else:
        st.markdown(f"<div class='chat-bubble'>🤖 <b>知音：</b>{chat['content']}</div>", unsafe_allow_html=True)
