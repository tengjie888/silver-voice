import streamlit as st
import dashscope
from dashscope.audio.asr import Transcription
from dashscope import Generation
import json
import os

# ==========================================
# 配置区：从 Streamlit Secrets 获取 Key
# 安全起见，不要在代码里直接写死 Key
# ==========================================
try:
    # 尝试从云端配置读取
    api_key = st.secrets["DASHSCOPE_API_KEY"]
except:
    # 如果本地运行没有配置secrets，可以使用硬编码（仅限本地测试）
    api_key = "sk-xxxxxxxxxxxxxxxxxxxxxxxx" 

dashscope.api_key = api_key

st.set_page_config(page_title="银龄知音", page_icon="👵", layout="centered")

# 注入手机端友好的 CSS
st.markdown("""
    <style>
    /* 隐藏顶部菜单和底部 */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* 按钮大一点，适合老人 */
    .stButton>button {
        height: 3em; 
        width: 100%; 
        font-size: 20px !important; 
        border-radius: 15px; 
        background-color: #FF5733; 
        color: white; 
        border: none;
    }
    
    /* 聊天气泡 */
    .chat-bubble {
        background: #f0f2f6; 
        padding: 15px; 
        border-radius: 15px; 
        margin-top: 10px; 
        font-size: 18px;
    }
    </style>
""", unsafe_allow_html=True)

st.title("👵 银龄知音")
st.info("点下方麦克风录音，说完点停止")

# 初始化历史记录
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

# ==========================================
# 核心逻辑：使用 st.audio_input (手机网页版专用)
# ==========================================
audio_value = st.audio_input("点击录音")

if audio_value:
    st.success("正在处理录音...")
    
    # 1. 保存临时文件
    with open("temp_audio.wav", "wb") as f:
        f.write(audio_value.getvalue())
    
    try:
        # 2. 调用通义千问语音转写 (Paraformer)
        # 注意：这里用的是文件转写API，不是流式API，更适合网页环境
        file_url = f"file://{os.path.abspath('temp_audio.wav')}"
        
        task_response = Transcription.async_call(
            model='paraformer-v1',
            file_urls=[file_url],
            language_hints=['zh']  # 提示是中文
        )
        
        # 等待结果
        transcribe_response = Transcription.wait(task=task_response.task_id, api_key=api_key)
        
        if transcribe_response.status_code == 200:
            # 提取文本
            user_text = ""
            if 'results' in transcribe_response.output and transcribe_response.output['results']:
                for sent in transcribe_response.output['results'][0]['sentences']:
                    user_text += sent['text']
            
            if user_text:
                # 3. 只有当识别出新内容，且不与上次重复时才处理
                # (防止Streamlit刷新导致的重复提交)
                if 'last_processed_audio' not in st.session_state or st.session_state.last_processed_audio != audio_value:
                    st.session_state.last_processed_audio = audio_value
                    
                    # --- 调用大模型 ---
                    # 简单的意图判断 Prompt
                    prompt = "你是一个温暖的老年人陪伴助手，请简短、亲切地回复。不要使用复杂术语。"
                    if any(k in user_text for k in ["查", "问", "怎么去", "在哪里"]):
                        prompt = "你是一个生活助手，请直接给出简单的办事建议。不要啰嗦。"

                    messages = [
                        {'role': 'system', 'content': prompt},
                        {'role': 'user', 'content': user_text}
                    ]
                    
                    llm_resp = Generation.call(model="qwen-turbo", messages=messages, result_format='message')
                    
                    if llm_resp.status_code == 200:
                        reply = llm_resp.output.choices[0].message.content
                        
                        # 存入历史
                        st.session_state.chat_history.insert(0, {"role": "bot", "content": reply})
                        st.session_state.chat_history.insert(0, {"role": "user", "content": user_text})
                        
                        # 强制刷新页面以显示新消息
                        st.rerun()
                    else:
                        st.error("大模型太累了，没理我。")
            else:
                st.warning("好像没听到声音，请大声一点。")
        else:
            st.error("语音识别服务出错了。")
            
    except Exception as e:
        st.error(f"处理出错: {e}")

# ==========================================
# 显示对话历史
# ==========================================
for chat in st.session_state.chat_history:
    if chat["role"] == "user":
        st.markdown(f"**👴 您说：** {chat['content']}")
    else:
        st.markdown(f"<div class='chat-bubble'>🤖 <b>知音：</b>{chat['content']}</div>", unsafe_allow_html=True)
