import streamlit as st
import dashscope
from dashscope.audio.asr import Transcription
from dashscope import Generation
import json

# =================配置区=================
# 👇 请务必替换成你的真实 API Key
API_KEY = "sk-3132c4eed8694648a1bb55ae6cc25d25"
dashscope.api_key = API_KEY
# =======================================

st.set_page_config(page_title="银龄知音·手机版", page_icon="📱", layout="centered")

# CSS 适配手机竖屏，按钮变大
# 修改 mobile.py 中的 CSS 部分
st.markdown("""
    <style>
    /* 1. 隐藏 Streamlit 默认的菜单、页脚和红色的 Deploy 按钮 */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    .stDeployButton {display:none;}
    header {visibility: hidden;} /* 隐藏顶部的彩色条 */

    /* 2. 按钮样式优化：超大、圆角、阴影，像原生App的按钮 */
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

    /* 3. 聊天气泡优化 */
    .chat-bubble {
        background: #ffffff; 
        padding: 18px; 
        border-radius: 18px; 
        margin-top: 15px; 
        box-shadow: 0px 2px 5px rgba(0,0,0,0.05);
        font-size: 18px;
    }

    /* 4. 整体背景微调 (让它不像网页白底) */
    .stApp {
        background-color: #F7F7F7;
    }
    </style>
""", unsafe_allow_html=True)

st.title("👵 银龄知音")
st.write("请按下方红色按钮说话，说完点击停止")

# 初始化状态
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

# --- 核心功能区 ---
# 1. 手机端录音组件
audio_value = st.audio_input("点此开始录音")

if audio_value:
    st.success("录音完成，正在处理...")

    # 2. 将录音转为文字 (调用 DashScope 文件转写 API)
    try:
        # DashScope 需要文件路径或二进制流，这里我们先把音频存为临时文件
        with open("temp_audio.wav", "wb") as f:
            f.write(audio_value.getvalue())

        # 调用离线文件转写服务 (Paraformer 模型)
        task_response = Transcription.async_call(
            model='paraformer-v1',
            file_urls=[f"file://{json.os.path.abspath('temp_audio.wav')}"],
            language_hints=['zh']
        )

        # 等待转写结果
        transcribe_response = Transcription.wait(task=task_response.task_id, api_key=API_KEY)

        if transcribe_response.status_code == 200:
            # 提取转写的文本
            user_text = ""
            for sent in transcribe_response.output['results'][0]['sentences']:
                user_text += sent['text']

            if user_text:
                st.session_state.last_user_text = user_text

                # 3. 调用大模型 (LLM) 进行回答
                # 这里简单判断意图：如果包含"查"或"问"走工具模式，否则走聊天模式
                mode_prompt = "你是一个温暖的老年人陪伴助手，请简短、亲切地回复。"
                if any(k in user_text for k in ["查", "问", "怎么", "哪里"]):
                    mode_prompt = "你是一个生活助手，请直接给出简单的办事建议。"

                messages = [
                    {'role': 'system', 'content': mode_prompt},
                    {'role': 'user', 'content': user_text}
                ]

                llm_resp = Generation.call(model="qwen-turbo", messages=messages, result_format='message')

                if llm_resp.status_code == 200:
                    reply = llm_resp.output.choices[0].message.content

                    # 存入历史
                    st.session_state.chat_history.insert(0, {"role": "bot", "content": reply})
                    st.session_state.chat_history.insert(0, {"role": "user", "content": user_text})
                else:
                    st.error("大模型开小差了，请重试")
        else:
            st.error("语音识别失败，请大声一点重试")

    except Exception as e:
        st.error(f"发生错误: {e}")

# --- 展示对话流 ---
st.markdown("---")
for chat in st.session_state.chat_history:
    if chat["role"] == "user":
        st.markdown(f"**👴 您说：** {chat['content']}")
    else:
        st.markdown(f"<div class='chat-bubble'>🤖 <b>知音：</b>{chat['content']}</div>", unsafe_allow_html=True)
