import streamlit as st
from dashscope import MultiModalConversation, Generation
import os

# =================配置区=================
# ⚠️⚠️⚠️ 请务必在此处填入您的真实 API Key ⚠️⚠️⚠️
if "DASHSCOPE_API_KEY" in st.secrets:
    API_KEY = st.secrets["DASHSCOPE_API_KEY"]
else:
    API_KEY = "sk-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx" 

# =======================================

st.set_page_config(page_title="银龄知音", page_icon="👴", layout="centered")

# --- CSS 美化：让网页尽量像 App ---
st.markdown("""
    <style>
    /* 隐藏网页杂项 */
    #MainMenu, footer, header {visibility: hidden;}
    
    /* 模式选择器的样式优化 */
    div[role="radiogroup"] > label {
        background-color: #f0f2f6;
        padding: 10px 20px;
        border-radius: 20px;
        margin-right: 10px;
        border: 1px solid #ddd;
    }
    div[role="radiogroup"] {
        flex-direction: row;
        gap: 10px;
    }
    
    /* 录音按钮样式微调 */
    .stAudioInput { margin-top: 20px; }
    
    /* 聊天气泡 */
    .chat-bubble {
        background: #ffffff; 
        padding: 18px; 
        border-radius: 18px; 
        margin-top: 15px; 
        box-shadow: 0px 2px 8px rgba(0,0,0,0.08);
        font-size: 18px;
        line-height: 1.6;
        color: #333;
    }
    .user-bubble {
        color: #666; font-size: 16px; margin-top: 20px;
    }
    
    .stApp { background-color: #F8F9FA; }
    h1 { color: #E74C3C; text-align: center; font-weight: 800; }
    </style>
""", unsafe_allow_html=True)

st.title("👵 银龄知音")

# --- 1. 模式选择 (还原桌面版功能) ---
mode = st.radio(
    "请选择模式：",
    ("🟢 陪我聊聊", "🔵 帮我查查"),
    horizontal=True,
    label_visibility="collapsed"
)

st.write(f"当前模式：**{mode}**")
st.write("请按下方按钮录音，说完点停止：")

if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

# --- 2. 录音组件 ---
audio_value = st.audio_input("点此开始录音", label_visibility="collapsed")

if audio_value:
    # 简单检查文件大小
    if len(audio_value.getvalue()) < 1000:
        st.warning("⏳ 录音太短啦，请多说几句~")
    else:
        st.info("知音正在听...")
        
        try:
            # A. 保存临时文件
            temp_filename = "temp_audio.wav"
            with open(temp_filename, "wb") as f:
                f.write(audio_value.getvalue())
            
            # B. 第一步：用【Qwen-Audio-Turbo】来“听” (ASR)
            # 这个模型能直接理解音频文件，不需要转码，非常稳
            asr_messages = [{
                "role": "user",
                "content": [
                    {"audio": f"file://{os.path.abspath(temp_filename)}"},
                    {"text": "请将这段语音转写为文字，不要包含任何标点符号之外的解释。"}
                ]
            }]
            
            asr_response = MultiModalConversation.call(
                api_key=API_KEY,
                model='qwen-audio-turbo',
                messages=asr_messages
            )
            
            # 检查“耳朵”是否好使
            if asr_response.status_code == 200:
                user_text = asr_response.output.choices[0].message.content[0]['text']
                
                if user_text and len(user_text) > 1:
                    st.success("听清啦！")
                    
                    # C. 第二步：用【Qwen-Turbo】来“想” (LLM)
                    # 根据模式设定不同的人设
                    if "聊聊" in mode:
                        system_prompt = "你是一个温暖的老年人陪伴助手“知音”。请用亲切、尊重的口吻，像晚辈一样陪老人聊天。回复要简短暖心，多给予情感支持。"
                    else:
                        system_prompt = "你是一个生活助手。请忽略老人的口语废话，直接提取核心需求，给出最简单、直接的办事建议或信息。不要长篇大论。"
                    
                    chat_messages = [
                        {'role': 'system', 'content': system_prompt},
                        {'role': 'user', 'content': user_text}
                    ]
                    
                    llm_resp = Generation.call(
                        api_key=API_KEY, 
                        model="qwen-turbo", 
                        messages=chat_messages, 
                        result_format='message'
                    )
                    
                    if llm_resp.status_code == 200:
                        reply = llm_resp.output.choices[0].message.content
                        # 存入历史并刷新
                        st.session_state.chat_history.insert(0, {"role": "bot", "content": reply})
                        st.session_state.chat_history.insert(0, {"role": "user", "content": user_text})
                        st.rerun()
                    else:
                        st.error(f"大脑思考时卡住了: {llm_resp.message}")
                else:
                    st.warning("😓 好像没听到声音，请大声一点~")
            else:
                st.error(f"耳朵听不见了: {asr_response.message}")
                
        except Exception as e:
            st.error(f"内部错误: {str(e)}")

# --- 3. 历史记录显示 ---
st.markdown("---")
for chat in st.session_state.chat_history:
    if chat["role"] == "user":
        st.markdown(f"<div class='user-bubble'>👴 <b>您说：</b>{chat['content']}</div>", unsafe_allow_html=True)
    else:
        st.markdown(f"<div class='chat-bubble'>🤖 <b>知音：</b>{chat['content']}</div>", unsafe_allow_html=True)
