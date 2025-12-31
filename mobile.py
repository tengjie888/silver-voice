import streamlit as st
import dashscope
from dashscope.audio.asr import Recognition
from dashscope import Generation
import json

# =================配置区=================
# ⚠️⚠️⚠️ 请务必在此处填入您的真实 API Key ⚠️⚠️⚠️
if "DASHSCOPE_API_KEY" in st.secrets:
    API_KEY = st.secrets["DASHSCOPE_API_KEY"]
else:
    API_KEY = "sk-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx" 

dashscope.api_key = API_KEY
# =======================================

st.set_page_config(page_title="银龄知音", page_icon="👴", layout="centered")

# --- CSS 样式 ---
st.markdown("""
    <style>
    #MainMenu, footer, header {visibility: hidden;}
    div[role="radiogroup"] > label {
        background-color: #f0f2f6; padding: 10px 20px; border-radius: 20px;
        margin-right: 10px; border: 1px solid #ddd;
    }
    div[role="radiogroup"] { flex-direction: row; gap: 10px; }
    .stAudioInput { margin-top: 20px; }
    .chat-bubble {
        background: #ffffff; padding: 18px; border-radius: 18px; 
        margin-top: 15px; box-shadow: 0px 2px 8px rgba(0,0,0,0.08);
        font-size: 18px; line-height: 1.6; color: #333;
    }
    .user-bubble { color: #666; font-size: 16px; margin-top: 20px; }
    .stApp { background-color: #F8F9FA; }
    h1 { color: #E74C3C; text-align: center; font-weight: 800; }
    </style>
""", unsafe_allow_html=True)

st.title("👵 银龄知音")

# --- 1. 模式选择 ---
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
    if len(audio_value.getvalue()) < 1000:
        st.warning("⏳ 录音太短啦，请多说几句~")
    else:
        st.info("正在识别...")
        
        try:
            # 🔥 核心修改：直接读取二进制数据流
            # 不存文件，直接把数据喂给 Recognition 接口
            # 使用 paraformer-realtime-v1 模型（这个通常有大量免费额度）
            audio_bytes = audio_value.getvalue()
            
            recognition = Recognition(
                model='paraformer-realtime-v1',
                format='wav',
                sample_rate=16000,
                callback=None
            )
            
            # 直接调用 call 方法传入音频数据
            response = recognition.call(
                audio_bytes,
                language_hints=['zh']
            )
            
            # 检查结果
            if response.status_code == 200:
                user_text = ""
                # 兼容不同的返回格式
                if 'sentences' in response.output:
                    user_text = "".join([s['text'] for s in response.output['sentences']])
                elif 'text' in response.output:
                    user_text = response.output['text']
                
                if user_text:
                    st.success("听清啦！")
                    
                    # --- 思考逻辑 ---
                    if "聊聊" in mode:
                        system_prompt = "你是一个温暖的老年人陪伴助手“知音”。请用亲切、尊重的口吻，像晚辈一样陪老人聊天。回复要简短暖心，多给予情感支持。"
                    else:
                        system_prompt = "你是一个生活助手。请忽略老人的口语废话，直接提取核心需求，给出最简单、直接的办事建议或信息。不要长篇大论。"
                    
                    messages = [
                        {'role': 'system', 'content': system_prompt},
                        {'role': 'user', 'content': user_text}
                    ]
                    
                    # 调用 LLM (qwen-turbo 也很便宜)
                    llm_resp = Generation.call(
                        api_key=API_KEY, 
                        model="qwen-turbo", 
                        messages=messages, 
                        result_format='message'
                    )
                    
                    if llm_resp.status_code == 200:
                        reply = llm_resp.output.choices[0].message.content
                        st.session_state.chat_history.insert(0, {"role": "bot", "content": reply})
                        st.session_state.chat_history.insert(0, {"role": "user", "content": user_text})
                        st.rerun()
                    else:
                        st.error(f"大脑思考失败: {llm_resp.message}")
                else:
                    st.warning("👂 好像没听到声音，请大声一点~")
            else:
                # 如果还报错，直接把错误显示出来
                st.error(f"识别服务报错: {response.code} - {response.message}")
                
        except Exception as e:
            st.error(f"内部错误: {str(e)}")

# --- 3. 历史记录 ---
st.markdown("---")
for chat in st.session_state.chat_history:
    if chat["role"] == "user":
        st.markdown(f"<div class='user-bubble'>👴 <b>您说：</b>{chat['content']}</div>", unsafe_allow_html=True)
    else:
        st.markdown(f"<div class='chat-bubble'>🤖 <b>知音：</b>{chat['content']}</div>", unsafe_allow_html=True)
