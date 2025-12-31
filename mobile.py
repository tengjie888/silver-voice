import streamlit as st
import dashscope
from dashscope.audio.asr import Recognition # 使用短语音识别接口
from dashscope import Generation
import os

# =================配置区=================
# ⚠️⚠️⚠️ 请务必在此处填入您的真实 API Key ⚠️⚠️⚠️
if "DASHSCOPE_API_KEY" in st.secrets:
    API_KEY = st.secrets["DASHSCOPE_API_KEY"]
else:
    API_KEY = "sk-3132c4eed8694648a1bb55ae6cc25d25" 

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
        st.info("正在听...")
        
        try:
            # A. 保存文件 (最稳妥的方式)
            temp_filename = "temp_audio.wav"
            with open(temp_filename, "wb") as f:
                f.write(audio_value.getvalue())
            
            # B. 使用 Paraformer 模型 (❌不要用 gummy，✅用 paraformer)
            # paraformer-realtime-v1 既支持流式，也支持这种短文件识别，且兼容性好
            rec_response = Recognition.call(
                model='paraformer-realtime-v1', 
                file=temp_filename, 
                format='wav',
                language_hints=['zh']
            )
            
            # C. 检查结果
            if rec_response.status_code == 200:
                user_text = ""
                # 兼容提取
                if 'sentences' in rec_response.output:
                    user_text = "".join([s['text'] for s in rec_response.output['sentences']])
                elif 'text' in rec_response.output:
                    user_text = rec_response.output['text']
                
                if user_text:
                    st.success("听清啦！")
                    
                    # D. 大模型思考 (这里可以实现翻译功能)
                    # 如果您原本想用 gummy 做翻译，这里改 Prompt 就行了
                    if "聊聊" in mode:
                        system_prompt = "你是一个温暖的老年人陪伴助手“知音”。请用亲切、尊重的口吻，像晚辈一样陪老人聊天。回复要简短暖心，多给予情感支持。"
                    else:
                        system_prompt = "你是一个生活助手。请忽略老人的口语废话，直接提取核心需求，给出最简单、直接的办事建议或信息。不要长篇大论。"
                    
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
                        st.session_state.chat_history.insert(0, {"role": "bot", "content": reply})
                        st.session_state.chat_history.insert(0, {"role": "user", "content": user_text})
                        st.rerun()
                    else:
                        st.error(f"大脑思考失败: {llm_resp.message}")
                else:
                    st.warning("👂 好像没听到声音，请大声一点~")
            else:
                st.error(f"识别失败: {rec_response.code} - {rec_response.message}")
                
        except Exception as e:
            st.error(f"内部错误: {str(e)}")

# --- 3. 历史记录 ---
st.markdown("---")
for chat in st.session_state.chat_history:
    if chat["role"] == "user":
        st.markdown(f"<div class='user-bubble'>👴 <b>您说：</b>{chat['content']}</div>", unsafe_allow_html=True)
    else:
        st.markdown(f"<div class='chat-bubble'>🤖 <b>知音：</b>{chat['content']}</div>", unsafe_allow_html=True)
