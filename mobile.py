import streamlit as st
import dashscope
from dashscope.audio.asr import Transcription
from dashscope import Generation
import os
import time

# =================配置区=================
# ⚠️⚠️⚠️ 请务必在此处填入您的真实 API Key ⚠️⚠️⚠️
if "DASHSCOPE_API_KEY" in st.secrets:
    API_KEY = st.secrets["DASHSCOPE_API_KEY"]
else:
    API_KEY = "sk-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx" 

dashscope.api_key = API_KEY
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
    # 检查文件大小，防止空录音
    file_size = len(audio_value.getvalue())
    if file_size < 1000:
        st.warning("⏳ 录音太短啦，请多说几句~")
    else:
        st.info("正在上传并识别...")
        
        try:
            # A. 保存临时文件
            temp_filename = "temp_audio.wav"
            with open(temp_filename, "wb") as f:
                f.write(audio_value.getvalue())
                
            # B. 使用【文件转写】接口 (兼容性最强，不挑采样率)
            # 注意：Streamlit Cloud 需要用绝对路径
            abs_path = os.path.abspath(temp_filename)
            file_url = f"file://{abs_path}"
            
            task_response = Transcription.async_call(
                model='paraformer-v1',
                file_urls=[file_url],
                language_hints=['zh']
            )
            
            if task_response.status_code == 200:
                task_id = task_response.output.task_id
                
                # C. 等待识别结果
                status = "PENDING"
                while status in ["PENDING", "RUNNING"]:
                    time.sleep(1) # 稍微等一下
                    wait_resp = Transcription.wait(task=task_id, api_key=API_KEY)
                    status = wait_resp.output.task_status
                    
                    if status == "SUCCEEDED":
                        # 提取文字
                        user_text = ""
                        results = wait_resp.output.get('results', [])
                        if results:
                            # 尝试多种提取方式，防止格式变动
                            res0 = results[0]
                            if 'sentences' in res0:
                                user_text = "".join([s['text'] for s in res0['sentences']])
                            elif 'text' in res0:
                                user_text = res0['text']
                        
                        if user_text:
                            st.success("听清啦！")
                            
                            # D. 根据模式设定 Prompt
                            if "聊聊" in mode:
                                system_prompt = "你是一个温暖的老年人陪伴助手“知音”。请用亲切、尊重的口吻，像晚辈一样陪老人聊天。回复要简短暖心，多给予情感支持。"
                            else:
                                system_prompt = "你是一个生活助手。请忽略老人的口语废话，直接提取核心需求，给出最简单、直接的办事建议或信息。不要长篇大论。"
                            
                            # E. 调用大模型
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
                                st.rerun() # 强制刷新显示
                            else:
                                st.error("大脑思考时卡住了，请重试")
                        else:
                            st.warning("😓 好像全是杂音，没听清您说了什么，请再试一次")
                    elif status == "FAILED":
                        st.error(f"识别失败: {wait_resp.output.message}")
                        break
            else:
                st.error(f"上传失败: {task_response.message} (请检查API Key)")
                
        except Exception as e:
            st.error(f"内部错误: {str(e)}")

# --- 3. 历史记录显示 ---
st.markdown("---")
for chat in st.session_state.chat_history:
    if chat["role"] == "user":
        st.markdown(f"<div class='user-bubble'>👴 <b>您说：</b>{chat['content']}</div>", unsafe_allow_html=True)
    else:
        st.markdown(f"<div class='chat-bubble'>🤖 <b>知音：</b>{chat['content']}</div>", unsafe_allow_html=True)
