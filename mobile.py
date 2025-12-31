import streamlit as st
import threading
import time
import pyaudio
import dashscope
from dashscope.audio.asr import TranslationRecognizerRealtime, TranslationRecognizerCallback
from dashscope import Generation
# ---------------------------------------------------------
# 【修复关键 1】引入 Streamlit 的上下文管理工具
# ---------------------------------------------------------
from streamlit.runtime.scriptrunner import add_script_run_ctx

# ==========================================
# 1. 适老化界面配置
# ==========================================
st.set_page_config(
    page_title="银龄知音",
    page_icon="👴",
    layout="centered"
)

st.markdown("""
    <style>
    p, div, label, input { font-size: 20px !important; }
    h1 { font-size: 42px !important; color: #E74C3C; text-align: center; }
    .stButton>button {
        height: 80px;
        width: 100%;
        font-size: 28px !important;
        border-radius: 15px;
        font-weight: bold;
    }
    .chat-box {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
        border-left: 5px solid #E74C3C;
        margin-bottom: 20px;
    }
    .status-text { color: #888; font-size: 18px; text-align: center; }
    </style>
""", unsafe_allow_html=True)

# ==========================================
# 2. Prompt 设计
# ==========================================
PROMPT_CHAT_SYSTEM = """
你是一个名为“知音”的老年人陪伴助手。你的用户是老年人。
【你的性格】：温暖、耐心、尊老、像一个懂事的晚辈。
【处理规则】：
1. 老年人说话可能会啰嗦、重复、带有大量“嗯、啊、这个”等口语，请完全包容。
2. 即使老人的话没有实质内容，也要给予情感上的回应（如：“我在听呢，您慢慢说”）。
3. 回复要短（不超过100字），语气要亲切。
"""

PROMPT_ASK_SYSTEM = """
你是一个高效的老年人生活助手。
【任务目标】：从老年人冗长、含糊的口语中提取核心需求，并给出直接、简单的答案。
【处理规则】：
1. 【去除噪音】：自动过滤掉“麻烦你”、“请问一下”等客套话。
2. 【直接回答】：如果问路，直接说几路车；如果问药，直接说禁忌。
3. 【字体友好】：分段清晰，重点内容可以加粗。
"""

# ==========================================
# 3. 后端逻辑
# ==========================================
API_KEY = "sk-3132c4eed8694648a1bb55ae6cc25d25"  # 请确保Key正确
LLM_MODEL = "qwen-plus"
dashscope.api_key = API_KEY


class ASRCallback(TranslationRecognizerCallback):
    def __init__(self):
        super().__init__()
        self.sentence_buffer = []
        self.current_text = ""
        self.lock = threading.Lock()

    def on_open(self) -> None:
        self.mic = pyaudio.PyAudio()
        self.stream = self.mic.open(
            format=pyaudio.paInt16, channels=1, rate=16000, input=True, frames_per_buffer=3200
        )

    def on_close(self) -> None:
        # 为了防止线程卡死，这里加了 try-except
        try:
            if hasattr(self, 'stream') and self.stream:
                self.stream.stop_stream()
                self.stream.close()
            if hasattr(self, 'mic') and self.mic:
                self.mic.terminate()
        except Exception:
            pass

    def on_event(self, request_id, transcription_result, translation_result, usage) -> None:
        if transcription_result:
            text = getattr(transcription_result, "text", "")
            is_final = getattr(transcription_result, "is_sentence_end", False)

            with self.lock:
                self.current_text = text
                if is_final and text:
                    self.sentence_buffer.append(text)

    def get_full_transcript(self):
        with self.lock:
            return "".join(self.sentence_buffer) + self.current_text

    def clear(self):
        with self.lock:
            self.sentence_buffer = []
            self.current_text = ""


def start_asr():
    if 'callback' not in st.session_state:
        st.session_state.callback = ASRCallback()

    # 初始化 ASR
    recognizer = TranslationRecognizerRealtime(
        model="gummy-realtime-v1",
        format="pcm",
        sample_rate=16000,
        transcription_enabled=True,
        translation_enabled=False,
        callback=st.session_state.callback
    )

    st.session_state.recognizer = recognizer
    recognizer.start()
    st.session_state.is_listening = True

    # ---------------------------------------------------------
    # 【修复关键 2】修改线程启动逻辑
    # ---------------------------------------------------------
    def audio_loop():
        # 这里为了稳健，我们再次获取 callback
        # 注意：因为加了 ctx，这里能访问 session_state 了
        cb = st.session_state.callback
        rec = st.session_state.recognizer

        while st.session_state.get('is_listening', False):
            if cb.stream:
                try:
                    data = cb.stream.read(3200, exception_on_overflow=False)
                    rec.send_audio_frame(data)
                except Exception:
                    break
        rec.stop()

    t = threading.Thread(target=audio_loop, daemon=True)
    # 这一步是关键：把当前页面的上下文传给线程
    add_script_run_ctx(t)
    t.start()


def stop_asr_and_get_result():
    st.session_state.is_listening = False
    time.sleep(0.5)
    text = st.session_state.callback.get_full_transcript()
    st.session_state.callback.clear()
    return text


# 用这段代码替换原来的 call_llm 函数
def call_llm(user_text, mode):
    if not user_text.strip():
        return "哎呀，我好像没听清您说了什么，能再说一遍吗？"

    system_prompt = PROMPT_CHAT_SYSTEM if mode == "chat" else PROMPT_ASK_SYSTEM

    messages = [
        {'role': 'system', 'content': system_prompt},
        {'role': 'user', 'content': f"老年人语音输入：{user_text}"}
    ]

    try:
        # 【关键修改】增加了 result_format='message'
        resp = Generation.call(
            api_key=API_KEY,
            model=LLM_MODEL,
            messages=messages,
            result_format='message'
        )

        # 打印一下原始内容方便调试（会在终端显示）
        print(f"DEBUG: Status={resp.status_code}, Msg={resp.message}")

        if resp.status_code == 200:
            if resp.output and resp.output.choices:
                return resp.output.choices[0].message.content
            else:
                # 如果还是空，打印出到底返回了啥
                return f"模型返回了空内容，原始数据是：{resp}"
        else:
            return f"出错了（代码{resp.code}）：{resp.message}"

    except Exception as e:
        return f"程序内部报错：{e}"

# ==========================================
# 4. 前端页面
# ==========================================
if 'is_listening' not in st.session_state:
    st.session_state.is_listening = False
if 'mode' not in st.session_state:
    st.session_state.mode = None
if 'last_response' not in st.session_state:
    st.session_state.last_response = ""
if 'user_transcript' not in st.session_state:
    st.session_state.user_transcript = ""

st.title("👵 银龄知音")
st.markdown("<div class='status-text'>您的贴心智能伴侣</div>", unsafe_allow_html=True)
st.markdown("---")

col1, col2 = st.columns(2)

with col1:
    if not st.session_state.is_listening:
        if st.button("🟢 陪我聊聊", use_container_width=True):
            st.session_state.mode = "chat"
            st.session_state.last_response = ""
            start_asr()
            st.rerun()
    else:
        if st.session_state.mode == "chat":
            if st.button("🛑 说完了", type="primary", use_container_width=True):
                transcript = stop_asr_and_get_result()
                st.session_state.user_transcript = transcript
                with st.spinner("知音正在思考中..."):
                    reply = call_llm(transcript, "chat")
                    st.session_state.last_response = reply
                st.session_state.mode = None
                st.rerun()
        else:
            st.button("🚫 忙碌中", disabled=True, use_container_width=True)

with col2:
    if not st.session_state.is_listening:
        if st.button("🔵 帮我查查", use_container_width=True):
            st.session_state.mode = "ask"
            st.session_state.last_response = ""
            start_asr()
            st.rerun()
    else:
        if st.session_state.mode == "ask":
            if st.button("🛑 问完了", type="primary", use_container_width=True):
                transcript = stop_asr_and_get_result()
                st.session_state.user_transcript = transcript
                with st.spinner("正在查询..."):
                    reply = call_llm(transcript, "ask")
                    st.session_state.last_response = reply
                st.session_state.mode = None
                st.rerun()
        else:
            st.button("🚫 忙碌中", disabled=True, use_container_width=True)

if st.session_state.is_listening:
    st.markdown("### 👂 我正在听：")
    placeholder = st.empty()
    # 循环刷新字幕
    while st.session_state.is_listening:
        if hasattr(st.session_state, 'callback'):
            txt = st.session_state.callback.get_full_transcript()
            placeholder.markdown(f"<div style='font-size:24px; color:#555;'>{txt}</div>", unsafe_allow_html=True)
        time.sleep(0.1)

if st.session_state.last_response:
    st.markdown("---")
    st.markdown(f"**👴 您刚才说：** {st.session_state.user_transcript}")
    st.markdown("### 🤖 知音回应：")
    st.markdown(f"<div class='chat-box'>{st.session_state.last_response}</div>", unsafe_allow_html=True)

st.markdown("---")
st.markdown("<div style='text-align:center; color:#999;'>按住按钮说话，再次点击结束</div>", unsafe_allow_html=True)
