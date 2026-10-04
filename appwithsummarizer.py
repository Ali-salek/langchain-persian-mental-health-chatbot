import os

import redis
import streamlit as st
from langchain_community.chat_message_histories import RedisChatMessageHistory
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI

st.set_page_config(page_title="سامانه محرمانه سلامت روان ")
st.title("سامانه محرمانه سلامت روان صنایع غذایی سلامت ...")

REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")
MAX_RECENT = 5
SYSTEM_PROMPT = "تو یک دستیار محرمانه سلامت روان برای همکاران شرکت صنایع غذایی سلامت اندیشان نو هستی، با لحن آران و  پاسخ ببده و راهکار های عملی برای مدیریت استرس و بهزیستی روانی پیشنهاد کن. اگر کاربر از خطر  یا بحران جدی صحبت کرد او را به متخصص انسانی ارجاع بده"


def get_history(session_id: str):
    return RedisChatMessageHistory(
        session_id=session_id,
        url=REDIS_URL,
        key_prefix="salamat:chatwithhistory:",
    )


def build_lines(messages):
    lines = []
    for msg in messages:
        speaker = "کاربر" if msg.type == "human" else "دستیار"
        lines.append(f"{speaker}: {msg.content}")
    return "\n".join(lines)


def summary_key(session_id):
    return f"salamat:summary:{session_id}"


def cursor_key(session_id):
    return f"salamat:cursor:{session_id}"


@st.cache_resource
def load_chain():
    llm = ChatOpenAI(
        model="openrouter/free",
        temperature=0.4,
        api_key=OPENROUTER_API_KEY,
        base_url="https://openrouter.ai/api/v1",
    )
    summary_llm = ChatOpenAI(
        model="openrouter/free",
        temperature=0.1,
        api_key=OPENROUTER_API_KEY,
        base_url="https://openrouter.ai/api/v1",
    )
    chat_prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        ("system", "خلاصه گفتگو تا این لحظه: \n {conversation_summary}"),
        MessagesPlaceholder("recent_history"),
        ("human", "{user_input}"),
    ])
    summary_prompt = ChatPromptTemplate.from_messages([
        ("system", "تو خلاصه‌ساز دقیق و فارسی هستی که نکات احساسی و عملی را برجسته می‌کند."),
        ("human", "خلاصه فعلی:\n{existing}\n\nبخش جدید برای ادغام:\n{chunk}\n\nخلاصه به‌روزشده را ارائه کن."),
    ])

    chat_chain = chat_prompt | llm
    summary_chain = summary_prompt | summary_llm
    redis_client = redis.from_url(REDIS_URL)

    return chat_chain, summary_chain, redis_client


def fetch_summary_text(session_id, client):
    data = client.get(summary_key(session_id))
    return data.decode("utf-8") if data else ""


def update_summary(session_id, summary_chain, client):
    history = get_history(session_id)
    messages = history.messages
    total = len(messages)
    if total <= MAX_RECENT:
        return
    summarized_upto = int(client.get(cursor_key(session_id)) or 0)
    summarizable_end = total - MAX_RECENT
    if summarizable_end <= summarized_upto:
        return
    chunk_msgs = messages[summarized_upto:summarizable_end]
    chunk_text = build_lines(chunk_msgs)
    if not chunk_text.strip():
        client.set(cursor_key(session_id), summarizable_end)
        return
    existing_text = fetch_summary_text(session_id, client)
    summary = summary_chain.invoke({"existing": existing_text, "chunk": chunk_text})
    client.set(summary_key(session_id), summary.content.strip())
    client.set(cursor_key(session_id), summarizable_end)


if "authenticated" not in st.session_state:
    st.session_state.authenticated = False
if "username" not in st.session_state:
    st.session_state.username = ""

if not st.session_state.authenticated:
    st.subheader("ورود به اپ")
    with st.form("login_form", clear_on_submit=False):
        username = st.text_input("نام کاربری (فقط حروف انگلیسی)")
        password = st.text_input("کلمه عبور", type="password")
        submitted = st.form_submit_button("ورود")
        if submitted:
            if username.strip() and password:
                st.session_state.authenticated = True
                st.session_state.username = username.strip()
                st.rerun()
            else:
                st.error("نام کاربری و کلمه عبور را کامل وارد کنید")

else:
    chat_chain, summary_chain, redis_client = load_chain()
    username = st.session_state.username
    st.success(f"خوش آمدی {username}")
    user_history = get_history(username)
    convo_summary = fetch_summary_text(username, redis_client)
    with st.sidebar:
        st.header("خلاصه فشرده")
        st.write(convo_summary or "هنوز خلاصه‌ای تولید نشده است.")
        st.header("مرور سریع پیام‌ها")
        sidebar_messages = user_history.messages[-10:]
        for msg in sidebar_messages:
            speaker = "کاربر" if msg.type == "human" else "دستیار"
            st.write(f"{speaker}: {msg.content}")
        if st.button("خروج"):
            st.session_state.authenticated = False
            st.session_state.username = ""
            st.rerun()

    for msg in user_history.messages:
        role = "user" if msg.type == "human" else "assistant"
        with st.chat_message(role):
            st.markdown(msg.content)

    prompt_text = st.chat_input("احساس خود را اینجا بنویس")
    if prompt_text:
        recent_messages = user_history.messages[-MAX_RECENT:]
        summary_for_prompt = convo_summary if convo_summary else "هنوز خلاصه‌ای تولید نشده است."
        response = chat_chain.invoke({
            "conversation_summary": summary_for_prompt,
            "recent_history": recent_messages,
            "user_input": prompt_text,
        })
        user_history.add_message(HumanMessage(content=prompt_text))
        user_history.add_message(AIMessage(content=response.content.strip()))
        update_summary(username, summary_chain, redis_client)
        st.rerun()
