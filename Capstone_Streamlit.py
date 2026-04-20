
import streamlit as st
from agent import ask

st.set_page_config(page_title="Capstone AI Assistant")

st.title("🤖 Capstone AI Assistant")

# ================= SESSION =================
if "messages" not in st.session_state:
    st.session_state.messages = []

if "thread_id" not in st.session_state:
    st.session_state.thread_id = "thread_1"


# ================= SIDEBAR =================
with st.sidebar:
    st.header("About")
    st.write("Replace this with your domain description")

    if st.button("New Conversation"):
        st.session_state.messages = []
        st.session_state.thread_id = "thread_new"


# ================= CHAT =================
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])


# ================= INPUT =================
user_input = st.chat_input("Ask your question...")

if user_input:
    # user message
    st.session_state.messages.append({"role": "user", "content": user_input})

    with st.chat_message("user"):
        st.write(user_input)

    # assistant response
    response = ask(user_input, st.session_state.thread_id)

    st.session_state.messages.append({"role": "assistant", "content": response})

    with st.chat_message("assistant"):
        st.write(response)
