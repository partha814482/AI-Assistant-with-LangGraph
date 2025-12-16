import os
import base64
import streamlit as st
from typing import Annotated
from typing_extensions import TypedDict

from langchain_groq import ChatGroq
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain_core.messages import HumanMessage

# =====================================================
# 🔐 API KEY (use env variable in real projects)
# =====================================================
os.environ["GROQ_API_KEY"] = ""

# =====================================================
# 🤖 LLM
# =====================================================
llm = ChatGroq(
    model_name="llama-3.3-70b-versatile",
    temperature=0.4,
    groq_api_key=os.environ["GROQ_API_KEY"]
)

# =====================================================
# 🧠 State
# =====================================================
class State(TypedDict):
    messages: Annotated[list, add_messages]
    sentiment: str

# =====================================================
# 🔹 LangGraph Nodes
# =====================================================
def preprocess(state: State) -> State:
    state["messages"][-1].content = state["messages"][-1].content.strip()
    return state

def analyze_sentiment(state: State) -> State:
    msg = state["messages"][-1].content.lower()
    if "good" in msg or "great" in msg:
        state["sentiment"] = "Positive 😊"
    elif "bad" in msg or "sad" in msg:
        state["sentiment"] = "Negative 😞"
    else:
        state["sentiment"] = "Neutral 😐"
    return state

def chatbot(state: State) -> State:
    response = llm.invoke(state["messages"])
    return {"messages": state["messages"] + [response]}

def logger(state: State) -> State:
    print(f"LOG ➜ {state['messages'][-1].content} | Sentiment: {state['sentiment']}")
    return state

# =====================================================
# 🔁 Build LangGraph
# =====================================================
builder = StateGraph(State)

builder.add_node("preprocess", preprocess)
builder.add_node("sentiment", analyze_sentiment)
builder.add_node("chatbot", chatbot)
builder.add_node("logger", logger)

builder.add_edge(START, "preprocess")
builder.add_edge("preprocess", "sentiment")
builder.add_edge("sentiment", "chatbot")
builder.add_edge("chatbot", "logger")
builder.add_edge("logger", END)

graph = builder.compile()

# =====================================================
# 🎨 BACKGROUND IMAGE FUNCTION
# =====================================================
def add_bg_from_local(image_file):
    with open(image_file, "rb") as f:
        encoded = base64.b64encode(f.read()).decode()

    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url("data:image/jpg;base64,{encoded}");
            background-size: cover;
            background-position: center;
            background-repeat: no-repeat;
            background-attachment: fixed;
        }}

        /* Glass container */
        .block-container {{
            background-color: rgba(0, 0, 0, 0.65);
            padding: 2rem;
            border-radius: 18px;
        }}

        /* User chat bubble */
        .chat-user {{
            background-color: rgba(220, 248, 198, 0.95);
            color: #000000;
            padding: 12px;
            border-radius: 12px;
            text-align: right;
            margin-bottom: 10px;
        }}

        /* Bot chat bubble */
        .chat-bot {{
            background-color: rgba(255, 255, 255, 0.95);
            color: #000000;
            padding: 12px;
            border-radius: 12px;
            margin-bottom: 10px;
        }}

        /* Input box */
        input {{
            background-color: rgba(255,255,255,0.95) !important;
            color: black !important;
        }}

        /* Button */
        button {{
            color: white !important;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

# APPLY BACKGROUND
add_bg_from_local(r"C:\Users\HP\OneDrive\Documents\data science\Langchain\Langgraph\istockphoto-1288199537-612x612.jpg")

# =====================================================
# 🖥️ Streamlit UI
# =====================================================
st.set_page_config(
    page_title="AI Assistant with LangGraph",
    page_icon="🤖",
    layout="centered"
)

st.title("😊 AI Assistant with LangGraph")
st.caption("Powered by Groq + LangGraph + Streamlit")

# =====================================================
# 💬 Session State
# =====================================================
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# =====================================================
# 💬 Input
# =====================================================
user_input = st.text_input("Ask something:")

if st.button("Send 🚀") and user_input:
    result = graph.invoke(
        {"messages": [HumanMessage(content=user_input)]}
    )

    bot_reply = result["messages"][-1].content
    sentiment = result["sentiment"]

    st.session_state.chat_history.append(("user", user_input))
    st.session_state.chat_history.append(("bot", bot_reply, sentiment))

# =====================================================
# 🗨️ Chat Display
# =====================================================
for msg in st.session_state.chat_history:
    if msg[0] == "user":
        st.markdown(
            f"<div class='chat-user'>👤 {msg[1]}</div>",
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            f"<div class='chat-bot'>🤖 {msg[1]}<br><b>Sentiment:</b> {msg[2]}</div>",
            unsafe_allow_html=True
        )

st.markdown("---")
st.caption("🧠 Multi-Node LangGraph • 🎨 Background Image • 🚀 Production UI")
