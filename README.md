# 🤖 AI Assistant with LangGraph, Groq & Streamlit

This project is an **end-to-end AI chatbot application** built using **LangGraph**, **Groq LLM**, and **Streamlit**.
It demonstrates how to design a **multi-node AI workflow**, analyze **user sentiment**, and present results in a **modern UI** with a background image and chat bubbles.

---

## 📌 Project Architecture (Working Flow)

```
User Input (Streamlit UI)
        ↓
LangGraph StateGraph
        ↓
[ Preprocess Node ]
        ↓
[ Sentiment Analyzer Node ]
        ↓
[ LLM Chatbot Node (Groq) ]
        ↓
[ Logger Node ]
        ↓
Response + Sentiment
        ↓
Displayed in Streamlit UI
```

---

## 🧠 Core Components

### 1️⃣ Streamlit (Frontend/UI)

* Handles **user input**
* Displays **chat messages**
* Applies **custom CSS & background image**
* Maintains **chat history** using session state

### 2️⃣ LangGraph (Workflow Engine)

* Manages **step-by-step execution**
* Controls how data flows between nodes
* Ensures clean, modular, and scalable AI pipelines

### 3️⃣ Groq LLM (Brain)

* Uses `llama-3.3-70b-versatile`
* Generates intelligent responses
* Integrated using `ChatGroq`

---

## 🧠 State Definition

```python
class State(TypedDict):
    messages: Annotated[list, add_messages]
    sentiment: str
```

### Why State?

* Holds **conversation messages**
* Stores **sentiment result**
* Shared across all LangGraph nodes

---

## 🔹 LangGraph Nodes Explained

### 🔹 1. Preprocess Node

```python
def preprocess(state: State) -> State:
    state["messages"][-1].content = state["messages"][-1].content.strip()
    return state
```

✅ Cleans user input
✅ Removes unwanted spaces
✅ Improves response quality

---

### 🔹 2. Sentiment Analyzer Node

```python
def analyze_sentiment(state: State) -> State:
```

**Logic:**

* Checks keywords like `good`, `great`, `bad`, `sad`
* Assigns:

  * 😊 Positive
  * 😞 Negative
  * 😐 Neutral

✅ Lightweight sentiment analysis
✅ Fast and rule-based

---

### 🔹 3. Chatbot Node (Groq LLM)

```python
def chatbot(state: State) -> State:
    response = llm.invoke(state["messages"])
    return {"messages": state["messages"] + [response]}
```

✅ Sends messages to Groq LLM
✅ Receives AI-generated response
✅ Appends reply to conversation history

---

### 🔹 4. Logger Node

```python
def logger(state: State) -> State:
```

✅ Logs conversation in terminal
✅ Helps in debugging & monitoring

---

## 🔁 LangGraph Workflow

```python
START
 ↓
preprocess
 ↓
sentiment
 ↓
chatbot
 ↓
logger
 ↓
END
```

✔ Fully sequential
✔ Easy to extend (RAG, Memory, Tools, APIs)

---

## 🎨 UI & Styling

### Background Image

* Loaded from local system
* Converted to **Base64**
* Applied using CSS

### Chat Bubbles

* User: Right aligned (green)
* Bot: Left aligned (white)
* Glass-morphism container

---

## 🖥️ Streamlit Execution Flow

1. User types a message
2. Clicks **Send 🚀**
3. Input passed to LangGraph
4. Each node executes in order
5. Final response + sentiment returned
6. Displayed in chat UI
7. Chat history preserved

---

## 🔐 API Key Handling (Important)

⚠️ **Do NOT push API keys to GitHub**

Instead use:

```bash
export GROQ_API_KEY="your_api_key_here"
```

or `.env` file.

---

## 📦 End-to-End Execution Summary

| Step | Action                     |
| ---- | -------------------------- |
| 1    | User enters message        |
| 2    | Input cleaned (Preprocess) |
| 3    | Sentiment detected         |
| 4    | LLM generates response     |
| 5    | Interaction logged         |
| 6    | Response displayed in UI   |

---

## 🚀 How to Run the Project

```bash
pip install streamlit langchain langgraph langchain-groq
streamlit run app.py
```

---

## 📌 Features

* ✅ Multi-node LangGraph pipeline
* ✅ Groq LLM integration
* ✅ Sentiment analysis
* ✅ Modern Streamlit UI
* ✅ Background image & chat bubbles
* ✅ Session-based chat history

---

## 🔮 Future Enhancements

* 🔍 RAG (PDF / Docs / CSV)
* 🧠 Memory & conversation summarization
* 🗣️ Voice input/output
* 🌐 Deployment (Docker / Cloud)

---

## 🖥️ Application Screenshots


<img width="1916" height="875" alt="Screenshot 2025-12-16 124610" src="https://github.com/user-attachments/assets/7bca9563-4adc-4c91-82e0-6ad8c8e81181" />
<img width="1919" height="857" alt="Screenshot 2025-12-16 124645" src="https://github.com/user-attachments/assets/4caaf3d5-42c4-46cf-8a89-5e422d89c5ec" />




## 👨‍💻 Author

**Parthasarathi Behera**
Data Analyst | AI & LangChain Enthusiast


