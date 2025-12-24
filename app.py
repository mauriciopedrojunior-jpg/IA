import os
from dotenv import load_dotenv

load_dotenv()  # carrega OPENAI_API_KEY do .env


from langchain_openai import ChatOpenAI

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory

from langchain_community.chat_message_histories import ChatMessageHistory


# ==============================
# MODELO
# ==============================
llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0.7
)


# ==============================
# PROMPT (CORRETO)
# ==============================
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "Você é um Hacker com mais 20 anos experiência em  invasao de sistemas"),
        MessagesPlaceholder(variable_name="history"),
        ("human", "{input}")
    ]
)


# ==============================
# CHAIN
# ==============================
chain = prompt | llm


# ==============================
# MEMÓRIA DE CONVERSA
# ==============================
store = {}

def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]


chat_with_history = RunnableWithMessageHistory(
    chain,
    get_session_history,
    input_messages_key="input",
    history_messages_key="history",
)


# ==============================
# LOOP DE CHAT
# ==============================
print("📐 Hacker iniciado! Digite 'sair' para encerrar.\n")

session_id = "aluno_1"

while True:
    user_input = input("Aluno: ")

    if user_input.lower() in ["sair", "exit", "quit"]:
        print("👋 Até mais!")
        break

    response = chat_with_history.invoke(
        {"input": user_input},
        config={"configurable": {"session_id": session_id}}
    )

    print(f"📘 Professor: {response.content}\n")

