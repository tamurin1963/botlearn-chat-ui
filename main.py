# ライブラリのインポート
from openai import AzureOpenAI
from openai.types.chat import (
    ChatCompletionUserMessageParam,
    ChatCompletionSystemMessageParam,
    ChatCompletionAssistantMessageParam,
    ChatCompletionMessageParam,
    ChatCompletionChunk
)
from dotenv import load_dotenv
import streamlit as st
import os
from typing import cast, List

# 環境変数の読み込み
load_dotenv()
api_version = os.getenv("CHATBOT_AZURE_OPENAI_API_VERSION", "")
azure_endpoint = os.getenv("CHATBOT_AZURE_OPENAI_ENDPOINT", "")
api_key = os.getenv("CHATBOT_AZURE_OPENAI_API_KEY", "")
deployment_name = os.getenv("CHATBOT_AZURE_OPENAI_DEPLOYMENT_NAME", "")

from pathlib import Path
print("📁 .env exists?", Path(".env").exists())
print("🔍 raw deployment:", os.environ.get("CHATBOT_AZURE_OPENAI_DEPLOYMENT_NAME"))
print("🔍 via getenv:", os.getenv("CHATBOT_AZURE_OPENAI_DEPLOYMENT_NAME"))

assert deployment_name, "環境変数 CHATBOT_AZURE_OPENAI_DEPLOYMENT_NAME が未設定です"

# デバッグログ
print("🔍 api_version:", api_version)
print("🔍 endpoint:", azure_endpoint)
print("🔍 deployment_name:", deployment_name)

# Azure OpenAI クライアントの初期化（← これが大事！）
client = AzureOpenAI(
    api_key=api_key,
    azure_endpoint=azure_endpoint,
    api_version=api_version
)

# チャット履歴の初期化
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# 応答生成関数
def get_response(prompt: str) -> str:
    st.session_state.chat_history.append(
        ChatCompletionUserMessageParam(role="user", content=prompt)
    )

    system_message = ChatCompletionSystemMessageParam(
        role="system", content="You are a helpful assistant."
    )

    response_stream = client.chat.completions.create(
        model=deployment_name,  # ← 環境変数から読み込んだ変数を使用
        messages=[system_message] + st.session_state.chat_history,
        stream=True
    )

    full_response = ""
    for raw_chunk in response_stream:
        chunk = cast(ChatCompletionChunk, raw_chunk)
        if chunk.choices and chunk.choices[0].delta and chunk.choices[0].delta.content:
            full_response += chunk.choices[0].delta.content

    return full_response

# 応答履歴の追加関数
def add_history(response: str):
    st.session_state.chat_history.append(
        ChatCompletionAssistantMessageParam(role="assistant", content=response)
    )

# Streamlit UI
st.title("💬 ChatGPT-like clone（Azure OpenAI × Streamlit）")

for chat in st.session_state.chat_history:
    with st.chat_message(chat.role):
        st.markdown(chat.content)

prompt = st.chat_input("何か話しかけてみてください！")
if prompt:
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        response_text = get_response(prompt)
        st.markdown(response_text)
        add_history(response_text)
