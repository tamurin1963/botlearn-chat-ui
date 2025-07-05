# ライブラリのインポート
from openai import AzureOpenAI
from openai.types.chat import (
    ChatCompletionUserMessageParam,
    ChatCompletionSystemMessageParam,
    ChatCompletionAssistantMessageParam,
    ChatCompletionMessageParam
)
from dotenv import load_dotenv
import streamlit as st
import os
from typing import cast
from openai.types.chat import ChatCompletionCh
# 環境変数の読み込み
load_dotenv()

# Azure OpenAI 接続情報（.envに準拠！）
client = AzureOpenAI(
    api_version=os.getenv("CHATBOT_AZURE_OPENAI_API_VERSION", ""),
    azure_endpoint=os.getenv("CHATBOT_AZURE_OPENAI_ENDPOINT", ""),
    api_key=os.getenv("CHATBOT_AZURE_OPENAI_API_KEY", "")
)

deployment_name = os.getenv("CHATBOT_AZURE_OPENAI_DEPLOYMENT_NAME")  # ← 上書きなし！

# チャット履歴の初期化
if "chat_history" not in st.session_state:
    st.session_state.chat_history: List[ChatCompletionMessageParam] = []

# 応答生成関数
def get_response(prompt: str) -> str:
    st.session_state.chat_history.append(
        ChatCompletionUserMessageParam(role="user", content=prompt)
    )

    system_message = ChatCompletionSystemMessageParam(
        role="system", content="You are a helpful assistant."
    )

    response_stream = client.chat.completions.create(
    model=deployment_name,  # ← ここは str 型が保証されているので安心！
    messages=[system_message] + st.session_state.chat_history,
    stream=True
)


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
