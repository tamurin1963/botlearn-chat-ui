# ライブラリのインポート
from openai import AzureOpenAI
from openai.types.chat import (
    ChatCompletionUserMessageParam,
    ChatCompletionAssistantMessageParam,
    ChatCompletionSystemMessageParam,
    ChatCompletionMessageParam
)
from dotenv import load_dotenv
import streamlit as st
import os

# 環境変数の読み込み
load_dotenv()

# Azure OpenAI 接続情報
azure_endpoint = os.environ["CHATBOT_AZURE_OPENAI_ENDPOINT"]
api_key = os.environ["CHATBOT_AZURE_OPENAI_API_KEY"]
deployment_name = "Botlearn-gpt-4o-mini"
api_version = "2024-08-01-preview"

# クライアントの初期化
client = AzureOpenAI(
    azure_endpoint=azure_endpoint,
    api_key=api_key,
    api_version=api_version
)

# チャット履歴の初期化（型注釈はコメントで対応）
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []  # type: ignore
# or 型だけ伝えたい場合：
# st.session_state.chat_history = []  # type: list[ChatCompletionMessageParam]st.session_state.chat_history = []  # type: list[ChatCompletionMessageParam]

# 応答生成関数
def get_response(prompt: str) -> str:
    # ユーザーメッセージの型付きオブジェクトとして履歴に追加
    st.session_state.chat_history.append(
        ChatCompletionUserMessageParam(role="user", content=prompt)
    )

    # システムメッセージ（Botの性格定義）
    system_message = ChatCompletionSystemMessageParam(
        role="system", content="You are a helpful assistant."
    )

    # ストリームレスポンスの生成
    response_stream = client.chat.completions.create(
        model=deployment_name,
        messages=[system_message] + st.session_state.chat_history,
        stream=True
    )

    # delta.content を順次抽出して文字列化
    full_response = ""
    for chunk in response_stream:
        if chunk.choices and chunk.choices[0].delta and chunk.choices[0].delta.content:
            full_response += chunk.choices[0].delta.content


    return full_response

# Bot応答の履歴追加関数
def add_history(response: str):
    st.session_state.chat_history.append(
        ChatCompletionAssistantMessageParam(role="assistant", content=response)
    )

# UI表示：タイトル
st.title("💬 ChatGPT-like clone（Azure OpenAI × Streamlit）")

# チャット履歴の表示
for chat in st.session_state.chat_history:
    with st.chat_message(chat.role):
        st.markdown(chat.content)

# ユーザー入力の受付と処理
prompt = st.chat_input("何か話しかけてみてください！")
if prompt:
    # 入力表示
    with st.chat_message("user"):
        st.markdown(prompt)

    # Bot応答の生成と表示
    with st.chat_message("assistant"):
        response_text = get_response(prompt)
        st.markdown(response_text)

    # 応答履歴に追加
    add_history(response_text)

