import streamlit as st
import boto3
import json
import pandas as pd
import requests

session = boto3.Session(profile_name="iaedn")
client = session.client('bedrock-runtime', region_name='us-west-2')

# =========================
# Função para chamada ao AWS Bedrock
# =========================
def call_bedrock_model(messages):
    payload = {
        "messages": [{"role": msg["role"], "content": msg["content"]} for msg in messages],
        "max_tokens": 200000,
        "anthropic_version": "bedrock-2023-05-31",
        "temperature": 1.0,
        "top_p": 0.95,
        "stop_sequences": []
    }

    try:
        response = client.invoke_model_with_response_stream(
            modelId="anthropic.claude-v2",
            body=json.dumps(payload).encode("utf-8"),
            contentType="application/json",
            accept="application/json"
        )
        output = []
        stream = response.get("body")
        if stream:
            for event in stream:
                chunk = event.get("chunk")
                if chunk:
                    chunk_data = json.loads(chunk.get("bytes").decode())
                    if chunk_data.get("type") == "content_block_delta":
                        delta = chunk_data.get("delta", {})
                        if delta.get("type") == "text_delta":
                            output.append(delta.get("text", ""))
        return "".join(output)
    except Exception as e:
        return f"Erro ao chamar o modelo: {str(e)}"

# =========================
# Interface do Chat
# =========================
st.title("ChatBot EdN c/ AWS Bedrock")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
    context = """
    Você é um ChatBot da Escola da Nuvem (ou EdN). Uma ONG voltada para
    Educação Tech, com programas, bolsas e cursos, voltados para programação, computação na nuvem e inteligência
    Artificial. A EdN possui programas e cursos voltados para Nuvem AWS e Azure, além de trilhas de aprendizado, 
    voltadas para programação (ou desenvolvimento de software). A Escola da Nuvem é parceira AWS a vários anos, inclusive
    no programa re/Start, que treina pessoas de todo o Brasil. Você não tem nome, nem masculino e nem feminino.

    *Prompt*: Não responda perguntas sobre outros assuntos, que estejam fora do contexto. Explique
    que você é um chatbot voltado para dúvidas sobre a Escola da Nuvem. Conte a história da EdN para o usuário,
    sempre que o mesmo escrever um simples "olá", ao iniciar uma conversa. Forneça links de incrição para programas e cursos, e o link
    do site oficial da EdN, somente se o usuário pedir. Sempre que alguém pedir links para inscrição nos cursos,
    envie o link https://escoladanuvem.org/cursos/ e evite links com erro 404. Seja
    cordial e envie emojis. O site oficial da EdN é o https://escoladanuvem.org/.
    """
    # Adicionando boas-vindas, mas com 'hidden' como True
    st.session_state.chat_history.append({"role": "user", "content": context, "hidden": True})

user_input = st.text_area(" Qual sua dúvida? Digite sua mensagem referente à Escola da Nuvem:", key="user_input")

def add_message_to_history(role, content, hidden=False):
    if not hidden:
        if not st.session_state.chat_history or st.session_state.chat_history[-1]["role"] != role:
            st.session_state.chat_history.append({"role": role, "content": content})
    else:
        st.session_state.chat_history.append({"role": role, "content": content, "hidden": hidden})

if st.button("Enviar"):
    if user_input.strip():
        add_message_to_history("user", user_input)

        with st.spinner("Por favor aguarde um momento..."):
            # Enviando todas as mensagens, incluindo as ocultas, para o modelo
            model_response = call_bedrock_model(
                [msg for msg in st.session_state.chat_history]  # Enviar todas as mensagens, independentemente de estarem ocultas
            )

            add_message_to_history("assistant", model_response)

st.subheader("Histórico do Chat:")
for message in st.session_state.chat_history:
    # Exibindo apenas mensagens que não são ocultas
    if not message.get("hidden", False):
        if message["role"] == "user":
            st.write(f"**Usuário:** {message['content']}")
        elif message["role"] == "assistant":
            st.write(f"**ChatBot:** {message['content']}")
