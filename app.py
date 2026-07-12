import streamlit as st
import streamlit.components.v1 as components
import os
import requests
import time
import hashlib
import logging
import json
from typing import Dict, List, Tuple, Any, Optional
import sqlite3
from contextlib import contextmanager

# Configuração de logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("chatbot_edn")

# Configurações
MAX_HISTORY_TOKENS = 800
MAX_INPUT_LENGTH = 512
CACHE_EXPIRY = 3600  # Cache expira em 1 hora (segundos)
MAX_RETRIES = 3
API_TIMEOUT = 30
MAX_CONVERSATION_TURNS = 10

# API OpenRouter (substitui o AWS Bedrock)
API_URL = "https://openrouter.ai/api/v1/chat/completions"
MODEL = "openai/gpt-3.5-turbo"
MAX_RESPONSE_TOKENS = 400

SYSTEM_PROMPT = (
    "Você é o ChatBot da Escola da Nuvem (EdN), uma ONG brasileira de educação em "
    "tecnologia. A EdN oferece programas, bolsas e cursos gratuitos voltados para "
    "computação em nuvem (AWS e Azure), inteligência artificial e programação/desenvolvimento "
    "de software, além de trilhas de aprendizado. A EdN é parceira da AWS há vários anos, "
    "inclusive no programa AWS re/Start, que treina pessoas de todo o Brasil. "
    "Você não tem nome, nem gênero masculino ou feminino.\n\n"
    "REGRAS:\n"
    "1. Responda SEMPRE em português do Brasil, de forma cordial e com emojis.\n"
    "2. Responda apenas sobre a Escola da Nuvem. Se perguntarem sobre outro assunto, "
    "explique gentilmente que você é um chatbot voltado a dúvidas sobre a EdN.\n"
    "3. Quando o usuário apenas cumprimentar (ex.: 'olá', 'oi'), conte brevemente a "
    "história da EdN e convide-o a perguntar sobre os cursos.\n"
    "4. Envie links somente quando o usuário pedir. Use apenas estes dois links, "
    "nunca invente outros: cursos e inscrições em https://escoladanuvem.org/cursos/ "
    "e site oficial em https://escoladanuvem.org/\n"
    "5. Nunca invente datas, valores, requisitos ou nomes de cursos. Se não souber, "
    "oriente o usuário a consultar o site oficial.\n"
    "6. Seja conciso: 2 a 5 frases por resposta."
)

# Banco de dados para cache
DB_PATH = "chat_cache.db"


@contextmanager
def get_db_connection():
    """Gerencia conexões com o banco de dados."""
    conn = None
    try:
        conn = sqlite3.connect(DB_PATH)
        conn.row_factory = sqlite3.Row
        yield conn
    finally:
        if conn:
            conn.close()


def init_db():
    """Inicializa o banco de dados para o cache."""
    try:
        with get_db_connection() as conn:
            conn.execute('''
            CREATE TABLE IF NOT EXISTS cache (
                query_hash TEXT PRIMARY KEY,
                response TEXT NOT NULL,
                timestamp INTEGER NOT NULL
            )
            ''')
            conn.commit()
            logger.info("Banco de dados para cache inicializado")
    except Exception as e:
        logger.error(f"Erro ao inicializar banco de dados: {e}")
        # Continua sem cache se não conseguir inicializar o banco de dados


class SecretManager:
    """Gerencia o acesso seguro aos segredos."""
    @staticmethod
    def get_api_key() -> Optional[str]:
        """Obtém a chave da API do secrets.toml ou da variável de ambiente."""
        api_key = None

        try:
            api_key = st.secrets["openrouter"]["OPENROUTER_API_KEY"]
        except Exception:
            logger.info("Chave não encontrada em st.secrets; tentando variável de ambiente")

        if not api_key:
            api_key = os.getenv("OPENROUTER_API_KEY")

        if not api_key:
            logger.error("Chave de API não encontrada")
            return None

        if len(api_key) < 8:
            logger.warning("Chave de API suspeita - muito curta")

        return api_key


class TokenManager:
    """Gerencia o tamanho do histórico de conversas."""
    @staticmethod
    def estimate_tokens(text: str) -> int:
        """Estimativa aproximada de tokens em um texto (~4 caracteres por token)."""
        return len(text) // 4

    @staticmethod
    def truncate_history(
        past_inputs: List[str],
        past_responses: List[str],
        max_tokens: int = MAX_HISTORY_TOKENS
    ) -> Tuple[List[str], List[str]]:
        """Trunca o histórico para não exceder o máximo de tokens, preservando o mais recente."""
        inputs_copy = past_inputs.copy()
        responses_copy = past_responses.copy()

        if len(inputs_copy) > MAX_CONVERSATION_TURNS:
            start_idx = len(inputs_copy) - MAX_CONVERSATION_TURNS
            inputs_copy = inputs_copy[start_idx:]
            responses_copy = responses_copy[start_idx:]
            logger.info(f"Histórico limitado a {MAX_CONVERSATION_TURNS} turnos")
            return inputs_copy, responses_copy

        total_tokens = sum(TokenManager.estimate_tokens(msg) for msg in inputs_copy + responses_copy)

        if total_tokens <= max_tokens:
            return inputs_copy, responses_copy

        while total_tokens > max_tokens and inputs_copy and responses_copy:
            oldest_input = inputs_copy.pop(0)
            oldest_response = responses_copy.pop(0)
            total_tokens -= (TokenManager.estimate_tokens(oldest_input) +
                             TokenManager.estimate_tokens(oldest_response))

        logger.info(f"Histórico truncado para {len(inputs_copy)} interações ({total_tokens} tokens est.)")
        return inputs_copy, responses_copy


class APICache:
    """Implementa cache para chamadas à API."""
    @staticmethod
    def compute_hash(data: str) -> str:
        return hashlib.md5(data.encode()).hexdigest()

    @staticmethod
    def get_cached_response(query: str) -> Optional[str]:
        """Recupera resposta em cache se existir e estiver válida."""
        try:
            query_hash = APICache.compute_hash(query)
            current_time = int(time.time())

            with get_db_connection() as conn:
                result = conn.execute(
                    "SELECT response, timestamp FROM cache WHERE query_hash = ?",
                    (query_hash,)
                ).fetchone()

                if result and (current_time - result['timestamp']) < CACHE_EXPIRY:
                    logger.info("Cache hit!")
                    return result['response']
        except Exception as e:
            logger.error(f"Erro ao buscar no cache: {e}")

        logger.info("Cache miss")
        return None

    @staticmethod
    def cache_response(query: str, response: str) -> None:
        """Armazena uma resposta no cache."""
        try:
            query_hash = APICache.compute_hash(query)
            current_time = int(time.time())

            with get_db_connection() as conn:
                conn.execute(
                    "INSERT OR REPLACE INTO cache (query_hash, response, timestamp) VALUES (?, ?, ?)",
                    (query_hash, response, current_time)
                )
                conn.commit()
                logger.info("Resposta armazenada em cache")
        except Exception as e:
            logger.error(f"Erro ao armazenar no cache: {e}")


class OpenRouterClient:
    """Cliente para a API do OpenRouter (chat completions)."""
    def __init__(self):
        self.api_key = SecretManager.get_api_key()
        self.api_url = API_URL

        if not self.api_key:
            st.error("❌ Chave de API não configurada corretamente")
            st.stop()

    def query(self, messages: List[Dict[str, str]]) -> str:
        """Envia a conversa para a API com cache, tratamento de erros e retentativas."""
        cache_key = json.dumps(messages, ensure_ascii=False)
        cached_response = APICache.get_cached_response(cache_key)
        if cached_response:
            return cached_response

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        payload = {
            "model": MODEL,
            "messages": messages,
            "max_tokens": MAX_RESPONSE_TOKENS,
            "temperature": 0.5,
        }

        for attempt in range(1, MAX_RETRIES + 1):
            try:
                logger.info(f"Tentativa {attempt} de {MAX_RETRIES}")

                response = requests.post(
                    self.api_url,
                    headers=headers,
                    json=payload,
                    timeout=API_TIMEOUT
                )

                if response.status_code == 400:
                    logger.error(f"Erro 400: {response.text}")
                    return "⚠️ Erro na requisição: formato inválido. Tente uma mensagem mais curta."

                response.raise_for_status()

                data = response.json()
                result = self._extract_response(data)

                if not result or len(result) < 2:
                    logger.warning("Resposta muito curta ou vazia")
                    if attempt < MAX_RETRIES:
                        time.sleep(2 ** attempt)
                        continue
                    result = "Desculpe, não consegui formular uma boa resposta. Pode reformular a pergunta? 🙂"

                APICache.cache_response(cache_key, result)
                return result

            except requests.exceptions.HTTPError as e:
                if e.response.status_code == 429:
                    logger.warning("Rate limit excedido. Aguardando antes de tentar novamente.")
                    time.sleep(2 ** attempt)
                elif e.response.status_code >= 500:
                    logger.error(f"Erro no servidor: {e}")
                    if attempt < MAX_RETRIES:
                        time.sleep(2 ** attempt)
                    else:
                        return "⚠️ Erro no servidor da API. Tente novamente mais tarde."
                else:
                    logger.error(f"Erro na requisição HTTP: {e}")
                    return f"⚠️ Erro na requisição: {e.response.status_code}"

            except requests.exceptions.Timeout:
                logger.error("Timeout na requisição")
                if attempt < MAX_RETRIES:
                    time.sleep(2 ** attempt)
                else:
                    return "⚠️ Tempo esgotado ao aguardar resposta da API."

            except requests.exceptions.RequestException as e:
                logger.error(f"Erro de conexão: {e}")
                return "⚠️ Erro de conexão com a API."

            except json.JSONDecodeError:
                logger.error("Resposta não é um JSON válido")
                if attempt < MAX_RETRIES:
                    time.sleep(2 ** attempt)
                else:
                    return "⚠️ Resposta inválida da API."

            except Exception as e:
                logger.error(f"Erro desconhecido: {e}")
                return f"⚠️ Ocorreu um erro: {str(e)}"

        return "⚠️ Não foi possível obter uma resposta após várias tentativas."

    def _extract_response(self, data: Any) -> str:
        """Extrai o texto da resposta no formato chat completions."""
        try:
            choices = data.get("choices") if isinstance(data, dict) else None
            if choices and isinstance(choices, list):
                content = choices[0].get("message", {}).get("content", "")
                if content:
                    return content.strip()
            if isinstance(data, dict) and "error" in data:
                error_msg = data["error"]
                if isinstance(error_msg, dict):
                    error_msg = error_msg.get("message", str(error_msg))
                logger.error(f"API retornou erro: {error_msg}")
                return f"⚠️ Erro da API: {error_msg}"
            logger.warning(f"Formato de resposta inesperado: {json.dumps(data)[:200]}")
        except Exception as e:
            logger.error(f"Erro ao extrair resposta: {e}")

        return "⚠️ Formato de resposta inesperado da API."


class ChatSession:
    """Gerencia a sessão de chat e o histórico."""
    def __init__(self):
        self.client = OpenRouterClient()

        if "user_inputs" not in st.session_state:
            st.session_state.user_inputs = []
            st.session_state.bot_responses = []

    def add_message(self, user_input: str) -> str:
        """Adiciona uma mensagem do usuário, gera e devolve a resposta."""
        if not user_input or len(user_input.strip()) == 0:
            return "⚠️ Por favor, digite uma mensagem válida."

        if len(user_input) > MAX_INPUT_LENGTH:
            user_input = user_input[:MAX_INPUT_LENGTH]
            logger.warning(f"Entrada truncada para {MAX_INPUT_LENGTH} caracteres")

        st.session_state.user_inputs.append(user_input)

        truncated_inputs, truncated_responses = TokenManager.truncate_history(
            st.session_state.user_inputs[:-1],
            st.session_state.bot_responses
        )

        messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        for past_input, past_response in zip(truncated_inputs, truncated_responses):
            messages.append({"role": "user", "content": past_input})
            messages.append({"role": "assistant", "content": past_response})
        messages.append({"role": "user", "content": user_input})

        try:
            reply = self.client.query(messages)
        except Exception as e:
            logger.error(f"Erro ao gerar resposta: {e}")
            reply = "⚠️ Ocorreu um erro ao processar sua mensagem. Tente novamente ou reformule."

        st.session_state.bot_responses.append(reply)
        return reply

    def get_history(self) -> List[dict]:
        """Retorna o histórico de conversas em formato estruturado."""
        history = []
        for i, user_msg in enumerate(st.session_state.user_inputs):
            history.append({"role": "user", "content": user_msg})
            if i < len(st.session_state.bot_responses):
                history.append({"role": "assistant", "content": st.session_state.bot_responses[i]})
        return history


class ChatUI:
    """Gerencia a interface do usuário do chat."""
    def __init__(self):
        st.set_page_config(
            page_title="ChatBot EdN c/ IA",
            page_icon="⛅",
            layout="centered",
            initial_sidebar_state="auto"
        )

        # Fallback estático para a cor da barra de status mobile
        st.markdown('<meta name="theme-color" content="#000000">', unsafe_allow_html=True)

        st.markdown("""
        <style>
        /* Barra de rolagem preta (fallback; injeção principal via JS) */
        * { scrollbar-color: #000000 #f0f0f0; scrollbar-width: thin; }
        *::-webkit-scrollbar { width: 8px; height: 8px; }
        *::-webkit-scrollbar-thumb { background-color: #000000; border-radius: 4px; }
        *::-webkit-scrollbar-track { background: #f0f0f0; border-radius: 4px; }
        .user-message {
            background-color: #e6f7ff;
            border-radius: 15px;
            padding: 10px 15px;
            margin: 5px 0;
            border-bottom-right-radius: 5px;
            text-align: right;
            margin-left: 20%;
        }
        .bot-message {
            background-color: #f0f0f0;
            border-radius: 15px;
            padding: 10px 15px;
            margin: 5px 0;
            border-bottom-left-radius: 5px;
            margin-right: 20%;
        }
        .chat-header {
            text-align: center;
            margin-bottom: 20px;
        }
        .stButton button {
            background-color: #FF9900;
            color: white;
            border-radius: 5px;
            border: none;
            width: 100%;
        }
        .error-message {
            color: #721c24;
            background-color: #f8d7da;
            border: 1px solid #f5c6cb;
            border-radius: 5px;
            padding: 10px;
            margin: 10px 0;
        }
        </style>
        """, unsafe_allow_html=True)

    def render_header(self):
        """Renderiza o cabeçalho da página."""
        st.markdown("""
        <div class="chat-header">
            <h1><span style="font-family: 'Segoe UI Emoji', 'Apple Color Emoji', 'Noto Color Emoji', sans-serif;">⛅</span> ChatBot EdN (c/ IA)</h1>
            <p>Tire suas dúvidas sobre a Escola da Nuvem, cursos e programas</p>
        </div>
        """, unsafe_allow_html=True)

    def render_conversation(self, history: List[dict]):
        """Renderiza a conversa com estilos CSS."""
        for message in history:
            if message["role"] == "user":
                st.markdown(
                    f'<div class="user-message">👤 {message["content"]}</div>',
                    unsafe_allow_html=True
                )
            else:
                content = message["content"]
                if content.startswith("⚠️"):
                    st.markdown(
                        f'<div class="error-message">⛅ {content}</div>',
                        unsafe_allow_html=True
                    )
                else:
                    st.markdown(
                        f'<div class="bot-message">⛅ {content}</div>',
                        unsafe_allow_html=True
                    )

    def render_input_form(self) -> Tuple[str, bool]:
        """Renderiza o formulário de entrada."""
        with st.form("chat_form", clear_on_submit=True):
            user_input = st.text_input(
                "Sua dúvida:",
                placeholder="Digite sua mensagem sobre a Escola da Nuvem...",
                key="user_input"
            )
            col1, col2 = st.columns([4, 1])
            with col2:
                submitted = st.form_submit_button("Enviar")

        return user_input, submitted

    def focus_input(self):
        """Mantém o cursor no campo de digitação e a barra de status mobile preta."""
        # O comentário com timestamp força o iframe a recarregar (e o script a
        # rodar de novo) a cada rerun do Streamlit
        components.html(f"""
        <script>
            // rerun: {time.time()}
            function focusChatInput() {{
                const input = window.parent.document.querySelector('div[data-testid="stTextInput"] input');
                if (input) {{
                    input.focus();
                }} else {{
                    setTimeout(focusChatInput, 100);
                }}
            }}
            focusChatInput();

            // Barra de status preta nos navegadores móveis: injeta a meta
            // theme-color em todos os níveis de document (iframe, parent, top)
            (function() {{
                function setThemeColor(d) {{
                    d.querySelectorAll('meta[name="theme-color"]').forEach(function(el) {{ el.remove(); }});
                    var meta = d.createElement('meta');
                    meta.name = 'theme-color';
                    meta.content = '#000000';
                    d.head.insertBefore(meta, d.head.firstChild);
                }}
                setThemeColor(document);
                setThemeColor(window.parent.document);
                try {{
                    if (window.top.document !== window.parent.document) {{
                        setThemeColor(window.top.document);
                    }}
                }} catch(e) {{}}

                // Scrollbar preta global
                const doc = window.parent.document;
                if (!doc.getElementById('chatbotedn-global-css')) {{
                    const style = doc.createElement('style');
                    style.id = 'chatbotedn-global-css';
                    style.textContent = `
                        * {{ scrollbar-color: #000000 #f0f0f0 !important; scrollbar-width: thin !important; }}
                        *::-webkit-scrollbar {{ width: 8px !important; height: 8px !important; }}
                        *::-webkit-scrollbar-thumb {{ background-color: #000000 !important; border-radius: 4px !important; }}
                        *::-webkit-scrollbar-track {{ background: #f0f0f0 !important; border-radius: 4px !important; }}
                    `;
                    doc.head.appendChild(style);
                }}
            }})();
        </script>
        """, height=0, width=0)

    def render_sidebar(self):
        """Renderiza a barra lateral com informações e controles."""
        with st.sidebar:
            st.header("Sobre")
            st.info(
                "Chatbot em homenagem à Escola da Nuvem (EdN), ONG de educação tech. "
                "Usa o modelo GPT-3.5 Turbo através da API do OpenRouter, com cache "
                "de respostas para melhorar o desempenho."
            )
            st.markdown("🔗 [Site oficial da EdN](https://escoladanuvem.org/)")
            st.markdown("🎓 [Cursos e inscrições](https://escoladanuvem.org/cursos/)")

            if st.button("Limpar Conversa", key="clear_btn"):
                st.session_state.user_inputs = []
                st.session_state.bot_responses = []
                st.rerun()

            st.subheader("Configurações")
            st.slider(
                "Contexto máximo (tokens)",
                min_value=100,
                max_value=1000,
                value=MAX_HISTORY_TOKENS,
                step=100,
                key="max_tokens"
            )

            if st.checkbox("Mostrar debug info", value=False):
                st.write(f"Versão Streamlit: {st.__version__}")
                st.write(f"Turnos na conversa: {len(st.session_state.get('user_inputs', []))}")
                st.write(f"Modelo: {MODEL}")


def main():
    """Função principal da aplicação."""
    try:
        init_db()

        ui = ChatUI()
        session = ChatSession()

        ui.render_header()
        ui.render_sidebar()

        history = session.get_history()
        ui.render_conversation(history)

        user_input, submitted = ui.render_input_form()

        # Devolve o foco ao campo de digitação (inclusive após enter/envio)
        ui.focus_input()

        if submitted and user_input:
            with st.spinner("Por favor aguarde um momento..."):
                try:
                    global MAX_HISTORY_TOKENS
                    MAX_HISTORY_TOKENS = st.session_state.get("max_tokens", MAX_HISTORY_TOKENS)

                    session.add_message(user_input)
                    st.rerun()
                except Exception as e:
                    logger.error(f"Erro ao processar mensagem: {e}")
                    st.error(f"Ocorreu um erro ao processar sua mensagem: {str(e)}")

    except Exception as e:
        logger.error(f"Erro na execução principal: {e}")
        st.error(f"Ocorreu um erro na aplicação: {str(e)}")
        st.info("Detalhes para suporte técnico foram registrados no log.")

        if st.button("Reiniciar Aplicação"):
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.rerun()


if __name__ == "__main__":
    main()

st.markdown("""
<hr>
<div style="text-align: center;">
    <h4>⛅ ChatBot de IA em homenagem à Escola da Nuvem</h4>
    <em>Projeto em Python e Streamlit, com o modelo GPT-3.5 Turbo via API do OpenRouter.</em><br>
    Por <strong>Ary Ribeiro</strong>: <a href="https://www.linkedin.com/in/aryribeiro" target="_blank">LinkedIn</a>
</div>
""", unsafe_allow_html=True)

st.markdown("""
<style>
    .main {
        background-color: #ffffff;
        color: #333333;
    }
    .block-container {
        padding-top: 1rem;
        padding-bottom: 0rem;
    }
    /* Esconde o chrome padrão do Streamlit, mas mantém o <header> no DOM
       para preservar a seta de abrir/fechar o sidebar */
    header[data-testid="stHeader"] {
        background: transparent !important;
        box-shadow: none !important;
    }
    /* Esconde só os itens à direita do toolbar (menu, Deploy, status).
       NÃO esconder o stToolbar inteiro: é dentro dele que o Streamlit
       renderiza a seta de expandir a sidebar (stExpandSidebarButton). */
    [data-testid="stToolbarActions"],
    [data-testid="stAppDeployButton"],
    [data-testid="stMainMenu"],
    [data-testid="stStatusWidget"],
    [data-testid="stDecoration"] {
        display: none !important;
    }
    footer {display: none !important;}
    #MainMenu {display: none !important;}
    /* Remove qualquer espaço em branco adicional */
    div[data-testid="stAppViewBlockContainer"] {
        padding-top: 0 !important;
        padding-bottom: 0 !important;
    }
    div[data-testid="stVerticalBlock"] {
        gap: 0 !important;
        padding-top: 0 !important;
        padding-bottom: 0 !important;
    }
    /* Remove quaisquer margens extras */
    .element-container {
        margin-top: 0 !important;
        margin-bottom: 0 !important;
    }
</style>
""", unsafe_allow_html=True)
