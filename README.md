Obs.: caso o app esteja no modo "sleeping" (dormindo) ao entrar, basta clicar no botão que estará disponível e aguardar, para ativar o mesmo.
<img width="912" height="787" alt="image" src="https://github.com/user-attachments/assets/11872e37-8513-424c-a9dd-ba7ba304648b" />

## ChatBot EdN com IA

Chatbot interativo desenvolvido com [Streamlit](https://streamlit.io/) e integrado ao modelo GPT-3.5 Turbo através da API do [OpenRouter](https://openrouter.ai/). O chatbot responde dúvidas relacionadas à Escola da Nuvem (EdN), uma ONG que oferece programas e cursos voltados para tecnologia, computação em nuvem e inteligência artificial.

## Funcionalidades

- Interface de chat com histórico e balões de mensagem
- Respostas restritas ao contexto da Escola da Nuvem (system prompt com regras anti-alucinação)
- Cache de respostas em SQLite (expira em 1 hora)
- Retentativas com backoff exponencial em caso de rate limit ou erro de servidor
- Truncamento automático do histórico para controlar o consumo de tokens

## Requisitos

- Python 3.8 ou superior
- Chave da API do OpenRouter em `.streamlit/secrets.toml`

## Como rodar localmente

1. Crie um arquivo `.streamlit/secrets.toml` com o seguinte conteúdo:

```toml
[openrouter]
OPENROUTER_API_KEY = "sua-chave-do-openrouter"
```

Alternativamente, defina a variável de ambiente `OPENROUTER_API_KEY`.

2. Instale as dependências:

```
pip install -r requirements.txt
```

3. Execute o app:

```
streamlit run app.py
```

## Licença

Este projeto é de código aberto e foi criado em homenagem à Escola da Nuvem.

Por **Ary Ribeiro**: [LinkedIn](https://www.linkedin.com/in/aryribeiro)
