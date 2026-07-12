Obs.: caso o app esteja no modo "sleeping" (dormindo) ao entrar, basta clicar no botão que estará disponível e aguardar, para ativar o mesmo.
![print chatbot edn](https://github.com/user-attachments/assets/a7e1cbef-f5b2-48ee-8b5e-64a60a68eab4)

## ChatBot EdN com IA

Chatbot interativo desenvolvido com [Streamlit](https://streamlit.io/) e integrado ao modelo GPT-3.5 Turbo através da API do [OpenRouter](https://openrouter.ai/). O chatbot responde dúvidas relacionadas à Escola da Nuvem (EdN), uma ONG que oferece programas e cursos voltados para tecnologia, computação em nuvem e inteligência artificial.

## Funcionalidades

- Interface de chat com histórico e balões de mensagem
- **Base de conhecimento (RAG)** extraída do site oficial: o chatbot responde apenas com base nos documentos de `base_conhecimento.json`
- Respostas restritas à Escola da Nuvem e sempre em português do Brasil, mesmo que a pergunta venha em outro idioma
- Regras anti-alucinação: não inventa datas, vagas, requisitos nem links; quando não sabe, direciona ao site oficial
- Cache de respostas em SQLite (expira em 1 hora)
- Retentativas com backoff exponencial em caso de rate limit ou erro de servidor
- Truncamento automático do histórico para controlar o consumo de tokens

## Base de conhecimento

O arquivo `base_conhecimento.json` contém documentos curados a partir do site oficial da EdN
(institucional, cursos, processo seletivo, empregabilidade, certificações, voluntariado,
parcerias, contato e transparência), cada um com suas `palavras_chave` e a `fonte` (URL).

A cada pergunta, o app pontua os documentos por sobreposição de termos e injeta os mais
relevantes no prompt do sistema. O modelo é instruído a responder **exclusivamente** com base
nesse contexto. Para atualizar o conteúdo do chatbot, basta editar o JSON — não é preciso
mexer no código.

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
