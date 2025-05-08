Obs.: caso o app esteja no modo "sleeping" (dormindo) ao entrar, basta clicar no botão que estará disponível e aguardar, para ativar o mesmo.
![print chatbot edn](https://github.com/user-attachments/assets/a7e1cbef-f5b2-48ee-8b5e-64a60a68eab4)

## ChatBot EdN com AWS Bedrock

Este é um chatbot interativo desenvolvido com [Streamlit](https://streamlit.io/) e integrado à AWS Bedrock usando o modelo Claude (anthropic.claude-v2.1). O chatbot responde dúvidas relacionadas à Escola da Nuvem (EdN), uma ONG que oferece programas e cursos voltados para tecnologia, computação em nuvem e inteligência artificial.

## Funcionalidades

- Interface de chat com histórico
- Respostas personalizadas sobre a Escola da Nuvem
- Integração com AWS Bedrock (modelo Claude)
- Utilização de streaming para resposta em tempo real

## Requisitos

- Python 3.8 ou superior
- Conta AWS com permissões adequadas para o serviço Bedrock
- Chaves AWS válidas no arquivo `.streamlit/secrets.toml`

## Como rodar localmente

1. Crie um arquivo `.streamlit/secrets.toml` com as seguintes variáveis:

```
[aws]
AWS_ACCESS_KEY_ID = "sua-access-key-id"
AWS_SECRET_ACCESS_KEY = "sua-secret-access-key"
```

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

Desenvolvido por Ary Ribeiro: aryribeiro@gmail.com