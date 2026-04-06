# 🐳 Dockerfile — Observabilidade de Anomalias (Astrofísica e Indústria)

# Use uma imagem leve do Python
FROM python:3.10-slim

# Evita que o Python gere arquivos .pyc e permite logs em tempo real
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

# Define o diretório de trabalho
WORKDIR /app

# Instala dependências do sistema necessárias para compilar bibliotecas
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    software-properties-common \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copia o requirements.txt primeiro para aproveitar o cache de camadas do Docker
COPY requirements.txt .

# Instala as dependências do Python
RUN pip install --no-cache-dir -r requirements.txt

# Copia o restante do código do projeto
COPY . .

# Cria as pastas de dados e resultados (se não existirem)
RUN mkdir -p data/cache results

# Expõe a porta padrão do Streamlit
EXPOSE 8501

# Configura o Healthcheck para garantir que o Streamlit está rodando
HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health

# Comando padrão: Iniciar o Dashboard
ENTRYPOINT ["streamlit", "run", "dashboard.py", "--server.port=8501", "--server.address=0.0.0.0"]
