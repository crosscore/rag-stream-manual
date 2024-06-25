# rag-streaming/s3_db/Dockerfile

FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

RUN echo "alias ls='ls --color=auto'" >> ~/.bashrc

COPY . .

ENV IS_DOCKER=true
