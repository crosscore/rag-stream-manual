# rag-streaming/pgvector_db/Dockerfile

FROM postgres:latest

RUN apt-get update && apt-get install -y \
    postgresql-server-dev-all \
    gcc \
    git \
    make \
    && rm -rf /var/lib/apt/lists/*

RUN cd /tmp && \
    git clone --branch v0.7.2 https://github.com/pgvector/pgvector.git && \
    cd pgvector && \
    make && \
    make install

RUN echo "alias ls='ls --color=auto'" >> ~/.bashrc && \
    echo "alias ll='ls -alF'" >> ~/.bashrc && \
    echo "export PS1='\[\033[01;32m\]\u@\h\[\033[00m\]:\[\033[01;34m\]\w\[\033[00m\]\$ '" >> ~/.bashrc

COPY init_pgvector.sql /docker-entrypoint-initdb.d/

RUN echo "CREATE EXTENSION IF NOT EXISTS vector;" > /docker-entrypoint-initdb.d/10-create-extension.sql

ENV IS_DOCKER=true
