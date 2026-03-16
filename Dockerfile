FROM python:3.12-slim

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        latexmk \
        texlive-latex-extra \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY texmf/ /usr/local/share/texmf/
RUN mktexlsr /usr/local/share/texmf
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

COPY . /app
