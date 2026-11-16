FROM python:3.13.1-slim-bookworm

ARG VCS_REF=unknown

LABEL org.opencontainers.image.title="KyxParser thesis experiments" \
      org.opencontainers.image.source="https://github.com/Mrthuancao/KyxParser" \
      org.opencontainers.image.revision="${VCS_REF}"

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    MPLBACKEND=Agg \
    PIP_NO_CACHE_DIR=1

WORKDIR /workspace

RUN apt-get update \
    && apt-get install --no-install-recommends -y libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN python -m pip install --index-url https://download.pytorch.org/whl/cpu torch==2.9.1 \
    && python -m pip install -r requirements.txt

COPY . .

ENTRYPOINT ["python"]
