# Raspberry Pi 5 (arm64) / x86_64 両対応
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

# libgomp1: numpy / scikit-learn の OpenMP
# tzdata:   supercronic を JST で回すため
RUN apt-get update \
 && apt-get install -y --no-install-recommends libgomp1 tzdata curl ca-certificates \
 && rm -rf /var/lib/apt/lists/*

# コンテナネイティブな cron。ログが stdout に出るので docker logs で追える
ARG SUPERCRONIC_VERSION=v0.2.48
RUN ARCH="$(dpkg --print-architecture)" \
 && curl -fsSL -o /usr/local/bin/supercronic \
      "https://github.com/aptible/supercronic/releases/download/${SUPERCRONIC_VERSION}/supercronic-linux-${ARCH}" \
 && chmod +x /usr/local/bin/supercronic

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY worker/common.py worker/telegram_ingest.py worker/transcribe_worker.py worker/embed_worker.py \
     worker/notify_worker.py worker/cluster_worker.py worker/theme_writer.py worker/research_worker.py \
     worker/digest_worker.py worker/supervisor.py worker/web.py worker/init_db.py worker/schema.sql worker/crontab ./

# ボリュームの所有者と揃える。Pi のデフォルトユーザーが 1000
RUN useradd -m -u 1000 app && chown -R app:app /app
USER app

CMD ["python", "telegram_ingest.py"]
