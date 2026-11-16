# BookCast production image
# Single-process Gradio app: one container, one persistent data directory.

FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    GRADIO_ANALYTICS_ENABLED=False \
    HF_HUB_DISABLE_TELEMETRY=1 \
    HOME=/home/bookcast

# ffmpeg does the chunk trimming/concatenation; espeak is the pyttsx3 fallback.
RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates \
        ffmpeg \
        espeak-ng \
    && rm -rf /var/lib/apt/lists/* \
    && ln -sf /usr/bin/espeak-ng /usr/bin/espeak

RUN useradd --create-home --uid 10001 --shell /usr/sbin/nologin bookcast

WORKDIR /app

COPY requirements.txt requirements-ui.txt ./
RUN pip install --upgrade pip \
    && pip install -r requirements.txt -r requirements-ui.txt

COPY . .

RUN chmod +x run_ui.sh \
    && mkdir -p /app/books /app/runs /app/models \
    && chown -R bookcast:bookcast /app /home/bookcast

USER bookcast

EXPOSE 7860

CMD ["./run_ui.sh", "--host", "0.0.0.0", "--port", "7860", "--no-browser"]
