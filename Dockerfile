FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
# install CPU-only torch wheels
RUN pip install --no-cache-dir --extra-index-url https://download.pytorch.org/whl/cpu -r requirements.txt
COPY app /app
ENV HF_HOME=/root/.cache/huggingface
ENV MODEL_ID=rednote-hilab/dots.ocr
ENV TORCH_DTYPE=float32
EXPOSE 8000
# enable auto-reload for development
CMD ["uvicorn","main:app","--host","0.0.0.0","--port","8000","--reload"]
