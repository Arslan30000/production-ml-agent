FROM python:3.12-slim
WORKDIR /app
RUN pip install --no-cache-dir "fastapi>=0.115,<1" "uvicorn>=0.30,<1"
COPY api/ /app/api/
COPY artifacts/model.json /app/artifacts/model.json
ENV MODEL_PATH=/app/artifacts/model.json
EXPOSE 8000
USER 10001
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
