FROM python:3.10.18 AS base

COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

WORKDIR /app

COPY . .

RUN uv sync

ENV PATH="/app/.venv/bin:$PATH"

# environment variables for streamlit file handling
ENV STREAMLIT_SERVER_ENABLE_XSRF_PROTECTION=false
ENV STREAMLIT_SERVER_ENABLE_CORS=false

EXPOSE 8000 8501

CMD ["sh", "-c", "uv run uvicorn src.ham_clf.backend.app:api --host 0.0.0.0 --port 8000 & streamlit run src/ham_clf/frontend/app.py --server.port=8501 --server.address=0.0.0.0 --server.enableXsrfProtection=false --server.enableCORS=false"]