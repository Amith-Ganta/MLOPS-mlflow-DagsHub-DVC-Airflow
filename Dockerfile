FROM python:3.11-slim

WORKDIR /app

# Install dependencies
COPY pyproject.toml ./
RUN pip install --no-cache-dir \
    fastapi>=0.115 \
    uvicorn[standard]>=0.30 \
    pydantic>=2.0 \
    scikit-learn \
    xgboost \
    pandas \
    numpy \
    joblib

# Copy source
COPY backend/ ./backend/
COPY artifacts/ ./artifacts/
COPY src/ ./src/

ENV PYTHONPATH=/app

EXPOSE 8000

CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8000"]
