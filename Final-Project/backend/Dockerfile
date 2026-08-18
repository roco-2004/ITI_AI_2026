FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HOUSE_MODEL_PATH=/app/models/house_price.pkl \
    HOUSE_LOCATIONS_PATH=/app/models/locations.json \
    HOUSE_METADATA_PATH=/app/models/model_metadata.json

WORKDIR /app

COPY backend/requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt \
    && addgroup --system appgroup \
    && adduser --system --ingroup appgroup appuser

COPY backend /app/backend
COPY models /app/models

USER appuser
EXPOSE 8000

CMD ["uvicorn", "backend.app.main:app", "--host", "0.0.0.0", "--port", "8000"]
