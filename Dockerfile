FROM python:3.11-slim

WORKDIR /app

# Install deps first for better layer caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# App code + already-trained model artifacts
COPY src/ ./src/
COPY models/ ./models/
COPY params.yaml .

EXPOSE 8000

# --app-dir puts src/ on sys.path so `from utils import ...` inside
# serve.py resolves without needing src to be an installed package.
CMD ["uvicorn", "serve:app", "--app-dir", "src", "--host", "0.0.0.0", "--port", "8000"]
