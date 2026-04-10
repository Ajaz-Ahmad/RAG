FROM python:3.11-slim

WORKDIR /app

# Install build tools needed by some packages (faiss, numpy, etc.)
RUN apt-get update && apt-get install -y --no-install-recommends gcc g++ && rm -rf /var/lib/apt/lists/*

# Install Python dependencies first (better layer caching)
COPY requirements-api.txt .
RUN pip install --no-cache-dir -r requirements-api.txt

# Pre-download the embedding model so the first recruiter request isn't slow
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"

# Copy source
COPY . .

# HuggingFace Spaces expects port 7860
EXPOSE 7860

CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "7860"]
