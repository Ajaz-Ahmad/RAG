FROM python:3.11-slim

WORKDIR /app

# Build tools needed by faiss-cpu and numpy
RUN apt-get update && apt-get install -y --no-install-recommends gcc g++ && rm -rf /var/lib/apt/lists/*

# Install CPU-only PyTorch FIRST.
# sentence-transformers depends on torch; without this it pulls the full
# CUDA build (~2 GB) which blows the build disk/memory budget.
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

# Install remaining dependencies
COPY requirements-api.txt .
RUN pip install --no-cache-dir -r requirements-api.txt

# Pre-download the embedding model so the first request isn't slow
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"

COPY . .

EXPOSE 7860

CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "7860"]
