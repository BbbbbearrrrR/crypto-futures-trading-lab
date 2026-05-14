FROM python:3.11-slim

# Prevent fork deadlock for numpy/pandas multiprocessing
ENV OMP_NUM_THREADS=1 \
    OPENBLAS_NUM_THREADS=1 \
    MKL_NUM_THREADS=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# Install supervisor (process manager) + build deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    supervisor \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Install Python deps first (layer cache)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source
COPY . .

# Runtime dirs (will be overridden by volumes in production)
RUN mkdir -p logs paper data results

EXPOSE 5050

CMD ["supervisord", "-c", "supervisord.conf", "-n"]
