FROM python:3.12-slim

# Set working directory
WORKDIR /app

# Install minimal build tools
RUN apt-get update && apt-get install -y \
        build-essential \
        git \
        wget \
        ca-certificates \
        && update-ca-certificates \
        && rm -rf /var/lib/apt/lists/*

# Upgrade pip and install prebuilt PyTorch wheel first
RUN pip install --upgrade pip

# Install PyTorch CPU first (prebuilt wheel speeds up sentence-transformers)
RUN pip install torch --index-url https://download.pytorch.org/whl/cpu

# Copy pyproject.toml and install other dependencies
COPY pyproject.toml .
RUN pip install --no-cache-dir --prefer-binary .

# Copy files needed
COPY app.css app.py rag_filters.py rag_pipeline.py ./
COPY data ./data

# Expose Gradio port
EXPOSE 7860

ENV PYTHONUNBUFFERED=1

# Run app
CMD ["python3", "-u", "app.py"]
