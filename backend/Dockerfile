FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
  build-essential \
  default-libmysqlclient-dev \
  pkg-config \
  netcat-traditional \
  curl \
  ffmpeg \
  && rm -rf /var/lib/apt/lists/*

# Copy requirements file
COPY requirements.txt .

# Install CPU-only torch before requirements.txt so sentence-transformers
# doesn't pull the full CUDA stack (~2GB of nvidia/triton wheels).
# RUN pip install --no-cache-dir \
#     torch torchvision --index-url https://download.pytorch.org/whl/cpu

# # Install Python packages
# RUN pip install --no-cache-dir -r requirements.txt \
#         --extra-index-url https://download.pytorch.org/whl/cpu

# Copy entrypoint script first
COPY entrypoint.sh .
RUN chmod +x entrypoint.sh

# Copy application files
COPY . .

# Create uploads directory
RUN mkdir -p uploads

# Set Python path and environment
ENV PYTHONPATH=/app
ENV ENVIRONMENT=production

# Run the application via entrypoint (migrations + readiness + uvicorn)
ENTRYPOINT ["./entrypoint.sh"]
