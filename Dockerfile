FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app \
    PORT=8501

WORKDIR /app

# Install system dependencies and Python packages as root
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential \
        curl \
        software-properties-common \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install as root (global installation)
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# Create non-root user after installing packages
RUN adduser --disabled-password --gecos '' --shell /bin/bash user \
    && chown -R user:user /app

# Switch to user for running the app
USER user

# Copy application code
COPY --chown=user:user . .

EXPOSE 8501

CMD ["streamlit", "run", "app.py", "--server.address", "0.0.0.0", "--server.port", "8501", "--server.headless", "true"]