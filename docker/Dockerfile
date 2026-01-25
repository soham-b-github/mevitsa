FROM python:3.10-slim

RUN apt-get update && apt-get install -y \
    git \
    build-essential \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --upgrade pip && pip install --no-cache-dir -r requirements.txt

# Copy folders individually to maintain your structure
COPY configs/ ./configs/
COPY data/ ./data/
COPY src/ ./src/
COPY assets/ ./assets/

# Ensure the mount point exists
RUN mkdir -p /app/src/trained

ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1

EXPOSE 8501

# Run streamlit from the root to keep paths consistent
CMD ["streamlit", "run", "src/app.py", "--server.port=8501", "--server.address=0.0.0.0"]
