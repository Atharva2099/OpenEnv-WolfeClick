FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
  && rm -rf /var/lib/apt/lists/*

# Copy only what the Space needs
COPY space_app.py .
COPY battle_logs/ ./battle_logs/
COPY requirements_space.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements_space.txt

# Healthcheck against the Gradio app
HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
  CMD curl -f http://localhost:7860/ || exit 1

EXPOSE 7860

CMD ["python", "space_app.py"]
