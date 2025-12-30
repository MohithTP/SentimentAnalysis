FROM python:3.13-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV MLFLOW_TRACKING_URI=sqlite:///mlflow.db

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libsqlite3-dev \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
RUN pip install --no-cache-dir \
    flask \
    gunicorn \
    pandas \
    tensorflow \
    "keras==3.13.0" \
    mlflow \
    scikit-learn \
    numpy

# Copy application code
COPY inference.py /app/
COPY DataSetup.py /app/

# Copy model artifacts
COPY tokenizer.pickle /app/
COPY label_encoder.pickle /app/
COPY model /app/model

# Expose port 8080 (SageMaker default)
EXPOSE 8080

# Serve with Gunicorn
ENTRYPOINT ["gunicorn", "--timeout", "120", "-b", "0.0.0.0:8080", "inference:app"]
