FROM python:3.9-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy Python requirements
COPY Back-End/requirements.txt .

# Install libraries
RUN pip install --no-cache-dir -r requirements.txt

# Copy the remaining files
COPY Back-End/ .

# Create models directory if it does not exist
RUN mkdir -p models

# Expose port 5000
EXPOSE 5000

# Run the application
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "app:app"]