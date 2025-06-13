# Use Python 3.9 base image
FROM python:3.9-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create startup script
RUN echo '#!/bin/bash\n\
    echo "Starting Flask app..."\n\
    python app.py\n\
    ' > start.sh && chmod +x start.sh

# Expose port (Railway will map this automatically)
EXPOSE 5001

# Start the services
CMD ["./start.sh"]