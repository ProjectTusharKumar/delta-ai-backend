# Use an official Python runtime as a parent image
FROM python:3.12-slim

# Set work directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt /app/

# Install Python dependencies
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY . /app/

# Add entrypoint script for easy startup
COPY start.sh /app/start.sh
RUN chmod +x /app/start.sh

# Expose the port
EXPOSE 5000

# Default command: run the start.sh script
ENTRYPOINT ["/app/start.sh"]