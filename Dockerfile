# Start from lightweight Python base
FROM python:3.12-slim

# Set working directory
WORKDIR /app

# Install system dependencies (needed for pandas, numpy, openpyxl, etc.)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    python3-dev \
    gfortran \
    libopenblas-dev \
    && rm -rf /var/lib/apt/lists/*


# Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the code
COPY . .

# Default command (runs your pipeline)
ENTRYPOINT ["python", "app.py"]
