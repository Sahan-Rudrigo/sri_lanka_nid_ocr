FROM python:3.9-slim

# Install system dependencies
RUN apt-get update && \
    apt-get install -y \
    tesseract-ocr \
    libtesseract-dev \
    libleptonica-dev \
    && rm -rf /var/lib/apt/lists/*

# Set the Tesseract path environment variable
ENV TESSERACT_PATH=/usr/bin/tesseract

WORKDIR /app

COPY requirements.txt .
RUN apt-get update && apt-get install -y libgl1-mesa-glx
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ["python", "app.py"]
