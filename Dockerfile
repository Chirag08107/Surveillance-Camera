FROM python:3.11-slim

# System deps for opencv, dlib (face_recognition) and easyocr
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential cmake \
    libgl1 libglib2.0-0 \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8000

CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8000"]
