FROM python:3.10-slim

WORKDIR /app

# Dependencias del sistema
RUN apt-get update && apt-get install -y \
    libglib2.0-0 \
    libgl1-mesa-glx \
    && rm -rf /var/lib/apt/lists/*

# Instalar dependencias Python
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copiar el servidor
COPY nudenet_server.py .

# Puerto
EXPOSE 5000

CMD ["python", "nudenet_server.py"]
