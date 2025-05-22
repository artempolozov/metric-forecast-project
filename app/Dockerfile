FROM python:3.9

WORKDIR /app

# Создаем директорию для логов
RUN mkdir -p /app/logs

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY main.py .

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]