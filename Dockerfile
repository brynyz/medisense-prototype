# Dockerfile
FROM python:3.12-slim

EXPOSE 8000

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ["gunicorn", "medisense.wsgi:application", "--bind", "0.0.0.0:8000"]