FROM python:3.12-slim

WORKDIR /app

# Установка системных зависимостей
RUN apt-get update && \
    apt-get install -y --no-install-recommends libgomp1 && \
    rm -rf /var/lib/apt/lists/*


# Создание директории для логов
RUN mkdir -p /app/logs && \
    touch /app/logs/service.log && \
    chmod -R 777 /app/logs  # Права на запись для всех пользователей

# Установка зависимостей
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Копирование исходного кода
COPY . .

# Установка прав на директорию приложения
RUN mkdir -p /app/input /app/output && \
    chmod -R 777 /app/input /app/output


# Точки монтирования
VOLUME /app/input
VOLUME /app/output

CMD ["python", "./app/app.py"]