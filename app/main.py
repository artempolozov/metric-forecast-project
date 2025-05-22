import logging
import os
import random
import time
from datetime import datetime, timedelta

import pandas as pd
import psycopg2
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, validator, Field

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),  # Логи в консоль
        logging.FileHandler("app.log")  # Логи в файл
    ]
)
logger = logging.getLogger(__name__)

app = FastAPI()

# Конфигурация БД из переменных окружения
DB_CONFIG = {
    "host": os.getenv("DB_HOST", "timescaledb"),  # Имя сервиса в docker-compose
    "database": os.getenv("DB_NAME", "postgres"),
    "user": os.getenv("DB_USER", "postgres"),
    "password": os.getenv("DB_PASSWORD", "password")
}

CREATE_TABLE_INFRASTRUCTURE = """
CREATE TABLE IF NOT EXISTS infrastructure_metrics (
    time TIMESTAMPTZ NOT NULL,
    team_name TEXT,
    service_name TEXT,
    metric_type TEXT CHECK (metric_type IN ('CPU', 'Disk', 'Memory', 'Network')),
    load_percent DOUBLE PRECISION,
    response_time_ms DOUBLE PRECISION,
    recovery_time_ms DOUBLE PRECISION,
    measurement_interval_ms INTEGER
);
SELECT create_hypertable('infrastructure_metrics', 'time');
"""

CREATE_TABLE_HTTP = """
CREATE TABLE IF NOT EXISTS http_metrics (
    time TIMESTAMPTZ NOT NULL,
    service_name TEXT,
    endpoint TEXT,
    request_type TEXT CHECK (request_type IN ('GET', 'POST', 'PUT', 'DELETE')),
    request_count INTEGER,
    avg_response_time_ms DOUBLE PRECISION,
    error_rate DOUBLE PRECISION
);
SELECT create_hypertable('http_metrics', 'time');
"""

CREATE_TABLE_BUSINESS = """
CREATE TABLE IF NOT EXISTS business_metrics (
    time TIMESTAMPTZ NOT NULL,
    project_name TEXT,
    endpoint TEXT,
    request_type TEXT CHECK (request_type IN ('GET', 'POST', 'PUT', 'DELETE')),
    request_count INTEGER,
    avg_response_time_ms DOUBLE PRECISION,
    error_rate DOUBLE PRECISION
);
SELECT create_hypertable('http_metrics', 'time');
"""

CREATE_TABLE_SERVICE_HEALTH = """
-- Создаем обычную таблицу без первичного ключа
CREATE TABLE IF NOT EXISTS service_health (
    time TIMESTAMPTZ NOT NULL,
    service_name TEXT NOT NULL,
    status TEXT NOT NULL CHECK (status IN ('healthy', 'degraded', 'unavailable')),
    last_incident TIMESTAMPTZ,
    uptime_percent DOUBLE PRECISION,
    UNIQUE (time, service_name)  -- Добавляем уникальное ограничение
);

-- Преобразуем в гибертаблицу БЕЗ уникальных ограничений
SELECT create_hypertable('service_health', 'time', if_not_exists => TRUE);

-- Создаем обычный индекс (не уникальный) для поиска
CREATE INDEX IF NOT EXISTS service_health_name_time_idx 
ON service_health (service_name, time DESC);
"""

# Infrastructure Metrics - генерация для всех типов
metrics_data = [
    {
        "type": "CPU",
        "load_range": (0.1, 100.0),
        "response_range": (50, 500),
        "recovery_range": (100, 1000)
    },
    {
        "type": "Memory",
        "load_range": (10.0, 95.0),  # % использования памяти
        "response_range": (30, 300),  # время ответа в ms
        "recovery_range": (200, 1500)  # время восстановления
    },
    {
        "type": "Disk",
        "load_range": (5.0, 90.0),  # % использования диска
        "response_range": (100, 800),  # IO latency
        "recovery_range": (300, 2000)
    },
    {
        "type": "Network",
        "load_range": (1.0, 80.0),  # % использования сети
        "response_range": (20, 200),  # latency
        "recovery_range": (50, 500)
    }
]


# Модель для запроса метрик
class MetricRequest(BaseModel):
    metric_name: str
    start_date: str
    end_date: str

    @validator('start_date', 'end_date')
    def validate_date_format(cls, v):
        try:
            datetime.strptime(v, '%Y-%m-%d')
            return v
        except ValueError:
            raise ValueError("Дата должна быть в формате YYYY-MM-DD")


class GenerateMetricsRequest(BaseModel):
    days: int = Field(..., gt=0, le=5000, description="Количество дней для генерации")


# Подключение к БД с ретраями
def get_db_connection(retries=5, delay=3):
    for i in range(retries):
        try:
            conn = psycopg2.connect(**DB_CONFIG)
            logger.info("Успешное подключение к БД")
            return conn
        except psycopg2.OperationalError as e:
            logger.warning(f"Попытка {i + 1}/{retries}. Ошибка подключения: {e}")
            if i == retries - 1:
                raise
            time.sleep(delay)


# Инициализация БД
def init_db():
    max_retries = 5

    conn = get_db_connection()
    cursor = conn.cursor()

    for attempt in range(max_retries):
        try:

            # Удаляем таблицу если существует (для чистого старта)
            cursor.execute("DROP TABLE IF EXISTS service_health CASCADE")
            conn.commit()

            # Создаем таблицу service_health
            cursor.execute(CREATE_TABLE_SERVICE_HEALTH)

            # Создаем остальные таблицы
            cursor.execute(CREATE_TABLE_INFRASTRUCTURE)
            cursor.execute(CREATE_TABLE_HTTP)

            # Создаем дополнительные индексы
            cursor.execute("""
                                CREATE INDEX IF NOT EXISTS infra_team_idx 
                                ON infrastructure_metrics (team_name, metric_type);
                            """)

            conn.commit()
            logger.info("Таблицы успешно созданы")
            return True
        except Exception as e:
            conn.rollback()
            logger.error(f"Ошибка инициализации БД (попытка {attempt + 1}): {e}")
            if attempt == max_retries - 1:
                raise
            time.sleep(3)


# Генерация тестовых данных
def generate_fake_metrics(days=180, points_per_day=24):
    conn = get_db_connection()
    cursor = conn.cursor()
    try:
        end_time = datetime.now()
        start_time = end_time - timedelta(days=days)
        teams = [f"team-{i}" for i in range(1, 6)]
        services = [f"microservice-{i}" for i in range(1, 11)]
        endpoints = ["/price", "/users", "/orders", "/inventory", "/payment"]

        # Интервал между точками данных в минутах
        interval_minutes = 24 * 60 // points_per_day  # Равномерно распределяем точки в течение дня

        logger.info(f"Генерация данных с {start_time} по {end_time}...")
        cursor.execute("TRUNCATE infrastructure_metrics, http_metrics, service_health;")

        total_points = days * points_per_day
        for i in range(total_points):
            # Правильно рассчитываем current_time для каждого шага
            current_time = start_time + timedelta(minutes=i * interval_minutes)
            team = random.choice(teams)
            service = random.choice(services)

            # Infrastructure metrics (генерируем для всех типов)
            for metric in metrics_data:
                cursor.execute("""
                    INSERT INTO infrastructure_metrics (
                        time, team_name, service_name, metric_type,
                        load_percent, response_time_ms, recovery_time_ms, 
                        measurement_interval_ms
                    ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                    """, (
                    current_time, team, service, metric["type"],
                    round(random.uniform(*metric["load_range"]), 1),
                    random.randint(*metric["response_range"]),
                    random.randint(*metric["recovery_range"]),
                    5000
                ))

            # HTTP metrics
            cursor.execute("""
                                INSERT INTO http_metrics (
                                    time, service_name, endpoint, request_type,
                                    request_count, avg_response_time_ms, error_rate
                                ) VALUES (%s, %s, %s, %s, %s, %s, %s)
                                """, (
                current_time, service, random.choice(endpoints),
                random.choice(["GET", "POST", "PUT"]),
                random.randint(100, 5000),
                random.randint(50, 300),
                round(random.uniform(0.0, 5.0), 2)
            ))

            status = random.choices(["healthy", "degraded"], weights=[99, 1])[0]
            last_incident = (current_time - timedelta(hours=random.randint(1, 72))
                             if status == "degraded" and random.random() < 0.2 else None)
            uptime = round(random.uniform(99.5, 99.99), 2) if status == "healthy" \
                else round(random.uniform(95.0, 99.4), 2)

            # Service health
            cursor.execute("""
                                INSERT INTO service_health (
                                    time, service_name, status,
                                    last_incident, uptime_percent
                                ) VALUES (%s, %s, %s, %s, %s)
                                ON CONFLICT (time, service_name) DO UPDATE SET
                                    status = EXCLUDED.status,
                                    last_incident = EXCLUDED.last_incident,
                                    uptime_percent = EXCLUDED.uptime_percent
                                """, (
                current_time, service,
                status,
                last_incident,
                uptime
            ))

            if i % 100 == 0:
                conn.commit()
                logger.debug(f"Добавлено {i}/{total_points} записей...")

        conn.commit()
        logger.info(f"Сгенерировано данных: {total_points} временных точек")
        return True
    except Exception as e:
        logger.error(f"Ошибка генерации данных: {e}")
        raise
    finally:
        if conn:
            conn.close()


# Получение метрик
def get_metrics(metric_name: str, start_date: str, end_date: str):
    try:
        conn = get_db_connection()

        # Форматируем даты с указанием времени
        start_datetime = f"{start_date} 00:00:00"
        end_datetime = f"{end_date} 23:59:59"

        query = """
            SELECT time, value 
            FROM metrics 
            WHERE metric_name = %s 
            AND time::timestamp BETWEEN %s AND %s
            ORDER BY time;
        """

        logger.info(f"SQL запрос: {query}")
        logger.info(f"Параметры: metric_name={metric_name}, start={start_datetime}, end={end_datetime}")

        df = pd.read_sql(query, conn, params=(metric_name, start_datetime, end_datetime))

        logger.info(f"Получено {len(df)} записей для {metric_name}")
        return df
    except Exception as e:
        logger.error(f"Ошибка получения метрик: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if conn:
            conn.close()


# Инициализация при старте
@app.on_event("startup")
async def startup():
    logger.info("Запуск инициализации...")
    try:
        init_db()
        generate_fake_metrics(days=1)  # Для теста генерируем данные за 1 день
    except Exception as e:
        logger.critical(f"Ошибка инициализации: {e}")
        raise


# Эндпоинты
@app.post("/generate-metrics")
async def generate_metrics(request: GenerateMetricsRequest):
    logger.info(f"Необходимо сгенерировать данные за {request.days} дней")
    try:
        generate_fake_metrics(days=request.days)
        return {"message": f"Метрики за {request.days} дней сгенерированы!"}
    except Exception as e:
        logger.error(f"Ошибка в /generate-metrics: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/get-metrics")
async def fetch_metrics(request: MetricRequest):
    try:
        df = get_metrics(request.metric_name, request.start_date, request.end_date)
        return df.to_dict(orient="records")
    except Exception as e:
        logger.error(f"Ошибка в /get-metrics: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/metrics/list")
async def list_metrics():
    return {"metrics": ["cpu_usage", "memory_usage", "disk_usage"]}


@app.get("/health")
async def health_check():
    try:
        conn = get_db_connection()
        conn.close()
        return {"status": "healthy"}
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(status_code=500, detail="Database connection error")
