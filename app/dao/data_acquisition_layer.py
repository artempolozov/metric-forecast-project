import logging
import os
from sqlalchemy import create_engine

import pandas as pd

logger = logging.getLogger(__name__)

# Конфигурация БД из переменных окружения
DB_CONFIG = {
    "host": os.getenv("DB_HOST", "localhost"),  # Имя сервиса в docker-compose
    "database": os.getenv("DB_NAME", "postgres"),
    "user": os.getenv("DB_USER", "postgres"),
    "password": os.getenv("DB_PASSWORD", "password")
}

# Список всех таблиц
tables = ["infrastructure_metrics", "http_metrics", "service_health"]  # Ваши таблицы


def create_engine_by_params():
    # 1. Формируем строку подключения для SQLAlchemy
    DB_URL = f"postgresql+psycopg2://{DB_CONFIG['user']}:{DB_CONFIG['password']}@{DB_CONFIG['host']}/{DB_CONFIG['database']}"

    # 2. Создаем движок SQLAlchemy
    return create_engine(DB_URL, pool_size=10, max_overflow=20)


# 3. получение примера данных из таблиц
def get_data_from_table(table_name, engine):
    """Экспорт таблицы из БД в CSV"""
    try:
        df = pd.read_sql(f"SELECT * FROM {table_name}", engine,  index_col='time', parse_dates=True)
        return df
    except Exception as e:
        print(f"Table reading error {table_name}: {e}")


def get_all_data():
    print(f'Getting all data...')
    table_with_df_dict = {}
    engine = create_engine_by_params()

    for table in tables:
        table_with_df_dict[table] = get_data_from_table(table, engine)

    return table_with_df_dict



