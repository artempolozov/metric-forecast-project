import pandas as pd
import numpy as np

from sklearn.preprocessing import MinMaxScaler


# -------------------------
# GENERAL
# -------------------------


def transform_to_datetime(df):
    df.index = pd.to_datetime(df.index, utc=True)


def remove_columns(df, cols_for_dropping):
    for col in cols_for_dropping:
        if col in df.columns:
            df.drop(col, axis=1, inplace=True)


def prepare_data(target, df, type, n_steps=60, verbose=False):
    df = df.copy()

    if type == 'service_health':
        if verbose:
            print("Continue Service health modeling...")
    elif type == 'infrastructure' or type == 'infrastructure_metrics':
        remove_columns(df, ("team_name", "measurement_interval_ms"))
    elif type == 'http' or type == 'http_metrics':
        if verbose:
            print("Continue HTTP modeling...")
    else:
        if verbose:
            print("Wrong type")

    # Удаление выбросов
    df = df[(df[target] > df[target].quantile(0.05)) &
            (df[target] < df[target].quantile(0.95))]

    # Логарифмическое преобразование для стабилизации дисперсии
    df.loc[:, target] = np.log1p(df[target])

    # Делаем target последним столбцом
    target_col = target
    cols = [col for col in df.columns if col != target_col] + [target_col]
    df = df[cols]

    original_df = df[[target]].copy()

    # Нормализация
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(df)

    # Создание последовательностей
    X, y = [], []
    for i in range(n_steps, len(scaled_data)):
        X.append(scaled_data[i-n_steps:i, :])
        y.append(scaled_data[i, -1])  # target - последний столбец

    return np.array(X), np.array(y), scaler, original_df


# -------------------------
# SERVICE HEALTH
# -------------------------

def lay_out_service_health(df):
    transform_to_datetime(df)

    df['status_code'] = df['status'].map({"healthy": 1, "degraded": 0})
    df['last_incident'] = df['last_incident'].fillna('0')

    # Преобразуем в дату и сразу извлекаем месяц
    months = []
    for date_str in df['last_incident']:
        if date_str == '0':
            months.append(0)
        else:
            try:
                months.append(pd.to_datetime(date_str).month)
            except:
                months.append(0)

    df['month'] = months

    services = df.service_name.unique()

    dfs_by_service_name = {}

    for service in services:
        temp_df = df[df.service_name == service].copy()
        remove_columns(temp_df, ('service_name', 'last_incident', 'status'))
        dfs_by_service_name[service] = temp_df

    return dfs_by_service_name


# -------------------------
# HTTP
# -------------------------

def lay_out_http(df):
    transform_to_datetime(df)

    dfs_by_service_and_endpoints_and_request_types = {}

    services = df.service_name.unique()
    endpoints = df.endpoint.unique()
    request_types = df.request_type.unique()

    for service in services:
        dfs_by_service_and_endpoints_and_request_types[service] = {}
        for endpoint in endpoints:
            dfs_by_service_and_endpoints_and_request_types[service][endpoint] = {}
            for request_type in request_types:
                temp_df = df[
                    (df.service_name == service) & (df.endpoint == endpoint) & (df.request_type == request_type)].copy()
                remove_columns(temp_df, ("service_name", "endpoint", "request_type"))
                dfs_by_service_and_endpoints_and_request_types[service][endpoint][request_type] = temp_df

    return dfs_by_service_and_endpoints_and_request_types


# -------------------------
# INFRASTRUCTURE
# -------------------------

def lay_out_infrastructure(df):
    transform_to_datetime(df)

    dfs_by_service_and_metric_type = {}

    services = df.service_name.unique()
    metric_types = df.metric_type.unique()

    for service in services:
        dfs_by_service_and_metric_type[service] = {}
        for metric_type in metric_types:
            temp_df = df[(df.service_name == service) & (df.metric_type == metric_type)].copy()
            remove_columns(temp_df, ("service_name", "metric_type"))
            dfs_by_service_and_metric_type[service][metric_type] = temp_df

    return dfs_by_service_and_metric_type

