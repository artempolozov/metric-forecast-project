import os

import pandas as pd

from app.config.common_data import HTTP_METRICS_ERROR_RATE, HTTP_METRICS_AVG_RESPONSE_TIME, TARGETS, \
    INFRASTRUCTURE_METRICS_LOAD_PERCENT, INFRASTRUCTURE_METRICS_RESPONSE_TIME, INFRASTRUCTURE_METRICS_RECOVERY_TIME
from app.processing.main_processing import INFRASTRUCTURE_METRICS, HTTP_METRICS, SERVICE_HEALTH_METRICS

FORECAST_FOLDER = 'data_forecast'


def save_results(result, service_name):
    create_and_check_folder()

    for metric_name, data in result.items():
        data_result = []
        output_path = os.path.join(FORECAST_FOLDER, f"forecast_{service_name}_{metric_name}.csv")
        columns = []
        if (metric_name == INFRASTRUCTURE_METRICS_LOAD_PERCENT or
                metric_name == INFRASTRUCTURE_METRICS_RESPONSE_TIME
                or metric_name == INFRASTRUCTURE_METRICS_RECOVERY_TIME):
            for i_m_name, i_m_data in data.items():
                for day, value in i_m_data.items():
                    data_result.append([day, i_m_name, value])
            columns = ['Day', 'Type', TARGETS[metric_name]]

        elif metric_name == HTTP_METRICS_ERROR_RATE or metric_name == HTTP_METRICS_AVG_RESPONSE_TIME:
            for h_endpoint, endpoint_data in data.items():
                for h_method, endpoint_method_data in endpoint_data.items():
                    for day, value in endpoint_method_data.items():
                        data_result.append([day, h_endpoint, h_method, value])
            columns = ['Day', 'Endpoint', 'Method', TARGETS[metric_name]]

        elif metric_name == SERVICE_HEALTH_METRICS:
            for day, value in data.items():
                data_result.append([day, value])
            columns = ['Day', TARGETS[metric_name]]

        df = pd.DataFrame(data_result, columns=columns)
        save_file(df, output_path)


def create_and_check_folder():
    os.makedirs(FORECAST_FOLDER, exist_ok=True)


def save_file(df, output_path):
    try:
        df.to_csv(output_path, index=False)
        print(f"Success saving in {output_path}")
    except Exception as e:
        print(f"Error saving: {e}")
