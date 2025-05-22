import torch
import pandas as pd
import numpy as np

from torch import nn
from prophet import Prophet
from statsmodels.tsa.statespace.sarimax import SARIMAX

from app.config.common_data import TARGETS, INFRASTRUCTURE_METRIC_NAMES, ENDPOINTS, METHODS, INFRASTRUCTURE_METRICS, \
    HTTP_METRICS, SERVICE_HEALTH_METRICS, HTTP_METRICS_ERROR_RATE, HTTP_METRICS_AVG_RESPONSE_TIME, \
    INFRASTRUCTURE_METRICS_LOAD_PERCENT, INFRASTRUCTURE_METRICS_RESPONSE_TIME, INFRASTRUCTURE_METRICS_RECOVERY_TIME
from app.config.model_config import MODEL_PARAMS
from app.metrics.metrics import print_metrics, calculate_metrics
from app.model.RNN_model import RNNModel
from app.model.model_fitting_and_predicting import train_model, forecast_with_days
from app.preprocessing.data_preprocessing_layer import prepare_data, lay_out_service_health, lay_out_infrastructure, \
    lay_out_http


def process(df, target, metric_type, verbose=False, extended=False):
    # Устройство
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    X, y, scaler, original_df = prepare_data(target, df=df, type=metric_type, n_steps=MODEL_PARAMS['N_STEPS'])

    # Разделение данных
    train_size = int(len(X) * 0.8)
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]
    original_train = original_df.iloc[:train_size + MODEL_PARAMS['N_STEPS']]
    original_test = original_df.iloc[train_size + MODEL_PARAMS['N_STEPS']:]

    # Конвертация в тензоры
    X_train_t = torch.FloatTensor(X_train).to(device)
    y_train_t = torch.FloatTensor(y_train).to(device).view(-1, 1)
    X_test_t = torch.FloatTensor(X_test).to(device)
    y_test_t = torch.FloatTensor(y_test).to(device).view(-1, 1)

    # DataLoader
    train_dataset = torch.utils.data.TensorDataset(X_train_t, y_train_t)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=MODEL_PARAMS['BATCH_SIZE'], shuffle=True)

    # Инициализация моделей
    input_size = X_train.shape[2]  # Количество признаков
    output_size = 1

    rnn_model = RNNModel(input_size, MODEL_PARAMS['HIDDEN_SIZE'], MODEL_PARAMS['NUM_LAYERS'], output_size).to(device)
    # lstm_model = LSTMModel(input_size, HIDDEN_SIZE, NUM_LAYERS, output_size).to(device)

    # Критерий и оптимизатор
    criterion = nn.MSELoss()
    rnn_optimizer = torch.optim.Adam(rnn_model.parameters(), lr=MODEL_PARAMS['LEARNING_RATE'])
    # lstm_optimizer = torch.optim.Adam(lstm_model.parameters(), lr=LEARNING_RATE)

    # Обучение моделей
    if verbose:
        print("Training RNN...")

    train_model(rnn_model, train_loader, criterion, rnn_optimizer, MODEL_PARAMS['N_EPOCHS'], verbose)

    result = {
        "rnn_model": rnn_model,
        "X_test": X_test,
        "y_test": y_test,
        "scaler": scaler
    }

    if extended:
        # Получаем фактические значения (обрезаем N_STEPS для соответствия)
        y_test_original = original_test.values.flatten()

        # Прогноз RNN
        with torch.no_grad():
            rnn_model.eval()
            rnn_pred_scaled = rnn_model(X_test_t).cpu().numpy().flatten()

        # Создаем массив с фиктивными признаками для обратного преобразования
        dummy_features = np.zeros((len(rnn_pred_scaled), X.shape[2] - 1))
        rnn_pred = scaler.inverse_transform(
            np.column_stack([dummy_features, rnn_pred_scaled])
        )[:, -1]

        if verbose:
            print("\nTraining Prophet...")

        prophet_model = train_prophet(original_train, target)
        # Прогноз Prophet (на длину тестовой выборки)
        future = prophet_model.make_future_dataframe(periods=len(y_test_original), freq='h')
        prophet_pred = prophet_model.predict(future)['yhat'].values[-len(y_test_original):]

        result["prophet_model"] = prophet_model

        if verbose:
            print("\nTraining SARIMA...")

        sarima_model = train_sarima(original_train, target)
        # Прогноз SARIMA
        if sarima_model is not None:
            sarima_pred = sarima_model.get_forecast(steps=len(y_test_original)).predicted_mean
            sarima_pred = sarima_pred[:len(y_test_original)]
        else:
            sarima_pred = np.zeros(len(y_test))  # Запасной вариант

        result["sarima_model"] = sarima_model

        # Проверка размерностей
        print(f"\nShapes - True: {y_test_original.shape}, RNN: {rnn_pred.shape}, "
              f"Prophet: {prophet_pred.shape}, SARIMA: {sarima_pred.shape}")

        # Вычисление метрик для всех моделей
        metrics = {
            'RNN': calculate_metrics(y_test_original, rnn_pred),
            'Prophet': calculate_metrics(y_test_original, prophet_pred),
            'SARIMA': calculate_metrics(y_test_original, sarima_pred.values)
        }

        print("\nModel Comparison:")
        for model_name, model_metrics in metrics.items():
            print(f"{model_name}:")
            print(f"  MAE: {model_metrics['MAE']}")
            print(f"  RMSE: {model_metrics['RMSE']}")
            print(f"  SMAPE: {model_metrics['SMAPE']}%")

    return result


def train_prophet(df, target):
    """Обучение модели Prophet (для временных рядов)."""
    # Создаем копию DataFrame, чтобы не менять исходные данные
    prophet_df = df.copy()

    # Если временные метки в индексе
    if isinstance(prophet_df.index, pd.DatetimeIndex):
        prophet_df = prophet_df.reset_index()
        # Проверяем автоматическое имя колонки
        time_col = prophet_df.columns[0]  # Первая колонка после reset_index
        prophet_df = prophet_df.rename(columns={time_col: 'ds', target: 'y'})
    else:
        # Если временные метки уже в колонке (проверяем существующие колонки)
        time_cols = [col for col in prophet_df.columns if col.lower() in ['timestamp', 'time', 'date', 'datetime']]
        if time_cols:
            prophet_df = prophet_df.rename(columns={time_cols[0]: 'ds', target: 'y'})
        else:
            raise ValueError("Не найдена колонка с временными метками")

    # Удаление информации о часовом поясе
    prophet_df['ds'] = pd.to_datetime(prophet_df['ds']).dt.tz_localize(None)

    # Проверка и преобразование числовых значений
    prophet_df['y'] = pd.to_numeric(prophet_df['y'], errors='coerce')
    prophet_df = prophet_df.dropna(subset=['y'])

    # Проверка наличия обязательных колонок
    if not {'ds', 'y'}.issubset(prophet_df.columns):
        missing = {'ds', 'y'} - set(prophet_df.columns)
        raise ValueError(f"Отсутствуют обязательные колонки: {missing}")

    # Проверка пустых значений
    if prophet_df[['ds', 'y']].isnull().values.any():
        raise ValueError("Обнаружены пустые значения в колонках 'ds' или 'y'")

    model = Prophet(seasonality_mode='multiplicative')
    model.fit(prophet_df)
    return model


def train_sarima(df, target):
    """Обучение модели SARIMA с явным указанием частоты"""
    try:
        # Убедимся, что индекс имеет частоту
        ts = df[target]
        if not hasattr(ts.index, 'freq') or ts.index.freq is None:
            ts = ts.asfreq('h')  # или другая подходящая частота ('D', 'H', 'T' и т.д.)
            ts = ts.ffill()  # заполнение пропусков

        # Параметры SARIMA (подберите под ваши данные)
        order = (1, 1, 1)
        seasonal_order = (1, 1, 1, 12)  # 24 для часовых данных

        model = SARIMAX(
            ts,
            order=order,
            seasonal_order=seasonal_order,
            enforce_stationarity=False
        )
        fitted_model = model.fit(disp=False)
        return fitted_model
    except Exception as e:
        print(f"SARIMA training failed: {str(e)}")
        return None


def get_all_forecast_data_by_metric_type(df, metric_type, target, verbose=False, days=180, extended=False):
    result_data = process(target=target, df=df, metric_type=metric_type, verbose=verbose, extended=extended)
    # if verbose:
    #     print_metrics(result_data['rnn_model'], result_data['X_test'], result_data['y_test'])
        # if extended:
        #     print_metrics(result_data['prophet_model'], result_data['X_test'], result_data['y_test'])
        #     print_metrics(result_data['sarima_model'], result_data['X_test'], result_data['y_test'])

    return forecast_with_days(X_test=result_data['X_test'], model=result_data['rnn_model'],
                              scaler=result_data['scaler'],
                              days_period=days, verbose=verbose)


def get_all_data_for_forecast(dfs, service_name, verbose=False, days=180, extended=False):
    infra_predictions_1 = {}
    infra_predictions_2 = {}
    infra_predictions_3 = {}
    http_predictions_1 = {}
    http_predictions_2 = {}
    sh_predictions = {}

    result = {}
    if verbose:
        print(f'\n===================={service_name.upper()}====================\n')

    for df_name, df in dfs.items():
        if verbose:
            print(f"Get all forecasts for {service_name} based on {df_name}")

        if df_name == 'infrastructure_metrics':
            layed_out_data = lay_out_infrastructure(df)

            for i_metric_name in INFRASTRUCTURE_METRIC_NAMES:
                if verbose:
                    print(f'--------{i_metric_name.upper()}--------\n')

                m_i_data = layed_out_data[service_name][i_metric_name]

                infra_predictions_1[i_metric_name] = get_all_forecast_data_by_metric_type(
                    m_i_data,
                    df_name,
                    TARGETS[INFRASTRUCTURE_METRICS_LOAD_PERCENT],
                    verbose,
                    days,
                    extended=extended)

                infra_predictions_2[i_metric_name] = get_all_forecast_data_by_metric_type(
                    m_i_data,
                    df_name,
                    TARGETS[INFRASTRUCTURE_METRICS_RESPONSE_TIME],
                    verbose,
                    days,
                    extended=extended)

                infra_predictions_3[i_metric_name] = get_all_forecast_data_by_metric_type(
                    m_i_data,
                    df_name,
                    TARGETS[INFRASTRUCTURE_METRICS_RECOVERY_TIME],
                    verbose,
                    days,
                    extended=extended)

        elif df_name == 'http_metrics':
            layed_out_http_data = lay_out_http(df)

            for h_endpoint in ENDPOINTS:
                if verbose:
                    print(f"\tForecasting for endpoint: {h_endpoint.upper()}")

                http_predictions_1[h_endpoint] = {}
                http_predictions_2[h_endpoint] = {}
                for h_method in METHODS:
                    if verbose:
                        print(f"\t\tForecasting for method: {h_method.upper()}")

                    m_h_data = layed_out_http_data[service_name][h_endpoint][h_method]

                    http_predictions_1[h_endpoint][h_method] = get_all_forecast_data_by_metric_type(
                        m_h_data,
                        df_name,
                        TARGETS[HTTP_METRICS_ERROR_RATE],
                        verbose,
                        days,
                        extended=extended)

                    http_predictions_2[h_endpoint][h_method] = get_all_forecast_data_by_metric_type(
                        m_h_data,
                        df_name,
                        TARGETS[HTTP_METRICS_AVG_RESPONSE_TIME],
                        verbose,
                        days,
                        extended=extended)

        elif df_name == 'service_health':
            layed_out_service_health_data = lay_out_service_health(df)

            m_data = layed_out_service_health_data[service_name]

            sh_predictions = get_all_forecast_data_by_metric_type(
                m_data,
                df_name,
                TARGETS[df_name],
                verbose,
                days,
                extended=extended)
        else:
            print("Unknown dataframe name")

    result[INFRASTRUCTURE_METRICS_LOAD_PERCENT] = infra_predictions_1
    result[INFRASTRUCTURE_METRICS_RESPONSE_TIME] = infra_predictions_2
    result[INFRASTRUCTURE_METRICS_RECOVERY_TIME] = infra_predictions_3
    result[HTTP_METRICS_ERROR_RATE] = http_predictions_1
    result[HTTP_METRICS_AVG_RESPONSE_TIME] = http_predictions_2
    result[SERVICE_HEALTH_METRICS] = sh_predictions

    return result


def print_result(result):
    for metric_name, data in result.items():
        print(f"\n\n{metric_name.upper()}\n=========================")
        if (metric_name == INFRASTRUCTURE_METRICS_LOAD_PERCENT or
                metric_name == INFRASTRUCTURE_METRICS_RESPONSE_TIME
                or metric_name == INFRASTRUCTURE_METRICS_RECOVERY_TIME):
            for i_m_name, i_m_data in data.items():
                print(f'\n\t{i_m_name.upper()}\n\t-------------------')
                for day, value in i_m_data.items():
                    if day % 10 == 0:
                        print(f'\t\t{day}: {value}')
        elif metric_name == HTTP_METRICS_ERROR_RATE or metric_name == HTTP_METRICS_AVG_RESPONSE_TIME:
            for h_endpoint, endpoint_data in data.items():
                print(f'\n\t{h_endpoint.upper()}\n\t-------------------')
                for h_method, endpoint_method_data in endpoint_data.items():
                    print(f'\n\t\t{h_method.upper()}\n\t\t-------------------')
                    for day, value in endpoint_method_data.items():
                        if day % 10 == 0:
                            print(f'\t\t\t{day}: {value}')
        elif metric_name == SERVICE_HEALTH_METRICS:
            for day, value in data.items():
                if day % 10 == 0:
                    print(f'\t{day}: {value}')
