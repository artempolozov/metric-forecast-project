import numpy as np
import torch
from sklearn.metrics import mean_absolute_error, mean_squared_error


def evaluate(model, X_test, y_test):
    model.eval()
    with torch.no_grad():
        test_preds = model(X_test)
        mae = mean_absolute_error(y_test.cpu(), test_preds.cpu())
        mse = np.sqrt(mean_squared_error(y_test.cpu(), test_preds.cpu()))
    return mae, mse


def print_metrics(model, X_test, y_test):
    rnn_mae, rnn_mse = evaluate(model, X_test, y_test)
    print(f"\nRNN Test MAE: {rnn_mae:.4f}\tMSE: {rnn_mse:.4f}")


def calculate_metrics(y_true, y_pred, scaler=None):
    """Улучшенная функция расчета метрик с проверкой размерностей"""
    try:
        # Приведение к правильной форме
        y_true = np.array(y_true).reshape(-1, 1)
        y_pred = np.array(y_pred).reshape(-1, 1)

        if scaler:
            # Проверка совместимости scaler
            if y_true.shape[1] != scaler.n_features_in_:
                y_true = y_true.reshape(-1, scaler.n_features_in_)
                y_pred = y_pred.reshape(-1, scaler.n_features_in_)

            y_true = scaler.inverse_transform(y_true).flatten()
            y_pred = scaler.inverse_transform(y_pred).flatten()

        # Расчет метрик
        mae = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        denominator = (np.abs(y_true) + np.abs(y_pred))
        mask = denominator > 0  # Игнорируем нулевые знаменатели
        smape = 100 * np.mean(2 * np.abs(y_pred[mask] - y_true[mask]) / denominator[mask])

        return {'MAE': mae, 'RMSE': rmse, 'SMAPE': smape}

    except Exception as e:
        print(f"Metrics calculation error: {str(e)}")
        return {'MAE': None, 'RMSE': None, 'SMAPE': None}

