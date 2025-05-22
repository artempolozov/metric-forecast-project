import numpy as np
import torch


def train_model(model, train_loader, criterion, optimizer, n_epochs, verbose=False):
    model.train()
    for epoch in range(n_epochs):
        total_loss = 0

        for batch_x, batch_y in train_loader:
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        if verbose:
            print(f'Epoch {epoch + 1} - Avg Loss: {total_loss / len(train_loader):.4f}')


def predict_future(model, last_sequence, scaler, days_ahead):
    model.eval()
    predictions = []
    # Преобразуем в тензор если нужно
    if not isinstance(last_sequence, torch.Tensor):
        current_seq = torch.FloatTensor(last_sequence).to(model.device)
    else:
        current_seq = last_sequence.clone().to(model.device)

    for _ in range(days_ahead):
        with torch.no_grad():
            pred = model(current_seq.unsqueeze(0))
            predictions.append(pred.item())

            # Создаем новую строку
            last_features = current_seq[-1, :-1]
            new_row = torch.cat([
                last_features,
                pred.view(-1)  # Преобразуем [1,1] -> [1]
            ])

            # Обновляем последовательность
            current_seq = torch.cat([
                current_seq[1:],
                new_row.unsqueeze(0)
            ])

    # Обратное масштабирование
    dummy = np.zeros((len(predictions), scaler.n_features_in_))
    dummy[:, -1] = predictions
    predictions = scaler.inverse_transform(dummy)[:, -1]

    return predictions


def forecast_with_days(model, X_test, scaler, days_period, verbose=False):
    if not isinstance(X_test, torch.Tensor):
        X_test = torch.FloatTensor(X_test)

    # Перенос данных на то же устройство, что и модель
    device = next(model.parameters()).device
    X_test = X_test.to(device)

    last_sequence = X_test[-1]  # Берем последнюю последовательность

    result = {}

    if verbose:
        print("\nFuture predictions...")

    # Прогнозируем сразу весь период
    predictions = predict_future(model, last_sequence, scaler, days_period)

    # Заполняем словарь результата
    for day in range(1, days_period + 1):
        result[day] = predictions[day - 1]

    return result
