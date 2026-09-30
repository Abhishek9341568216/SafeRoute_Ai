import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def evaluate_regression(y_true, y_pred, model_name="Model"):
    """
    Computes regression evaluation metrics:
    - MAE: Mean Absolute Error
    - MSE: Mean Squared Error
    - RMSE: Root Mean Squared Error
    - R2: Coefficient of Determination
    """
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)
    
    metrics = {
        'model': model_name,
        'mae': round(float(mae), 4),
        'mse': round(float(mse), 4),
        'rmse': round(float(rmse), 4),
        'r2': round(float(r2), 4)
    }
    return metrics
