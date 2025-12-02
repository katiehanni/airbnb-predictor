"""
Machine Learning Models for Airbnb price prediction.

This module implements baseline, linear regression, random forest, and XGBoost
models for predicting Airbnb listing prices, with evaluation metrics.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Try to import XGBoost (optional)
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except (ImportError, Exception) as e:
    XGBOOST_AVAILABLE = False
    print(f"XGBoost not available: {type(e).__name__}")
    print("The notebook will run without XGBoost. Other models (Baseline, Linear Regression, Random Forest) will still work.")


def prepare_features(df, target_col='price'):
    """
    Prepare feature matrix and target vector from dataframe.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Preprocessed dataframe
    target_col : str
        Name of target column
        
    Returns:
    --------
    X : pd.DataFrame
        Feature matrix
    y : pd.Series
        Target vector
    """
    # Select features (exclude target)
    exclude_cols = [target_col]
    
    # Get all columns except target
    feature_cols = [col for col in df.columns if col not in exclude_cols]
    
    X = df[feature_cols].copy()
    y = df[target_col].copy()
    
    # Remove rows with missing target
    mask = ~y.isna()
    X = X[mask]
    y = y[mask]
    
    # Fill any remaining NaN in features with 0
    X = X.fillna(0)
    
    # Ensure all features are numeric
    for col in X.columns:
        if X[col].dtype == 'object':
            # Try to convert to numeric
            X[col] = pd.to_numeric(X[col], errors='coerce')
            X[col] = X[col].fillna(0)
    
    return X, y


def evaluate_model(y_true, y_pred, model_name='Model'):
    """
    Evaluate model predictions and return metrics.
    
    Parameters:
    -----------
    y_true : array-like
        True target values
    y_pred : array-like
        Predicted target values
    model_name : str
        Name of the model
        
    Returns:
    --------
    dict
        Dictionary of evaluation metrics
    """
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    
    metrics = {
        'Model': model_name,
        'RMSE': rmse,
        'MAE': mae,
        'R²': r2
    }
    
    return metrics


def baseline_mean_model(X_train, y_train, X_test, y_test):
    """
    Baseline model: predict the mean of training data.
    
    Parameters:
    -----------
    X_train : pd.DataFrame
        Training features
    y_train : pd.Series
        Training target
    X_test : pd.DataFrame
        Test features
    y_test : pd.Series
        Test target
        
    Returns:
    --------
    dict
        Evaluation metrics
    """
    mean_pred = np.full(len(y_test), y_train.mean())
    return evaluate_model(y_test, mean_pred, 'Baseline (Mean)')


def linear_regression_model(X_train, y_train, X_test, y_test):
    """
    Linear regression model.
    
    Parameters:
    -----------
    X_train : pd.DataFrame
        Training features
    y_train : pd.Series
        Training target
    X_test : pd.DataFrame
        Test features
    y_test : pd.Series
        Test target
        
    Returns:
    --------
    tuple
        (evaluation_metrics, trained_model)
    """
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    metrics = evaluate_model(y_test, y_pred, 'Linear Regression')
    
    return metrics, model


def random_forest_model(X_train, y_train, X_test, y_test, n_estimators=100, max_depth=10):
    """
    Random Forest regressor model.
    
    Parameters:
    -----------
    X_train : pd.DataFrame
        Training features
    y_train : pd.Series
        Training target
    X_test : pd.DataFrame
        Test features
    y_test : pd.Series
        Test target
    n_estimators : int
        Number of trees
    max_depth : int
        Maximum depth of trees
        
    Returns:
    --------
    tuple
        (evaluation_metrics, trained_model)
    """
    model = RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    metrics = evaluate_model(y_test, y_pred, 'Random Forest')
    
    return metrics, model


def xgboost_model(X_train, y_train, X_test, y_test, n_estimators=100, max_depth=6):
    """
    XGBoost regressor model.
    
    Parameters:
    -----------
    X_train : pd.DataFrame
        Training features
    y_train : pd.Series
        Training target
    X_test : pd.DataFrame
        Test features
    y_test : pd.Series
        Test target
    n_estimators : int
        Number of boosting rounds
    max_depth : int
        Maximum depth of trees
        
    Returns:
    --------
    tuple
        (evaluation_metrics, trained_model)
    """
    if not XGBOOST_AVAILABLE:
        return None, None
    
    model = xgb.XGBRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=0.1,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    metrics = evaluate_model(y_test, y_pred, 'XGBoost')
    
    return metrics, model


def train_models(df, test_size=0.2, target_col='price'):
    """
    Train all models and return evaluation results.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Preprocessed dataframe
    test_size : float
        Proportion of data for testing
    target_col : str
        Name of target column
        
    Returns:
    --------
    tuple
        (results_dataframe, models_dictionary)
    """
    print("Preparing features...")
    X, y = prepare_features(df, target_col=target_col)
    
    print(f"Features: {X.shape[1]}")
    print(f"Samples: {X.shape[0]}")
    
    # Split data randomly (for Airbnb, random split is fine)
    print(f"\nSplitting data (test_size={test_size})...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42
    )
    
    print(f"Training samples: {len(X_train)}")
    print(f"Test samples: {len(X_test)}")
    
    # Store results
    results = []
    models = {}
    
    # Baseline model
    print("\n" + "=" * 50)
    print("Training Baseline Model (Mean)...")
    print("=" * 50)
    baseline_metrics = baseline_mean_model(X_train, y_train, X_test, y_test)
    results.append(baseline_metrics)
    print(f"RMSE: ${baseline_metrics['RMSE']:.2f}, MAE: ${baseline_metrics['MAE']:.2f}, R²: {baseline_metrics['R²']:.3f}")
    
    # Linear Regression
    print("\n" + "=" * 50)
    print("Training Linear Regression...")
    print("=" * 50)
    lr_metrics, lr_model = linear_regression_model(X_train, y_train, X_test, y_test)
    results.append(lr_metrics)
    models['linear_regression'] = lr_model
    print(f"RMSE: ${lr_metrics['RMSE']:.2f}, MAE: ${lr_metrics['MAE']:.2f}, R²: {lr_metrics['R²']:.3f}")
    
    # Random Forest
    print("\n" + "=" * 50)
    print("Training Random Forest...")
    print("=" * 50)
    rf_metrics, rf_model = random_forest_model(X_train, y_train, X_test, y_test)
    results.append(rf_metrics)
    models['random_forest'] = rf_model
    print(f"RMSE: ${rf_metrics['RMSE']:.2f}, MAE: ${rf_metrics['MAE']:.2f}, R²: {rf_metrics['R²']:.3f}")
    
    # XGBoost (if available)
    if XGBOOST_AVAILABLE:
        print("\n" + "=" * 50)
        print("Training XGBoost...")
        print("=" * 50)
        xgb_metrics, xgb_model = xgboost_model(X_train, y_train, X_test, y_test)
        if xgb_metrics:
            results.append(xgb_metrics)
            models['xgboost'] = xgb_model
            print(f"RMSE: ${xgb_metrics['RMSE']:.2f}, MAE: ${xgb_metrics['MAE']:.2f}, R²: {xgb_metrics['R²']:.3f}")
    else:
        print("\nSkipping XGBoost (not installed)")
    
    # Convert results to dataframe
    results_df = pd.DataFrame(results)
    
    print("\n" + "=" * 50)
    print("Model Comparison:")
    print("=" * 50)
    print(results_df.to_string(index=False))
    
    # Save models
    models_dir = Path(__file__).parent.parent / 'output' / 'models'
    models_dir.mkdir(parents=True, exist_ok=True)
    
    import pickle
    for name, model in models.items():
        model_path = models_dir / f'{name}.pkl'
        with open(model_path, 'wb') as f:
            pickle.dump(model, f)
        print(f"Saved model: {model_path}")
    
    return results_df, models


if __name__ == '__main__':
    # Test models
    from load_data import load_data
    from preprocess import preprocess
    
    df_raw = load_data()
    df = preprocess(df_raw)
    results, models = train_models(df)
    print("\nResults:")
    print(results)
