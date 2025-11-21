import pandas as pd
import numpy as np
import lightgbm as lgb
import yfinance as yf
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error

def load_and_process_data(filepath):
    """Loads data and resamples to quarterly sum. Fetches exogenous data."""
    df = pd.read_csv(filepath)
    df['DATE'] = pd.to_datetime(df['DATE'])
    df.set_index('DATE', inplace=True)
    
    # Resample to quarterly sum
    df_quarterly = df.resample('Q').sum()
    df_quarterly.columns = ['Value']
    
    # Check if last quarter is incomplete.
    df_quarterly = df_quarterly.iloc[:-1]
    
    # Initialize SP500 with 0
    df_quarterly['SP500'] = 0.0
    
    # Fetch Exogenous Data (S&P 500)
    start_date = df_quarterly.index[0]
    end_date = df_quarterly.index[-1] + pd.DateOffset(days=1)
    exo_start = start_date - pd.DateOffset(years=1)
    
    try:
        print(f"Fetching S&P 500 data from {exo_start} to {end_date}...")
        # sp500 = yf.download("^GSPC", start=exo_start, end=end_date, progress=False)
        raise Exception("Disabled for debugging")
        
        if not sp500.empty:
            # Check columns. yfinance might return MultiIndex or just 'Close'
            # If MultiIndex, it might be ('Close', '^GSPC')
            if isinstance(sp500.columns, pd.MultiIndex):
                # Try to find Close
                if 'Close' in sp500.columns.get_level_values(0):
                    sp500_close = sp500['Close']
                    # If it's still a DataFrame (multiple tickers?), take first
                    if isinstance(sp500_close, pd.DataFrame):
                        sp500_close = sp500_close.iloc[:, 0]
                else:
                    # Fallback
                    sp500_close = sp500.iloc[:, 0]
            elif 'Close' in sp500.columns:
                sp500_close = sp500['Close']
            else:
                sp500_close = sp500.iloc[:, 0]
                
            # Resample
            sp500_q = sp500_close.resample('Q').mean()
            
            # Update df_quarterly using direct assignment (aligns on index)
            # This is safer than join and avoids overwrite issues
            df_quarterly['SP500'] = sp500_q
            
            # Fill missing
            df_quarterly['SP500'] = df_quarterly['SP500'].ffill().bfill().fillna(0.0)
            print("S&P 500 data merged successfully. Columns:", df_quarterly.columns)
        else:
            print("Warning: S&P 500 data download returned empty.")
            
    except Exception as e:
        print(f"Error fetching exogenous data: {e}")
        # SP500 is already 0.0
def create_features_and_targets(df, input_lags=8, output_horizon=4):
    """
    Creates features and targets based on the requirement:
    Use y-8, ..., y-1 to predict y+1, ..., y+4.
    Skip current quarter y.
    Adds rolling and diff features.
    Adds exogenous features (S&P 500 lags).
    """
    data = df['Value'].values
    exo_data = df['SP500'].values
    
    X = []
    y = []
    dates = []
    
    # Feature names for dataframe reconstruction
    feature_names = [f'lag_{i}' for i in range(input_lags, 0, -1)] # lag_8 ... lag_1
    feature_names += ['rolling_mean_4', 'rolling_std_4', 'diff_lag1_lag2', 'diff_lag1_lag5']
    feature_names += [f'exo_lag_{i}' for i in range(input_lags, 0, -1)] # exo_lag_8 ... exo_lag_1
    
    for t in range(input_lags, len(data) - output_horizon):
        # Features: t-8 to t-1
        features_raw = data[t-input_lags : t]
        exo_features_raw = exo_data[t-input_lags : t]
        
        # --- New Features ---
        # Rolling Mean/Std of last 4 lags (most recent 4 quarters in the feature set)
        roll_window = features_raw[-4:]
        roll_mean = np.mean(roll_window)
        roll_std = np.std(roll_window)
        
        # Diff features
        diff_1_2 = features_raw[-1] - features_raw[-2]
        diff_1_5 = features_raw[-1] - features_raw[-5]
        
        # Combine
        row_features = list(features_raw) + [roll_mean, roll_std, diff_1_2, diff_1_5] + list(exo_features_raw)
        
        # Targets: t+1 to t+4
        targets = data[t+1 : t+1+output_horizon]
        
        X.append(row_features)
        y.append(targets)
        dates.append(df.index[t]) 
        
    return np.array(X), np.array(y), dates, feature_names

def train_models(X_train, y_train):
    """Trains one LightGBM model per horizon step."""
    models = []
    for i in range(y_train.shape[1]):
        model = lgb.LGBMRegressor(n_estimators=100, random_state=42, verbose=-1)
        model.fit(X_train, y_train[:, i])
        models.append(model)
    return models

def conformal_prediction(models, X_calib, y_calib, X_test, alpha=0.1):
    """
    Applies Split Conformal Prediction.
    """
    intervals = []
    
    for i, model in enumerate(models):
        preds_calib = model.predict(X_calib)
        scores = np.abs(y_calib[:, i] - preds_calib)
        n = len(scores)
        q_val = np.quantile(scores, np.ceil((n + 1) * (1 - alpha)) / n)
        
        preds_test = model.predict(X_test)
        lower = preds_test - q_val
        upper = preds_test + q_val
        
        intervals.append({
            'horizon': i+1,
            'pred': preds_test,
            'lower': lower,
            'upper': upper,
            'q_val': q_val
        })
        
    return intervals

def run_forecasting_pipeline(filepath):
    df = load_and_process_data(filepath)
    X, y, dates, feature_names = create_features_and_targets(df)
    
    # Split: Train, Calibration, Test
    n_total = len(X)
    n_test = 8
    n_calib = 16
    n_train = n_total - n_test - n_calib
    
    if n_train < 10:
        n_test = 4
        n_calib = 8
        n_train = n_total - n_test - n_calib
        
    X_train = X[:n_train]
    y_train = y[:n_train]
    
    X_calib = X[n_train : n_train + n_calib]
    y_calib = y[n_train : n_train + n_calib]
    
    X_test = X[n_train + n_calib :]
    y_test = y[n_train + n_calib :]
    dates_test = dates[n_train + n_calib :]
    
    # Train
    models = train_models(X_train, y_train)
    
    # Conformal Prediction
    intervals = conformal_prediction(models, X_calib, y_calib, X_test, alpha=0.1)
    
    # Organize results
    results = []
    for idx, date in enumerate(dates_test):
        row = {'Date_Anchor': date}
        for i, interval in enumerate(intervals):
            h = i + 1
            row[f'Pred_h{h}'] = interval['pred'][idx]
            row[f'Lower_h{h}'] = interval['lower'][idx]
            row[f'Upper_h{h}'] = interval['upper'][idx]
            row[f'True_h{h}'] = y_test[idx, i]
        results.append(row)
        
    results_df = pd.DataFrame(results)
import pandas as pd
import numpy as np
import lightgbm as lgb
import yfinance as yf
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error

def load_and_process_data(filepath):
    """Loads data and resamples to quarterly sum. Fetches exogenous data."""
    df = pd.read_csv(filepath)
    df['DATE'] = pd.to_datetime(df['DATE'])
    df.set_index('DATE', inplace=True)
    
    # Resample to quarterly sum
    df_quarterly = df.resample('Q').sum()
    df_quarterly.columns = ['Value']
    
    # Check if last quarter is incomplete.
    df_quarterly = df_quarterly.iloc[:-1]
    
    # Initialize SP500 with 0
    df_quarterly['SP500'] = 0.0
    
    # Fetch Exogenous Data (S&P 500)
    start_date = df_quarterly.index[0]
    end_date = df_quarterly.index[-1] + pd.DateOffset(days=1)
    exo_start = start_date - pd.DateOffset(years=1)
    
    try:
        print(f"Fetching S&P 500 data from {exo_start} to {end_date}...")
        # sp500 = yf.download("^GSPC", start=exo_start, end=end_date, progress=False)
        raise Exception("Disabled for debugging")
        
        if not sp500.empty:
            # Check columns. yfinance might return MultiIndex or just 'Close'
            # If MultiIndex, it might be ('Close', '^GSPC')
            if isinstance(sp500.columns, pd.MultiIndex):
                # Try to find Close
                if 'Close' in sp500.columns.get_level_values(0):
                    sp500_close = sp500['Close']
                    # If it's still a DataFrame (multiple tickers?), take first
                    if isinstance(sp500_close, pd.DataFrame):
                        sp500_close = sp500_close.iloc[:, 0]
                else:
                    # Fallback
                    sp500_close = sp500.iloc[:, 0]
            elif 'Close' in sp500.columns:
                sp500_close = sp500['Close']
            else:
                sp500_close = sp500.iloc[:, 0]
                
            # Resample
            sp500_q = sp500_close.resample('Q').mean()
            
            # Update df_quarterly using direct assignment (aligns on index)
            # This is safer than join and avoids overwrite issues
            df_quarterly['SP500'] = sp500_q
            
            # Fill missing
            df_quarterly['SP500'] = df_quarterly['SP500'].ffill().bfill().fillna(0.0)
            print("S&P 500 data merged successfully. Columns:", df_quarterly.columns)
        else:
            print("Warning: S&P 500 data download returned empty.")
            
    except Exception as e:
        print(f"Error fetching exogenous data: {e}")
        # SP500 is already 0.0
def create_features_and_targets(df, input_lags=8, output_horizon=4):
    """
    Creates features and targets based on the requirement:
    Use y-8, ..., y-1 to predict y+1, ..., y+4.
    Skip current quarter y.
    Adds rolling and diff features.
    Adds exogenous features (S&P 500 lags).
    """
    data = df['Value'].values
    exo_data = df['SP500'].values
    
    X = []
    y = []
    dates = []
    
    # Feature names for dataframe reconstruction
    feature_names = [f'lag_{i}' for i in range(input_lags, 0, -1)] # lag_8 ... lag_1
    feature_names += ['rolling_mean_4', 'rolling_std_4', 'diff_lag1_lag2', 'diff_lag1_lag5']
    feature_names += [f'exo_lag_{i}' for i in range(input_lags, 0, -1)] # exo_lag_8 ... exo_lag_1
    
    for t in range(input_lags, len(data) - output_horizon):
        # Features: t-8 to t-1
        features_raw = data[t-input_lags : t]
        exo_features_raw = exo_data[t-input_lags : t]
        
        # --- New Features ---
        # Rolling Mean/Std of last 4 lags (most recent 4 quarters in the feature set)
        roll_window = features_raw[-4:]
        roll_mean = np.mean(roll_window)
        roll_std = np.std(roll_window)
        
        # Diff features
        diff_1_2 = features_raw[-1] - features_raw[-2]
        diff_1_5 = features_raw[-1] - features_raw[-5]
        
        # Combine
        row_features = list(features_raw) + [roll_mean, roll_std, diff_1_2, diff_1_5] + list(exo_features_raw)
        
        # Targets: t+1 to t+4
        targets = data[t+1 : t+1+output_horizon]
        
        X.append(row_features)
        y.append(targets)
        dates.append(df.index[t]) 
        
    return np.array(X), np.array(y), dates, feature_names

def train_models(X_train, y_train):
    """Trains one LightGBM model per horizon step."""
    models = []
    for i in range(y_train.shape[1]):
        model = lgb.LGBMRegressor(n_estimators=100, random_state=42, verbose=-1)
        model.fit(X_train, y_train[:, i])
        models.append(model)
    return models

def conformal_prediction(models, X_calib, y_calib, X_test, alpha=0.1):
    """
    Applies Split Conformal Prediction.
    """
    intervals = []
    
    for i, model in enumerate(models):
        preds_calib = model.predict(X_calib)
        scores = np.abs(y_calib[:, i] - preds_calib)
        n = len(scores)
        q_val = np.quantile(scores, np.ceil((n + 1) * (1 - alpha)) / n)
        
        preds_test = model.predict(X_test)
        lower = preds_test - q_val
        upper = preds_test + q_val
        
        intervals.append({
            'horizon': i+1,
            'pred': preds_test,
            'lower': lower,
            'upper': upper,
            'q_val': q_val
        })
        
    return intervals

def run_forecasting_pipeline(filepath):
    df = load_and_process_data(filepath)
    X, y, dates, feature_names = create_features_and_targets(df)
    
    # Split: Train, Calibration, Test
    n_total = len(X)
    n_test = 8
    n_calib = 16
    n_train = n_total - n_test - n_calib
    
    if n_train < 10:
        n_test = 4
        n_calib = 8
        n_train = n_total - n_test - n_calib
        
    X_train = X[:n_train]
    y_train = y[:n_train]
    
    X_calib = X[n_train : n_train + n_calib]
    y_calib = y[n_train : n_train + n_calib]
    
    X_test = X[n_train + n_calib :]
    y_test = y[n_train + n_calib :]
    dates_test = dates[n_train + n_calib :]
    
    # Train
    models = train_models(X_train, y_train)
    
    # Conformal Prediction
    intervals = conformal_prediction(models, X_calib, y_calib, X_test, alpha=0.1)
    
    # Organize results
    results = []
    for idx, date in enumerate(dates_test):
        row = {'Date_Anchor': date}
        for i, interval in enumerate(intervals):
            h = i + 1
            row[f'Pred_h{h}'] = interval['pred'][idx]
            row[f'Lower_h{h}'] = interval['lower'][idx]
            row[f'Upper_h{h}'] = interval['upper'][idx]
            row[f'True_h{h}'] = y_test[idx, i]
        results.append(row)
        
    results_df = pd.DataFrame(results)
    
    # Calculate Metrics
    all_preds = []
    all_true = []
    for i in range(4):
        all_preds.extend(results_df[f'Pred_h{i+1}'].values)
import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error

def load_and_process_data(filepath):
    """Loads data and resamples to quarterly sum."""
    df = pd.read_csv(filepath)
    df['DATE'] = pd.to_datetime(df['DATE'])
    df.set_index('DATE', inplace=True)
    
    # Resample to quarterly sum
    df_quarterly = df.resample('Q').sum()
    df_quarterly.columns = ['Value']
    
    # Check if last quarter is incomplete.
    # In this specific dataset, we know the last point is 2018-01-01.
    # So 2018Q1 is incomplete (only 1 month).
    # We can check the last index date.
    # Or we can just drop the last row as requested "last year production crashed... remove those points".
    # Dropping the last row is safe here.
    df_quarterly = df_quarterly.iloc[:-1]
    
    # Create Quarter ID for display (e.g., q1_1993)
    df_quarterly['Quarter_ID'] = df_quarterly.index.to_series().apply(
        lambda x: f"q{x.quarter}_{x.year}"
    )
    
    # Force SP500 to exist and print columns
    if 'SP500' not in df_quarterly.columns:
        df_quarterly['SP500'] = 0.0
        print("DEBUG: Added SP500 column (was missing).")
    else:
        print("DEBUG: SP500 column exists.")
        
    print("DEBUG: Final columns:", df_quarterly.columns)
    return df_quarterly

def create_features_and_targets(df, input_lags=8, output_horizon=4):
    """
    Creates features and targets based on the requirement:
    Use y-8, ..., y-1 to predict y+1, ..., y+4.
    Skip current quarter y.
    Adds rolling and diff features.
    """
    data = df['Value'].values
    X = []
    y = []
    dates = []
    
    # Feature names for dataframe reconstruction
    feature_names = [f'lag_{i}' for i in range(input_lags, 0, -1)] # lag_8 ... lag_1
    feature_names += ['rolling_mean_4', 'rolling_std_4', 'diff_lag1_lag2', 'diff_lag1_lag5']
    
    for t in range(input_lags, len(data) - output_horizon):
        # Features: t-8 to t-1
        features_raw = data[t-input_lags : t]
        
        # --- New Features ---
        # Rolling Mean/Std of last 4 lags (most recent 4 quarters in the feature set)
        # features_raw[-4:] are the lags t-4, t-3, t-2, t-1
        roll_window = features_raw[-4:]
        roll_mean = np.mean(roll_window)
        roll_std = np.std(roll_window)
        
        # Diff features
        # lag1 is features_raw[-1]
        # lag2 is features_raw[-2]
        # lag5 is features_raw[-5] (1 year ago from lag1)
        diff_1_2 = features_raw[-1] - features_raw[-2]
        diff_1_5 = features_raw[-1] - features_raw[-5]
        
        # Combine
        row_features = list(features_raw) + [roll_mean, roll_std, diff_1_2, diff_1_5]
        
        # Targets: t+1 to t+4
        targets = data[t+1 : t+1+output_horizon]
        
        X.append(row_features)
        y.append(targets)
        dates.append(df.index[t]) 
        
    return np.array(X), np.array(y), dates, feature_names

def train_models(X_train, y_train):
    """Trains one LightGBM model per horizon step."""
    models = []
    for i in range(y_train.shape[1]):
        model = lgb.LGBMRegressor(n_estimators=100, random_state=42, verbose=-1)
        model.fit(X_train, y_train[:, i])
        models.append(model)
    return models

def conformal_prediction(models, X_calib, y_calib, X_test, alpha=0.1):
    """
    Applies Split Conformal Prediction.
    """
    intervals = []
    
    for i, model in enumerate(models):
        preds_calib = model.predict(X_calib)
        scores = np.abs(y_calib[:, i] - preds_calib)
        n = len(scores)
        q_val = np.quantile(scores, np.ceil((n + 1) * (1 - alpha)) / n)
        
        preds_test = model.predict(X_test)
        lower = preds_test - q_val
        upper = preds_test + q_val
        
        intervals.append({
            'horizon': i+1,
            'pred': preds_test,
            'lower': lower,
            'upper': upper,
            'q_val': q_val
        })
        
    return intervals

def run_forecasting_pipeline(filepath):
    df = load_and_process_data(filepath)
    X, y, dates, feature_names = create_features_and_targets(df)
    
    # Split: Train, Calibration, Test
    n_total = len(X)
    n_test = 8
    n_calib = 16
    n_train = n_total - n_test - n_calib
    
    if n_train < 10:
        n_test = 4
        n_calib = 8
        n_train = n_total - n_test - n_calib
        
    X_train = X[:n_train]
    y_train = y[:n_train]
    
    X_calib = X[n_train : n_train + n_calib]
    y_calib = y[n_train : n_train + n_calib]
    
    X_test = X[n_train + n_calib :]
    y_test = y[n_train + n_calib :]
    dates_test = dates[n_train + n_calib :]
    
    # Train
    models = train_models(X_train, y_train)
    
    # Conformal Prediction
    intervals = conformal_prediction(models, X_calib, y_calib, X_test, alpha=0.1)
    
    # Organize results
    results = []
    for idx, date in enumerate(dates_test):
        row = {'Date_Anchor': date}
        for i, interval in enumerate(intervals):
            h = i + 1
            row[f'Pred_h{h}'] = interval['pred'][idx]
            row[f'Lower_h{h}'] = interval['lower'][idx]
            row[f'Upper_h{h}'] = interval['upper'][idx]
            row[f'True_h{h}'] = y_test[idx, i]
        results.append(row)
        
    results_df = pd.DataFrame(results)
    
    # Calculate Test Metrics
    all_preds_test = []
    all_true_test = []
    for i in range(4):
        all_preds_test.extend(results_df[f'Pred_h{i+1}'].values)
        all_true_test.extend(results_df[f'True_h{i+1}'].values)
        
    test_mape = mean_absolute_percentage_error(all_true_test, all_preds_test)
    test_mae = mean_absolute_error(all_true_test, all_preds_test)
    
    # Calculate Training Metrics
    # Predict on X_train
    train_preds_list = []
    train_true_list = []
    
    for i, model in enumerate(models):
        preds_train = model.predict(X_train)
        true_train = y_train[:, i]
        train_preds_list.extend(preds_train)
        train_true_list.extend(true_train)
        
    train_mape = mean_absolute_percentage_error(train_true_list, train_preds_list)
    train_mae = mean_absolute_error(train_true_list, train_preds_list)
    
    # Create Training Data DataFrame for display
    target_names = [f'Target_h{i+1}' for i in range(4)]
    full_data_df = pd.DataFrame(X, columns=feature_names)
    target_df = pd.DataFrame(y, columns=target_names)
    full_data_df = pd.concat([pd.Series(dates, name='Date_Anchor'), full_data_df, target_df], axis=1)
    
    # Calculate Feature Importance
    importances = np.zeros(len(feature_names))
    for model in models:
        importances += model.feature_importances_
    importances /= len(models)
    
    importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importances
    }).sort_values(by='Importance', ascending=False)
    
    return df, results_df, test_mape, test_mae, train_mape, train_mae, full_data_df, importance_df

if __name__ == "__main__":
    # Quick test
    df, res, t_mape, t_mae, tr_mape, tr_mae, full, imp = run_forecasting_pipeline("Electric_Production.csv")
    print(f"Test MAPE: {t_mape:.4f}, Test MAE: {t_mae:.4f}")
    print(f"Train MAPE: {tr_mape:.4f}, Train MAE: {tr_mae:.4f}")
    print(imp.head())
