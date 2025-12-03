"""
=============================================================================
올인원 템플릿 - 모든 모델 포함
Linear, Lasso, Ridge, RandomForest, XGBoost, ARIMA, DNN, CNN, RNN(LSTM)
=============================================================================
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder
from sklearn.linear_model import LinearRegression, Ridge, Lasso, LogisticRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import (accuracy_score, f1_score, roc_auc_score, 
                             mean_absolute_error, mean_squared_error, r2_score)
from imblearn.over_sampling import SMOTE
import xgboost as xgb
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.arima.model import ARIMA
import warnings
warnings.filterwarnings('ignore')

# GPU 설정
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")

SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)


# ============================================================================
# ⭐ 설정 (여기만 수정!)
# ============================================================================
PROBLEM_TYPE = 'classification'  # 'classification', 'regression', 'timeseries'
DATA_FILE = 'train.csv'
TARGET_COL = 'target'

# 모델 선택 (True로 설정하면 학습)
USE_LINEAR = True       # Linear/Logistic, Ridge, Lasso
USE_RF = True          # Random Forest
USE_XGB = True         # XGBoost
USE_ARIMA = False      # ARIMA (시계열 전용)
USE_DNN = False        # DNN (PyTorch)
USE_CNN = False        # CNN 1D (PyTorch, 시계열)
USE_LSTM = False       # LSTM (PyTorch, 시계열)


# ============================================================================
# [A] 분류/회귀 문제
# ============================================================================

if PROBLEM_TYPE in ['classification', 'regression']:
    print("="*60)
    print(f"[{PROBLEM_TYPE.upper()}] 모델 학습 시작")
    print("="*60)
    
    # ------------------------------------------------------------------------
    # 1. 데이터 로드 및 전처리
    # ------------------------------------------------------------------------
    df = pd.read_csv(DATA_FILE)
    print(f"데이터 shape: {df.shape}")
    
    # 결측치 처리
    df = df.fillna(df.median(numeric_only=True))
    for col in df.select_dtypes(include=['object']).columns:
        if col != TARGET_COL:
            df[col] = LabelEncoder().fit_transform(df[col].astype(str))
    
    # X, y 분리
    X = df.drop(columns=[TARGET_COL])
    y = df[TARGET_COL]
    
    # 타겟 인코딩 (분류)
    if PROBLEM_TYPE == 'classification' and y.dtype == 'object':
        le_target = LabelEncoder()
        y = le_target.fit_transform(y)
        n_classes = len(le_target.classes_)
    else:
        n_classes = len(np.unique(y)) if PROBLEM_TYPE == 'classification' else 1
    
    # 데이터 분할
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=SEED,
        stratify=y if PROBLEM_TYPE == 'classification' else None
    )
    
    # 스케일링
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # 불균형 처리 (분류)
    if PROBLEM_TYPE == 'classification':
        imbalance = max(np.bincount(y_train)) / min(np.bincount(y_train))
        if imbalance > 3:
            X_train_balanced, y_train_balanced = SMOTE(random_state=SEED).fit_resample(
                X_train_scaled, y_train
            )
            print(f"SMOTE 적용 (불균형 비율: {imbalance:.1f}:1)")
        else:
            X_train_balanced, y_train_balanced = X_train_scaled, y_train
    else:
        X_train_balanced, y_train_balanced = X_train_scaled, y_train
    
    print(f"Train: {X_train.shape}, Test: {X_test.shape}")
    
    results = {}
    
    # ------------------------------------------------------------------------
    # 2. Linear Models (Linear/Logistic, Ridge, Lasso)
    # ------------------------------------------------------------------------
    if USE_LINEAR:
        print("\n" + "="*60)
        print("[Linear Models]")
        print("="*60)
        
        if PROBLEM_TYPE == 'classification':
            # Logistic Regression
            print("\n[1] Logistic Regression")
            lr = LogisticRegression(max_iter=1000, random_state=SEED)
            lr.fit(X_train_balanced, y_train_balanced)
            lr_pred = lr.predict(X_test_scaled)
            results['Logistic'] = {
                'f1': f1_score(y_test, lr_pred, average='weighted'),
                'accuracy': accuracy_score(y_test, lr_pred)
            }
            print(f"F1: {results['Logistic']['f1']:.4f}, Acc: {results['Logistic']['accuracy']:.4f}")
            
        else:  # regression
            # Linear Regression
            print("\n[1] Linear Regression")
            lr = LinearRegression()
            lr.fit(X_train_scaled, y_train)
            lr_pred = lr.predict(X_test_scaled)
            results['Linear'] = {
                'mae': mean_absolute_error(y_test, lr_pred),
                'rmse': np.sqrt(mean_squared_error(y_test, lr_pred)),
                'r2': r2_score(y_test, lr_pred)
            }
            print(f"MAE: {results['Linear']['mae']:.4f}, RMSE: {results['Linear']['rmse']:.4f}, R2: {results['Linear']['r2']:.4f}")
            
            # Ridge
            print("\n[2] Ridge Regression")
            ridge = Ridge(alpha=1.0, random_state=SEED)
            ridge.fit(X_train_scaled, y_train)
            ridge_pred = ridge.predict(X_test_scaled)
            results['Ridge'] = {
                'mae': mean_absolute_error(y_test, ridge_pred),
                'rmse': np.sqrt(mean_squared_error(y_test, ridge_pred)),
                'r2': r2_score(y_test, ridge_pred)
            }
            print(f"MAE: {results['Ridge']['mae']:.4f}, RMSE: {results['Ridge']['rmse']:.4f}, R2: {results['Ridge']['r2']:.4f}")
            
            # Lasso
            print("\n[3] Lasso Regression")
            lasso = Lasso(alpha=0.1, random_state=SEED)
            lasso.fit(X_train_scaled, y_train)
            lasso_pred = lasso.predict(X_test_scaled)
            results['Lasso'] = {
                'mae': mean_absolute_error(y_test, lasso_pred),
                'rmse': np.sqrt(mean_squared_error(y_test, lasso_pred)),
                'r2': r2_score(y_test, lasso_pred)
            }
            print(f"MAE: {results['Lasso']['mae']:.4f}, RMSE: {results['Lasso']['rmse']:.4f}, R2: {results['Lasso']['r2']:.4f}")
    
    # ------------------------------------------------------------------------
    # 3. Random Forest
    # ------------------------------------------------------------------------
    if USE_RF:
        print("\n" + "="*60)
        print("[Random Forest]")
        print("="*60)
        
        if PROBLEM_TYPE == 'classification':
            rf = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=SEED, n_jobs=-1)
            rf.fit(X_train_balanced, y_train_balanced)
            rf_pred = rf.predict(X_test_scaled)
            results['RandomForest'] = {
                'f1': f1_score(y_test, rf_pred, average='weighted'),
                'accuracy': accuracy_score(y_test, rf_pred)
            }
            print(f"F1: {results['RandomForest']['f1']:.4f}, Acc: {results['RandomForest']['accuracy']:.4f}")
        else:
            rf = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=SEED, n_jobs=-1)
            rf.fit(X_train_scaled, y_train)
            rf_pred = rf.predict(X_test_scaled)
            results['RandomForest'] = {
                'mae': mean_absolute_error(y_test, rf_pred),
                'rmse': np.sqrt(mean_squared_error(y_test, rf_pred)),
                'r2': r2_score(y_test, rf_pred)
            }
            print(f"MAE: {results['RandomForest']['mae']:.4f}, RMSE: {results['RandomForest']['rmse']:.4f}, R2: {results['RandomForest']['r2']:.4f}")
    
    # ------------------------------------------------------------------------
    # 4. XGBoost
    # ------------------------------------------------------------------------
    if USE_XGB:
        print("\n" + "="*60)
        print("[XGBoost]")
        print("="*60)
        
        if PROBLEM_TYPE == 'classification':
            xgb_model = xgb.XGBClassifier(n_estimators=100, max_depth=6, learning_rate=0.1,
                                         random_state=SEED, eval_metric='logloss', n_jobs=-1)
            xgb_model.fit(X_train_balanced, y_train_balanced)
            xgb_pred = xgb_model.predict(X_test_scaled)
            results['XGBoost'] = {
                'f1': f1_score(y_test, xgb_pred, average='weighted'),
                'accuracy': accuracy_score(y_test, xgb_pred)
            }
            print(f"F1: {results['XGBoost']['f1']:.4f}, Acc: {results['XGBoost']['accuracy']:.4f}")
        else:
            xgb_model = xgb.XGBRegressor(n_estimators=100, max_depth=6, learning_rate=0.1,
                                        random_state=SEED, n_jobs=-1)
            xgb_model.fit(X_train_scaled, y_train)
            xgb_pred = xgb_model.predict(X_test_scaled)
            results['XGBoost'] = {
                'mae': mean_absolute_error(y_test, xgb_pred),
                'rmse': np.sqrt(mean_squared_error(y_test, xgb_pred)),
                'r2': r2_score(y_test, xgb_pred)
            }
            print(f"MAE: {results['XGBoost']['mae']:.4f}, RMSE: {results['XGBoost']['rmse']:.4f}, R2: {results['XGBoost']['r2']:.4f}")
    
    # ------------------------------------------------------------------------
    # 5. DNN (PyTorch)
    # ------------------------------------------------------------------------
    if USE_DNN:
        print("\n" + "="*60)
        print("[DNN - PyTorch]")
        print("="*60)
        
        # Validation split
        X_train_dnn, X_val_dnn, y_train_dnn, y_val_dnn = train_test_split(
            X_train_balanced, y_train_balanced, test_size=0.2, random_state=SEED,
            stratify=y_train_balanced if PROBLEM_TYPE == 'classification' else None
        )
        
        # Tensor 변환
        X_train_t = torch.FloatTensor(X_train_dnn).to(device)
        y_train_t = torch.LongTensor(y_train_dnn).to(device) if PROBLEM_TYPE == 'classification' else torch.FloatTensor(y_train_dnn).to(device)
        X_val_t = torch.FloatTensor(X_val_dnn).to(device)
        y_val_t = torch.LongTensor(y_val_dnn).to(device) if PROBLEM_TYPE == 'classification' else torch.FloatTensor(y_val_dnn).to(device)
        X_test_t = torch.FloatTensor(X_test_scaled).to(device)
        
        train_dataset = TensorDataset(X_train_t, y_train_t)
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        
        # DNN 모델
        class DNN(nn.Module):
            def __init__(self, input_dim, output_dim):
                super(DNN, self).__init__()
                self.layers = nn.Sequential(
                    nn.Linear(input_dim, 128), nn.ReLU(), nn.BatchNorm1d(128), nn.Dropout(0.3),
                    nn.Linear(128, 64), nn.ReLU(), nn.BatchNorm1d(64), nn.Dropout(0.3),
                    nn.Linear(64, 32), nn.ReLU(), nn.Dropout(0.2),
                    nn.Linear(32, output_dim)
                )
            def forward(self, x):
                return self.layers(x)
        
        input_dim = X_train.shape[1]
        output_dim = n_classes if PROBLEM_TYPE == 'classification' and n_classes > 2 else 1
        dnn_model = DNN(input_dim, output_dim).to(device)
        
        # 손실 함수 및 옵티마이저
        if PROBLEM_TYPE == 'classification':
            criterion = nn.CrossEntropyLoss() if n_classes > 2 else nn.BCEWithLogitsLoss()
        else:
            criterion = nn.MSELoss()
        optimizer = optim.Adam(dnn_model.parameters(), lr=0.001)
        
        # 학습
        best_val_loss = float('inf')
        patience = 15
        patience_counter = 0
        
        for epoch in range(100):
            dnn_model.train()
            for X_batch, y_batch in train_loader:
                optimizer.zero_grad()
                outputs = dnn_model(X_batch).squeeze()
                loss = criterion(outputs, y_batch)
                loss.backward()
                optimizer.step()
            
            dnn_model.eval()
            with torch.no_grad():
                val_outputs = dnn_model(X_val_t).squeeze()
                val_loss = criterion(val_outputs, y_val_t).item()
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                torch.save(dnn_model.state_dict(), 'best_dnn.pth')
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    break
        
        dnn_model.load_state_dict(torch.load('best_dnn.pth'))
        
        # 평가
        dnn_model.eval()
        with torch.no_grad():
            test_outputs = dnn_model(X_test_t).squeeze()
            
            if PROBLEM_TYPE == 'classification':
                if n_classes > 2:
                    dnn_pred = torch.argmax(test_outputs, dim=1).cpu().numpy()
                else:
                    dnn_pred = (torch.sigmoid(test_outputs) > 0.5).cpu().numpy().astype(int)
                results['DNN'] = {
                    'f1': f1_score(y_test, dnn_pred, average='weighted'),
                    'accuracy': accuracy_score(y_test, dnn_pred)
                }
                print(f"F1: {results['DNN']['f1']:.4f}, Acc: {results['DNN']['accuracy']:.4f}")
            else:
                dnn_pred = test_outputs.cpu().numpy()
                results['DNN'] = {
                    'mae': mean_absolute_error(y_test, dnn_pred),
                    'rmse': np.sqrt(mean_squared_error(y_test, dnn_pred)),
                    'r2': r2_score(y_test, dnn_pred)
                }
                print(f"MAE: {results['DNN']['mae']:.4f}, RMSE: {results['DNN']['rmse']:.4f}, R2: {results['DNN']['r2']:.4f}")
    
    # ------------------------------------------------------------------------
    # 결과 요약
    # ------------------------------------------------------------------------
    print("\n" + "="*60)
    print("[전체 결과 요약]")
    print("="*60)
    results_df = pd.DataFrame(results).T
    print(results_df)
    
    if PROBLEM_TYPE == 'classification':
        best_model = results_df['f1'].idxmax()
        print(f"\n🏆 최고 성능: {best_model} (F1={results_df.loc[best_model, 'f1']:.4f})")
    else:
        best_model = results_df['mae'].idxmin()
        print(f"\n🏆 최고 성능: {best_model} (MAE={results_df.loc[best_model, 'mae']:.4f})")


# ============================================================================
# [B] 시계열 문제
# ============================================================================

if PROBLEM_TYPE == 'timeseries':
    print("="*60)
    print("[TIMESERIES] 시계열 예측")
    print("="*60)
    
    # ------------------------------------------------------------------------
    # 1. 데이터 로드
    # ------------------------------------------------------------------------
    df = pd.read_csv(DATA_FILE)
    df['date'] = pd.to_datetime(df['date'])  # ⭐ 날짜 컬럼명 수정 필요
    df = df.set_index('date').sort_index()
    series = df[TARGET_COL].values
    
    print(f"시계열 길이: {len(series)}")
    
    # 데이터 분할
    train_size = int(len(series) * 0.8)
    train_series = series[:train_size]
    test_series = series[train_size:]
    
    ts_results = {}
    
    # ------------------------------------------------------------------------
    # 2. ARIMA
    # ------------------------------------------------------------------------
    if USE_ARIMA:
        print("\n" + "="*60)
        print("[ARIMA]")
        print("="*60)
        
        # 정상성 검정
        def check_stationary(s):
            result = adfuller(s)
            return result[1] <= 0.05
        
        d = 0
        diff_series = pd.Series(train_series)
        if not check_stationary(train_series):
            diff_series = pd.Series(train_series).diff().dropna()
            d = 1
            if not check_stationary(diff_series):
                diff_series = diff_series.diff().dropna()
                d = 2
        
        print(f"차분 차수 d = {d}")
        
        # 자동 파라미터 탐색
        best_aic = np.inf
        best_order = None
        
        for p in range(3):
            for q in range(3):
                if p == 0 and d == 0 and q == 0:
                    continue
                try:
                    model = ARIMA(train_series, order=(p, d, q))
                    fitted = model.fit()
                    if fitted.aic < best_aic:
                        best_aic = fitted.aic
                        best_order = (p, d, q)
                except:
                    continue
        
        print(f"최적 파라미터: ARIMA{best_order}, AIC={best_aic:.2f}")
        
        # 학습 및 예측
        arima_model = ARIMA(train_series, order=best_order).fit()
        arima_forecast = arima_model.forecast(steps=len(test_series))
        
        ts_results['ARIMA'] = {
            'mae': mean_absolute_error(test_series, arima_forecast),
            'rmse': np.sqrt(mean_squared_error(test_series, arima_forecast))
        }
        print(f"MAE: {ts_results['ARIMA']['mae']:.4f}, RMSE: {ts_results['ARIMA']['rmse']:.4f}")
    
    # ------------------------------------------------------------------------
    # 3. CNN 1D (PyTorch)
    # ------------------------------------------------------------------------
    if USE_CNN:
        print("\n" + "="*60)
        print("[CNN 1D - PyTorch]")
        print("="*60)
        
        # 스케일링
        scaler_ts = MinMaxScaler()
        scaled_series = scaler_ts.fit_transform(series.reshape(-1, 1))
        
        # 시퀀스 생성
        N_STEPS = 10
        X_seq, y_seq = [], []
        for i in range(len(scaled_series) - N_STEPS):
            X_seq.append(scaled_series[i:i + N_STEPS])
            y_seq.append(scaled_series[i + N_STEPS])
        X_seq, y_seq = np.array(X_seq), np.array(y_seq)
        
        # 데이터 분할
        train_size_seq = int(len(X_seq) * 0.8)
        X_train_seq, X_test_seq = X_seq[:train_size_seq], X_seq[train_size_seq:]
        y_train_seq, y_test_seq = y_seq[:train_size_seq], y_seq[train_size_seq:]
        
        # Tensor 변환
        X_train_cnn = torch.FloatTensor(X_train_seq).to(device)
        y_train_cnn = torch.FloatTensor(y_train_seq).to(device)
        X_test_cnn = torch.FloatTensor(X_test_seq).to(device)
        
        train_dataset_cnn = TensorDataset(X_train_cnn, y_train_cnn)
        train_loader_cnn = DataLoader(train_dataset_cnn, batch_size=32, shuffle=False)
        
        # CNN 모델
        class CNN1D(nn.Module):
            def __init__(self):
                super(CNN1D, self).__init__()
                self.conv1 = nn.Conv1d(1, 64, kernel_size=3, padding=1)
                self.conv2 = nn.Conv1d(64, 128, kernel_size=3, padding=1)
                self.conv3 = nn.Conv1d(128, 64, kernel_size=3, padding=1)
                self.pool = nn.MaxPool1d(2)
                self.dropout = nn.Dropout(0.3)
                self.fc1 = nn.Linear(64 * (N_STEPS // 8), 128)
                self.fc2 = nn.Linear(128, 1)
                self.relu = nn.ReLU()
            
            def forward(self, x):
                x = x.permute(0, 2, 1)  # (batch, features, seq_len)
                x = self.relu(self.conv1(x))
                x = self.pool(x)
                x = self.dropout(x)
                x = self.relu(self.conv2(x))
                x = self.pool(x)
                x = self.dropout(x)
                x = self.relu(self.conv3(x))
                x = self.pool(x)
                x = x.flatten(1)
                x = self.relu(self.fc1(x))
                return self.fc2(x)
        
        cnn_model = CNN1D().to(device)
        criterion_cnn = nn.MSELoss()
        optimizer_cnn = optim.Adam(cnn_model.parameters(), lr=0.001)
        
        # 학습
        best_loss_cnn = float('inf')
        patience_cnn = 15
        patience_counter_cnn = 0
        
        for epoch in range(100):
            cnn_model.train()
            for X_batch, y_batch in train_loader_cnn:
                optimizer_cnn.zero_grad()
                outputs = cnn_model(X_batch).squeeze()
                loss = criterion_cnn(outputs, y_batch.squeeze())
                loss.backward()
                optimizer_cnn.step()
            
            if loss.item() < best_loss_cnn:
                best_loss_cnn = loss.item()
                patience_counter_cnn = 0
                torch.save(cnn_model.state_dict(), 'best_cnn.pth')
            else:
                patience_counter_cnn += 1
                if patience_counter_cnn >= patience_cnn:
                    break
        
        cnn_model.load_state_dict(torch.load('best_cnn.pth'))
        
        # 평가
        cnn_model.eval()
        with torch.no_grad():
            cnn_pred = cnn_model(X_test_cnn).cpu().numpy()
        
        y_test_inv = scaler_ts.inverse_transform(y_test_seq)
        cnn_pred_inv = scaler_ts.inverse_transform(cnn_pred)
        
        ts_results['CNN'] = {
            'mae': mean_absolute_error(y_test_inv, cnn_pred_inv),
            'rmse': np.sqrt(mean_squared_error(y_test_inv, cnn_pred_inv))
        }
        print(f"MAE: {ts_results['CNN']['mae']:.4f}, RMSE: {ts_results['CNN']['rmse']:.4f}")
    
    # ------------------------------------------------------------------------
    # 4. LSTM (PyTorch)
    # ------------------------------------------------------------------------
    if USE_LSTM:
        print("\n" + "="*60)
        print("[LSTM - PyTorch]")
        print("="*60)
        
        # 스케일링
        scaler_ts = MinMaxScaler()
        scaled_series = scaler_ts.fit_transform(series.reshape(-1, 1))
        
        # 시퀀스 생성
        N_STEPS = 10
        X_seq, y_seq = [], []
        for i in range(len(scaled_series) - N_STEPS):
            X_seq.append(scaled_series[i:i + N_STEPS])
            y_seq.append(scaled_series[i + N_STEPS])
        X_seq, y_seq = np.array(X_seq), np.array(y_seq)
        
        # 데이터 분할
        train_size_seq = int(len(X_seq) * 0.8)
        X_train_seq, X_test_seq = X_seq[:train_size_seq], X_seq[train_size_seq:]
        y_train_seq, y_test_seq = y_seq[:train_size_seq], y_seq[train_size_seq:]
        
        # Tensor 변환
        X_train_lstm = torch.FloatTensor(X_train_seq).to(device)
        y_train_lstm = torch.FloatTensor(y_train_seq).to(device)
        X_test_lstm = torch.FloatTensor(X_test_seq).to(device)
        
        train_dataset_lstm = TensorDataset(X_train_lstm, y_train_lstm)
        train_loader_lstm = DataLoader(train_dataset_lstm, batch_size=32, shuffle=False)
        
        # LSTM 모델
        class LSTM(nn.Module):
            def __init__(self, input_dim=1, hidden_dim=64, num_layers=2):
                super(LSTM, self).__init__()
                self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, 
                                   batch_first=True, dropout=0.2)
                self.fc = nn.Linear(hidden_dim, 1)
            
            def forward(self, x):
                lstm_out, _ = self.lstm(x)
                return self.fc(lstm_out[:, -1, :])
        
        lstm_model = LSTM().to(device)
        criterion_lstm = nn.MSELoss()
        optimizer_lstm = optim.Adam(lstm_model.parameters(), lr=0.001)
        
        # 학습
        best_loss_lstm = float('inf')
        patience_lstm = 15
        patience_counter_lstm = 0
        
        for epoch in range(100):
            lstm_model.train()
            for X_batch, y_batch in train_loader_lstm:
                optimizer_lstm.zero_grad()
                outputs = lstm_model(X_batch).squeeze()
                loss = criterion_lstm(outputs, y_batch.squeeze())
                loss.backward()
                optimizer_lstm.step()
            
            if loss.item() < best_loss_lstm:
                best_loss_lstm = loss.item()
                patience_counter_lstm = 0
                torch.save(lstm_model.state_dict(), 'best_lstm.pth')
            else:
                patience_counter_lstm += 1
                if patience_counter_lstm >= patience_lstm:
                    break
        
        lstm_model.load_state_dict(torch.load('best_lstm.pth'))
        
        # 평가
        lstm_model.eval()
        with torch.no_grad():
            lstm_pred = lstm_model(X_test_lstm).cpu().numpy()
        
        y_test_inv = scaler_ts.inverse_transform(y_test_seq)
        lstm_pred_inv = scaler_ts.inverse_transform(lstm_pred)
        
        ts_results['LSTM'] = {
            'mae': mean_absolute_error(y_test_inv, lstm_pred_inv),
            'rmse': np.sqrt(mean_squared_error(y_test_inv, lstm_pred_inv))
        }
        print(f"MAE: {ts_results['LSTM']['mae']:.4f}, RMSE: {ts_results['LSTM']['rmse']:.4f}")
    
    # ------------------------------------------------------------------------
    # 결과 요약
    # ------------------------------------------------------------------------
    print("\n" + "="*60)
    print("[시계열 결과 요약]")
    print("="*60)
    ts_results_df = pd.DataFrame(ts_results).T
    print(ts_results_df)
    
    best_model_ts = ts_results_df['mae'].idxmin()
    print(f"\n🏆 최고 성능: {best_model_ts} (MAE={ts_results_df.loc[best_model_ts, 'mae']:.4f})")


print("\n" + "="*60)
print("완료!")
print("="*60)
