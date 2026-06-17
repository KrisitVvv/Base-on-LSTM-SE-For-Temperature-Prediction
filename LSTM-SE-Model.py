import os
import sys
import argparse
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from datetime import datetime, timedelta
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from tqdm import tqdm
import json

warnings.filterwarnings('ignore')


FONT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'msyh.ttc')
if os.path.exists(FONT_PATH):
    FONT_PROP = FontProperties(fname=FONT_PATH)
else:
    FONT_PROP = FontProperties(family='SimHei')

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)

# Device configuration
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Model hyperparameters 
SEQ_LEN = 30          # 输入序列长度（天）
HIDDEN_SIZE = 128     # LSTM隐藏层大小
NUM_LAYERS = 2        # LSTM层数
DROPOUT = 0.2         # Dropout率
LEARNING_RATE = 0.001 # 学习率
EPOCHS = 200          # 训练轮数
BATCH_SIZE = 64       # 批大小
PATIENCE = 20         # 早停耐心值
PREDICTION_HORIZON = 3650  # 预测天数

# Data processing module
def load_data(read_path):
    max_path = os.path.join(read_path, 'MaxTemperature.xlsx')
    min_path = os.path.join(read_path, 'MinTemperature.xlsx')
    if not os.path.exists(max_path):
        raise FileNotFoundError(f"未找到最高温数据文件: {max_path}")
    if not os.path.exists(min_path):
        raise FileNotFoundError(f"未找到最低温数据文件: {min_path}")
    df_max = pd.read_excel(max_path)
    df_min = pd.read_excel(min_path)
    return df_max, df_min


def wide_to_long(df, temp_type):
    records = []
    year_col = df.columns[0]
    date_cols = df.columns[1:]
    for _, row in df.iterrows():
        year = int(row[year_col])
        for col in date_cols:
            try:
                month, day = col.split('-')
                month, day = int(month), int(day)
                # Skip invalid dates
                try:
                    date = pd.Timestamp(year=year, month=month, day=day)
                except ValueError:
                    continue
                val = row[col]
                if pd.notna(val):
                    records.append({
                        'date': date,
                        'temperature': float(val),
                        'type': temp_type
                    })
            except (ValueError, AttributeError):
                continue

    result = pd.DataFrame(records)
    result = result.sort_values('date').reset_index(drop=True)
    return result

def create_features(df):
    df = df.copy()
    df['month'] = df['date'].dt.month
    df['day'] = df['date'].dt.day
    df['dayofyear'] = df['date'].dt.dayofyear
    df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
    df['day_sin'] = np.sin(2 * np.pi * df['dayofyear'] / 365)
    df['day_cos'] = np.cos(2 * np.pi * df['dayofyear'] / 365)
    return df

def prepare_time_series(df, seq_len=SEQ_LEN):
    """准备时间序列数据，构建滑动窗口样本"""
    temps = df['temperature'].values
    month_sin = df['month_sin'].values
    month_cos = df['month_cos'].values
    day_sin = df['day_sin'].values
    day_cos = df['day_cos'].values
    features = np.column_stack([temps, month_sin, month_cos, day_sin, day_cos])

    X, y = [], []
    for i in range(len(features) - seq_len):
        X.append(features[i:i + seq_len])
        y.append(temps[i + seq_len])

    return np.array(X), np.array(y)

# dataset configuration
class TemperatureDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# LSTM-SE
class SqueezeExcitation(nn.Module):
    def __init__(self, channel, reduction=4):
        super(SqueezeExcitation, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        """x: (batch, channel, seq_len) -> (batch, channel, seq_len)"""
        b, c, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1)
        return x * y.expand_as(x)


class LSTMSEModel(nn.Module):
    def __init__(self, input_size=5, hidden_size=HIDDEN_SIZE, num_layers=NUM_LAYERS,
                 dropout=DROPOUT, se_reduction=4):
        super(LSTMSEModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        # LSTM
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=False
        )
        # SE
        self.se = SqueezeExcitation(hidden_size, reduction=se_reduction)
        # FC
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, 1)
        )

        self.attention_weights = None

    def forward(self, x):
        # LSTM
        lstm_out, _ = self.lstm(x)  # (batch, seq_len, hidden_size)
        lstm_out_t = lstm_out.permute(0, 2, 1)
        # SE
        se_out = self.se(lstm_out_t)  # (batch, hidden_size, seq_len)
        with torch.no_grad():
            b, c, _ = se_out.size()
            pooled = self.se.avg_pool(se_out).view(b, c)
            self.attention_weights = self.se.fc(pooled).cpu().numpy()  # (batch, hidden_size)

        se_out_last = se_out[:, :, -1]  # (batch, hidden_size)

        output = self.fc(se_out_last)  # (batch, 1)
        return output.squeeze(-1)

# train_model and eval_model
def train_model(model, train_loader, val_loader, epochs=EPOCHS,
                lr=LEARNING_RATE, patience=PATIENCE, model_name='model'):
    model = model.to(DEVICE)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=10, verbose=False
    )

    history = {
        'train_loss': [], 'val_loss': [],
        'train_mae': [], 'val_mae': [],
        'train_rmse': [], 'val_rmse': [],
        'train_r2': [], 'val_r2': []
    }

    best_val_loss = float('inf')
    best_model_state = None
    patience_counter = 0

    for epoch in range(epochs):
        # train phase
        model.train()
        train_losses = []
        train_preds = []
        train_targets = []

        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(DEVICE), y_batch.to(DEVICE)
            optimizer.zero_grad()
            output = model(X_batch)
            loss = criterion(output, y_batch)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            train_losses.append(loss.item())
            train_preds.extend(output.detach().cpu().numpy())
            train_targets.extend(y_batch.detach().cpu().numpy())

        # eval phase
        model.eval()
        val_losses = []
        val_preds = []
        val_targets = []

        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(DEVICE), y_batch.to(DEVICE)
                output = model(X_batch)
                loss = criterion(output, y_batch)
                val_losses.append(loss.item())
                val_preds.extend(output.cpu().numpy())
                val_targets.extend(y_batch.cpu().numpy())

        # calculate metrics
        train_loss = np.mean(train_losses)
        val_loss = np.mean(val_losses)

        train_mae = mean_absolute_error(train_targets, train_preds)
        val_mae = mean_absolute_error(val_targets, val_preds)

        train_rmse = np.sqrt(mean_squared_error(train_targets, train_preds))
        val_rmse = np.sqrt(mean_squared_error(val_targets, val_preds))

        train_r2 = r2_score(train_targets, train_preds) if len(train_targets) > 1 else 0
        val_r2 = r2_score(val_targets, val_preds) if len(val_targets) > 1 else 0

        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_mae'].append(train_mae)
        history['val_mae'].append(val_mae)
        history['train_rmse'].append(train_rmse)
        history['val_rmse'].append(val_rmse)
        history['train_r2'].append(train_r2)
        history['val_r2'].append(val_r2)

        scheduler.step(val_loss)

        # early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1

        if (epoch + 1) % 20 == 0:
            print(f"  Epoch {epoch+1}/{epochs} - "
                  f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, "
                  f"Train MAE: {train_mae:.4f}, Val MAE: {val_mae:.4f}, "
                  f"Train R2: {train_r2:.4f}, Val R2: {val_r2:.4f}")

        if patience_counter >= patience:
            print(f"  早停于第 {epoch+1} 轮，最佳验证损失: {best_val_loss:.4f}")
            break

    # load best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    model = model.to(DEVICE)

    return model, history


def evaluate_model(model, test_loader):
    model.eval()
    predictions = []
    targets = []

    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch = X_batch.to(DEVICE)
            output = model(X_batch)
            predictions.extend(output.cpu().numpy())
            targets.extend(y_batch.numpy())

    predictions = np.array(predictions)
    targets = np.array(targets)

    mae = mean_absolute_error(targets, predictions)
    rmse = np.sqrt(mean_squared_error(targets, predictions))
    r2 = r2_score(targets, predictions)
    mse = mean_squared_error(targets, predictions)

    return {
        'MAE': mae,
        'RMSE': rmse,
        'R2': r2,
        'MSE': mse,
        'predictions': predictions,
        'targets': targets
    }

# future prediction module
def predict_future(model, df_history, temp_scaler, seq_len=SEQ_LEN,
                   years=10, temp_type='max'):
    model.eval()

    df = df_history.copy()
    temps = df['temperature'].values
    month_sin = df['month_sin'].values
    month_cos = df['month_cos'].values
    day_sin = df['day_sin'].values
    day_cos = df['day_cos'].values

    features = np.column_stack([temps, month_sin, month_cos, day_sin, day_cos])
    features[:, 0] = temp_scaler.transform(temps.reshape(-1, 1)).flatten()

    current_seq = features[-seq_len:]  # (seq_len, 5)

    last_date = df['date'].iloc[-1]
    future_dates = []
    current_date = last_date + timedelta(days=1)

    total_days = years * 365
    for _ in range(total_days):
        future_dates.append(current_date)
        current_date += timedelta(days=1)

    predictions = []
    with torch.no_grad():
        for i in range(total_days):
            x_input = torch.FloatTensor(current_seq).unsqueeze(0).to(DEVICE)  # (1, seq_len, 5)
            pred = model(x_input).cpu().numpy()[0]

            pred_original = temp_scaler.inverse_transform([[pred]])[0, 0]
            predictions.append(pred_original)

            next_date = future_dates[i]
            next_month = next_date.month
            next_dayofyear = next_date.timetuple().tm_yday

            next_features = [
                pred, 
                np.sin(2 * np.pi * next_month / 12),
                np.cos(2 * np.pi * next_month / 12),
                np.sin(2 * np.pi * next_dayofyear / 365),
                np.cos(2 * np.pi * next_dayofyear / 365)
            ]

            current_seq = np.vstack([current_seq[1:], next_features])

    result_df = pd.DataFrame({
        'date': future_dates[:len(predictions)],
        'temperature': predictions,
        'type': temp_type
    })
    return result_df

# display module
def plot_training_history(history, output_dir, temp_type):
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    # Loss
    axes[0, 0].plot(history['train_loss'], label='Train Loss', color='#1f77b4')
    axes[0, 0].plot(history['val_loss'], label='Val Loss', color='#ff7f0e')
    axes[0, 0].set_xlabel('Epoch', fontproperties=FONT_PROP)
    axes[0, 0].set_ylabel('Loss (MSE)', fontproperties=FONT_PROP)
    axes[0, 0].set_title('Loss Curve', fontproperties=FONT_PROP, fontsize=14)
    axes[0, 0].legend(prop=FONT_PROP)
    axes[0, 0].grid(True, alpha=0.3)

    # MAE
    axes[0, 1].plot(history['train_mae'], label='Train MAE', color='#1f77b4')
    axes[0, 1].plot(history['val_mae'], label='Val MAE', color='#ff7f0e')
    axes[0, 1].set_xlabel('Epoch', fontproperties=FONT_PROP)
    axes[0, 1].set_ylabel('MAE', fontproperties=FONT_PROP)
    axes[0, 1].set_title('MAE Curve', fontproperties=FONT_PROP, fontsize=14)
    axes[0, 1].legend(prop=FONT_PROP)
    axes[0, 1].grid(True, alpha=0.3)

    # RMSE
    axes[1, 0].plot(history['train_rmse'], label='Train RMSE', color='#1f77b4')
    axes[1, 0].plot(history['val_rmse'], label='Val RMSE', color='#ff7f0e')
    axes[1, 0].set_xlabel('Epoch', fontproperties=FONT_PROP)
    axes[1, 0].set_ylabel('RMSE', fontproperties=FONT_PROP)
    axes[1, 0].set_title('RMSE Curve', fontproperties=FONT_PROP, fontsize=14)
    axes[1, 0].legend(prop=FONT_PROP)
    axes[1, 0].grid(True, alpha=0.3)

    # R2
    axes[1, 1].plot(history['train_r2'], label='Train R2', color='#1f77b4')
    axes[1, 1].plot(history['val_r2'], label='Val R2', color='#ff7f0e')
    axes[1, 1].set_xlabel('Epoch', fontproperties=FONT_PROP)
    axes[1, 1].set_ylabel('R2 Score', fontproperties=FONT_PROP)
    axes[1, 1].set_title('R2 Score Curve', fontproperties=FONT_PROP, fontsize=14)
    axes[1, 1].legend(prop=FONT_PROP)
    axes[1, 1].grid(True, alpha=0.3)

    type_label = '最高温' if temp_type == 'max' else '最低温'
    fig.suptitle(f'LSTM-SE模型训练评价指标 - {type_label}',
                 fontproperties=FONT_PROP, fontsize=16)
    plt.tight_layout()

    save_path = os.path.join(output_dir, f'training_metrics_{temp_type}.png')
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  训练指标图已保存: {save_path}")


def plot_attention_weights(model, output_dir, temp_type, feature_names=None):
    if model.attention_weights is None:
        print(" 警告: 无注意力权重数据，跳过可视化")
        return

    weights = model.attention_weights  # (batch, hidden_size)
    if weights.ndim > 1:
        weights = weights.mean(axis=0)  # (hidden_size,)

    n_channels = len(weights)
    group_size = max(1, n_channels // 16)
    n_groups = (n_channels + group_size - 1) // group_size
    grouped_weights = []
    for i in range(n_groups):
        start = i * group_size
        end = min(start + group_size, n_channels)
        grouped_weights.append(weights[start:end].mean())
    grouped_weights = np.array(grouped_weights)

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # heatmap
    ax = axes[0]
    weight_matrix = weights.reshape(1, -1)
    im = ax.imshow(weight_matrix, aspect='auto', cmap='YlOrRd', interpolation='nearest')
    ax.set_xlabel('Hidden Channel Index', fontproperties=FONT_PROP)
    ax.set_ylabel('Sample', fontproperties=FONT_PROP)
    type_label = '最高温' if temp_type == 'max' else '最低温'
    ax.set_title(f'SE Attention Weights Heatmap - {type_label}',
                 fontproperties=FONT_PROP, fontsize=13)
    plt.colorbar(im, ax=ax, label='Attention Weight')

    # bar chart
    ax = axes[1]
    colors = plt.cm.YlOrRd(grouped_weights / (grouped_weights.max() + 1e-8))
    bars = ax.bar(range(n_groups), grouped_weights, color=colors, edgecolor='gray', linewidth=0.5)
    ax.set_xlabel('Channel Group', fontproperties=FONT_PROP)
    ax.set_ylabel('Attention Weight', fontproperties=FONT_PROP)
    ax.set_title(f'SE Channel Attention Distribution - {type_label}',
                 fontproperties=FONT_PROP, fontsize=13)
    ax.grid(True, alpha=0.3, axis='y')

    fig.suptitle(f'LSTM-SE注意力机制可视化 - {type_label}',
                 fontproperties=FONT_PROP, fontsize=15)
    plt.tight_layout()

    save_path = os.path.join(output_dir, f'attention_weights_{temp_type}.png')
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f" 注意力权重图已保存: {save_path}")


def plot_attention_features(model, output_dir, temp_type, seq_len=SEQ_LEN):
    model.eval()
    if model.attention_weights is None:
        return

    weights = model.attention_weights.mean(axis=0)  # (hidden_size,)

    fig, ax = plt.subplots(figsize=(14, 5))

    n_show = min(64, len(weights))
    x = np.arange(n_show)
    colors = plt.cm.coolwarm(weights[:n_show] / (weights[:n_show].max() + 1e-8))
    ax.bar(x, weights[:n_show], color=colors, edgecolor='gray', linewidth=0.3)
    ax.set_xlabel('Hidden Channel Index', fontproperties=FONT_PROP)
    ax.set_ylabel('Attention Weight', fontproperties=FONT_PROP)
    type_label = '最高温' if temp_type == 'max' else '最低温'
    ax.set_title(f'SE Attention Weight per Hidden Channel - {type_label}',
                 fontproperties=FONT_PROP, fontsize=13)
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    save_path = os.path.join(output_dir, f'attention_channels_{temp_type}.png')
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  通道注意力图已保存: {save_path}")


def plot_yearly_prediction(pred_max_df, pred_min_df, output_dir, year):
    year_max = pred_max_df[pred_max_df['date'].dt.year == year]
    year_min = pred_min_df[pred_min_df['date'].dt.year == year]

    if len(year_max) == 0 or len(year_min) == 0:
        print(f"  警告: {year}年无预测数据")
        return

    fig, ax = plt.subplots(figsize=(14, 6))

    ax.plot(year_max['date'], year_max['temperature'],
            color='#d62728', linewidth=1.2, label='最高温预测', alpha=0.9)
    ax.plot(year_min['date'], year_min['temperature'],
            color='#1f77b4', linewidth=1.2, label='最低温预测', alpha=0.9)

    ax.fill_between(year_max['date'], year_min['temperature'], year_max['temperature'],
                    alpha=0.15, color='#9467bd', label='温度范围')

    ax.set_xlabel('日期', fontproperties=FONT_PROP, fontsize=12)
    ax.set_ylabel('温度 (°C)', fontproperties=FONT_PROP, fontsize=12)
    ax.set_title(f'{year}年温度预测 - LSTM-SE模型',
                 fontproperties=FONT_PROP, fontsize=14)
    ax.legend(prop=FONT_PROP, fontsize=11)
    ax.grid(True, alpha=0.3)

    import matplotlib.dates as mdates
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    plt.xticks(rotation=45)

    plt.tight_layout()
    save_path = os.path.join(output_dir, f'prediction_{year}.png')
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close()


def plot_overall_max_prediction(pred_max_df, output_dir):
    fig, ax = plt.subplots(figsize=(16, 6))

    years = sorted(pred_max_df['date'].dt.year.unique())
    cmap = plt.cm.tab10
    for i, year in enumerate(years):
        year_data = pred_max_df[pred_max_df['date'].dt.year == year]
        month_day = year_data['date'].dt.strftime('%m-%d')
        day_of_year = year_data['date'].dt.dayofyear
        ax.plot(day_of_year, year_data['temperature'],
                color=cmap(i % 10), linewidth=0.8, label=str(year), alpha=0.8)

    ax.set_xlabel('Day of Year', fontproperties=FONT_PROP, fontsize=12)
    ax.set_ylabel('温度 (°C)', fontproperties=FONT_PROP, fontsize=12)
    ax.set_title('未来十年最高温度预测曲线 - LSTM-SE模型',
                 fontproperties=FONT_PROP, fontsize=14)
    ax.legend(prop=FONT_PROP, fontsize=9, ncol=2, loc='upper right')
    ax.grid(True, alpha=0.3)

    month_starts = [1, 32, 60, 91, 121, 152, 182, 213, 244, 274, 305, 335]
    month_labels = ['1月', '2月', '3月', '4月', '5月', '6月',
                    '7月', '8月', '9月', '10月', '11月', '12月']
    ax.set_xticks(month_starts)
    ax.set_xticklabels(month_labels, fontproperties=FONT_PROP)

    plt.tight_layout()
    save_path = os.path.join(output_dir, 'overall_max_temperature.png')
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  整体最高温预测图已保存: {save_path}")


def plot_overall_min_prediction(pred_min_df, output_dir):
    fig, ax = plt.subplots(figsize=(16, 6))

    years = sorted(pred_min_df['date'].dt.year.unique())
    cmap = plt.cm.tab10
    for i, year in enumerate(years):
        year_data = pred_min_df[pred_min_df['date'].dt.year == year]
        day_of_year = year_data['date'].dt.dayofyear
        ax.plot(day_of_year, year_data['temperature'],
                color=cmap(i % 10), linewidth=0.8, label=str(year), alpha=0.8)

    ax.set_xlabel('Day of Year', fontproperties=FONT_PROP, fontsize=12)
    ax.set_ylabel('温度 (°C)', fontproperties=FONT_PROP, fontsize=12)
    ax.set_title('未来十年最低温度预测曲线 - LSTM-SE模型',
                 fontproperties=FONT_PROP, fontsize=14)
    ax.legend(prop=FONT_PROP, fontsize=9, ncol=2, loc='upper right')
    ax.grid(True, alpha=0.3)

    month_starts = [1, 32, 60, 91, 121, 152, 182, 213, 244, 274, 305, 335]
    month_labels = ['1月', '2月', '3月', '4月', '5月', '6月',
                    '7月', '8月', '9月', '10月', '11月', '12月']
    ax.set_xticks(month_starts)
    ax.set_xticklabels(month_labels, fontproperties=FONT_PROP)

    plt.tight_layout()
    save_path = os.path.join(output_dir, 'overall_min_temperature.png')
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  整体最低温预测图已保存: {save_path}")


def plot_test_comparison(metrics, output_dir, temp_type):
    preds = metrics['predictions']
    targets = metrics['targets']

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ax = axes[0]
    ax.scatter(targets, preds, alpha=0.3, s=10, color='#2ca02c')
    lim_min = min(targets.min(), preds.min())
    lim_max = max(targets.max(), preds.max())
    ax.plot([lim_min, lim_max], [lim_min, lim_max], 'r--', linewidth=1.5, label='Ideal')
    ax.set_xlabel('真实值', fontproperties=FONT_PROP)
    ax.set_ylabel('预测值', fontproperties=FONT_PROP)
    type_label = '最高温' if temp_type == 'max' else '最低温'
    ax.set_title(f'预测 vs 真实 - {type_label}', fontproperties=FONT_PROP)
    ax.legend(prop=FONT_PROP)
    ax.grid(True, alpha=0.3)

    # time series plot
    ax = axes[1]
    n_show = min(500, len(targets))
    ax.plot(targets[:n_show], label='真实值', color='#1f77b4', linewidth=0.8, alpha=0.8)
    ax.plot(preds[:n_show], label='预测值', color='#d62728', linewidth=0.8, alpha=0.8)
    ax.set_xlabel('Time Step', fontproperties=FONT_PROP)
    ax.set_ylabel('温度 (°C)', fontproperties=FONT_PROP)
    ax.set_title(f'时序预测对比 - {type_label}', fontproperties=FONT_PROP)
    ax.legend(prop=FONT_PROP)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    save_path = os.path.join(output_dir, f'test_comparison_{temp_type}.png')
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  测试对比图已保存: {save_path}")


def main():
    parser = argparse.ArgumentParser(description='LSTM-SE Model Temperature Forecast')
    parser.add_argument('-r', '--read-path', type=str, default='./reptile/output/pivot',
                        help='数据文件路径, 默认: ./reptile/output/pivot')
    args = parser.parse_args()

    print("=" * 60)
    print("LSTM-SE 温度预测模型")
    print("=" * 60)
    print(f"设备: {DEVICE}")
    print(f"数据路径: {args.read_path}")

    # make output directory if not exist
    output_dir = os.path.join('.', 'lstm_se_output')
    img_dir = os.path.join(output_dir, 'images')
    os.makedirs(img_dir, exist_ok=True)
    print(f"输出目录: {output_dir}")

    print("\n[1/6] 数据加载与预处理...")
    try:
        df_max_wide, df_min_wide = load_data(args.read_path)
    except FileNotFoundError as e:
        print(f"错误: {e}")
        sys.exit(1)

    df_max = wide_to_long(df_max_wide, 'max')
    df_min = wide_to_long(df_min_wide, 'min')
    print(f"  最高温数据: {len(df_max)} 条记录, 日期范围 {df_max['date'].min()} ~ {df_max['date'].max()}")
    print(f"  最低温数据: {len(df_min)} 条记录, 日期范围 {df_min['date'].min()} ~ {df_min['date'].max()}")

    df_max = create_features(df_max)
    df_min = create_features(df_min)

    print("\n[2/6] 构建数据集...")

    results = {}

    for temp_type, df in [('max', df_max), ('min', df_min)]:
        type_label = '最高温' if temp_type == 'max' else '最低温'
        print(f"\n--- 处理 {type_label} 数据 ---")

        temp_scaler = MinMaxScaler()
        temp_values = df['temperature'].values.reshape(-1, 1)
        temp_scaler.fit(temp_values)

        X, y = prepare_time_series(df, seq_len=SEQ_LEN)
        print(f"  样本数: {len(X)}, 输入形状: {X.shape}")

        X[:, :, 0] = temp_scaler.transform(X[:, :, 0].reshape(-1, 1)).reshape(X[:, :, 0].shape)
        y = temp_scaler.transform(y.reshape(-1, 1)).flatten()

        # split data into train, val, test sets
        n = len(X)
        train_end = int(n * 0.7)
        val_end = int(n * 0.85)

        X_train, y_train = X[:train_end], y[:train_end]
        X_val, y_val = X[train_end:val_end], y[train_end:val_end]
        X_test, y_test = X[val_end:], y[val_end:]

        print(f" 训练集: {len(X_train)}, 验证集: {len(X_val)}, 测试集: {len(X_test)}")

        # create DataLoader loaders
        train_dataset = TemperatureDataset(X_train, y_train)
        val_dataset = TemperatureDataset(X_val, y_val)
        test_dataset = TemperatureDataset(X_test, y_test)

        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=False)
        val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

        print(f"\n[3/6] 训练 {type_label} LSTM-SE模型...")
        model = LSTMSEModel(
            input_size=5,
            hidden_size=HIDDEN_SIZE,
            num_layers=NUM_LAYERS,
            dropout=DROPOUT,
            se_reduction=4
        )
        total_params = sum(p.numel() for p in model.parameters())
        print(f"  模型参数量: {total_params:,}")

        model, history = train_model(
            model, train_loader, val_loader,
            epochs=EPOCHS, lr=LEARNING_RATE, patience=PATIENCE,
            model_name=f'lstm_se_{temp_type}'
        )

        print(f"\n[4/6] 评估 {type_label} 模型...")
        metrics = evaluate_model(model, test_loader)
        print(f"  MAE: {metrics['MAE']:.4f}")
        print(f"  RMSE: {metrics['RMSE']:.4f}")
        print(f"  R2: {metrics['R2']:.4f}")
        print(f"  MSE: {metrics['MSE']:.4f}")

        print(f"\n[5/6] 预测 {type_label} 未来10年温度...")
        pred_df = predict_future(
            model, df, temp_scaler,
            seq_len=SEQ_LEN, years=10, temp_type=temp_type
        )
        print(f"  预测数据: {len(pred_df)} 条, 日期范围 {pred_df['date'].min()} ~ {pred_df['date'].max()}")

        pred_save_path = os.path.join(output_dir, f'prediction_{temp_type}.xlsx')
        pred_df.to_excel(pred_save_path, index=False)
        print(f"  预测结果已保存: {pred_save_path}")

        model_save_path = os.path.join(output_dir, f'lstm_se_{temp_type}.pth')
        torch.save(model.state_dict(), model_save_path)
        print(f"  模型已保存: {model_save_path}")

        results[temp_type] = {
            'model': model,
            'history': history,
            'metrics': metrics,
            'predictions': pred_df,
            'scaler': temp_scaler,
            'df': df
        }

    print("\n[6/6] 生成可视化图表...")

    for temp_type in ['max', 'min']:
        type_label = '最高温' if temp_type == 'max' else '最低温'
        r = results[temp_type]

        plot_training_history(r['history'], img_dir, temp_type)

        model = r['model']
        model.eval()
        sample_X = torch.FloatTensor(r['df']['temperature'].values[-SEQ_LEN:].reshape(1, SEQ_LEN, 1))
        df_temp = r['df'].tail(SEQ_LEN).copy()
        feat = np.column_stack([
            df_temp['temperature'].values,
            df_temp['month_sin'].values,
            df_temp['month_cos'].values,
            df_temp['day_sin'].values,
            df_temp['day_cos'].values
        ])
        feat[:, 0] = r['scaler'].transform(feat[:, 0].reshape(-1, 1)).flatten()
        sample_input = torch.FloatTensor(feat).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            _ = model(sample_input)

        plot_attention_weights(model, img_dir, temp_type)
        plot_attention_features(model, img_dir, temp_type)

        plot_test_comparison(r['metrics'], img_dir, temp_type)

    pred_max = results['max']['predictions']
    pred_min = results['min']['predictions']

    print(" 生成年度预测图...")
    for year in range(2026, 2036):
        plot_yearly_prediction(pred_max, pred_min, img_dir, year)
    print(f" 年度预测图已保存: {img_dir}")

    plot_overall_max_prediction(pred_max, img_dir)
    plot_overall_min_prediction(pred_min, img_dir)

    eval_results = {
        'max': {k: float(v) if isinstance(v, (np.floating, float)) else v
                for k, v in results['max']['metrics'].items() if k not in ['predictions', 'targets']},
        'min': {k: float(v) if isinstance(v, (np.floating, float)) else v
                for k, v in results['min']['metrics'].items() if k not in ['predictions', 'targets']}
    }
    eval_path = os.path.join(output_dir, 'evaluation_metrics.json')
    with open(eval_path, 'w', encoding='utf-8') as f:
        json.dump(eval_results, f, indent=2, ensure_ascii=False)
    print(f"评估指标已保存: {eval_path}")

    print("\n" + "=" * 60)
    print("LSTM-SE 温度预测完成!")
    print(f"所有输出保存在: {output_dir}")
    print(f"  - 预测结果: prediction_max.xlsx, prediction_min.xlsx")
    print(f"  - 模型文件: lstm_se_max.pth, lstm_se_min.pth")
    print(f"  - 可视化图片: {img_dir}")
    print(f"  - 评估指标: evaluation_metrics.json")
    print("=" * 60)


if __name__ == '__main__':
    main()