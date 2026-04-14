# -*- coding: utf-8 -*-
"""
train_daily.py — Reentrenamiento diario IBEX 35
================================================
COMO USARLO:
  py train_daily.py          <- ejecutar cada dia hasta el 31/03/2026

LOGICA:
  - Si hoy > 31/03/2026  -> avisa que el periodo termino y sale
  - Descarga datos frescos del IBEX hasta hoy
  - Carga checkpoints existentes y hace fine-tuning (continua desde donde se quedo)
  - Guarda checkpoints actualizados (solo si mejoran el val_loss previo)
  - Registra en training_log.csv: fecha, epocas, val_loss, RMSE, MAE por modelo

OBJETIVO FINAL:
  El 10 de abril ejecutar evaluate_april10.py para ver como predijo el modelo
  el IBEX durante los dias 01/04 - 10/04/2026
"""

import os, csv, warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from datetime import datetime, date

import yfinance as yf
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error

# ─────────────────────────────────────────────────────────────
# CONFIGURACION
# ─────────────────────────────────────────────────────────────
DEVICE      = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
SEQ_LEN     = 60
HORIZON     = 10      # predecir los 10 dias siguientes
EPOCHS_DAY  = 20      # epocas de fine-tuning por sesion diaria
BATCH       = 256
LR          = 0.0005  # lr mas bajo para fine-tuning
PATIENCE    = 10
START_DATE  = '1993-07-12'
END_DATE_TRAIN = date(2026, 3, 31)   # ultimo dia de entrenamiento
LOG_FILE    = 'training_log.csv'

CKPT = {
    "RNN Simple": "ckpt_rnn.pt",
    "LSTM":       "ckpt_lstm.pt",
    "GRU":        "ckpt_gru.pt",
}
COLORS = {"RNN Simple": "tomato", "LSTM": "steelblue", "GRU": "seagreen"}

# ─────────────────────────────────────────────────────────────
# GUARDAR SNAPSHOT DEL MODELO EL ULTIMO DIA DE ENTRENAMIENTO
# (para usar el 10 de abril en evaluate_april10.py)
# ─────────────────────────────────────────────────────────────
SNAPSHOT = {
    "RNN Simple": "snap_rnn_31mar.pt",
    "LSTM":       "snap_lstm_31mar.pt",
    "GRU":        "snap_gru_31mar.pt",
}

# ─────────────────────────────────────────────────────────────
# COMPROBACION DE FECHA
# ─────────────────────────────────────────────────────────────
hoy = date.today()
print("=" * 60)
print(f"  train_daily.py  —  {hoy.strftime('%d/%m/%Y')}")
print(f"  Dispositivo: {DEVICE}" + (f" ({torch.cuda.get_device_name(0)})" if DEVICE.type=='cuda' else ""))
print("=" * 60)

if hoy > END_DATE_TRAIN:
    print(f"\n  Periodo de entrenamiento finalizado el {END_DATE_TRAIN}.")
    print(f"  Ejecuta evaluate_april10.py el 10 de abril para ver los resultados.")
    exit(0)

dias_restantes = (END_DATE_TRAIN - hoy).days
print(f"\n  Dias restantes de entrenamiento: {dias_restantes + 1} (hoy incluido)")
print(f"  Fin planificado: {END_DATE_TRAIN.strftime('%d/%m/%Y')}")
print(f"  Evaluacion final: 10/04/2026\n")

# ─────────────────────────────────────────────────────────────
# 1. DATOS
# ─────────────────────────────────────────────────────────────
end_str = hoy.strftime('%Y-%m-%d')
raw = yf.download('^IBEX', start=START_DATE, end=end_str, progress=True)
raw.dropna(inplace=True)
raw.columns = raw.columns.get_level_values(0)
df = raw[['Open','High','Low','Close','Volume']].copy()

# Indicadores tecnicos
df['MA10']  = df['Close'].rolling(10).mean()
df['MA30']  = df['Close'].rolling(30).mean()
df['MA60']  = df['Close'].rolling(60).mean()
df['Std10'] = df['Close'].rolling(10).std()
delta = df['Close'].diff()
gain  = delta.clip(lower=0).rolling(14).mean()
loss  = (-delta.clip(upper=0)).rolling(14).mean()
df['RSI']   = 100 - (100 / (1 + gain / (loss + 1e-9)))
df['Mom10'] = df['Close'] - df['Close'].shift(10)
df.dropna(inplace=True)

print(f"  Datos: {df.index[0].date()} -> {df.index[-1].date()}  ({len(df)} sesiones)")

feature_cols = list(df.columns)
n_features   = len(feature_cols)
close_idx    = feature_cols.index('Close')

scaler = MinMaxScaler(feature_range=(0, 1))
scaled = scaler.fit_transform(df.values)

close_scaler = MinMaxScaler(feature_range=(0, 1))
close_scaler.fit(df[['Close']].values)

# ─────────────────────────────────────────────────────────────
# 2. SECUENCIAS MULTI-STEP (target = 10 dias siguientes)
# ─────────────────────────────────────────────────────────────
def make_sequences_ms(arr, seq_len, horizon):
    """Crea secuencias X e y donde y son los proximos 'horizon' cierres."""
    X, y = [], []
    for i in range(seq_len, len(arr) - horizon + 1):
        X.append(arr[i - seq_len:i])
        y.append(arr[i:i + horizon, close_idx])
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)

X_all, y_all = make_sequences_ms(scaled, SEQ_LEN, HORIZON)
n      = len(X_all)
va_end = int(n * 0.85)

X_train, y_train = X_all[:va_end], y_all[:va_end]
X_val,   y_val   = X_all[va_end:], y_all[va_end:]

def make_loader(X, y, shuffle=False):
    ds = TensorDataset(torch.tensor(X), torch.tensor(y))
    return DataLoader(ds, batch_size=BATCH, shuffle=shuffle,
                      pin_memory=(DEVICE.type == 'cuda'))

train_loader = make_loader(X_train, y_train, shuffle=True)
val_loader   = make_loader(X_val, y_val)

print(f"  Train: {len(X_train)}  |  Val: {len(X_val)}  |  Horizonte: {HORIZON} dias")

# ─────────────────────────────────────────────────────────────
# 3. ARQUITECTURAS (salida = HORIZON dias)
# ─────────────────────────────────────────────────────────────
class RNNModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.rnn = nn.RNN(n_features, 128, num_layers=3,
                          batch_first=True, dropout=0.3, nonlinearity='tanh')
        self.fc  = nn.Sequential(nn.Linear(128, 64), nn.ReLU(),
                                  nn.Linear(64, HORIZON))
    def forward(self, x):
        out, _ = self.rnn(x)
        return self.fc(out[:, -1, :])

class LSTMModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(n_features, 256, num_layers=3,
                            batch_first=True, dropout=0.3)
        self.fc   = nn.Sequential(
            nn.Linear(256, 128), nn.ReLU(),
            nn.Linear(128, 64),  nn.ReLU(),
            nn.Linear(64, HORIZON)
        )
    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])

class GRUModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.gru = nn.GRU(n_features, 256, num_layers=3,
                          batch_first=True, dropout=0.3)
        self.fc  = nn.Sequential(
            nn.Linear(256, 128), nn.ReLU(),
            nn.Linear(128, 64),  nn.ReLU(),
            nn.Linear(64, HORIZON)
        )
    def forward(self, x):
        out, _ = self.gru(x)
        return self.fc(out[:, -1, :])

BUILDERS = {"RNN Simple": RNNModel, "LSTM": LSTMModel, "GRU": GRUModel}

# ─────────────────────────────────────────────────────────────
# 4. FINE-TUNING DIARIO
# ─────────────────────────────────────────────────────────────
def fine_tune(name, ModelClass, ckpt_path):
    model = ModelClass().to(DEVICE)

    best_val    = float('inf')
    start_epoch = 0

    if os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location=DEVICE, weights_only=True)
        try:
            model.load_state_dict(ckpt['model_state'])
            best_val    = ckpt.get('best_val', float('inf'))
            start_epoch = ckpt.get('epoch', 0)
            print(f"  [OK] {name}: checkpoint epoch={start_epoch}, best_val={best_val:.6f}")
        except Exception:
            print(f"  [WARN] {name}: checkpoint incompatible (cambio de arquitectura). Entrenando desde cero.")
    else:
        print(f"  [NEW] {name}: sin checkpoint. Entrenando desde cero.")

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

    no_improve   = 0
    epocas_reales = 0
    tr_losses, va_losses = [], []

    for epoch in range(EPOCHS_DAY):
        model.train()
        ep_loss = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            ep_loss += loss.item() * len(xb)
        tr_loss = ep_loss / len(X_train)

        model.eval()
        va_loss = 0.0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                va_loss += criterion(model(xb), yb).item() * len(xb)
        va_loss /= len(X_val)

        scheduler.step(va_loss)
        tr_losses.append(tr_loss)
        va_losses.append(va_loss)
        epocas_reales += 1

        if va_loss < best_val:
            best_val = va_loss
            no_improve = 0
            torch.save({'model_state': model.state_dict(),
                        'best_val':    best_val,
                        'epoch':       start_epoch + epoch + 1,
                        'horizon':     HORIZON}, ckpt_path)
        else:
            no_improve += 1

        if no_improve >= PATIENCE:
            print(f"    Early stopping en epoch {epoch+1}")
            break

    # Cargar mejores pesos para evaluacion
    best = torch.load(ckpt_path, map_location=DEVICE, weights_only=True)
    model.load_state_dict(best['model_state'])

    # Metricas en val (desnormalizadas, promediadas sobre los 10 dias)
    model.eval()
    preds = []
    with torch.no_grad():
        for xb, _ in val_loader:
            preds.append(model(xb.to(DEVICE)).cpu().numpy())
    pred_sc = np.vstack(preds)    # (N_val, HORIZON)
    true_sc = y_val               # (N_val, HORIZON)

    # Desnormalizar cada horizonte
    pred_real = close_scaler.inverse_transform(pred_sc.reshape(-1, 1)).reshape(-1, HORIZON)
    true_real = close_scaler.inverse_transform(true_sc.reshape(-1, 1)).reshape(-1, HORIZON)

    rmse = np.sqrt(mean_squared_error(true_real.flatten(), pred_real.flatten()))
    mae  = mean_absolute_error(true_real.flatten(), pred_real.flatten())

    # RMSE por dia de horizonte
    rmse_by_day = [np.sqrt(mean_squared_error(true_real[:, d], pred_real[:, d]))
                   for d in range(HORIZON)]

    print(f"  {name}: RMSE={rmse:,.2f}  MAE={mae:,.2f}  best_val={best_val:.6f}  epocas={epocas_reales}")
    print(f"    RMSE por dia: {[f'{v:.0f}' for v in rmse_by_day]}")

    # Snapshot del ultimo dia de entrenamiento
    if hoy == END_DATE_TRAIN:
        snap_path = SNAPSHOT[name]
        torch.save({'model_state': model.state_dict(),
                    'best_val':    best_val,
                    'epoch':       start_epoch + epocas_reales,
                    'horizon':     HORIZON,
                    'fecha_snap':  str(hoy)}, snap_path)
        print(f"  [SNAPSHOT] Guardado {snap_path} — listo para evaluacion el 10/04")

    return best_val, rmse, mae, epocas_reales

# ─────────────────────────────────────────────────────────────
# 5. EJECUTAR
# ─────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print(f"FINE-TUNING  {hoy.strftime('%d/%m/%Y')}  ({EPOCHS_DAY} epocas max por modelo)")
print("=" * 60)

log_rows = []
for name, ModelClass in BUILDERS.items():
    print(f"\n--- {name} ---")
    best_val, rmse, mae, epocas = fine_tune(name, ModelClass, CKPT[name])
    log_rows.append({
        'fecha':     str(hoy),
        'modelo':    name,
        'epocas':    epocas,
        'best_val':  round(best_val, 8),
        'rmse':      round(rmse, 2),
        'mae':       round(mae, 2),
        'horizonte': HORIZON,
    })

# ─────────────────────────────────────────────────────────────
# 6. LOG CSV
# ─────────────────────────────────────────────────────────────
fieldnames = ['fecha','modelo','epocas','best_val','rmse','mae','horizonte']
write_header = not os.path.exists(LOG_FILE)

with open(LOG_FILE, 'a', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    if write_header:
        writer.writeheader()
    writer.writerows(log_rows)

print(f"\n  Log actualizado: {LOG_FILE}")

# Mostrar historico completo del log
log_df = pd.read_csv(LOG_FILE)
print("\n  Historico de sesiones:")
print(log_df.to_string(index=False))

# ─────────────────────────────────────────────────────────────
# 7. RESUMEN
# ─────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("RESUMEN")
print("=" * 60)
print(f"  Fecha hoy         : {hoy.strftime('%d/%m/%Y')}")
print(f"  Ultimo dia train  : {END_DATE_TRAIN.strftime('%d/%m/%Y')}")
print(f"  Dias restantes    : {(END_DATE_TRAIN - hoy).days}")
print(f"  Evaluacion final  : 10/04/2026  (ejecutar evaluate_april10.py)")
print(f"  Horizonte pred    : {HORIZON} dias")
print(f"  GPU               : {torch.cuda.get_device_name(0) if DEVICE.type=='cuda' else 'CPU'}")
print("=" * 60)

if hoy == END_DATE_TRAIN:
    print("\n  ENTRENAMIENTO COMPLETADO. Snapshots guardados.")
    print("  El 10 de abril ejecuta: py evaluate_april10.py")
else:
    print(f"\n  Vuelve a ejecutar manana: py train_daily.py")
