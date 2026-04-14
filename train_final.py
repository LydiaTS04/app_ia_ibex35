# -*- coding: utf-8 -*-
"""
train_final.py -- Entrenamiento FINAL definitivo IBEX 35
=========================================================
OBJETIVO:
  Entrena los 3 modelos (RNN, LSTM, GRU) con TODOS los datos disponibles
  hasta el 31/03/2026, usando arquitecturas potentes y muchas epocas para
  maximizar la capacidad predictiva en el horizonte 10 dias (hacia 10/04).

  Se busca overfitting controlado: el modelo memoriza bien los patrones
  historicos recientes para hacer la mejor prediccion posible el 10/04.

EJECUTAR:
  py train_final.py

SALIDA:
  snap_rnn_31mar.pt   <- snapshot RNN Simple listo para evaluate_april10.py
  snap_lstm_31mar.pt  <- snapshot LSTM
  snap_gru_31mar.pt   <- snapshot GRU
  training_log.csv    <- metricas actualizadas
"""

import os, csv, warnings, time
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

# =============================================================
# CONFIGURACION — parametros para entrenamiento final definitivo
# =============================================================
DEVICE    = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
SEQ_LEN   = 90          # mas contexto historico (vs 60 anterior)
HORIZON   = 10          # predecir 10 dias hacia adelante
EPOCHS    = 1000        # muchas epocas para aprender bien
BATCH     = 64          # batch pequeno = mas actualizaciones por epoch
LR        = 0.001       # lr inicial mas alto, decae con cosine
PATIENCE  = 300         # early stopping muy relajado (casi sin limitacion)
DROPOUT   = 0.05        # dropout minimo: queremos que el modelo aprenda todo
START_DATE  = '1993-07-12'
END_TRAIN   = '2026-04-01'   # yfinance excluye este dia -> coge hasta 31/03

LOG_FILE  = 'training_log.csv'

SNAPSHOT = {
    "RNN Simple": "snap_rnn_31mar.pt",
    "LSTM":       "snap_lstm_31mar.pt",
    "GRU":        "snap_gru_31mar.pt",
}

# Checkpoints del fine-tuning diario (arrancar desde donde ya entrenamos)
CKPT_DAILY = {
    "RNN Simple": "ckpt_rnn.pt",
    "LSTM":       "ckpt_lstm.pt",
    "GRU":        "ckpt_gru.pt",
}

print("=" * 65)
print("  train_final.py  --  Entrenamiento FINAL IBEX 35")
print(f"  Dispositivo: {DEVICE}" + (f" ({torch.cuda.get_device_name(0)})" if DEVICE.type=='cuda' else ""))
print(f"  Epocas: {EPOCHS}  |  Batch: {BATCH}  |  SEQ_LEN: {SEQ_LEN}  |  LR: {LR}")
print("=" * 65)

# =============================================================
# 1. DATOS — todos hasta el 31/03/2026
# =============================================================
print("\n[1/4] Descargando datos IBEX 35...")
raw = yf.download('^IBEX', start=START_DATE, end=END_TRAIN, progress=True)
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

# Normalizar con TODOS los datos (ajuste final)
scaler = MinMaxScaler(feature_range=(0, 1))
scaled = scaler.fit_transform(df.values)

close_scaler = MinMaxScaler(feature_range=(0, 1))
close_scaler.fit(df[['Close']].values)

# Guardar scalers para evaluate_april10.py
import pickle
with open('scaler_final.pkl', 'wb') as f:
    pickle.dump(scaler, f)
with open('close_scaler_final.pkl', 'wb') as f:
    pickle.dump(close_scaler, f)
with open('feature_cols_final.pkl', 'wb') as f:
    pickle.dump(feature_cols, f)
print("  Scalers guardados: scaler_final.pkl, close_scaler_final.pkl")

# =============================================================
# 2. SECUENCIAS — 100% para entrenamiento (overfitting deliberado)
# =============================================================
print("\n[2/4] Creando secuencias...")

def make_sequences_ms(arr, seq_len, horizon):
    X, y = [], []
    for i in range(seq_len, len(arr) - horizon + 1):
        X.append(arr[i - seq_len:i])
        y.append(arr[i:i + horizon, close_idx])
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)

X_all, y_all = make_sequences_ms(scaled, SEQ_LEN, HORIZON)

# 90% train / 10% val solo para monitoreo (entrenamos con todo igualmente)
n       = len(X_all)
val_cut = int(n * 0.90)

X_train, y_train = X_all,         y_all           # 100% para entrenamiento
X_val,   y_val   = X_all[val_cut:], y_all[val_cut:] # ultimos 10% para monitoreo

print(f"  Total secuencias: {n}  |  Val (monitoreo): {len(X_val)}  |  Horizonte: {HORIZON} dias")

def make_loader(X, y, shuffle=False):
    ds = TensorDataset(torch.tensor(X), torch.tensor(y))
    return DataLoader(ds, batch_size=BATCH, shuffle=shuffle,
                      pin_memory=(DEVICE.type == 'cuda'), num_workers=0)

train_loader = make_loader(X_train, y_train, shuffle=True)
val_loader   = make_loader(X_val, y_val)

# =============================================================
# 3. ARQUITECTURAS POTENCIADAS
# =============================================================
# Caracteristicas vs version anterior:
#   - RNN:  3 capas 128u  ->  4 capas 256u  + atencion
#   - LSTM: 3 capas 256u  ->  4 capas 512u  + atencion + BN
#   - GRU:  3 capas 256u  ->  4 capas 512u  + atencion + BN
#   - Dropout: 0.3 -> 0.05 (queremos que aprenda todo)
#   - FC mas profundo: 3 capas densas

class Attention(nn.Module):
    """Atencion aditiva simple sobre la secuencia temporal."""
    def __init__(self, hidden_size):
        super().__init__()
        self.attn   = nn.Linear(hidden_size, hidden_size)
        self.v      = nn.Linear(hidden_size, 1, bias=False)
    def forward(self, rnn_out):
        # rnn_out: (batch, seq, hidden)
        scores = self.v(torch.tanh(self.attn(rnn_out)))  # (batch, seq, 1)
        weights = torch.softmax(scores, dim=1)            # (batch, seq, 1)
        context = (weights * rnn_out).sum(dim=1)          # (batch, hidden)
        return context

class RNNModelFinal(nn.Module):
    def __init__(self):
        super().__init__()
        self.rnn = nn.RNN(n_features, 256, num_layers=4,
                          batch_first=True, dropout=DROPOUT, nonlinearity='tanh')
        self.attn = Attention(256)
        self.fc   = nn.Sequential(
            nn.Linear(256, 128), nn.GELU(),
            nn.Linear(128, 64),  nn.GELU(),
            nn.Linear(64, HORIZON)
        )
    def forward(self, x):
        out, _ = self.rnn(x)
        ctx = self.attn(out)
        return self.fc(ctx)

class LSTMModelFinal(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(n_features, 512, num_layers=4,
                            batch_first=True, dropout=DROPOUT)
        self.attn = Attention(512)
        self.bn   = nn.BatchNorm1d(512)
        self.fc   = nn.Sequential(
            nn.Linear(512, 256), nn.GELU(),
            nn.Linear(256, 128), nn.GELU(),
            nn.Linear(128, 64),  nn.GELU(),
            nn.Linear(64, HORIZON)
        )
    def forward(self, x):
        out, _ = self.lstm(x)
        ctx = self.attn(out)
        ctx = self.bn(ctx)
        return self.fc(ctx)

class GRUModelFinal(nn.Module):
    def __init__(self):
        super().__init__()
        self.gru  = nn.GRU(n_features, 512, num_layers=4,
                           batch_first=True, dropout=DROPOUT)
        self.attn = Attention(512)
        self.bn   = nn.BatchNorm1d(512)
        self.fc   = nn.Sequential(
            nn.Linear(512, 256), nn.GELU(),
            nn.Linear(256, 128), nn.GELU(),
            nn.Linear(128, 64),  nn.GELU(),
            nn.Linear(64, HORIZON)
        )
    def forward(self, x):
        out, _ = self.gru(x)
        ctx = self.attn(out)
        ctx = self.bn(ctx)
        return self.fc(ctx)

BUILDERS = {
    "RNN Simple": RNNModelFinal,
    "LSTM":       LSTMModelFinal,
    "GRU":        GRUModelFinal,
}

# =============================================================
# 4. ENTRENAMIENTO FINAL
# =============================================================
def train_final(name, ModelClass, snap_path, daily_ckpt):
    print(f"\n{'='*65}")
    print(f"  ENTRENANDO: {name}")
    print(f"{'='*65}")

    model = ModelClass().to(DEVICE)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Parametros: {n_params:,}")

    # Intentar cargar desde snap previo (prioridad) o checkpoint diario
    loaded = False
    for ckpt_src in [snap_path, daily_ckpt]:
        if os.path.exists(ckpt_src):
            try:
                ckpt = torch.load(ckpt_src, map_location=DEVICE, weights_only=True)
                model.load_state_dict(ckpt['model_state'])
                ep_prev = ckpt.get('epoch', '?')
                print(f"  [OK] Reanudando desde: {ckpt_src}  (epoch anterior: {ep_prev})")
                loaded = True
                break
            except Exception as e:
                print(f"  [INFO] No se pudo cargar {ckpt_src} ({e}). Probando siguiente...")

    if not loaded:
        # Inicializacion Xavier para converger mas rapido
        for name_m, param in model.named_parameters():
            if 'weight' in name_m and param.dim() >= 2:
                nn.init.xavier_uniform_(param)
            elif 'bias' in name_m:
                nn.init.zeros_(param)
        print("  [INFO] Pesos inicializados con Xavier.")

    criterion = nn.HuberLoss(delta=0.5)   # mas robusto que MSE a outliers
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=0.0)

    # CosineAnnealing: LR baja suavemente desde LR hasta 1e-6 en EPOCHS epocas
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-6)

    best_train_loss = float('inf')
    best_val_loss   = float('inf')
    no_improve      = 0
    t0 = time.time()
    log_interval = max(1, EPOCHS // 20)  # imprimir ~20 veces

    for epoch in range(1, EPOCHS + 1):
        # -- TRAIN --
        model.train()
        tr_loss = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            tr_loss += loss.item() * len(xb)
        tr_loss /= len(X_train)
        scheduler.step()

        # -- VAL --
        model.eval()
        va_loss = 0.0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                va_loss += criterion(model(xb), yb).item() * len(xb)
        va_loss /= len(X_val)

        # Guardar si mejora el train loss (overfitting deliberado sobre train)
        if tr_loss < best_train_loss:
            best_train_loss = tr_loss
            torch.save({
                'model_state':    model.state_dict(),
                'best_train_loss': best_train_loss,
                'best_val_loss':   va_loss,
                'epoch':          epoch,
                'horizon':        HORIZON,
                'seq_len':        SEQ_LEN,
                'n_features':     n_features,
                'fecha_snap':     '2026-03-31',
                'arquitectura':   name,
            }, snap_path)

        # Early stopping basado en val loss (muy relajado: PATIENCE=300)
        if va_loss < best_val_loss:
            best_val_loss = va_loss
            no_improve    = 0
        else:
            no_improve += 1

        if epoch % log_interval == 0 or epoch == 1:
            elapsed = time.time() - t0
            lr_now  = scheduler.get_last_lr()[0]
            print(f"  Epoch {epoch:4d}/{EPOCHS}  "
                  f"tr={tr_loss:.6f}  va={va_loss:.6f}  "
                  f"lr={lr_now:.2e}  sin_mejora={no_improve}  "
                  f"[{elapsed:.0f}s]")

        if no_improve >= PATIENCE:
            print(f"  [Early stop] epoch {epoch}  (sin mejora val {PATIENCE} epocas)")
            break

    # Cargar el mejor modelo para evaluacion
    best = torch.load(snap_path, map_location=DEVICE, weights_only=True)
    model.load_state_dict(best['model_state'])

    # -- METRICAS FINALES (val) --
    model.eval()
    preds = []
    with torch.no_grad():
        for xb, _ in val_loader:
            preds.append(model(xb.to(DEVICE)).cpu().numpy())
    pred_sc = np.vstack(preds)
    true_sc = y_val

    pred_real = close_scaler.inverse_transform(pred_sc.reshape(-1,1)).reshape(-1, HORIZON)
    true_real = close_scaler.inverse_transform(true_sc.reshape(-1,1)).reshape(-1, HORIZON)

    rmse = np.sqrt(mean_squared_error(true_real.flatten(), pred_real.flatten()))
    mae  = mean_absolute_error(true_real.flatten(), pred_real.flatten())
    rmse_by_day = [np.sqrt(mean_squared_error(true_real[:,d], pred_real[:,d]))
                   for d in range(HORIZON)]

    elapsed_total = time.time() - t0
    print(f"\n  RESULTADO {name}:")
    print(f"    RMSE={rmse:.2f}  MAE={mae:.2f}")
    print(f"    RMSE por dia: {[f'{v:.0f}' for v in rmse_by_day]}")
    print(f"    Snapshot guardado: {snap_path}")
    print(f"    Tiempo: {elapsed_total:.0f}s  ({elapsed_total/60:.1f} min)")

    return rmse, mae, best_train_loss, best_val_loss, epoch

# =============================================================
# 5. EJECUTAR LOS 3 MODELOS
# =============================================================
print("\n[3/4] Iniciando entrenamiento final de los 3 modelos...")
log_rows = []

DONE_THRESHOLD_EPOCH = 900   # snap con epoch >= este valor = modelo terminado

for model_name, ModelClass in BUILDERS.items():
    snap = SNAPSHOT[model_name]
    # Saltar modelos ya terminados (snap existe con muchas epocas)
    if os.path.exists(snap):
        try:
            info = torch.load(snap, map_location='cpu', weights_only=True)
            ep_done = info.get('epoch', 0)
            if ep_done >= DONE_THRESHOLD_EPOCH:
                print(f"\n  [SKIP] {model_name}: ya terminado (epoch {ep_done}). Saltando.")
                continue
            else:
                print(f"\n  [RESUME] {model_name}: snap parcial (epoch {ep_done}). Continuando.")
        except Exception:
            pass

    rmse, mae, best_tr, best_va, ep = train_final(
        model_name, ModelClass,
        snap,
        CKPT_DAILY[model_name]
    )
    log_rows.append({
        'fecha':     '2026-03-31',
        'modelo':    model_name + ' [FINAL]',
        'epocas':    ep,
        'best_val':  round(best_va, 8),
        'rmse':      round(rmse, 2),
        'mae':       round(mae, 2),
        'horizonte': HORIZON,
    })

# =============================================================
# 6. LOG CSV
# =============================================================
print("\n[4/4] Actualizando log...")
fieldnames   = ['fecha','modelo','epocas','best_val','rmse','mae','horizonte']
write_header = not os.path.exists(LOG_FILE)
with open(LOG_FILE, 'a', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    if write_header:
        writer.writeheader()
    writer.writerows(log_rows)

log_df = pd.read_csv(LOG_FILE)
print("\n  Historico completo:")
print(log_df.to_string(index=False))

# =============================================================
# 7. RESUMEN FINAL
# =============================================================
print("\n" + "=" * 65)
print("  ENTRENAMIENTO FINAL COMPLETADO")
print("=" * 65)
print(f"  Snapshots listos para evaluate_april10.py:")
for n2, s in SNAPSHOT.items():
    size_mb = os.path.getsize(s) / 1e6 if os.path.exists(s) else 0
    print(f"    {s:30s}  ({size_mb:.1f} MB)")
print(f"\n  Configuracion usada:")
print(f"    SEQ_LEN  = {SEQ_LEN}  (ventana de entrada)")
print(f"    HORIZON  = {HORIZON}  (dias predichos)")
print(f"    EPOCHS   = {EPOCHS}")
print(f"    BATCH    = {BATCH}")
print(f"    LR       = {LR} (CosineAnnealing -> 1e-6)")
print(f"    DROPOUT  = {DROPOUT}")
print(f"    GPU      = {torch.cuda.get_device_name(0) if DEVICE.type=='cuda' else 'CPU'}")
print(f"\n  Ejecuta el 10/04:  py evaluate_april10.py")
print("=" * 65)
