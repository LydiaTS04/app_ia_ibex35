# -*- coding: utf-8 -*-
"""
train_ultra.py -- Modelos ULTRA definitivos IBEX 35
====================================================
Mejoras vs train_final.py:
  - 17 features (MACD, Bollinger, ATR, VolMA)
  - Residual connections entre capas LSTM/GRU
  - Atencion multi-cabeza (4 heads)
  - 100% datos para training (sin split de validacion)
  - Perdida ponderada temporalmente (datos recientes x3 mas peso)
  - 2000 epocas con CosineAnnealingWarmRestarts
  - PATIENCE = 600 (practicamente sin early stop)
  - SEQ_LEN = 120 dias (4 meses de contexto)
  - Batch = 128
  - Snapshots: snap_rnn_ultra.pt / snap_lstm_ultra.pt / snap_gru_ultra.pt
"""

import os, csv, warnings, time, pickle
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
from datetime import date

import yfinance as yf
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error

# ─────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────
DEVICE     = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
SEQ_LEN    = 120        # 4 meses de contexto
HORIZON    = 10
EPOCHS     = 800        # reducido para terminar en ~1.5h
BATCH      = 256        # batch mayor = epochs mas rapidas
LR         = 0.001
PATIENCE   = 200        # early stop moderado
DROPOUT    = 0.03       # minimo: maximo overfitting
START_DATE = '1993-07-12'
END_TRAIN  = '2026-04-01'  # excluye este dia -> coge hasta 31/03

SNAP = {
    "RNN Simple": "snap_rnn_ultra.pt",
    "LSTM":       "snap_lstm_ultra.pt",
    "GRU":        "snap_gru_ultra.pt",
}
LOG_FILE = 'training_log.csv'

print("=" * 70)
print("  train_ultra.py -- Modelos ULTRA IBEX 35")
print(f"  GPU: {torch.cuda.get_device_name(0) if DEVICE.type=='cuda' else 'CPU'}")
print(f"  Epocas: {EPOCHS}  SEQ_LEN: {SEQ_LEN}  BATCH: {BATCH}  LR: {LR}")
print("=" * 70)

# ─────────────────────────────────────────────────────────────
# 1. DATOS + 17 FEATURES
# ─────────────────────────────────────────────────────────────
print("\n[1/4] Descargando datos y calculando features...")
raw = yf.download('^IBEX', start=START_DATE, end=END_TRAIN, progress=True)
raw.dropna(inplace=True)
raw.columns = raw.columns.get_level_values(0)
df = raw[['Open','High','Low','Close','Volume']].copy()

# --- Medias moviles ---
df['MA10']  = df['Close'].rolling(10).mean()
df['MA30']  = df['Close'].rolling(30).mean()
df['MA60']  = df['Close'].rolling(60).mean()
df['Std10'] = df['Close'].rolling(10).std()

# --- RSI ---
delta = df['Close'].diff()
gain  = delta.clip(lower=0).rolling(14).mean()
loss  = (-delta.clip(upper=0)).rolling(14).mean()
df['RSI']   = 100 - (100 / (1 + gain / (loss + 1e-9)))
df['Mom10'] = df['Close'] - df['Close'].shift(10)

# --- MACD ---
ema12 = df['Close'].ewm(span=12, adjust=False).mean()
ema26 = df['Close'].ewm(span=26, adjust=False).mean()
df['MACD']   = ema12 - ema26
df['MACDsig'] = df['MACD'].ewm(span=9, adjust=False).mean()

# --- Bollinger Bands (20 dias, 2 sigma) ---
ma20      = df['Close'].rolling(20).mean()
std20     = df['Close'].rolling(20).std()
df['BollUp']  = ma20 + 2 * std20
df['BollLow'] = ma20 - 2 * std20

# --- ATR (Average True Range) ---
tr1 = df['High'] - df['Low']
tr2 = (df['High'] - df['Close'].shift()).abs()
tr3 = (df['Low']  - df['Close'].shift()).abs()
df['ATR'] = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1).rolling(14).mean()

# --- Volume MA ---
df['VolMA10'] = df['Volume'].rolling(10).mean()

df.dropna(inplace=True)

feature_cols = list(df.columns)
n_features   = len(feature_cols)
close_idx    = feature_cols.index('Close')
dates_all    = df.index

print(f"  Datos: {df.index[0].date()} -> {df.index[-1].date()}  ({len(df)} sesiones)")
print(f"  Features: {n_features}  ->  {feature_cols}")

# Normalizar
scaler = MinMaxScaler(feature_range=(0, 1))
scaled = scaler.fit_transform(df.values)

close_scaler = MinMaxScaler(feature_range=(0, 1))
close_scaler.fit(df[['Close']].values)

# Guardar scalers
with open('scaler_ultra.pkl', 'wb') as f:       pickle.dump(scaler, f)
with open('close_scaler_ultra.pkl', 'wb') as f: pickle.dump(close_scaler, f)
with open('feature_cols_ultra.pkl', 'wb') as f: pickle.dump(feature_cols, f)

# ─────────────────────────────────────────────────────────────
# 2. SECUENCIAS -- 100% para training, pesos temporales
# ─────────────────────────────────────────────────────────────
print("\n[2/4] Creando secuencias con pesos temporales...")

def make_seq(arr, sl, hz):
    X, y = [], []
    for i in range(sl, len(arr) - hz + 1):
        X.append(arr[i - sl:i])
        y.append(arr[i:i + hz, close_idx])
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)

X_all, y_all = make_seq(scaled, SEQ_LEN, HORIZON)
n = len(X_all)

# Pesos temporales: los datos mas recientes pesan exponencialmente mas
# El ultimo dato tiene peso 1.0, el primero ~0.22 (factor 3x los recientes)
raw_weights = np.exp(np.linspace(np.log(0.22), 0.0, n)).astype(np.float32)
# Normalizar para que sumen n (mismo gradiente total)
sample_weights = (raw_weights / raw_weights.mean()).astype(np.float32)

print(f"  Secuencias: {n}  |  Peso min={raw_weights.min():.3f}  max={raw_weights.max():.3f}")
print(f"  100% datos para training  |  Horizonte: {HORIZON} dias")

def make_loader_weighted(X, y, weights, shuffle=True):
    ds      = TensorDataset(torch.tensor(X), torch.tensor(y),
                            torch.tensor(weights))
    loader  = DataLoader(ds, batch_size=BATCH, shuffle=shuffle,
                         pin_memory=(DEVICE.type == 'cuda'), num_workers=0)
    return loader

def make_loader(X, y, shuffle=False):
    ds = TensorDataset(torch.tensor(X), torch.tensor(y))
    return DataLoader(ds, batch_size=BATCH, shuffle=shuffle,
                      pin_memory=(DEVICE.type == 'cuda'), num_workers=0)

train_loader = make_loader_weighted(X_all, y_all, sample_weights, shuffle=True)
eval_loader  = make_loader(X_all[-500:], y_all[-500:])   # ultimo año para monitoreo

# ─────────────────────────────────────────────────────────────
# 3. ARQUITECTURAS ULTRA
# ─────────────────────────────────────────────────────────────

class MultiHeadAttention(nn.Module):
    """Atencion multi-cabeza sobre la secuencia temporal."""
    def __init__(self, hidden, n_heads=4):
        super().__init__()
        assert hidden % n_heads == 0
        self.n_heads = n_heads
        self.dh      = hidden // n_heads
        self.Q = nn.Linear(hidden, hidden, bias=False)
        self.K = nn.Linear(hidden, hidden, bias=False)
        self.V = nn.Linear(hidden, hidden, bias=False)
        self.out = nn.Linear(hidden, hidden)
    def forward(self, x):
        B, T, H = x.shape
        Q = self.Q(x).view(B, T, self.n_heads, self.dh).transpose(1,2)
        K = self.K(x).view(B, T, self.n_heads, self.dh).transpose(1,2)
        V = self.V(x).view(B, T, self.n_heads, self.dh).transpose(1,2)
        scores  = torch.matmul(Q, K.transpose(-2,-1)) / (self.dh**0.5)
        weights = torch.softmax(scores, dim=-1)
        ctx = torch.matmul(weights, V).transpose(1,2).contiguous().view(B, T, H)
        return self.out(ctx).mean(dim=1)   # (B, H)

# ── RNN Ultra: 4 capas, 256u, MH-attention ──────────────────
class RNNUltra(nn.Module):
    def __init__(self):
        super().__init__()
        self.rnn  = nn.RNN(n_features, 256, num_layers=4,
                           batch_first=True, dropout=DROPOUT, nonlinearity='tanh')
        self.attn = MultiHeadAttention(256, n_heads=4)
        self.norm = nn.LayerNorm(256)
        self.fc   = nn.Sequential(
            nn.Linear(256, 128), nn.GELU(),
            nn.Linear(128, 64),  nn.GELU(),
            nn.Linear(64, HORIZON)
        )
    def forward(self, x):
        out, _ = self.rnn(x)
        ctx    = self.attn(out)
        ctx    = self.norm(ctx)
        return self.fc(ctx)

# ── LSTM Ultra: 5 capas, 512u, residual + MH-attention ───────
class LSTMLayer(nn.Module):
    """LSTM con residual connection si dim coincide."""
    def __init__(self, in_dim, hidden):
        super().__init__()
        self.lstm = nn.LSTM(in_dim, hidden, batch_first=True)
        self.proj = nn.Linear(in_dim, hidden) if in_dim != hidden else nn.Identity()
        self.norm = nn.LayerNorm(hidden)
        self.drop = nn.Dropout(DROPOUT)
    def forward(self, x):
        out, _ = self.lstm(x)
        res    = self.proj(x)
        return self.norm(self.drop(out) + res)

class LSTMUltra(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = LSTMLayer(n_features, 512)
        self.layer2 = LSTMLayer(512, 512)
        self.layer3 = LSTMLayer(512, 512)
        self.layer4 = LSTMLayer(512, 512)
        self.layer5 = LSTMLayer(512, 512)
        self.attn   = MultiHeadAttention(512, n_heads=4)
        self.bn     = nn.BatchNorm1d(512)
        self.fc     = nn.Sequential(
            nn.Linear(512, 256), nn.GELU(),
            nn.Linear(256, 128), nn.GELU(),
            nn.Linear(128, 64),  nn.GELU(),
            nn.Linear(64, HORIZON)
        )
    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        ctx = self.attn(x)
        ctx = self.bn(ctx)
        return self.fc(ctx)

# ── GRU Ultra: 5 capas, 512u, residual + MH-attention ────────
class GRULayer(nn.Module):
    def __init__(self, in_dim, hidden):
        super().__init__()
        self.gru  = nn.GRU(in_dim, hidden, batch_first=True)
        self.proj = nn.Linear(in_dim, hidden) if in_dim != hidden else nn.Identity()
        self.norm = nn.LayerNorm(hidden)
        self.drop = nn.Dropout(DROPOUT)
    def forward(self, x):
        out, _ = self.gru(x)
        res    = self.proj(x)
        return self.norm(self.drop(out) + res)

class GRUUltra(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = GRULayer(n_features, 512)
        self.layer2 = GRULayer(512, 512)
        self.layer3 = GRULayer(512, 512)
        self.layer4 = GRULayer(512, 512)
        self.layer5 = GRULayer(512, 512)
        self.attn   = MultiHeadAttention(512, n_heads=4)
        self.bn     = nn.BatchNorm1d(512)
        self.fc     = nn.Sequential(
            nn.Linear(512, 256), nn.GELU(),
            nn.Linear(256, 128), nn.GELU(),
            nn.Linear(128, 64),  nn.GELU(),
            nn.Linear(64, HORIZON)
        )
    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        ctx = self.attn(x)
        ctx = self.bn(ctx)
        return self.fc(ctx)

BUILDERS = {
    "RNN Simple": RNNUltra,
    "LSTM":       LSTMUltra,
    "GRU":        GRUUltra,
}

# ─────────────────────────────────────────────────────────────
# 4. ENTRENAMIENTO
# ─────────────────────────────────────────────────────────────
def weighted_huber_loss(pred, target, weights, delta=0.5):
    """HuberLoss ponderado por muestra."""
    err  = (pred - target).abs()
    loss = torch.where(err < delta, 0.5 * err**2, delta * (err - 0.5 * delta))
    # weights shape (B,), loss shape (B, HORIZON)
    return (loss.mean(dim=1) * weights).mean()

def train_ultra(name, ModelClass, snap_path):
    print(f"\n{'='*70}")
    print(f"  ENTRENANDO: {name}")
    print(f"{'='*70}")

    model   = ModelClass().to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Parametros: {n_params:,}")

    # Reanudar desde snap si existe
    best_loss = float('inf')
    if os.path.exists(snap_path):
        try:
            ckpt = torch.load(snap_path, map_location=DEVICE, weights_only=True)
            model.load_state_dict(ckpt['model_state'])
            best_loss = ckpt.get('best_loss', float('inf'))
            ep_prev   = ckpt.get('epoch', 0)
            print(f"  [RESUME] Cargado snap epoch={ep_prev}  best_loss={best_loss:.6f}")
        except Exception as e:
            print(f"  [NUEVO] No se pudo cargar snap: {e}")
            for pname, param in model.named_parameters():
                if 'weight' in pname and param.dim() >= 2:
                    nn.init.xavier_uniform_(param)
                elif 'bias' in pname:
                    nn.init.zeros_(param)
    else:
        for pname, param in model.named_parameters():
            if 'weight' in pname and param.dim() >= 2:
                nn.init.xavier_uniform_(param)
            elif 'bias' in pname:
                nn.init.zeros_(param)
        print("  [NUEVO] Entrenando desde cero.")

    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=200, T_mult=2, eta_min=1e-6)
    no_imp    = 0
    t0        = time.time()
    log_int   = max(1, EPOCHS // 40)

    for epoch in range(1, EPOCHS + 1):
        model.train()
        ep_loss = 0.0
        for xb, yb, wb in train_loader:
            xb, yb, wb = xb.to(DEVICE), yb.to(DEVICE), wb.to(DEVICE)
            optimizer.zero_grad()
            loss = weighted_huber_loss(model(xb), yb, wb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            ep_loss += loss.item() * len(xb)
        ep_loss /= n
        scheduler.step()

        if ep_loss < best_loss:
            best_loss = ep_loss
            no_imp    = 0
            torch.save({
                'model_state': model.state_dict(),
                'best_loss':   best_loss,
                'epoch':       epoch,
                'horizon':     HORIZON,
                'seq_len':     SEQ_LEN,
                'n_features':  n_features,
                'fecha_snap':  '2026-03-31',
                'arquitectura': name,
            }, snap_path)
        else:
            no_imp += 1

        if epoch % log_int == 0 or epoch == 1:
            lr_now = scheduler.get_last_lr()[0]
            print(f"  Epoch {epoch:4d}/{EPOCHS}  loss={ep_loss:.6f}  "
                  f"best={best_loss:.6f}  lr={lr_now:.2e}  "
                  f"sin_mejora={no_imp}  [{time.time()-t0:.0f}s]")

        if no_imp >= PATIENCE:
            print(f"  [Early stop] epoch {epoch}")
            break

    # Cargar mejor y calcular metricas
    best = torch.load(snap_path, map_location=DEVICE, weights_only=True)
    model.load_state_dict(best['model_state'])
    model.eval()

    preds = []
    with torch.no_grad():
        for xb, _ in eval_loader:
            preds.append(model(xb.to(DEVICE)).cpu().numpy())
    pred_sc = np.vstack(preds)
    true_sc = y_all[-500:]

    pred_r = close_scaler.inverse_transform(pred_sc.reshape(-1,1)).reshape(-1,HORIZON)
    true_r = close_scaler.inverse_transform(true_sc.reshape(-1,1)).reshape(-1,HORIZON)

    rmse = np.sqrt(mean_squared_error(true_r.flatten(), pred_r.flatten()))
    mae  = mean_absolute_error(true_r.flatten(), pred_r.flatten())
    rmse_h = [np.sqrt(mean_squared_error(true_r[:,d], pred_r[:,d])) for d in range(HORIZON)]

    elapsed = time.time() - t0
    print(f"\n  RESULTADO {name}:")
    print(f"    RMSE={rmse:.2f}  MAE={mae:.2f}  (en ultimos 500 dias)")
    print(f"    RMSE por dia: {[f'{v:.0f}' for v in rmse_h]}")
    print(f"    Snapshot: {snap_path}  ({os.path.getsize(snap_path)/1e6:.1f} MB)")
    print(f"    Tiempo: {elapsed:.0f}s  ({elapsed/60:.1f} min)")
    return rmse, mae, best_loss, epoch

# ─────────────────────────────────────────────────────────────
# 5. EJECUTAR
# ─────────────────────────────────────────────────────────────
print("\n[3/4] Entrenando los 3 modelos ULTRA...")
log_rows = []
SKIP_EPOCH = 700   # si snap tiene >= este epoch -> ya terminado, saltar

for mname, Cls in BUILDERS.items():
    snap = SNAP[mname]
    if os.path.exists(snap):
        try:
            info = torch.load(snap, map_location='cpu', weights_only=True)
            ep_done = info.get('epoch', 0)
            if ep_done >= SKIP_EPOCH:
                print(f"\n  [SKIP] {mname}: ya terminado (epoch {ep_done}). Saltando.")
                continue
            else:
                print(f"\n  [RESUME] {mname}: continuando desde epoch {ep_done}.")
        except Exception:
            pass

    rmse, mae, bl, ep = train_ultra(mname, Cls, snap)
    log_rows.append({'fecha':'2026-03-31','modelo':mname+' [ULTRA]',
                     'epocas':ep,'best_val':round(bl,8),
                     'rmse':round(rmse,2),'mae':round(mae,2),'horizonte':HORIZON})

# ─────────────────────────────────────────────────────────────
# 6. LOG
# ─────────────────────────────────────────────────────────────
print("\n[4/4] Actualizando log y regenerando graficas...")
import csv as csvmod
fieldnames = ['fecha','modelo','epocas','best_val','rmse','mae','horizonte']
write_hdr  = not os.path.exists(LOG_FILE)
with open(LOG_FILE,'a',newline='') as f:
    w = csvmod.DictWriter(f, fieldnames=fieldnames)
    if write_hdr: w.writeheader()
    w.writerows(log_rows)

log_df = pd.read_csv(LOG_FILE)
print("\n  Historico completo:")
print(log_df.to_string(index=False))

print("\n" + "="*70)
print("  ENTRENAMIENTO ULTRA COMPLETADO")
print("="*70)
for mn, s in SNAP.items():
    mb = os.path.getsize(s)/1e6 if os.path.exists(s) else 0
    print(f"  {s:<30}  ({mb:.1f} MB)")
print(f"\n  Ahora ejecuta: py plot_ultra.py")
print("="*70)
