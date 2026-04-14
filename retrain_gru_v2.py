# -*- coding: utf-8 -*-
"""
retrain_gru_v2.py — GRU Ultra con prediccion de LOG-RETORNOS
=============================================================
FIX LINEA PLANA: predecir retornos diarios en lugar de precios absolutos.
  - Los retornos tienen media ~0 y std ~0.01 (distribucion estacionaria)
  - El modelo no puede colapsar al promedio de precios
  - Cada paso del horizonte es independiente → mas volatilidad visible

Cambios vs v1:
  - Target = log-retornos (StandardScaler) en vez de precios (MinMaxScaler)
  - SEQ_LEN = 60 (ventana mas corta, mejor memoria reciente)
  - DROPOUT = 0.10 (mas regularizacion → mas variabilidad en forecast)
  - Loss = MSE ponderado + penalizacion de baja volatilidad
  - 3 features extra: LogRet, LogRet5, VolRet
  - Datos: SOLO hasta 31/03/2026 (sin tocar abril)
  - Guarda: snap_gru_ultra.pt (reemplaza)
"""
import os, warnings, time, pickle
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
import yfinance as yf
import torch, torch.nn as nn, torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error

DEVICE     = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
SEQ_LEN    = 60
HORIZON    = 10
EPOCHS     = 1500
BATCH      = 128
LR         = 0.0008
PATIENCE   = 400
DROPOUT    = 0.10
START_DATE = '1993-07-12'
END_TRAIN  = '2026-04-01'   # SOLO hasta 31 marzo — no toca abril
SNAP       = 'snap_gru_ultra.pt'   # reemplaza el modelo activo

print("=" * 65)
print("  retrain_gru_v2 — GRU Ultra (LOG-RETORNOS, sin linea plana)")
print(f"  Device: {DEVICE}  {'GPU: '+torch.cuda.get_device_name(0) if DEVICE.type=='cuda' else ''}")
print(f"  SEQ_LEN={SEQ_LEN}  HORIZON={HORIZON}  DROPOUT={DROPOUT}")
print(f"  Datos hasta: {END_TRAIN}  CONFIRMADO (sin datos de abril)")
print("=" * 65)

# ── DATOS ────────────────────────────────────────────────────
print("\nDescargando datos IBEX 35...")
raw = yf.download('^IBEX', start=START_DATE, end=END_TRAIN, progress=True)
raw.dropna(inplace=True)
raw.columns = raw.columns.get_level_values(0)
df = raw[['Open','High','Low','Close','Volume']].copy()

df['MA10']   = df['Close'].rolling(10).mean()
df['MA30']   = df['Close'].rolling(30).mean()
df['MA60']   = df['Close'].rolling(60).mean()
df['Std10']  = df['Close'].rolling(10).std()
delta = df['Close'].diff()
gain  = delta.clip(lower=0).rolling(14).mean()
loss_ = (-delta.clip(upper=0)).rolling(14).mean()
df['RSI']    = 100 - (100/(1+gain/(loss_+1e-9)))
df['Mom10']  = df['Close'] - df['Close'].shift(10)
ema12 = df['Close'].ewm(span=12,adjust=False).mean()
ema26 = df['Close'].ewm(span=26,adjust=False).mean()
df['MACD']   = ema12 - ema26
df['MACDsig']= df['MACD'].ewm(span=9,adjust=False).mean()
ma20  = df['Close'].rolling(20).mean()
std20 = df['Close'].rolling(20).std()
df['BollUp'] = ma20 + 2*std20
df['BollLow']= ma20 - 2*std20
tr1 = df['High']-df['Low']
tr2 = (df['High']-df['Close'].shift()).abs()
tr3 = (df['Low'] -df['Close'].shift()).abs()
df['ATR']    = pd.concat([tr1,tr2,tr3],axis=1).max(axis=1).rolling(14).mean()
df['VolMA10']= df['Volume'].rolling(10).mean()

# Features de retorno (nuevas vs v1)
df['LogRet'] = np.log(df['Close'] / df['Close'].shift(1))
df['LogRet5']= df['LogRet'].rolling(5).mean()
df['VolRet'] = df['LogRet'].rolling(10).std()

df.dropna(inplace=True)

feature_cols = list(df.columns)
n_features   = len(feature_cols)
close_vals   = df['Close'].values.astype(np.float64)

print(f"  {df.index[0].date()} -> {df.index[-1].date()}")
print(f"  {len(df)} sesiones  |  {n_features} features")
print(f"  Ultimo cierre: {close_vals[-1]:,.2f}  (31 Mar 2026)")

# ── SCALERS ──────────────────────────────────────────────────
scaler_feat = MinMaxScaler(feature_range=(0,1))
scaled = scaler_feat.fit_transform(df.values).astype(np.float32)

# Target: log-retornos escalados a std≈1
log_rets   = np.log(close_vals[1:] / close_vals[:-1])
ret_scaler = StandardScaler()
ret_scaler.fit(log_rets.reshape(-1,1))

print(f"  Retorno diario IBEX: media={log_rets.mean():.5f}  "
      f"std={log_rets.std():.4f} ({log_rets.std()*100:.2f}%/dia)")

with open('scaler_ultra.pkl','wb') as f:       pickle.dump(scaler_feat, f)

# ── SECUENCIAS ───────────────────────────────────────────────
def make_sequences(feat_arr, close_arr, sl, hz):
    X, y = [], []
    for i in range(sl, len(feat_arr) - hz):
        X.append(feat_arr[i-sl:i])
        future = np.log(close_arr[i+1:i+hz+1] / close_arr[i:i+hz])
        y.append(ret_scaler.transform(future.reshape(-1,1)).flatten())
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)

X_all, y_all = make_sequences(scaled, close_vals, SEQ_LEN, HORIZON)
n = len(X_all)
weights = np.linspace(0.2, 1.0, n).astype(np.float32)
weights /= weights.mean()

ds = TensorDataset(torch.tensor(X_all), torch.tensor(y_all), torch.tensor(weights))
loader = DataLoader(ds, batch_size=BATCH, shuffle=True,
                    pin_memory=(DEVICE.type=='cuda'), num_workers=0)
eval_loader = DataLoader(
    TensorDataset(torch.tensor(X_all[-500:]), torch.tensor(y_all[-500:])),
    batch_size=BATCH, shuffle=False)
print(f"  Secuencias: {n} train  |  500 eval")

# ── ARQUITECTURA ─────────────────────────────────────────────
class MultiHeadAttention(nn.Module):
    def __init__(self, h, nh=4):
        super().__init__()
        self.nh=nh; self.dh=h//nh
        self.Q=nn.Linear(h,h,bias=False); self.K=nn.Linear(h,h,bias=False)
        self.V=nn.Linear(h,h,bias=False); self.out=nn.Linear(h,h)
    def forward(self,x):
        B,T,H=x.shape
        Q=self.Q(x).view(B,T,self.nh,self.dh).transpose(1,2)
        K=self.K(x).view(B,T,self.nh,self.dh).transpose(1,2)
        V=self.V(x).view(B,T,self.nh,self.dh).transpose(1,2)
        w=torch.softmax(torch.matmul(Q,K.transpose(-2,-1))/(self.dh**0.5),dim=-1)
        return self.out(torch.matmul(w,V).transpose(1,2).contiguous().view(B,T,H)).mean(dim=1)

class GRULayer(nn.Module):
    def __init__(self, ind, h):
        super().__init__()
        self.gru =nn.GRU(ind, h, batch_first=True)
        self.proj=nn.Linear(ind, h) if ind != h else nn.Identity()
        self.norm=nn.LayerNorm(h)
        self.drop=nn.Dropout(DROPOUT)
    def forward(self, x):
        out, _ = self.gru(x)
        return self.norm(self.drop(out) + self.proj(x))

class GRUUltra(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1=GRULayer(n_features, 512)
        self.layer2=GRULayer(512, 512)
        self.layer3=GRULayer(512, 512)
        self.layer4=GRULayer(512, 512)
        self.layer5=GRULayer(512, 512)
        self.attn  =MultiHeadAttention(512, 4)
        self.bn    =nn.BatchNorm1d(512)
        self.fc    =nn.Sequential(
            nn.Linear(512, 256), nn.GELU(), nn.Dropout(DROPOUT/2),
            nn.Linear(256, 128), nn.GELU(),
            nn.Linear(128, 64),  nn.GELU(),
            nn.Linear(64, HORIZON)
        )
    def forward(self, x):
        x=self.layer1(x); x=self.layer2(x); x=self.layer3(x)
        x=self.layer4(x); x=self.layer5(x)
        return self.fc(self.bn(self.attn(x)))

model = GRUUltra().to(DEVICE)
print(f"\n  Parametros: {sum(p.numel() for p in model.parameters()):,}")

for pname, param in model.named_parameters():
    if 'weight' in pname and param.dim() >= 2: nn.init.xavier_uniform_(param)
    elif 'bias' in pname: nn.init.zeros_(param)

# ── LOSS CON PENALIZACION DE VOLATILIDAD ────────────────────
def loss_fn(pred, target, w):
    mse      = ((pred - target)**2).mean(dim=1)
    wloss    = (mse * w).mean()
    pred_std = pred.std(dim=1).mean()
    tgt_std  = target.std(dim=1).mean()
    vol_pen  = torch.relu(tgt_std - pred_std) ** 2   # penaliza si pred < real
    return wloss + 0.15 * vol_pen

optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-5)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=5e-7)

# ── ENTRENAMIENTO ────────────────────────────────────────────
best_loss = float('inf')
no_imp = 0; t0 = time.time()
log_int = max(1, EPOCHS // 20)

print(f"\n{'='*65}")
print(f"  ENTRENANDO ({EPOCHS} epocas, CosineAnnealing, vol_penalty=0.15)")
print(f"{'='*65}")

for epoch in range(1, EPOCHS+1):
    model.train()
    ep_loss = 0.0
    for xb, yb, wb in loader:
        xb, yb, wb = xb.to(DEVICE), yb.to(DEVICE), wb.to(DEVICE)
        optimizer.zero_grad()
        loss = loss_fn(model(xb), yb, wb)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        ep_loss += loss.item() * len(xb)
    ep_loss /= n
    scheduler.step()

    if ep_loss < best_loss:
        best_loss = ep_loss; no_imp = 0
        torch.save({
            'model_state':  model.state_dict(),
            'best_loss':    best_loss,
            'epoch':        epoch,
            'horizon':      HORIZON,
            'seq_len':      SEQ_LEN,
            'n_features':   n_features,
            'feature_cols': feature_cols,
            'fecha_snap':   '2026-03-31',
            'arquitectura': 'GRU_V2_log_returns',
            'use_returns':  True,
            'ret_scaler':   ret_scaler,
            'last_close':   float(close_vals[-1]),
        }, SNAP)
    else:
        no_imp += 1

    if epoch % log_int == 0 or epoch == 1:
        print(f"  Epoch {epoch:4d}/{EPOCHS}  loss={ep_loss:.6f}  "
              f"best={best_loss:.6f}  lr={scheduler.get_last_lr()[0]:.2e}  "
              f"no_imp={no_imp}  [{time.time()-t0:.0f}s]")

    if no_imp >= PATIENCE:
        print(f"  [Early stop] epoch {epoch}"); break

# ── METRICAS ─────────────────────────────────────────────────
snap = torch.load(SNAP, map_location=DEVICE, weights_only=False)
model.load_state_dict(snap['model_state']); model.eval()
rs = snap['ret_scaler']

preds_r, trues_r = [], []
with torch.no_grad():
    for xb, yb in eval_loader:
        preds_r.append(model(xb.to(DEVICE)).cpu().numpy())
        trues_r.append(yb.numpy())
pred_sc = np.vstack(preds_r); true_sc = np.vstack(trues_r)

base_closes = close_vals[-(500+HORIZON):-HORIZON]
pred_p, true_p = [], []
for i in range(len(pred_sc)):
    b = base_closes[i]
    pr = rs.inverse_transform(pred_sc[i].reshape(-1,1)).flatten()
    tr = rs.inverse_transform(true_sc[i].reshape(-1,1)).flatten()
    pred_p.append([b*np.exp(pr[:k+1].sum()) for k in range(HORIZON)])
    true_p.append([b*np.exp(tr[:k+1].sum()) for k in range(HORIZON)])
pred_p = np.array(pred_p); true_p = np.array(true_p)

rmse = np.sqrt(mean_squared_error(true_p.flatten(), pred_p.flatten()))
mae  = mean_absolute_error(true_p.flatten(), pred_p.flatten())
mape = np.mean(np.abs((true_p-pred_p)/true_p))*100
r2   = 1-np.sum((true_p-pred_p)**2)/np.sum((true_p-true_p.mean())**2)
pred_std = pred_p.std(axis=1).mean()
true_std = true_p.std(axis=1).mean()

# Actualizar snap con metricas
snap_data = torch.load(SNAP, map_location='cpu', weights_only=False)
snap_data.update({'rmse':rmse,'mae':mae,'mape':mape,'r2':r2})
torch.save(snap_data, SNAP)

# Guardar close_scaler compatible con app (devuelve precios, no retornos)
# Para la app: usamos ret_scaler que esta dentro del snap
with open('close_scaler_ultra.pkl','wb') as f: pickle.dump(rs, f)

elapsed = time.time()-t0
print(f"\n{'='*65}")
print(f"  RESULTADO GRU Ultra V2 (log-retornos):")
print(f"    RMSE   = {rmse:.2f} pts")
print(f"    MAE    = {mae:.2f} pts")
print(f"    MAPE   = {mape:.3f}%")
print(f"    R2     = {r2:.4f}")
print(f"    Variabilidad pred: {pred_std:.1f} pts  (real: {true_std:.1f} pts)")
print(f"    Snapshot: {SNAP}  ({os.path.getsize(SNAP)/1e6:.1f} MB)")
print(f"    Datos hasta: 31/03/2026  CONFIRMADO")
print(f"    Tiempo: {elapsed:.0f}s ({elapsed/60:.1f} min)")
print(f"{'='*65}")
print(f"\n  Ejecuta a continuacion:")
print(f"    py plot_ultra.py       (actualizar PNGs)")
print(f"    py make_pptx.py        (actualizar PowerPoint)")
