# -*- coding: utf-8 -*-
"""
retrain_gru.py -- Reentrenamiento GRU Ultra con scheduler correcto
===================================================================
Igual que train_ultra.py pero:
  - Solo entrena el GRU
  - CosineAnnealing SIN warm restarts (como train_final.py que dio RMSE=94)
  - 1000 epocas
  - Guarda en snap_gru_ultra.pt (sobreescribe el malo)
"""
import os, warnings, time, pickle
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
import yfinance as yf
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error

DEVICE     = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
SEQ_LEN    = 120
HORIZON    = 10
EPOCHS     = 2000
BATCH      = 256
LR         = 0.001
PATIENCE   = 500
DROPOUT    = 0.03
START_DATE = '1993-07-12'
END_TRAIN  = '2026-04-01'
SNAP       = 'snap_gru_ultra.pt'

print("=" * 65)
print("  retrain_gru.py -- GRU Ultra (CosineAnnealing suave)")
print(f"  GPU: {torch.cuda.get_device_name(0) if DEVICE.type=='cuda' else 'CPU'}")
print(f"  Epocas: {EPOCHS}  Batch: {BATCH}  Patience: {PATIENCE}")
print("=" * 65)

# ── DATOS ────────────────────────────────────────────────────
print("\nDescargando datos...")
raw = yf.download('^IBEX', start=START_DATE, end=END_TRAIN, progress=True)
raw.dropna(inplace=True)
raw.columns = raw.columns.get_level_values(0)
df = raw[['Open','High','Low','Close','Volume']].copy()
df['MA10']  = df['Close'].rolling(10).mean()
df['MA30']  = df['Close'].rolling(30).mean()
df['MA60']  = df['Close'].rolling(60).mean()
df['Std10'] = df['Close'].rolling(10).std()
delta = df['Close'].diff()
gain  = delta.clip(lower=0).rolling(14).mean()
loss  = (-delta.clip(upper=0)).rolling(14).mean()
df['RSI']     = 100 - (100/(1+gain/(loss+1e-9)))
df['Mom10']   = df['Close'] - df['Close'].shift(10)
ema12 = df['Close'].ewm(span=12,adjust=False).mean()
ema26 = df['Close'].ewm(span=26,adjust=False).mean()
df['MACD']    = ema12 - ema26
df['MACDsig'] = df['MACD'].ewm(span=9,adjust=False).mean()
ma20 = df['Close'].rolling(20).mean(); std20 = df['Close'].rolling(20).std()
df['BollUp']  = ma20 + 2*std20
df['BollLow'] = ma20 - 2*std20
tr1 = df['High']-df['Low']
tr2 = (df['High']-df['Close'].shift()).abs()
tr3 = (df['Low'] -df['Close'].shift()).abs()
df['ATR']     = pd.concat([tr1,tr2,tr3],axis=1).max(axis=1).rolling(14).mean()
df['VolMA10'] = df['Volume'].rolling(10).mean()
df.dropna(inplace=True)

feature_cols = list(df.columns)
n_features   = len(feature_cols)
close_idx    = feature_cols.index('Close')
print(f"  {df.index[0].date()} -> {df.index[-1].date()}  ({len(df)} sesiones)  {n_features} features")

with open('scaler_ultra.pkl','rb') as f:       scaler = pickle.load(f)
with open('close_scaler_ultra.pkl','rb') as f: close_scaler = pickle.load(f)
scaled = scaler.transform(df.values)

# ── SECUENCIAS con pesos temporales ──────────────────────────
def make_seq(arr, sl, hz):
    X, y = [], []
    for i in range(sl, len(arr)-hz+1):
        X.append(arr[i-sl:i]); y.append(arr[i:i+hz, close_idx])
    return np.array(X,dtype=np.float32), np.array(y,dtype=np.float32)

X_all, y_all = make_seq(scaled, SEQ_LEN, HORIZON)
n = len(X_all)
weights = np.exp(np.linspace(np.log(0.22), 0.0, n)).astype(np.float32)
weights /= weights.mean()

ds = TensorDataset(torch.tensor(X_all), torch.tensor(y_all), torch.tensor(weights))
loader = DataLoader(ds, batch_size=BATCH, shuffle=True,
                    pin_memory=(DEVICE.type=='cuda'), num_workers=0)
eval_loader = DataLoader(TensorDataset(torch.tensor(X_all[-500:]), torch.tensor(y_all[-500:])),
                         batch_size=BATCH, shuffle=False)
print(f"  Secuencias: {n}  |  100% training  |  Horizonte: {HORIZON}")

# ── MODELO ───────────────────────────────────────────────────
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
    def __init__(self,ind,h):
        super().__init__()
        self.gru=nn.GRU(ind,h,batch_first=True)
        self.proj=nn.Linear(ind,h) if ind!=h else nn.Identity()
        self.norm=nn.LayerNorm(h); self.drop=nn.Dropout(DROPOUT)
    def forward(self,x):
        out,_=self.gru(x); return self.norm(self.drop(out)+self.proj(x))

class GRUUltra(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1=GRULayer(n_features,512); self.layer2=GRULayer(512,512)
        self.layer3=GRULayer(512,512); self.layer4=GRULayer(512,512)
        self.layer5=GRULayer(512,512)
        self.attn=MultiHeadAttention(512,4); self.bn=nn.BatchNorm1d(512)
        self.fc=nn.Sequential(nn.Linear(512,256),nn.GELU(),nn.Linear(256,128),
                               nn.GELU(),nn.Linear(128,64),nn.GELU(),nn.Linear(64,HORIZON))
    def forward(self,x):
        x=self.layer1(x);x=self.layer2(x);x=self.layer3(x)
        x=self.layer4(x);x=self.layer5(x)
        return self.fc(self.bn(self.attn(x)))

# ── ENTRENAMIENTO ────────────────────────────────────────────
model = GRUUltra().to(DEVICE)
n_params = sum(p.numel() for p in model.parameters())
print(f"\n  GRU Ultra: {n_params:,} parametros")

for pname, param in model.named_parameters():
    if 'weight' in pname and param.dim() >= 2: nn.init.xavier_uniform_(param)
    elif 'bias' in pname: nn.init.zeros_(param)

def weighted_huber(pred, target, w, delta=0.5):
    err  = (pred-target).abs()
    loss = torch.where(err<delta, 0.5*err**2, delta*(err-0.5*delta))
    return (loss.mean(dim=1)*w).mean()

optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-5)
# CosineAnnealing SUAVE: sin restarts, baja de LR a 1e-6 en EPOCHS epocas
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-6)

best_loss = float('inf')
no_imp    = 0
t0        = time.time()
log_int   = max(1, EPOCHS // 20)

print(f"\n{'='*65}")
print(f"  ENTRENANDO GRU ({EPOCHS} epocas, CosineAnnealing suave)")
print(f"{'='*65}")

for epoch in range(1, EPOCHS+1):
    model.train()
    ep_loss = 0.0
    for xb, yb, wb in loader:
        xb, yb, wb = xb.to(DEVICE), yb.to(DEVICE), wb.to(DEVICE)
        optimizer.zero_grad()
        loss = weighted_huber(model(xb), yb, wb)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        ep_loss += loss.item() * len(xb)
    ep_loss /= n
    scheduler.step()

    if ep_loss < best_loss:
        best_loss = ep_loss
        no_imp    = 0
        torch.save({'model_state': model.state_dict(),
                    'best_loss':   best_loss,
                    'epoch':       epoch,
                    'horizon':     HORIZON,
                    'seq_len':     SEQ_LEN,
                    'n_features':  n_features,
                    'fecha_snap':  '2026-03-31',
                    'arquitectura':'GRU'}, SNAP)
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

# ── METRICAS ─────────────────────────────────────────────────
best = torch.load(SNAP, map_location=DEVICE, weights_only=True)
model.load_state_dict(best['model_state']); model.eval()

preds = []
with torch.no_grad():
    for xb,_ in eval_loader:
        preds.append(model(xb.to(DEVICE)).cpu().numpy())
pred_sc = np.vstack(preds)
true_sc = y_all[-500:]
pred_r  = close_scaler.inverse_transform(pred_sc.reshape(-1,1)).reshape(-1,HORIZON)
true_r  = close_scaler.inverse_transform(true_sc.reshape(-1,1)).reshape(-1,HORIZON)
rmse    = np.sqrt(mean_squared_error(true_r.flatten(), pred_r.flatten()))
mae     = mean_absolute_error(true_r.flatten(), pred_r.flatten())
rmse_h  = [np.sqrt(mean_squared_error(true_r[:,d], pred_r[:,d])) for d in range(HORIZON)]
elapsed = time.time()-t0

print(f"\n{'='*65}")
print(f"  RESULTADO GRU Ultra (reentrenado):")
print(f"    RMSE={rmse:.2f}  MAE={mae:.2f}")
print(f"    RMSE por dia: {[f'{v:.0f}' for v in rmse_h]}")
print(f"    Snapshot: {SNAP}  ({os.path.getsize(SNAP)/1e6:.1f} MB)")
print(f"    Tiempo: {elapsed:.0f}s  ({elapsed/60:.1f} min)")
print(f"{'='*65}")
print(f"\n  Ahora ejecuta: py plot_ultra.py")
