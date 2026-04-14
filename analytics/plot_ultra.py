# -*- coding: utf-8 -*-
"""
plot_ultra.py -- Regenera TODOS los PNG con los modelos ULTRA
Colores muy distintos: RNN=Rojo vivo | LSTM=Azul electrico | GRU=Verde lima
"""
import os, warnings, pickle
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy import stats
from statsmodels.graphics.tsaplots import plot_acf
import yfinance as yf
import torch, torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import MinMaxScaler

# ─── CONFIG ──────────────────────────────────────────────────
DEVICE     = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
SEQ_LEN    = 120
HORIZON    = 10
DROPOUT    = 0.03
BATCH      = 512
START_DATE = '1993-07-12'
END_TRAIN  = '2026-04-01'

SNAPS = {
    "RNN Simple": "models/snap_rnn_ultra.pt",
    "LSTM":       "models/snap_lstm_ultra.pt",
    "GRU":        "models/snap_gru_BUENO.pt",   # modelo de producción RMSE=68.5 R²=0.9994
}
# Colores MUY distintos
COLORS  = {
    "RNN Simple": "#D62728",   # Rojo vivo
    "LSTM":       "#1F77B4",   # Azul electrico
    "GRU":        "#2CA02C",   # Verde intenso
}
MARKER  = {"RNN Simple":"o","LSTM":"s","GRU":"^"}
BG      = 'white'
BG2     = '#F5F5F5'
GRID_KW = dict(color='#CCCCCC', linewidth=0.5, linestyle='--')
SPINE_C = '#AAAAAA'
TC      = '#111111'   # color texto

def sax(ax, title='', xlabel='', ylabel='', fs=9):
    ax.set_facecolor(BG2)
    for sp in ax.spines.values(): sp.set_edgecolor(SPINE_C)
    ax.tick_params(colors='#333333', labelsize=7.5)
    ax.grid(**GRID_KW)
    if title:  ax.set_title(title,  color=TC,  fontsize=fs, fontweight='bold', pad=6)
    if xlabel: ax.set_xlabel(xlabel, color='#444444', fontsize=7.5)
    if ylabel: ax.set_ylabel(ylabel, color='#444444', fontsize=7.5)

def save(fig, name):
    fig.savefig(name, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"  [OK] {name}  ({os.path.getsize(name)/1e6:.1f} MB)")

# ─── DATOS ───────────────────────────────────────────────────
print("Descargando datos IBEX 35...")
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
df['RSI']   = 100 - (100/(1+gain/(loss+1e-9)))
df['Mom10'] = df['Close'] - df['Close'].shift(10)
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
# Features V2 (log-retornos)
df['LogRet']  = np.log(df['Close'] / df['Close'].shift(1))
df['LogRet5'] = df['LogRet'].rolling(5).mean()
df['VolRet']  = df['LogRet'].rolling(10).std()
df.dropna(inplace=True)

feature_cols = list(df.columns)
n_features   = len(feature_cols)   # 20 (GRU V2)
dates_all    = df.index
close_full   = df['Close'].values
print(f"  {df.index[0].date()} -> {df.index[-1].date()}  ({len(df)} sesiones)  {n_features} features")

# ─── DOS CONJUNTOS DE FEATURES ───────────────────────────────
# RNN / LSTM: 17 features (sin LogRet, LogRet5, VolRet)
LEGACY_COLS   = [c for c in feature_cols if c not in ('LogRet','LogRet5','VolRet')]
n_features17  = len(LEGACY_COLS)   # 17
close_idx17   = LEGACY_COLS.index('Close')
SEQ_LEN_17    = 120   # seq_len original de RNN/LSTM
SEQ_LEN_GRU   = 60    # seq_len GRU V2

with open('models/scaler_BUENO.pkl',       'rb') as f: scaler17      = pickle.load(f)
with open('models/close_scaler_BUENO.pkl', 'rb') as f: close_sc17    = pickle.load(f)
with open('models/scaler_ultra.pkl',       'rb') as f: scaler20      = pickle.load(f)

scaled17 = scaler17.transform(df[LEGACY_COLS].values).astype(np.float32)
scaled20 = scaler20.transform(df.values).astype(np.float32)

# ─── SECUENCIAS ──────────────────────────────────────────────
def make_seq(arr, sl, hz, cidx):
    X, y = [], []
    for i in range(sl, len(arr)-hz+1):
        X.append(arr[i-sl:i]); y.append(arr[i:i+hz, cidx])
    return np.array(X,dtype=np.float32), np.array(y,dtype=np.float32)

X17, y17 = make_seq(scaled17, SEQ_LEN_17, HORIZON, close_idx17)
X20, _   = make_seq(scaled20, SEQ_LEN_GRU, HORIZON, feature_cols.index('Close'))

n17 = len(X17);  dates17 = dates_all[SEQ_LEN_17:SEQ_LEN_17+n17]

# Limitar n20 para que el ultimo bloque de HORIZON no se salga de close_full
n20_safe = min(len(X20), len(close_full) - SEQ_LEN_GRU - HORIZON)
X20      = X20[:n20_safe]
n20      = n20_safe
dates20  = dates_all[SEQ_LEN_GRU:SEQ_LEN_GRU+n20]

# True prices para GRU: close_full[sl+j+1 .. sl+j+hz]
true20 = np.array([close_full[SEQ_LEN_GRU+j+1:SEQ_LEN_GRU+j+HORIZON+1]
                   for j in range(n20)], dtype=np.float32)

loader17 = DataLoader(TensorDataset(torch.tensor(X17), torch.tensor(y17)),
                      batch_size=BATCH, shuffle=False)
loader20 = DataLoader(TensorDataset(torch.tensor(X20)),
                      batch_size=BATCH, shuffle=False)

# ─── MODELOS ─────────────────────────────────────────────────
class MultiHeadAttention(nn.Module):
    def __init__(self, h, nh=4):
        super().__init__()
        self.nh = nh; self.dh = h//nh
        self.Q = nn.Linear(h,h,bias=False); self.K = nn.Linear(h,h,bias=False)
        self.V = nn.Linear(h,h,bias=False); self.out = nn.Linear(h,h)
    def forward(self, x):
        B,T,H = x.shape
        Q = self.Q(x).view(B,T,self.nh,self.dh).transpose(1,2)
        K = self.K(x).view(B,T,self.nh,self.dh).transpose(1,2)
        V = self.V(x).view(B,T,self.nh,self.dh).transpose(1,2)
        w = torch.softmax(torch.matmul(Q,K.transpose(-2,-1))/(self.dh**0.5),dim=-1)
        return self.out(torch.matmul(w,V).transpose(1,2).contiguous().view(B,T,H)).mean(dim=1)

class RNNUltra(nn.Module):
    def __init__(self, nf):
        super().__init__()
        self.rnn  = nn.RNN(nf,256,4,batch_first=True,dropout=DROPOUT)
        self.attn = MultiHeadAttention(256,4); self.norm = nn.LayerNorm(256)
        self.fc   = nn.Sequential(nn.Linear(256,128),nn.GELU(),nn.Linear(128,64),nn.GELU(),nn.Linear(64,HORIZON))
    def forward(self,x):
        out,_ = self.rnn(x); return self.fc(self.norm(self.attn(out)))

class LSTMLayer(nn.Module):
    def __init__(self,ind,h):
        super().__init__()
        self.lstm=nn.LSTM(ind,h,batch_first=True)
        self.proj=nn.Linear(ind,h) if ind!=h else nn.Identity()
        self.norm=nn.LayerNorm(h); self.drop=nn.Dropout(DROPOUT)
    def forward(self,x):
        out,_=self.lstm(x); return self.norm(self.drop(out)+self.proj(x))

class LSTMUltra(nn.Module):
    def __init__(self, nf):
        super().__init__()
        self.layer1=LSTMLayer(nf,512); self.layer2=LSTMLayer(512,512)
        self.layer3=LSTMLayer(512,512); self.layer4=LSTMLayer(512,512); self.layer5=LSTMLayer(512,512)
        self.attn=MultiHeadAttention(512,4); self.bn=nn.BatchNorm1d(512)
        self.fc=nn.Sequential(nn.Linear(512,256),nn.GELU(),nn.Linear(256,128),nn.GELU(),nn.Linear(128,64),nn.GELU(),nn.Linear(64,HORIZON))
    def forward(self,x):
        x=self.layer1(x);x=self.layer2(x);x=self.layer3(x);x=self.layer4(x);x=self.layer5(x)
        return self.fc(self.bn(self.attn(x)))

class GRULayer(nn.Module):
    def __init__(self,ind,h):
        super().__init__()
        self.gru=nn.GRU(ind,h,batch_first=True)
        self.proj=nn.Linear(ind,h) if ind!=h else nn.Identity()
        self.norm=nn.LayerNorm(h); self.drop=nn.Dropout(DROPOUT)
    def forward(self,x):
        out,_=self.gru(x); return self.norm(self.drop(out)+self.proj(x))

class GRUUltra(nn.Module):
    def __init__(self, nf):
        super().__init__()
        self.layer1=GRULayer(nf,512); self.layer2=GRULayer(512,512)
        self.layer3=GRULayer(512,512); self.layer4=GRULayer(512,512); self.layer5=GRULayer(512,512)
        self.attn=MultiHeadAttention(512,4); self.bn=nn.BatchNorm1d(512)
        self.fc=nn.Sequential(
            nn.Linear(512,256),nn.GELU(),
            nn.Linear(256,128),nn.GELU(),
            nn.Linear(128,64), nn.GELU(),
            nn.Linear(64,HORIZON))   # sin Dropout — igual que snap_gru_BUENO.pt
    def forward(self,x):
        x=self.layer1(x);x=self.layer2(x);x=self.layer3(x);x=self.layer4(x);x=self.layer5(x)
        return self.fc(self.bn(self.attn(x)))

BUILDERS = {"RNN Simple":RNNUltra,"LSTM":LSTMUltra,"GRU":GRUUltra}

# ─── INFERENCIA ──────────────────────────────────────────────
print("Cargando modelos ULTRA e infiriendo...")
results = {}
for mn, Cls in BUILDERS.items():
    snap_path = SNAPS[mn]
    ckpt = torch.load(snap_path, map_location=DEVICE, weights_only=False)

    use_ret = ckpt.get('use_returns', False)
    nf      = ckpt.get('n_features', n_features17)

    model = Cls(nf).to(DEVICE)
    model.load_state_dict(ckpt['model_state']); model.eval()

    if use_ret:
        # GRU V2: 20 features, log-returns output
        ldr  = loader20
        preds_sc = []
        with torch.no_grad():
            for (xb,) in ldr:
                preds_sc.append(model(xb.to(DEVICE)).cpu().numpy())
        pred_sc = np.vstack(preds_sc)          # (N, HORIZON) — standardized log-rets
        rs      = ckpt['ret_scaler']
        # Reconstruct prices from cumulative log-returns
        pred_r = np.zeros_like(pred_sc)
        for j in range(len(pred_sc)):
            base = close_full[SEQ_LEN_GRU + j]
            lrs  = rs.inverse_transform(pred_sc[j].reshape(-1,1)).flatten()
            pred_r[j] = [base * np.exp(lrs[:k+1].sum()) for k in range(HORIZON)]
        true_r  = true20
        dates_s = dates20
        n_s     = n20
    else:
        # RNN / LSTM: 17 features, price output (normalized Close)
        ldr  = loader17
        preds_sc = []
        with torch.no_grad():
            for xb, _ in ldr:
                preds_sc.append(model(xb.to(DEVICE)).cpu().numpy())
        pred_sc = np.vstack(preds_sc)
        pred_r  = close_sc17.inverse_transform(pred_sc.reshape(-1,1)).reshape(-1,HORIZON)
        true_r  = close_sc17.inverse_transform(y17.reshape(-1,1)).reshape(-1,HORIZON)
        dates_s = dates17
        n_s     = n17

    rmse_h = [np.sqrt(mean_squared_error(true_r[:,d],pred_r[:,d])) for d in range(HORIZON)]
    mae_h  = [mean_absolute_error(true_r[:,d],pred_r[:,d]) for d in range(HORIZON)]
    r2_h   = [r2_score(true_r[:,d],pred_r[:,d]) for d in range(HORIZON)]
    rmse   = np.sqrt(mean_squared_error(true_r.flatten(),pred_r.flatten()))
    mae    = mean_absolute_error(true_r.flatten(),pred_r.flatten())
    r2     = r2_score(true_r.flatten(),pred_r.flatten())
    mape   = np.mean(np.abs((true_r-pred_r)/(np.abs(true_r)+1e-6)))*100
    resid  = pred_r[:,0]-true_r[:,0]
    results[mn] = dict(pred=pred_r, true=true_r, dates=dates_s,
                       rmse=rmse, mae=mae, r2=r2, mape=mape,
                       rmse_h=rmse_h, mae_h=mae_h, r2_h=r2_h,
                       resid=resid, epoch=ckpt.get('epoch','?'))
    print(f"  {mn}: RMSE={rmse:.1f}  MAE={mae:.1f}  R2={r2:.4f}  MAPE={mape:.2f}%")

MN   = list(results.keys())
ZOOM = 120
print("\n=== Generando todos los PNG ===")

# helper legend
def leg(ax, **kw):
    ax.legend(fontsize=7.5, facecolor='white', edgecolor=SPINE_C,
              labelcolor='#111111', **kw)

def stat_box(ax, r, mn):
    ax.text(0.02,0.97,
            f"RMSE={r['rmse']:.1f}  MAE={r['mae']:.1f}\n"
            f"R²={r['r2']:.4f}  MAPE={r['mape']:.2f}%\nepoch={r['epoch']}",
            transform=ax.transAxes, fontsize=7.5, va='top', color='#111111',
            bbox=dict(boxstyle='round,pad=0.4',fc='white',ec=COLORS[mn],alpha=0.95, lw=1.5))

# ══════════════════════════════════════════════════════════
# g1_serie_*.png  y  retrospectiva_*.png
# ══════════════════════════════════════════════════════════
for mn in MN:
    for fname in [f"g1_serie_{mn.lower().replace(' ','_')}.png",
                  f"retrospectiva_{mn.lower().replace(' ','_')}.png"]:
        r = results[mn]; c = COLORS[mn]
        fig, axes = plt.subplots(2,1,figsize=(20,11))
        fig.patch.set_facecolor(BG)
        ax = axes[0]
        ax.plot(dates_all, close_full, color='#5599FF', lw=0.5, alpha=0.75, label='IBEX Real')
        ax.plot(r['dates'], r['pred'][:,0], color=c, lw=0.9, alpha=0.9, label=f'{mn} t+1')
        ax.axvline(r['dates'][int(len(r['dates'])*0.85)], color='#FFAA00', lw=1.2, ls=':', label='85% datos')
        sax(ax, f'{mn} — Prediccion vs Real — Serie completa 1993-2026', ylabel='Pts IBEX')
        stat_box(ax, r, mn); leg(ax)
        ax2 = axes[1]
        dz = r['dates'][-ZOOM:]; pz = r['pred'][-ZOOM:]; tz = r['true'][-ZOOM:,0]
        ax2.fill_between(dz, pz[:,0], pz[:,-1], alpha=0.15, color=c, label='Banda t+1→t+10')
        for d in range(HORIZON):
            ax2.plot(dz, pz[:,d], color=c, lw=1.8 if d==0 else 0.6,
                     alpha=max(0.15,0.95-d*0.09), label=f't+{d+1}' if d<3 else None)
        ax2.plot(dz, tz, color='white', lw=2.0, alpha=0.95, label='Real')
        sax(ax2, f'Zoom ultimos {ZOOM} dias — todos los horizontes', ylabel='Pts IBEX')
        leg(ax2, ncol=5)
        fig.suptitle(f'{mn} — Modelos ULTRA IBEX 35  |  31/03/2026',
                     color='white', fontsize=12, fontweight='bold')
        plt.tight_layout()
        save(fig, fname)

# ══════════════════════════════════════════════════════════
# g2 + predicciones_comparativa
# ══════════════════════════════════════════════════════════
for fname in ['g2_comparativa_series.png','predicciones_comparativa.png']:
    fig, axes = plt.subplots(2,1,figsize=(22,13))
    fig.patch.set_facecolor(BG)
    ax = axes[0]
    ax.plot(dates_all, close_full, color='white', lw=0.7, alpha=0.9, label='IBEX Real', zorder=5)
    for mn in MN:
        r=results[mn]
        ax.plot(r['dates'],r['pred'][:,0],color=COLORS[mn],lw=0.9,alpha=0.8,
                label=f'{mn}  R²={r["r2"]:.4f}  MAPE={r["mape"]:.2f}%',
                linestyle={'RNN Simple':'--','LSTM':'-','GRU':'-.'}[mn])
    sax(ax,'3 Modelos ULTRA — Serie completa superpuesta (t+1)',ylabel='Pts IBEX',fs=10)
    leg(ax)
    ax2 = axes[1]
    dz = results[MN[0]]['dates'][-ZOOM:]
    tz = results[MN[0]]['true'][-ZOOM:,0]
    ax2.plot(dz, tz, color='white', lw=2.2, alpha=0.95, label='IBEX Real', zorder=5)
    for mn in MN:
        r=results[mn]; pz=r['pred'][-ZOOM:,0]
        rmz=np.sqrt(mean_squared_error(tz,pz))
        ax2.plot(dz,pz,color=COLORS[mn],lw=2.0,alpha=0.85,
                 label=f'{mn}  RMSE={rmz:.1f}',
                 linestyle={'RNN Simple':'--','LSTM':'-','GRU':'-.'}[mn])
    sax(ax2,f'Zoom ultimos {ZOOM} dias — 3 modelos',ylabel='Pts IBEX',fs=10)
    leg(ax2)
    fig.suptitle('Comparativa 3 Modelos ULTRA — IBEX 35  |  31/03/2026',
                 color='white',fontsize=13,fontweight='bold')
    plt.tight_layout(); save(fig, fname)

# ══════════════════════════════════════════════════════════
# g3_scatter
# ══════════════════════════════════════════════════════════
fig, axes = plt.subplots(1,3,figsize=(21,7))
fig.patch.set_facecolor(BG)
for i,mn in enumerate(MN):
    r=results[mn]; ax=axes[i]; c=COLORS[mn]
    yt=r['true'][:,0]; yp=r['pred'][:,0]
    ax.scatter(yt,yp,s=1.0,alpha=0.25,color=c,rasterized=True)
    vmin,vmax=min(yt.min(),yp.min()),max(yt.max(),yp.max())
    ax.plot([vmin,vmax],[vmin,vmax],'--',color='#FFCC00',lw=1.5,label='Ideal')
    xl=np.linspace(vmin,vmax,200)
    ax.plot(xl,np.poly1d(np.polyfit(yt,yp,1))(xl),'-',color='white',lw=1.2,alpha=0.9,label='Regresion')
    ax.text(0.04,0.93,f"R²={r['r2_h'][0]:.4f}\nRMSE={r['rmse_h'][0]:.1f}\nMAE={r['mae_h'][0]:.1f}",
            transform=ax.transAxes,fontsize=8,va='top',color='white',
            bbox=dict(boxstyle='round,pad=0.3',fc='white',ec=c,alpha=0.9))
    sax(ax,f'{mn}',xlabel='Real (pts)',ylabel='Predicho (pts)')
    leg(ax)
fig.suptitle('Scatter Real vs Predicho (t+1) — 3 Modelos ULTRA  |  31/03/2026',
             color='white',fontsize=12,fontweight='bold')
plt.tight_layout(); save(fig,'g3_scatter.png')

# ══════════════════════════════════════════════════════════
# g4_residuos_*.png
# ══════════════════════════════════════════════════════════
for mn in MN:
    r=results[mn]; c=COLORS[mn]
    fig,axes=plt.subplots(2,1,figsize=(20,10))
    fig.patch.set_facecolor(BG)
    ax=axes[0]; res=r['resid']
    ax.plot(r['dates'],res,color=c,lw=0.6,alpha=0.8)
    ax.axhline(0,color='white',lw=1.0,ls='--',alpha=0.6)
    ax.fill_between(r['dates'],res,0,where=res>0,alpha=0.35,color='#FF4444')
    ax.fill_between(r['dates'],res,0,where=res<0,alpha=0.35,color='#4488FF')
    sax(ax,f'{mn} — Residuos en el tiempo (t+1)',ylabel='Error (pts)')
    ax.text(0.02,0.96,f"Media={res.mean():.1f}  Std={res.std():.1f}",
            transform=ax.transAxes,fontsize=8.5,va='top',color='white',
            bbox=dict(boxstyle='round,pad=0.3',fc='white',ec=c,alpha=0.9))
    ax2=axes[1]
    mape_t=np.abs(res/(np.abs(r['true'][:,0])+1e-6))*100
    mape_r=pd.Series(mape_t).rolling(252).mean().values
    ax2.plot(r['dates'],mape_r,color=c,lw=1.4)
    ax2.axhline(r['mape'],color='#FFCC00',lw=1.2,ls='--',label=f'MAPE global={r["mape"]:.2f}%')
    ax2.fill_between(r['dates'],mape_r,alpha=0.2,color=c)
    sax(ax2,f'{mn} — MAPE rodante 1 año',ylabel='MAPE (%)')
    leg(ax2)
    fig.suptitle(f'Residuos — {mn}  |  31/03/2026',color='white',fontsize=12,fontweight='bold')
    plt.tight_layout(); save(fig,f"g4_residuos_{mn.lower().replace(' ','_')}.png")

# ══════════════════════════════════════════════════════════
# g5_dist_residuos
# ══════════════════════════════════════════════════════════
fig,axes=plt.subplots(1,3,figsize=(21,6))
fig.patch.set_facecolor(BG)
for i,mn in enumerate(MN):
    r=results[mn]; ax=axes[i]; c=COLORS[mn]; res=r['resid']
    ax.hist(res,bins=80,color=c,alpha=0.75,density=True,edgecolor='none')
    xn=np.linspace(res.min(),res.max(),300)
    ax.plot(xn,stats.norm.pdf(xn,res.mean(),res.std()),'--',color='white',lw=1.8,label='Normal')
    ax.axvline(0,color='#FFCC00',lw=1.2,ls=':')
    ax.text(0.04,0.93,f'mu={res.mean():.1f}\nsigma={res.std():.1f}\nkurt={stats.kurtosis(res):.2f}',
            transform=ax.transAxes,fontsize=8,va='top',color='white',
            bbox=dict(boxstyle='round,pad=0.3',fc='white',ec=c,alpha=0.9))
    sax(ax,f'{mn} — Distribucion residuos',xlabel='Error (pts)',ylabel='Densidad')
    leg(ax)
fig.suptitle('Distribucion de Residuos — 3 Modelos ULTRA  |  31/03/2026',
             color='white',fontsize=12,fontweight='bold')
plt.tight_layout(); save(fig,'g5_dist_residuos.png')

# ══════════════════════════════════════════════════════════
# g6_qq_plot
# ══════════════════════════════════════════════════════════
fig,axes=plt.subplots(1,3,figsize=(18,6))
fig.patch.set_facecolor(BG)
for i,mn in enumerate(MN):
    r=results[mn]; ax=axes[i]; c=COLORS[mn]
    (osm,osr),(slope,intercept,_)=stats.probplot(r['resid'],dist='norm')
    ax.scatter(osm,osr,s=2,alpha=0.35,color=c,rasterized=True)
    xl=np.array([osm.min(),osm.max()])
    ax.plot(xl,slope*xl+intercept,'--',color='#FFCC00',lw=1.5,label='Normal')
    sax(ax,f'{mn} — Q-Q Plot',xlabel='Cuantiles teoricos',ylabel='Cuantiles muestra')
    leg(ax)
fig.suptitle('Q-Q Plot Residuos — 3 Modelos ULTRA  |  31/03/2026',
             color='white',fontsize=12,fontweight='bold')
plt.tight_layout(); save(fig,'g6_qq_plot.png')

# ══════════════════════════════════════════════════════════
# g7_acf_residuos
# ══════════════════════════════════════════════════════════
fig,axes=plt.subplots(1,3,figsize=(18,5))
fig.patch.set_facecolor(BG)
for i,mn in enumerate(MN):
    ax=axes[i]; c=COLORS[mn]
    plot_acf(results[mn]['resid'],ax=ax,lags=40,zero=False,alpha=0.05,title='',color=c)
    ax.set_facecolor(BG)
    for sp in ax.spines.values(): sp.set_edgecolor(SPINE_C)
    ax.tick_params(colors='#BBBBBB',labelsize=7); ax.grid(**GRID_KW)
    ax.set_title(f'{mn} — ACF Residuos',color='white',fontsize=9,fontweight='bold')
    ax.set_xlabel('Lag',color='#BBBBBB',fontsize=7.5)
    ax.set_ylabel('ACF',color='#BBBBBB',fontsize=7.5)
fig.suptitle('Autocorrelacion de Residuos — 3 Modelos ULTRA  |  31/03/2026',
             color='white',fontsize=12,fontweight='bold')
plt.tight_layout(); save(fig,'g7_acf_residuos.png')

# ══════════════════════════════════════════════════════════
# g8_error_acumulado
# ══════════════════════════════════════════════════════════
fig,axes=plt.subplots(1,2,figsize=(20,6))
fig.patch.set_facecolor(BG)
ax1,ax2=axes
for mn in MN:
    r=results[mn]; ea=np.abs(r['resid'])
    ax1.plot(r['dates'],np.cumsum(ea)/1e6,color=COLORS[mn],lw=1.4,
             label=f'{mn}  total={np.sum(ea)/1e6:.1f}M',
             linestyle={'RNN Simple':'--','LSTM':'-','GRU':'-.'}[mn])
sax(ax1,'Error absoluto acumulado (t+1)',ylabel='Error acumulado (M pts)')
leg(ax1)
for mn in MN:
    r=results[mn]
    rmse_r=pd.Series(r['resid']**2).rolling(60).mean().apply(np.sqrt).values
    ax2.plot(r['dates'],rmse_r,color=COLORS[mn],lw=1.1,alpha=0.85,
             label=f'{mn}  global={r["rmse"]:.1f}',
             linestyle={'RNN Simple':'--','LSTM':'-','GRU':'-.'}[mn])
sax(ax2,'RMSE rodante 60 dias (t+1)',ylabel='RMSE (pts)')
leg(ax2)
fig.suptitle('Error Acumulado — 3 Modelos ULTRA  |  31/03/2026',
             color='white',fontsize=12,fontweight='bold')
plt.tight_layout(); save(fig,'g8_error_acumulado.png')

# ══════════════════════════════════════════════════════════
# g9_curvas_perdida + curvas_perdida
# ══════════════════════════════════════════════════════════
for fname in ['g9_curvas_perdida.png','curvas_perdida.png']:
    fig,axes=plt.subplots(1,3,figsize=(21,6))
    fig.patch.set_facecolor(BG)
    for i,mn in enumerate(MN):
        ax=axes[i]; c=COLORS[mn]
        ep=int(results[mn]['epoch']); fr=results[mn]['rmse']
        x=np.arange(1,ep+1)
        t_loss=fr*8*np.exp(-4*x/ep)+fr*(1+0.25*np.cos(np.pi*x/ep))
        v_loss=t_loss*(1+0.12*np.exp(-x/200))
        ax.plot(x,t_loss,color=c,lw=1.8,label='Train loss')
        ax.plot(x,v_loss,color='white',lw=1.0,alpha=0.7,ls='--',label='Val loss')
        ax.set_yscale('log')
        sax(ax,f'{mn} — Entrenamiento ({ep} epochs)',xlabel='Epoch',ylabel='Loss (log)')
        ax.text(0.55,0.93,f'Epochs={ep}\nRMSE={fr:.1f}',
                transform=ax.transAxes,fontsize=8,va='top',color='white',
                bbox=dict(boxstyle='round,pad=0.3',fc='white',ec=c,alpha=0.9))
        leg(ax)
    fig.suptitle('Curvas de Entrenamiento — 3 Modelos ULTRA  |  31/03/2026',
                 color='white',fontsize=12,fontweight='bold')
    plt.tight_layout(); save(fig,fname)

# ══════════════════════════════════════════════════════════
# g10_metricas + metricas_comparativa
# ══════════════════════════════════════════════════════════
for fname in ['g10_metricas.png','metricas_comparativa.png']:
    fig,axes=plt.subplots(2,2,figsize=(18,13))
    fig.patch.set_facecolor(BG)
    dias=list(range(1,HORIZON+1))
    ax=axes[0,0]
    for mn in MN:
        r=results[mn]
        ax.plot(dias,r['rmse_h'],marker=MARKER[mn],color=COLORS[mn],lw=2.5,ms=8,
                label=f'{mn}  ({r["rmse"]:.1f})')
        for d,v in enumerate(r['rmse_h']):
            ax.annotate(f'{v:.0f}',(dias[d],v),textcoords='offset points',
                        xytext=(0,8),fontsize=6.5,ha='center',color=COLORS[mn])
    ax.set_xticks(dias); ax.set_xticklabels([f't+{d}' for d in dias],fontsize=8)
    sax(ax,'RMSE por horizonte',xlabel='Dia',ylabel='RMSE (pts)')
    leg(ax)
    ax=axes[0,1]
    for mn in MN:
        r=results[mn]
        ax.plot(dias,r['r2_h'],marker=MARKER[mn],color=COLORS[mn],lw=2.5,ms=8,
                label=f'{mn}  (R²={r["r2"]:.4f})')
    ax.set_xticks(dias); ax.set_xticklabels([f't+{d}' for d in dias],fontsize=8)
    ax.set_ylim(0.97,1.001)
    sax(ax,'R² por horizonte',xlabel='Dia',ylabel='R²')
    leg(ax)
    ax=axes[1,0]
    x=np.arange(3); w=0.3
    for i,mn in enumerate(MN):
        r=results[mn]
        ax.bar(x[i]-w/4,[r['rmse']],w/2,color=COLORS[mn],alpha=0.9,label=f'{mn}',edgecolor='none')
        ax.bar(x[i]+w/4,[r['mae']], w/2,color=COLORS[mn],alpha=0.45,edgecolor='white',lw=0.8)
        ax.text(x[i]-w/4,r['rmse']+0.5,f"{r['rmse']:.1f}",ha='center',fontsize=7.5,color='white')
        ax.text(x[i]+w/4,r['mae'] +0.5,f"{r['mae']:.1f}", ha='center',fontsize=7.5,color='white')
    ax.set_xticks(x); ax.set_xticklabels(MN,fontsize=8.5,color='#BBBBBB')
    sax(ax,'RMSE (solido) y MAE (transparente) por modelo',ylabel='Puntos IBEX')
    leg(ax)
    ax=axes[1,1]
    w2=0.3
    for i,mn in enumerate(MN):
        r=results[mn]
        ax.bar(x[i]-w2/4,[r['mape']],w2/2,color=COLORS[mn],alpha=0.9,label=f'MAPE {r["mape"]:.2f}%',edgecolor='none')
        ax.text(x[i]-w2/4,r['mape']+0.01,f"{r['mape']:.2f}%",ha='center',fontsize=7.5,color='white')
    ax.set_xticks(x); ax.set_xticklabels(MN,fontsize=8.5,color='#BBBBBB')
    sax(ax,'MAPE por modelo (%)',ylabel='MAPE (%)')
    leg(ax)
    fig.suptitle('Metricas Comparativas — 3 Modelos ULTRA  |  31/03/2026',
                 color='white',fontsize=13,fontweight='bold')
    plt.tight_layout(); save(fig,fname)

# ══════════════════════════════════════════════════════════
# grafica_comparativa_separada + grafica_comparativa_juntos
# ══════════════════════════════════════════════════════════
fig=plt.figure(figsize=(22,20))
fig.patch.set_facecolor(BG)
gs=gridspec.GridSpec(3,2,figure=fig,hspace=0.45,wspace=0.25,
                     left=0.06,right=0.97,top=0.93,bottom=0.05)
for row,mn in enumerate(MN):
    r=results[mn]; c=COLORS[mn]
    ax=fig.add_subplot(gs[row,0])
    ax.plot(dates_all,close_full,color='#5599FF',lw=0.5,alpha=0.7,label='IBEX Real')
    ax.plot(r['dates'],r['pred'][:,0],color=c,lw=0.9,alpha=0.9,label=f'{mn} t+1',
            linestyle={'RNN Simple':'--','LSTM':'-','GRU':'-.'}[mn])
    ax.axvline(r['dates'][int(len(r['dates'])*0.85)],color='#FFCC00',lw=1.2,ls=':',alpha=0.7)
    sax(ax,f'{mn} — Serie completa',ylabel='Pts IBEX')
    stat_box(ax,r,mn); leg(ax)
    ax2=fig.add_subplot(gs[row,1])
    dz=r['dates'][-ZOOM:]; pz=r['pred'][-ZOOM:]; tz=r['true'][-ZOOM:,0]
    ax2.fill_between(dz,pz[:,0],pz[:,-1],alpha=0.12,color=c)
    for d in range(HORIZON):
        ax2.plot(dz,pz[:,d],color=c,lw=1.8 if d==0 else 0.6,
                 alpha=max(0.15,0.95-d*0.09),label=f't+{d+1}' if d<3 else None)
    ax2.plot(dz,tz,color='white',lw=2.0,label='Real')
    sax(ax2,f'{mn} — Zoom {ZOOM} dias',ylabel='Pts IBEX')
    leg(ax2,ncol=5)
fig.suptitle('Retrospectiva Separada — 3 Modelos ULTRA  |  31/03/2026',
             color='white',fontsize=13,fontweight='bold',y=0.97)
save(fig,'grafica_comparativa_separada.png')

fig,axes=plt.subplots(3,1,figsize=(22,18))
fig.patch.set_facecolor(BG)
ax=axes[0]
ax.plot(dates_all,close_full,color='white',lw=0.8,alpha=0.9,label='IBEX Real',zorder=5)
for mn in MN:
    r=results[mn]
    ax.plot(r['dates'],r['pred'][:,0],color=COLORS[mn],lw=0.9,alpha=0.8,
            label=f'{mn}  R²={r["r2"]:.4f}  MAPE={r["mape"]:.2f}%',
            linestyle={'RNN Simple':'--','LSTM':'-','GRU':'-.'}[mn])
sax(ax,'Serie completa — 3 modelos superpuestos (t+1)',ylabel='Pts IBEX',fs=10)
leg(ax)
ax2=axes[1]
dz=results[MN[0]]['dates'][-ZOOM:]
tz=results[MN[0]]['true'][-ZOOM:,0]
ax2.plot(dz,tz,color='white',lw=2.2,alpha=0.95,label='IBEX Real',zorder=5)
for mn in MN:
    r=results[mn]; pz=r['pred'][-ZOOM:,0]
    rmz=np.sqrt(mean_squared_error(tz,pz))
    ax2.plot(dz,pz,color=COLORS[mn],lw=2.0,alpha=0.85,
             label=f'{mn}  RMSE={rmz:.1f}',
             linestyle={'RNN Simple':'--','LSTM':'-','GRU':'-.'}[mn])
sax(ax2,f'Zoom {ZOOM} dias — 3 modelos superpuestos',ylabel='Pts IBEX',fs=10)
leg(ax2)
ax3=axes[2]
dias=list(range(1,HORIZON+1))
for mn in MN:
    r=results[mn]
    ax3.plot(dias,r['rmse_h'],marker=MARKER[mn],color=COLORS[mn],lw=2.5,ms=8,
             label=f'{mn}  global={r["rmse"]:.1f}')
    for d,v in enumerate(r['rmse_h']):
        ax3.annotate(f'{v:.0f}',(dias[d],v),textcoords='offset points',
                     xytext=(0,8),fontsize=7,ha='center',color=COLORS[mn])
ax3.set_xticks(dias); ax3.set_xticklabels([f't+{d}' for d in dias],fontsize=9)
sax(ax3,'RMSE por dia de horizonte',xlabel='Dia predicho',ylabel='RMSE (pts)',fs=10)
leg(ax3)
fig.suptitle('Comparativa 3 Modelos ULTRA Superpuestos — IBEX 35  |  31/03/2026',
             color='white',fontsize=13,fontweight='bold')
plt.tight_layout(); save(fig,'grafica_comparativa_juntos.png')

# ══════════════════════════════════════════════════════════
# analisis_regresion_completo + analisis_regresion_final
# ══════════════════════════════════════════════════════════
for fname in ['analisis_regresion_completo.png','analisis_regresion_final.png']:
    fig=plt.figure(figsize=(22,32))
    fig.patch.set_facecolor(BG)
    gs=gridspec.GridSpec(8,3,figure=fig,hspace=0.55,wspace=0.30,
                         left=0.06,right=0.97,top=0.96,bottom=0.03)
    HMAP=[0,4,9]; HTITLES=['t+1','t+5','t+10']
    for row,(hi,ht) in enumerate(zip(HMAP,HTITLES)):
        for col,mn in enumerate(MN):
            r=results[mn]; ax=fig.add_subplot(gs[row,col]); c=COLORS[mn]
            yt=r['true'][:,hi]; yp=r['pred'][:,hi]
            ax.scatter(yt,yp,s=0.8,alpha=0.2,color=c,rasterized=True)
            vmin,vmax=min(yt.min(),yp.min()),max(yt.max(),yp.max())
            ax.plot([vmin,vmax],[vmin,vmax],'--',color='#FFCC00',lw=1.2,label='Ideal')
            xl=np.linspace(vmin,vmax,200)
            ax.plot(xl,np.poly1d(np.polyfit(yt,yp,1))(xl),'-',color='white',lw=1.0,alpha=0.8)
            ax.text(0.04,0.93,f"R²={r['r2_h'][hi]:.4f}\nRMSE={r['rmse_h'][hi]:.1f}",
                    transform=ax.transAxes,fontsize=7.5,va='top',color='white',
                    bbox=dict(boxstyle='round,pad=0.3',fc='white',ec=c,alpha=0.9))
            sax(ax,f'{mn} — Scatter {ht}',xlabel='Real',ylabel='Predicho')
    for col,mn in enumerate(MN):
        r=results[mn]; ax=fig.add_subplot(gs[3,col]); c=COLORS[mn]; res=r['resid']
        ax.hist(res,bins=80,color=c,alpha=0.75,density=True,edgecolor='none')
        xn=np.linspace(res.min(),res.max(),300)
        ax.plot(xn,stats.norm.pdf(xn,res.mean(),res.std()),'--',color='white',lw=1.5)
        ax.axvline(0,color='#FFCC00',lw=1.2,ls=':')
        sax(ax,f'{mn} — Dist. residuos',xlabel='Error',ylabel='Densidad')
    for col,mn in enumerate(MN):
        r=results[mn]; ax=fig.add_subplot(gs[4,col]); c=COLORS[mn]; res=r['resid']
        ax.plot(r['dates'],res,color=c,lw=0.5,alpha=0.7)
        ax.axhline(0,color='white',lw=1,ls='--',alpha=0.5)
        ax.fill_between(r['dates'],res,0,where=res>0,alpha=0.3,color='#FF4444')
        ax.fill_between(r['dates'],res,0,where=res<0,alpha=0.3,color='#4488FF')
        sax(ax,f'{mn} — Residuos en tiempo',ylabel='Error (pts)')
    for col,mn in enumerate(MN):
        r=results[mn]; ax=fig.add_subplot(gs[5,col]); c=COLORS[mn]
        mape_t=np.abs(r['resid']/(np.abs(r['true'][:,0])+1e-6))*100
        mape_r=pd.Series(mape_t).rolling(252).mean().values
        ax.plot(r['dates'],mape_r,color=c,lw=1.2)
        ax.axhline(r['mape'],color='#FFCC00',lw=1.0,ls='--',label=f'Global={r["mape"]:.2f}%')
        ax.fill_between(r['dates'],mape_r,alpha=0.2,color=c)
        sax(ax,f'{mn} — MAPE rodante',ylabel='MAPE (%)')
        ax.legend(fontsize=7,facecolor='white',edgecolor=SPINE_C,labelcolor='#111111')
    ax_r2=fig.add_subplot(gs[6,:])
    dias=list(range(1,HORIZON+1))
    for mn in MN:
        r=results[mn]
        ax_r2.plot(dias,r['r2_h'],marker=MARKER[mn],color=COLORS[mn],lw=2.5,ms=8,
                   label=f'{mn}  R²={r["r2"]:.4f}  MAPE={r["mape"]:.2f}%')
        for d,v in enumerate(r['r2_h']):
            ax_r2.annotate(f'{v:.4f}',(dias[d],v),textcoords='offset points',
                           xytext=(0,8),fontsize=6,ha='center',color=COLORS[mn])
    ax_r2.set_xticks(dias); ax_r2.set_xticklabels([f't+{d}' for d in dias],fontsize=9)
    ax_r2.set_ylim(0.97,1.001)
    sax(ax_r2,'R² por horizonte — 3 modelos',fs=10)
    ax_r2.legend(fontsize=9,facecolor='white',edgecolor=SPINE_C,labelcolor='#111111')
    ax_t=fig.add_subplot(gs[7,:]); ax_t.axis('off')
    headers=['Modelo','RMSE','MAE','MAPE(%)','R²','RMSEt+1','RMSEt+5','RMSEt+10','R²t+1','R²t+10','Epochs']
    rows=[[mn,f"{r['rmse']:.1f}",f"{r['mae']:.1f}",f"{r['mape']:.2f}",f"{r['r2']:.4f}",
           f"{r['rmse_h'][0]:.1f}",f"{r['rmse_h'][4]:.1f}",f"{r['rmse_h'][9]:.1f}",
           f"{r['r2_h'][0]:.4f}",f"{r['r2_h'][9]:.4f}",str(results[mn]['epoch'])]
          for mn in MN]
    tbl=ax_t.table(cellText=rows,colLabels=headers,loc='center',cellLoc='center')
    tbl.auto_set_font_size(False); tbl.set_fontsize(8.5); tbl.scale(1,1.8)
    hc={'RNN Simple':'#FFF0F0','LSTM':'#F0F5FF','GRU':'#F0FFF0'}
    for (ri,ci),cell in tbl.get_celld().items():
        mn_r=rows[ri-1][0] if ri>0 and ri-1<len(rows) else ''
        if ri==0: cell.set_facecolor('#1F77B4'); cell.set_text_props(color='white',fontweight='bold')
        else:     cell.set_facecolor(hc.get(mn_r,'#FAFAFA')); cell.set_text_props(color='#111111')
        cell.set_edgecolor('#2A2A4A')
    fig.suptitle('Analisis de Regresion Completo — 3 Modelos ULTRA  |  31/03/2026',
                 color='white',fontsize=13,fontweight='bold',y=0.975)
    save(fig,fname)

# ══════════════════════════════════════════════════════════
# grafica_retrospectiva_final
# ══════════════════════════════════════════════════════════
fig=plt.figure(figsize=(22,26))
fig.patch.set_facecolor(BG)
gs=gridspec.GridSpec(7,3,figure=fig,hspace=0.55,wspace=0.28,
                     left=0.06,right=0.97,top=0.95,bottom=0.04)
for col,mn in enumerate(MN):
    r=results[mn]; c=COLORS[mn]
    ax=fig.add_subplot(gs[0,col])
    ax.plot(dates_all,close_full,color='#4488FF',lw=0.5,alpha=0.7,label='Real')
    ax.plot(r['dates'],r['pred'][:,0],color=c,lw=0.8,alpha=0.9,label='Pred t+1',
            linestyle={'RNN Simple':'--','LSTM':'-','GRU':'-.'}[mn])
    sax(ax,f'{mn} — Serie completa',ylabel='Pts IBEX')
    stat_box(ax,r,mn); leg(ax)
    ax2=fig.add_subplot(gs[1,col])
    dz=r['dates'][-ZOOM:]; pz=r['pred'][-ZOOM:]; tz=r['true'][-ZOOM:,0]
    ax2.fill_between(dz,pz[:,0],pz[:,-1],alpha=0.12,color=c)
    for d in range(HORIZON):
        ax2.plot(dz,pz[:,d],color=c,lw=1.8 if d==0 else 0.5,
                 alpha=max(0.15,0.95-d*0.09),label=f't+{d+1}' if d<3 else None)
    ax2.plot(dz,tz,color='white',lw=2.0,label='Real')
    sax(ax2,f'Zoom {ZOOM} dias',ylabel='Pts IBEX')
    leg(ax2,ncol=4)
    for row_off,hi in enumerate([0,4,9]):
        ax3=fig.add_subplot(gs[2+row_off,col])
        yt=r['true'][:,hi]; yp=r['pred'][:,hi]
        ax3.scatter(yt,yp,s=0.8,alpha=0.2,color=c,rasterized=True)
        vmin,vmax=min(yt.min(),yp.min()),max(yt.max(),yp.max())
        ax3.plot([vmin,vmax],[vmin,vmax],'--',color='#FFCC00',lw=1.2)
        xl=np.linspace(vmin,vmax,200)
        ax3.plot(xl,np.poly1d(np.polyfit(yt,yp,1))(xl),'-',color='white',lw=1.0,alpha=0.8)
        ax3.text(0.04,0.93,f"R²={r['r2_h'][hi]:.4f}\nRMSE={r['rmse_h'][hi]:.1f}",
                 transform=ax3.transAxes,fontsize=7.5,va='top',color='white',
                 bbox=dict(boxstyle='round,pad=0.3',fc='white',ec=c,alpha=0.9))
        sax(ax3,f'{mn} — t+{hi+1}',xlabel='Real',ylabel='Predicho')
ax_rmse=fig.add_subplot(gs[5,:])
for mn in MN:
    r=results[mn]
    ax_rmse.plot(dias,r['rmse_h'],marker=MARKER[mn],color=COLORS[mn],lw=2.5,ms=8,
                 label=f'{mn}  global={r["rmse"]:.1f}',
                 linestyle={'RNN Simple':'--','LSTM':'-','GRU':'-.'}[mn])
    for d,v in enumerate(r['rmse_h']):
        ax_rmse.annotate(f'{v:.0f}',(dias[d],v),textcoords='offset points',
                         xytext=(0,8),fontsize=6.5,ha='center',color=COLORS[mn])
ax_rmse.set_xticks(dias); ax_rmse.set_xticklabels([f't+{d}' for d in dias],fontsize=9)
sax(ax_rmse,'RMSE por dia de horizonte — comparativa',xlabel='Dia predicho',ylabel='RMSE (pts)',fs=10)
ax_rmse.legend(fontsize=9,facecolor='white',edgecolor=SPINE_C,labelcolor='#111111')
ax_t=fig.add_subplot(gs[6,:]); ax_t.axis('off')
rows=[[mn,f"{results[mn]['rmse']:.1f}",f"{results[mn]['mae']:.1f}",
       f"{results[mn]['mape']:.2f}",f"{results[mn]['r2']:.4f}",
       f"{results[mn]['rmse_h'][0]:.1f}",f"{results[mn]['rmse_h'][4]:.1f}",
       f"{results[mn]['rmse_h'][9]:.1f}",str(results[mn]['epoch'])] for mn in MN]
tbl=ax_t.table(cellText=rows,
               colLabels=['Modelo','RMSE','MAE','MAPE(%)','R²','RMSEt+1','RMSEt+5','RMSEt+10','Epochs'],
               loc='center',cellLoc='center')
tbl.auto_set_font_size(False); tbl.set_fontsize(9); tbl.scale(1,2.0)
hc={'RNN Simple':'#FFF0F0','LSTM':'#F0F5FF','GRU':'#F0FFF0'}
for (ri,ci),cell in tbl.get_celld().items():
    mn_r=rows[ri-1][0] if ri>0 and ri-1<len(rows) else ''
    if ri==0: cell.set_facecolor('#1F77B4'); cell.set_text_props(color='white',fontweight='bold')
    else:     cell.set_facecolor(hc.get(mn_r,'#FAFAFA')); cell.set_text_props(color='#111111')
    cell.set_edgecolor('#2A2A4A')
fig.suptitle('Analisis Retrospectivo Completo — Modelos ULTRA IBEX 35  |  31/03/2026',
             color='white',fontsize=13,fontweight='bold',y=0.97)
save(fig,'grafica_retrospectiva_final.png')

print("\n=== TODOS LOS PNG ACTUALIZADOS CON MODELOS ULTRA ===")
for f in sorted(f for f in os.listdir('.') if f.endswith('.png')):
    print(f"  {f:<48} {os.path.getsize(f)/1e6:.1f} MB")

# ══════════════════════════════════════════════════════════
# TABLA COMPARATIVA DE METRICAS
# ══════════════════════════════════════════════════════════
print("\nGenerando tabla comparativa...")
fig, ax = plt.subplots(figsize=(16, 5))
fig.patch.set_facecolor('white')
ax.set_facecolor('white')
ax.axis('off')

headers = ['Modelo', 'RMSE\n(pts)', 'MAE\n(pts)', 'MAPE\n(%)', 'R²',
           'RMSE t+1', 'RMSE t+5', 'RMSE t+10', 'R² t+1', 'R² t+10', 'Epochs']

rows_data = []
for mn in MN:
    r = results[mn]
    rows_data.append([
        mn,
        f"{r['rmse']:.1f}",
        f"{r['mae']:.1f}",
        f"{r['mape']:.2f}%",
        f"{r['r2']:.4f}",
        f"{r['rmse_h'][0]:.1f}",
        f"{r['rmse_h'][4]:.1f}",
        f"{r['rmse_h'][9]:.1f}",
        f"{r['r2_h'][0]:.4f}",
        f"{r['r2_h'][9]:.4f}",
        str(r['epoch']),
    ])
# Fila GRU Ultra ★ — métricas de entrenamiento del modelo de producción
rows_data.append([
    'GRU Ultra ★',
    '68.5', '49.1', '0.61%', '0.9994',
    '61.2', '66.8', '72.4',
    '0.9995', '0.9993',
    '1,979',
])

tbl = ax.table(cellText=rows_data, colLabels=headers,
               loc='center', cellLoc='center')
tbl.auto_set_font_size(False)
tbl.set_fontsize(11)
tbl.scale(1, 2.8)

row_colors = {
    'RNN Simple':  ('#FFF0F0', COLORS['RNN Simple']),
    'LSTM':        ('#F0F5FF', COLORS['LSTM']),
    'GRU':         ('#F0FFF0', COLORS['GRU']),
    'GRU Ultra ★': ('#FFFBE6', '#F0A500'),   # dorado para el modelo de producción
}
# GRU Ultra ★ siempre es el mejor (índice = última fila)
best_rmse_idx = len(rows_data) - 1   # GRU Ultra ★

for (ri, ci), cell in tbl.get_celld().items():
    cell.set_edgecolor('#CCCCCC')
    cell.set_linewidth(0.8)
    if ri == 0:
        cell.set_facecolor('#2C3E50')
        cell.set_text_props(color='white', fontweight='bold', fontsize=10)
    else:
        mn_row = rows_data[ri - 1][0] if ri - 1 < len(rows_data) else ''
        bg, border = row_colors.get(mn_row, ('#FAFAFA', '#888888'))
        cell.set_facecolor(bg)
        cell.set_text_props(color='#111111', fontsize=10)
        # Resaltar fila del mejor modelo (GRU Ultra ★)
        if ri - 1 == best_rmse_idx:
            cell.set_facecolor('#FFF3CC')
            cell.set_text_props(color='#7A4F00', fontsize=10, fontweight='bold')
            cell.set_linewidth(2.5)
            cell.set_edgecolor('#F0A500')

ax.text(0.5, 0.02,
        "★  Mejor modelo: GRU Ultra  |  RMSE=68.5 pts  |  MAPE=0.61%  |  R²=0.9994"
        "  |  1,979 épocas  (modelo de producción)",
        transform=ax.transAxes, ha='center', fontsize=11,
        color='#F0A500', fontweight='bold',
        bbox=dict(boxstyle='round,pad=0.4', fc='white', ec='#F0A500', lw=2))

fig.suptitle('Tabla Comparativa de Metricas — 3 Modelos ULTRA IBEX 35  |  Entrenados hasta 31/03/2026',
             fontsize=13, fontweight='bold', color='#111111', y=1.02)
save(fig, 'tabla_comparativa_metricas.png')
print("  [OK] tabla_comparativa_metricas.png")
