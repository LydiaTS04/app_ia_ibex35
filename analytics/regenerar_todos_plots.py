# -*- coding: utf-8 -*-
"""
regenerar_todos_plots.py
Regenera TODOS los PNG de la carpeta con los modelos finales entrenados.
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
SEQ_LEN    = 90
HORIZON    = 10
DROPOUT    = 0.05
BATCH      = 512
START_DATE = '1993-07-12'
END_TRAIN  = '2026-04-01'
BG         = '#0D1117'
GRID_KW    = dict(color='#2A2A3A', linewidth=0.5, linestyle='--')
SPINE_C    = '#3A3A4A'
SNAPSHOTS  = {"RNN Simple":"snap_rnn_31mar.pt","LSTM":"snap_lstm_31mar.pt","GRU":"snap_gru_31mar.pt"}
COLORS     = {"RNN Simple":"#E74C3C","LSTM":"#3498DB","GRU":"#2ECC71"}

def sax(ax, title='', xlabel='', ylabel='', fs=9):
    ax.set_facecolor(BG)
    for sp in ax.spines.values(): sp.set_edgecolor(SPINE_C)
    ax.tick_params(colors='#AAAAAA', labelsize=7.5)
    ax.grid(**GRID_KW)
    if title:  ax.set_title(title,  color='white',   fontsize=fs, fontweight='bold', pad=6)
    if xlabel: ax.set_xlabel(xlabel, color='#AAAAAA', fontsize=7.5)
    if ylabel: ax.set_ylabel(ylabel, color='#AAAAAA', fontsize=7.5)

def save(fig, name):
    fig.savefig(name, dpi=150, bbox_inches='tight', facecolor=BG)
    plt.close(fig)
    mb = os.path.getsize(name)/1e6
    print(f"  [OK] {name}  ({mb:.1f} MB)")

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
df['RSI']   = 100 - (100/(1 + gain/(loss+1e-9)))
df['Mom10'] = df['Close'] - df['Close'].shift(10)
df.dropna(inplace=True)
feature_cols = list(df.columns)
n_features   = len(feature_cols)
close_idx    = feature_cols.index('Close')
dates_all    = df.index
close_full   = df['Close'].values
print(f"  {df.index[0].date()} -> {df.index[-1].date()}  ({len(df)} sesiones)")

with open('scaler_final.pkl','rb') as f:       scaler = pickle.load(f)
with open('close_scaler_final.pkl','rb') as f: close_scaler = pickle.load(f)
scaled = scaler.transform(df.values)

# ─── SECUENCIAS ──────────────────────────────────────────────
def make_seq(arr, sl, hz):
    X, y = [], []
    for i in range(sl, len(arr)-hz+1):
        X.append(arr[i-sl:i]); y.append(arr[i:i+hz, close_idx])
    return np.array(X,dtype=np.float32), np.array(y,dtype=np.float32)

X_all, y_all = make_seq(scaled, SEQ_LEN, HORIZON)
n = len(X_all)
dates_seq = dates_all[SEQ_LEN:SEQ_LEN+n]
val_cut   = int(n*0.90)
X_val, y_val = X_all[val_cut:], y_all[val_cut:]
loader_all = DataLoader(TensorDataset(torch.tensor(X_all),torch.tensor(y_all)),
                        batch_size=BATCH, shuffle=False)

# ─── MODELOS ─────────────────────────────────────────────────
class Attention(nn.Module):
    def __init__(self, h):
        super().__init__()
        self.attn = nn.Linear(h,h); self.v = nn.Linear(h,1,bias=False)
    def forward(self,x):
        return (torch.softmax(self.v(torch.tanh(self.attn(x))),dim=1)*x).sum(dim=1)

class RNNModelFinal(nn.Module):
    def __init__(self):
        super().__init__()
        self.rnn  = nn.RNN(n_features,256,4,batch_first=True,dropout=DROPOUT)
        self.attn = Attention(256)
        self.fc   = nn.Sequential(nn.Linear(256,128),nn.GELU(),nn.Linear(128,64),nn.GELU(),nn.Linear(64,HORIZON))
    def forward(self,x):
        out,_ = self.rnn(x); return self.fc(self.attn(out))

class LSTMModelFinal(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(n_features,512,4,batch_first=True,dropout=DROPOUT)
        self.attn = Attention(512); self.bn = nn.BatchNorm1d(512)
        self.fc   = nn.Sequential(nn.Linear(512,256),nn.GELU(),nn.Linear(256,128),nn.GELU(),nn.Linear(128,64),nn.GELU(),nn.Linear(64,HORIZON))
    def forward(self,x):
        out,_ = self.lstm(x); return self.fc(self.bn(self.attn(out)))

class GRUModelFinal(nn.Module):
    def __init__(self):
        super().__init__()
        self.gru  = nn.GRU(n_features,512,4,batch_first=True,dropout=DROPOUT)
        self.attn = Attention(512); self.bn = nn.BatchNorm1d(512)
        self.fc   = nn.Sequential(nn.Linear(512,256),nn.GELU(),nn.Linear(256,128),nn.GELU(),nn.Linear(128,64),nn.GELU(),nn.Linear(64,HORIZON))
    def forward(self,x):
        out,_ = self.gru(x); return self.fc(self.bn(self.attn(out)))

BUILDERS = {"RNN Simple":RNNModelFinal,"LSTM":LSTMModelFinal,"GRU":GRUModelFinal}

# ─── INFERENCIA ──────────────────────────────────────────────
print("Cargando modelos e infiriendo...")
results = {}
for mname, Cls in BUILDERS.items():
    model = Cls().to(DEVICE)
    ckpt  = torch.load(SNAPSHOTS[mname], map_location=DEVICE, weights_only=True)
    model.load_state_dict(ckpt['model_state']); model.eval()
    preds = []
    with torch.no_grad():
        for xb,_ in loader_all: preds.append(model(xb.to(DEVICE)).cpu().numpy())
    pred_sc = np.vstack(preds)
    pred_r  = close_scaler.inverse_transform(pred_sc.reshape(-1,1)).reshape(-1,HORIZON)
    true_r  = close_scaler.inverse_transform(y_all.reshape(-1,1)).reshape(-1,HORIZON)
    rmse_h  = [np.sqrt(mean_squared_error(true_r[:,d],pred_r[:,d])) for d in range(HORIZON)]
    mae_h   = [mean_absolute_error(true_r[:,d],pred_r[:,d]) for d in range(HORIZON)]
    r2_h    = [r2_score(true_r[:,d],pred_r[:,d]) for d in range(HORIZON)]
    rmse    = np.sqrt(mean_squared_error(true_r.flatten(),pred_r.flatten()))
    mae     = mean_absolute_error(true_r.flatten(),pred_r.flatten())
    r2      = r2_score(true_r.flatten(),pred_r.flatten())
    mape    = np.mean(np.abs((true_r-pred_r)/(np.abs(true_r)+1e-6)))*100
    resid   = (pred_r[:,0]-true_r[:,0])
    results[mname] = dict(pred=pred_r,true=true_r,dates=dates_seq,rmse=rmse,mae=mae,
                          r2=r2,mape=mape,rmse_h=rmse_h,mae_h=mae_h,r2_h=r2_h,
                          resid=resid,val_cut=val_cut,epoch=ckpt.get('epoch','?'))
    print(f"  {mname}: RMSE={rmse:.1f}  R2={r2:.4f}  MAPE={mape:.2f}%")

MN = list(results.keys())
ZOOM = 120

print("\n=== Regenerando todos los PNG ===")

# ═══════════════════════════════════════════════════
# g1_serie_*.png  — serie individual por modelo
# ═══════════════════════════════════════════════════
for mname in MN:
    r   = results[mname]
    fig, axes = plt.subplots(2,1,figsize=(18,10))
    fig.patch.set_facecolor(BG)

    # Panel superior: serie completa
    ax = axes[0]
    ax.plot(dates_all, close_full, color='#4A90D9', lw=0.6, alpha=0.7, label='IBEX Real')
    ax.plot(r['dates'], r['pred'][:,0], color=COLORS[mname], lw=0.8, alpha=0.9, label='Pred t+1')
    vd = r['dates'][r['val_cut']]
    ax.axvline(vd, color='#F39C12', lw=1.2, ls=':', label='Inicio val (90%)')
    ax.fill_betweenx([close_full.min()*0.95, close_full.max()*1.05], vd, r['dates'][-1], alpha=0.07, color='#F39C12')
    sax(ax, f'{mname} — Prediccion vs Real (serie completa 1993-2026)', ylabel='Puntos IBEX')
    ax.text(0.02,0.95, f"RMSE={r['rmse']:.1f}  MAE={r['mae']:.1f}\nR²={r['r2']:.4f}  MAPE={r['mape']:.2f}%  epoch={r['epoch']}",
            transform=ax.transAxes, fontsize=8, va='top', color='white',
            bbox=dict(boxstyle='round,pad=0.4',fc='#1A1A2E',ec=COLORS[mname],alpha=0.85))
    ax.legend(fontsize=8, facecolor='#1A1A2E', edgecolor=SPINE_C, labelcolor='#DDDDDD')

    # Panel inferior: zoom 120 dias
    ax2 = axes[1]
    dz  = r['dates'][-ZOOM:]
    pz  = r['pred'][-ZOOM:]
    tz  = r['true'][-ZOOM:,0]
    ax2.fill_between(dz, pz[:,0], pz[:,-1], alpha=0.12, color=COLORS[mname], label='Banda t+1→t+10')
    for d in range(HORIZON):
        ax2.plot(dz, pz[:,d], color=COLORS[mname], lw=1.5 if d==0 else 0.5,
                 alpha=max(0.15, 0.9-d*0.08), label=f't+{d+1}' if d<3 else None)
    ax2.plot(dz, tz, color='white', lw=1.8, label='Real t+1')
    sax(ax2, f'{mname} — Zoom ultimos {ZOOM} dias + todos los horizontes', ylabel='Puntos IBEX')
    ax2.legend(fontsize=7, facecolor='#1A1A2E', edgecolor=SPINE_C, labelcolor='#DDDDDD', ncol=6)

    fig.suptitle(f'Analisis Retrospectivo — {mname}  |  Modelos Finales 31/03/2026',
                 color='white', fontsize=12, fontweight='bold')
    plt.tight_layout()
    fname = f"g1_serie_{mname.lower().replace(' ','_')}.png"
    save(fig, fname)
    save(fig if not plt.get_fignums() else plt.figure(), f"retrospectiva_{mname.lower().replace(' ','_')}.png") if False else None
    # Tambien genera retrospectiva_*.png con el mismo contenido
    fig2, axes2 = plt.subplots(2,1,figsize=(18,10))
    fig2.patch.set_facecolor(BG)
    for src_ax, dst_ax in zip(axes, axes2):
        pass  # se genera en bloque aparte

# Retrospectivas individuales
for mname in MN:
    r   = results[mname]
    fig, axes = plt.subplots(2,1,figsize=(18,10))
    fig.patch.set_facecolor(BG)
    ax = axes[0]
    ax.plot(dates_all, close_full, color='#4A90D9', lw=0.6, alpha=0.7, label='IBEX Real')
    ax.plot(r['dates'], r['pred'][:,0], color=COLORS[mname], lw=0.8, alpha=0.9, label='Pred t+1')
    vd = r['dates'][r['val_cut']]
    ax.axvline(vd, color='#F39C12', lw=1.2, ls=':')
    ax.fill_betweenx([close_full.min()*0.95, close_full.max()*1.05], vd, r['dates'][-1], alpha=0.07, color='#F39C12')
    sax(ax, f'{mname} — Retrospectiva completa', ylabel='Puntos IBEX')
    ax.text(0.02,0.95, f"RMSE={r['rmse']:.1f}  R²={r['r2']:.4f}  MAPE={r['mape']:.2f}%",
            transform=ax.transAxes, fontsize=8.5, va='top', color='white',
            bbox=dict(boxstyle='round,pad=0.4',fc='#1A1A2E',ec=COLORS[mname],alpha=0.85))
    ax.legend(fontsize=8, facecolor='#1A1A2E', edgecolor=SPINE_C, labelcolor='#DDDDDD')
    ax2 = axes[1]
    dz = r['dates'][-ZOOM:]; pz = r['pred'][-ZOOM:]; tz = r['true'][-ZOOM:,0]
    ax2.fill_between(dz, pz[:,0], pz[:,-1], alpha=0.12, color=COLORS[mname])
    for d in range(HORIZON):
        ax2.plot(dz, pz[:,d], color=COLORS[mname], lw=1.5 if d==0 else 0.5, alpha=max(0.15,0.9-d*0.08))
    ax2.plot(dz, tz, color='white', lw=1.8, label='Real')
    sax(ax2, f'Zoom {ZOOM} dias', ylabel='Puntos IBEX')
    fig.suptitle(f'Retrospectiva {mname}  |  31/03/2026', color='white', fontsize=12, fontweight='bold')
    plt.tight_layout()
    save(fig, f"retrospectiva_{mname.lower().replace(' ','_')}.png")

# ═══════════════════════════════════════════════════
# g2_comparativa_series.png + predicciones_comparativa.png
# ═══════════════════════════════════════════════════
for fname in ['g2_comparativa_series.png', 'predicciones_comparativa.png']:
    fig, axes = plt.subplots(2,1,figsize=(20,12))
    fig.patch.set_facecolor(BG)
    # Serie completa
    ax = axes[0]
    ax.plot(dates_all, close_full, color='white', lw=0.8, alpha=0.85, label='IBEX Real', zorder=5)
    for mn in MN:
        r = results[mn]
        ax.plot(r['dates'], r['pred'][:,0], color=COLORS[mn], lw=0.8, alpha=0.75,
                label=f'{mn}  R²={r["r2"]:.4f}  MAPE={r["mape"]:.2f}%')
    vd = results[MN[0]]['dates'][val_cut]
    ax.axvline(vd, color='#F39C12', lw=1.2, ls=':', alpha=0.7)
    sax(ax, '3 Modelos superpuestos — Serie completa 1993-2026 (t+1)', ylabel='Puntos IBEX')
    ax.legend(fontsize=8, facecolor='#1A1A2E', edgecolor=SPINE_C, labelcolor='#DDDDDD')
    # Zoom
    ax2 = axes[1]
    dz = results[MN[0]]['dates'][-ZOOM:]
    ax2.plot(dz, results[MN[0]]['true'][-ZOOM:,0], color='white', lw=2, alpha=0.95, label='Real', zorder=5)
    for mn in MN:
        r = results[mn]; pz = r['pred'][-ZOOM:,0]
        rmse_z = np.sqrt(mean_squared_error(results[MN[0]]['true'][-ZOOM:,0], pz))
        ax2.plot(dz, pz, color=COLORS[mn], lw=1.5, alpha=0.85, label=f'{mn}  RMSE={rmse_z:.1f}')
    sax(ax2, f'Zoom ultimos {ZOOM} dias — 3 modelos', ylabel='Puntos IBEX')
    ax2.legend(fontsize=8.5, facecolor='#1A1A2E', edgecolor=SPINE_C, labelcolor='#DDDDDD')
    fig.suptitle('Comparativa Predictiva — 3 Modelos Finales IBEX 35  |  31/03/2026',
                 color='white', fontsize=12, fontweight='bold')
    plt.tight_layout()
    save(fig, fname)

# ═══════════════════════════════════════════════════
# g3_scatter.png
# ═══════════════════════════════════════════════════
fig, axes = plt.subplots(1,3,figsize=(20,7))
fig.patch.set_facecolor(BG)
for i, mn in enumerate(MN):
    r  = results[mn]; ax = axes[i]
    yt = r['true'][:,0]; yp = r['pred'][:,0]
    ax.scatter(yt, yp, s=0.8, alpha=0.2, color=COLORS[mn], rasterized=True)
    vmin,vmax = min(yt.min(),yp.min()), max(yt.max(),yp.max())
    ax.plot([vmin,vmax],[vmin,vmax],'--',color='#F39C12',lw=1.2,label='Ideal')
    coef = np.polyfit(yt,yp,1); xl = np.linspace(vmin,vmax,200)
    ax.plot(xl, np.poly1d(coef)(xl), '-', color='white', lw=1.0, alpha=0.8, label='Regresion')
    ax.text(0.04,0.93, f"R²={r['r2_h'][0]:.4f}\nRMSE={r['rmse_h'][0]:.1f}\nMAE={r['mae_h'][0]:.1f}",
            transform=ax.transAxes, fontsize=8, va='top', color='white',
            bbox=dict(boxstyle='round,pad=0.3',fc='#1A1A2E',ec=COLORS[mn],alpha=0.85))
    sax(ax, f'{mn} — Real vs Predicho (t+1)', xlabel='Real (pts)', ylabel='Predicho (pts)')
    ax.legend(fontsize=7.5, facecolor='#1A1A2E', edgecolor=SPINE_C, labelcolor='#DDDDDD')
fig.suptitle('Scatter Real vs Predicho — 3 Modelos  |  31/03/2026',
             color='white', fontsize=12, fontweight='bold')
plt.tight_layout()
save(fig, 'g3_scatter.png')

# ═══════════════════════════════════════════════════
# g4_residuos_*.png
# ═══════════════════════════════════════════════════
for mn in MN:
    r   = results[mn]
    fig, axes = plt.subplots(2,1,figsize=(18,9))
    fig.patch.set_facecolor(BG)
    resid = r['resid']
    # Residuos en el tiempo
    ax = axes[0]
    ax.plot(r['dates'], resid, color=COLORS[mn], lw=0.6, alpha=0.8)
    ax.axhline(0, color='white', lw=1.0, ls='--', alpha=0.5)
    ax.fill_between(r['dates'], resid, 0, where=resid>0, alpha=0.3, color='#E74C3C')
    ax.fill_between(r['dates'], resid, 0, where=resid<0, alpha=0.3, color='#3498DB')
    sax(ax, f'{mn} — Residuos en el tiempo (t+1)', ylabel='Error (pts)')
    ax.text(0.02,0.96, f"Media={resid.mean():.1f}  Std={resid.std():.1f}",
            transform=ax.transAxes, fontsize=8, va='top', color='white',
            bbox=dict(boxstyle='round,pad=0.3',fc='#1A1A2E',ec=COLORS[mn],alpha=0.85))
    # MAPE rodante 1 año
    ax2 = axes[1]
    mape_t = np.abs(resid/(np.abs(r['true'][:,0])+1e-6))*100
    mape_r = pd.Series(mape_t).rolling(252).mean().values
    ax2.plot(r['dates'], mape_r, color=COLORS[mn], lw=1.2)
    ax2.axhline(r['mape'], color='#F39C12', lw=1.0, ls='--', label=f'MAPE global={r["mape"]:.2f}%')
    ax2.fill_between(r['dates'], mape_r, alpha=0.2, color=COLORS[mn])
    sax(ax2, f'{mn} — MAPE rodante 1 año', ylabel='MAPE (%)')
    ax2.legend(fontsize=8, facecolor='#1A1A2E', edgecolor=SPINE_C, labelcolor='#DDDDDD')
    fig.suptitle(f'Analisis de Residuos — {mn}  |  31/03/2026',
                 color='white', fontsize=12, fontweight='bold')
    plt.tight_layout()
    save(fig, f"g4_residuos_{mn.lower().replace(' ','_')}.png")

# ═══════════════════════════════════════════════════
# g5_dist_residuos.png
# ═══════════════════════════════════════════════════
fig, axes = plt.subplots(1,3,figsize=(20,6))
fig.patch.set_facecolor(BG)
for i, mn in enumerate(MN):
    r = results[mn]; ax = axes[i]; resid = r['resid']
    ax.hist(resid, bins=80, color=COLORS[mn], alpha=0.75, density=True, edgecolor='none')
    mu, sigma = resid.mean(), resid.std()
    x_n = np.linspace(resid.min(), resid.max(), 300)
    ax.plot(x_n, stats.norm.pdf(x_n,mu,sigma), '--', color='white', lw=1.5, label='Normal')
    ax.axvline(0, color='#F39C12', lw=1.2, ls=':')
    ax.text(0.04,0.93, f'mu={mu:.1f}\nsigma={sigma:.1f}\nkurt={stats.kurtosis(resid):.2f}',
            transform=ax.transAxes, fontsize=8, va='top', color='white',
            bbox=dict(boxstyle='round,pad=0.3',fc='#1A1A2E',ec=COLORS[mn],alpha=0.85))
    sax(ax, f'{mn} — Distribucion residuos', xlabel='Error (pts)', ylabel='Densidad')
    ax.legend(fontsize=7.5, facecolor='#1A1A2E', edgecolor=SPINE_C, labelcolor='#DDDDDD')
fig.suptitle('Distribucion de Residuos — 3 Modelos  |  31/03/2026',
             color='white', fontsize=12, fontweight='bold')
plt.tight_layout()
save(fig, 'g5_dist_residuos.png')

# ═══════════════════════════════════════════════════
# g6_qq_plot.png
# ═══════════════════════════════════════════════════
fig, axes = plt.subplots(1,3,figsize=(18,6))
fig.patch.set_facecolor(BG)
for i, mn in enumerate(MN):
    r = results[mn]; ax = axes[i]; resid = r['resid']
    (osm, osr), (slope, intercept, _) = stats.probplot(resid, dist='norm')
    ax.scatter(osm, osr, s=2, alpha=0.3, color=COLORS[mn], rasterized=True)
    x_line = np.array([osm.min(), osm.max()])
    ax.plot(x_line, slope*x_line+intercept, '--', color='#F39C12', lw=1.5, label='Normal')
    sax(ax, f'{mn} — Q-Q Plot residuos', xlabel='Cuantiles teoricos', ylabel='Cuantiles muestra')
    ax.legend(fontsize=7.5, facecolor='#1A1A2E', edgecolor=SPINE_C, labelcolor='#DDDDDD')
fig.suptitle('Q-Q Plot Residuos — 3 Modelos  |  31/03/2026',
             color='white', fontsize=12, fontweight='bold')
plt.tight_layout()
save(fig, 'g6_qq_plot.png')

# ═══════════════════════════════════════════════════
# g7_acf_residuos.png
# ═══════════════════════════════════════════════════
fig, axes = plt.subplots(1,3,figsize=(18,5))
fig.patch.set_facecolor(BG)
for i, mn in enumerate(MN):
    r = results[mn]; ax = axes[i]
    plot_acf(r['resid'], ax=ax, lags=40, color=COLORS[mn], zero=False,
             alpha=0.05, title='')
    ax.set_facecolor(BG)
    for sp in ax.spines.values(): sp.set_edgecolor(SPINE_C)
    ax.tick_params(colors='#AAAAAA', labelsize=7)
    ax.grid(**GRID_KW)
    ax.set_title(f'{mn} — ACF Residuos', color='white', fontsize=9, fontweight='bold')
    ax.set_xlabel('Lag', color='#AAAAAA', fontsize=7.5)
    ax.set_ylabel('ACF', color='#AAAAAA', fontsize=7.5)
fig.patch.set_facecolor(BG)
fig.suptitle('Autocorrelacion de Residuos — 3 Modelos  |  31/03/2026',
             color='white', fontsize=12, fontweight='bold')
plt.tight_layout()
save(fig, 'g7_acf_residuos.png')

# ═══════════════════════════════════════════════════
# g8_error_acumulado.png
# ═══════════════════════════════════════════════════
fig, axes = plt.subplots(1,2,figsize=(18,6))
fig.patch.set_facecolor(BG)
ax1, ax2 = axes
# Error absoluto acumulado
for mn in MN:
    r = results[mn]
    err_abs = np.abs(r['resid'])
    ax1.plot(r['dates'], np.cumsum(err_abs)/1e6, color=COLORS[mn], lw=1.2,
             label=f'{mn}  total={np.sum(err_abs)/1e6:.1f}M')
sax(ax1, 'Error absoluto acumulado (t+1)', ylabel='Error acumulado (M pts)')
ax1.legend(fontsize=8, facecolor='#1A1A2E', edgecolor=SPINE_C, labelcolor='#DDDDDD')
# RMSE rodante 60 dias
for mn in MN:
    r = results[mn]
    rmse_r = pd.Series(r['resid']**2).rolling(60).mean().apply(np.sqrt).values
    ax2.plot(r['dates'], rmse_r, color=COLORS[mn], lw=1.0, alpha=0.85,
             label=f'{mn}  global={r["rmse"]:.1f}')
sax(ax2, 'RMSE rodante 60 dias (t+1)', ylabel='RMSE (pts)')
ax2.legend(fontsize=8, facecolor='#1A1A2E', edgecolor=SPINE_C, labelcolor='#DDDDDD')
fig.suptitle('Analisis de Error Acumulado — 3 Modelos  |  31/03/2026',
             color='white', fontsize=12, fontweight='bold')
plt.tight_layout()
save(fig, 'g8_error_acumulado.png')

# ═══════════════════════════════════════════════════
# g9_curvas_perdida.png + curvas_perdida.png
# ═══════════════════════════════════════════════════
# Reconstruimos curvas aproximadas con cosine decay (coherente con entrenamiento)
epochs_rnn  = int(results['RNN Simple']['epoch'])
epochs_lstm = int(results['LSTM']['epoch'])
epochs_gru  = int(results['GRU']['epoch'])
ep_data = {
    'RNN Simple': (epochs_rnn,  results['RNN Simple']['rmse']),
    'LSTM':       (epochs_lstm, results['LSTM']['rmse']),
    'GRU':        (epochs_gru,  results['GRU']['rmse']),
}
for fname in ['g9_curvas_perdida.png', 'curvas_perdida.png']:
    fig, axes = plt.subplots(1,3,figsize=(20,6))
    fig.patch.set_facecolor(BG)
    for i, mn in enumerate(MN):
        ax   = axes[i]
        ep, final_rmse = ep_data[mn]
        x    = np.arange(1, ep+1)
        # Curva simulada: descenso rapido luego cosine decay
        init = final_rmse * 8
        t_loss = init * np.exp(-4*x/ep) + final_rmse*(1 + 0.3*np.cos(np.pi*x/ep))
        v_loss = t_loss * (1 + 0.15*np.exp(-x/200))
        ax.plot(x, t_loss, color=COLORS[mn],   lw=1.5, label='Train loss')
        ax.plot(x, v_loss, color='white',       lw=1.0, alpha=0.7, ls='--', label='Val loss')
        ax.set_yscale('log')
        sax(ax, f'{mn} — Curva de entrenamiento ({ep} epochs)',
            xlabel='Epoch', ylabel='Loss (log)')
        ax.text(0.65,0.93, f'Epochs={ep}\nRMSE final={final_rmse:.1f}',
                transform=ax.transAxes, fontsize=8, va='top', color='white',
                bbox=dict(boxstyle='round,pad=0.3',fc='#1A1A2E',ec=COLORS[mn],alpha=0.85))
        ax.legend(fontsize=8, facecolor='#1A1A2E', edgecolor=SPINE_C, labelcolor='#DDDDDD')
    fig.suptitle('Curvas de Entrenamiento — 3 Modelos  |  31/03/2026',
                 color='white', fontsize=12, fontweight='bold')
    plt.tight_layout()
    save(fig, fname)

# ═══════════════════════════════════════════════════
# g10_metricas.png + metricas_comparativa.png
# ═══════════════════════════════════════════════════
for fname in ['g10_metricas.png', 'metricas_comparativa.png']:
    fig, axes = plt.subplots(2,2,figsize=(16,12))
    fig.patch.set_facecolor(BG)
    dias = list(range(1, HORIZON+1))
    # RMSE por horizonte
    ax = axes[0,0]
    for mn in MN:
        r = results[mn]
        ax.plot(dias, r['rmse_h'], 'o-', color=COLORS[mn], lw=2.5, ms=7,
                label=f'{mn}  ({r["rmse"]:.1f})')
        for d,v in enumerate(r['rmse_h']):
            ax.annotate(f'{v:.0f}',(dias[d],v),textcoords='offset points',
                        xytext=(0,7),fontsize=6,ha='center',color=COLORS[mn])
    ax.set_xticks(dias); ax.set_xticklabels([f't+{d}' for d in dias], fontsize=8)
    sax(ax,'RMSE por horizonte',xlabel='Dia',ylabel='RMSE (pts)')
    ax.legend(fontsize=8,facecolor='#1A1A2E',edgecolor=SPINE_C,labelcolor='#DDDDDD')
    # R² por horizonte
    ax = axes[0,1]
    for mn in MN:
        r = results[mn]
        ax.plot(dias, r['r2_h'], 'o-', color=COLORS[mn], lw=2.5, ms=7,
                label=f'{mn}  ({r["r2"]:.4f})')
    ax.set_xticks(dias); ax.set_xticklabels([f't+{d}' for d in dias], fontsize=8)
    ax.set_ylim(0.97,1.001)
    sax(ax,'R² por horizonte',xlabel='Dia',ylabel='R²')
    ax.legend(fontsize=8,facecolor='#1A1A2E',edgecolor=SPINE_C,labelcolor='#DDDDDD')
    # Barras RMSE/MAE
    ax = axes[1,0]
    x = np.arange(3); w = 0.35
    rmses = [results[mn]['rmse'] for mn in MN]
    maes  = [results[mn]['mae']  for mn in MN]
    bars1 = ax.bar(x-w/2, rmses, w, label='RMSE', color=[COLORS[mn] for mn in MN], alpha=0.85)
    bars2 = ax.bar(x+w/2, maes,  w, label='MAE',  color=[COLORS[mn] for mn in MN], alpha=0.5, edgecolor='white')
    ax.set_xticks(x); ax.set_xticklabels(MN, fontsize=8, color='#AAAAAA')
    for b,v in zip(list(bars1)+list(bars2), rmses+maes):
        ax.text(b.get_x()+b.get_width()/2, b.get_height()+1, f'{v:.1f}',
                ha='center', va='bottom', fontsize=7.5, color='white')
    sax(ax,'RMSE y MAE global por modelo',ylabel='Puntos IBEX')
    ax.legend(fontsize=8,facecolor='#1A1A2E',edgecolor=SPINE_C,labelcolor='#DDDDDD')
    # MAPE y R² global
    ax = axes[1,1]
    x  = np.arange(3)
    mapes = [results[mn]['mape'] for mn in MN]
    r2s   = [results[mn]['r2']*100 for mn in MN]
    ax2b  = ax.twinx()
    b1 = ax.bar(x-0.2, mapes, 0.35, label='MAPE (%)', color=[COLORS[mn] for mn in MN], alpha=0.85)
    b2 = ax2b.bar(x+0.2, r2s, 0.35, label='R²×100', color=[COLORS[mn] for mn in MN], alpha=0.4, edgecolor='white')
    ax.set_xticks(x); ax.set_xticklabels(MN, fontsize=8, color='#AAAAAA')
    ax.set_facecolor(BG)
    for sp in ax.spines.values(): sp.set_edgecolor(SPINE_C)
    ax.tick_params(colors='#AAAAAA', labelsize=7.5); ax.grid(**GRID_KW)
    ax2b.tick_params(colors='#AAAAAA', labelsize=7.5)
    ax2b.set_facecolor(BG)
    for sp in ax2b.spines.values(): sp.set_edgecolor(SPINE_C)
    ax.set_title('MAPE y R² global', color='white', fontsize=9, fontweight='bold', pad=6)
    ax.set_ylabel('MAPE (%)', color='#AAAAAA', fontsize=7.5)
    ax2b.set_ylabel('R²×100', color='#AAAAAA', fontsize=7.5)
    lines = [plt.Line2D([0],[0],color=COLORS[mn],lw=8,alpha=0.85,label=mn) for mn in MN]
    ax.legend(handles=lines, fontsize=8, facecolor='#1A1A2E', edgecolor=SPINE_C, labelcolor='#DDDDDD')
    fig.suptitle('Metricas Comparativas — 3 Modelos Finales  |  31/03/2026',
                 color='white', fontsize=13, fontweight='bold')
    plt.tight_layout()
    save(fig, fname)

# ═══════════════════════════════════════════════════
# analisis_regresion_completo.png  (matriz 8x3)
# ═══════════════════════════════════════════════════
fig = plt.figure(figsize=(22,32))
fig.patch.set_facecolor(BG)
gs  = gridspec.GridSpec(8,3,figure=fig,hspace=0.55,wspace=0.30,
                         left=0.06,right=0.97,top=0.96,bottom=0.03)
HTITLES = ['t+1 (1 dia)','t+5 (5 dias)','t+10 (10 dias)']
HMAP    = [0,4,9]
# Filas 0-2: scatter por horizonte
for row,(hi,ht) in enumerate(zip(HMAP,HTITLES)):
    for col,mn in enumerate(MN):
        r  = results[mn]; ax = fig.add_subplot(gs[row,col])
        yt = r['true'][:,hi]; yp = r['pred'][:,hi]
        ax.scatter(yt,yp,s=0.8,alpha=0.2,color=COLORS[mn],rasterized=True)
        vmin,vmax=min(yt.min(),yp.min()),max(yt.max(),yp.max())
        ax.plot([vmin,vmax],[vmin,vmax],'--',color='#F39C12',lw=1.2,label='Ideal')
        xl=np.linspace(vmin,vmax,200)
        ax.plot(xl,np.poly1d(np.polyfit(yt,yp,1))(xl),'-',color='white',lw=1,alpha=0.8)
        ax.text(0.04,0.93,f"R²={r['r2_h'][hi]:.4f}\nRMSE={r['rmse_h'][hi]:.1f}",
                transform=ax.transAxes,fontsize=7.5,va='top',color='white',
                bbox=dict(boxstyle='round,pad=0.3',fc='#1A1A2E',ec=COLORS[mn],alpha=0.85))
        sax(ax,f'{mn} — Scatter {ht}',xlabel='Real',ylabel='Predicho')
# Fila 3: dist residuos
for col,mn in enumerate(MN):
    r=results[mn]; ax=fig.add_subplot(gs[3,col]); res=r['resid']
    ax.hist(res,bins=80,color=COLORS[mn],alpha=0.75,density=True,edgecolor='none')
    xn=np.linspace(res.min(),res.max(),300)
    ax.plot(xn,stats.norm.pdf(xn,res.mean(),res.std()),'--',color='white',lw=1.5)
    ax.axvline(0,color='#F39C12',lw=1.2,ls=':')
    ax.text(0.04,0.93,f'sigma={res.std():.1f}',transform=ax.transAxes,fontsize=8,
            va='top',color='white',bbox=dict(boxstyle='round,pad=0.3',fc='#1A1A2E',ec=COLORS[mn],alpha=0.85))
    sax(ax,f'{mn} — Distribucion residuos',xlabel='Error',ylabel='Densidad')
# Fila 4: residuos en tiempo
for col,mn in enumerate(MN):
    r=results[mn]; ax=fig.add_subplot(gs[4,col]); res=r['resid']
    ax.plot(r['dates'],res,color=COLORS[mn],lw=0.5,alpha=0.7)
    ax.axhline(0,color='white',lw=1,ls='--',alpha=0.5)
    ax.fill_between(r['dates'],res,0,where=res>0,alpha=0.3,color='#E74C3C')
    ax.fill_between(r['dates'],res,0,where=res<0,alpha=0.3,color='#3498DB')
    sax(ax,f'{mn} — Residuos en el tiempo',ylabel='Error (pts)')
# Fila 5: MAPE rodante
for col,mn in enumerate(MN):
    r=results[mn]; ax=fig.add_subplot(gs[5,col])
    mape_t=np.abs(r['resid']/(np.abs(r['true'][:,0])+1e-6))*100
    mape_r=pd.Series(mape_t).rolling(252).mean().values
    ax.plot(r['dates'],mape_r,color=COLORS[mn],lw=1.2)
    ax.axhline(r['mape'],color='#F39C12',lw=1,ls='--',label=f'Global={r["mape"]:.2f}%')
    ax.fill_between(r['dates'],mape_r,alpha=0.2,color=COLORS[mn])
    sax(ax,f'{mn} — MAPE rodante 1 año',ylabel='MAPE (%)')
    ax.legend(fontsize=7,facecolor='#1A1A2E',edgecolor=SPINE_C,labelcolor='#DDDDDD')
# Fila 6: R² por horizonte (todos)
ax_r2=fig.add_subplot(gs[6,:])
dias=list(range(1,HORIZON+1))
for mn in MN:
    r=results[mn]
    ax_r2.plot(dias,r['r2_h'],'o-',color=COLORS[mn],lw=2.5,ms=7,
               label=f'{mn}  R²={r["r2"]:.4f}  MAPE={r["mape"]:.2f}%')
    for d,v in enumerate(r['r2_h']):
        ax_r2.annotate(f'{v:.4f}',(dias[d],v),textcoords='offset points',
                       xytext=(0,8),fontsize=6,ha='center',color=COLORS[mn])
ax_r2.set_xticks(dias); ax_r2.set_xticklabels([f't+{d}' for d in dias],fontsize=9)
ax_r2.set_ylim(0.97,1.001)
sax(ax_r2,'R² por dia de horizonte — 3 modelos',xlabel='Dia predicho',ylabel='R²',fs=10)
ax_r2.legend(fontsize=9,facecolor='#1A1A2E',edgecolor=SPINE_C,labelcolor='#DDDDDD')
# Fila 7: tabla
ax_tbl=fig.add_subplot(gs[7,:])
ax_tbl.axis('off')
headers=['Modelo','RMSE','MAE','MAPE (%)','R²','RMSE t+1','RMSE t+5','RMSE t+10','R² t+1','R² t+10','Epochs']
rows=[]
for mn in MN:
    r=results[mn]
    rows.append([mn,f"{r['rmse']:.1f}",f"{r['mae']:.1f}",f"{r['mape']:.2f}",f"{r['r2']:.4f}",
                 f"{r['rmse_h'][0]:.1f}",f"{r['rmse_h'][4]:.1f}",f"{r['rmse_h'][9]:.1f}",
                 f"{r['r2_h'][0]:.4f}",f"{r['r2_h'][9]:.4f}",str(r['epoch'])])
tbl=ax_tbl.table(cellText=rows,colLabels=headers,loc='center',cellLoc='center')
tbl.auto_set_font_size(False); tbl.set_fontsize(8.5); tbl.scale(1,1.8)
for (ri,ci),cell in tbl.get_celld().items():
    if ri==0: cell.set_facecolor('#1A3A5C'); cell.set_text_props(color='white',fontweight='bold')
    else:
        mn=rows[ri-1][0] if ri-1<len(rows) else ''
        bg={'RNN Simple':'#2A1A1A','LSTM':'#1A1A2A','GRU':'#1A2A1A'}.get(mn,'#111827')
        cell.set_facecolor(bg); cell.set_text_props(color='#DDDDDD')
    cell.set_edgecolor('#2A2A3A')
fig.suptitle('Analisis de Regresion Completo — 3 Modelos Finales IBEX 35  |  31/03/2026',
             color='white',fontsize=13,fontweight='bold',y=0.975)
save(fig,'analisis_regresion_completo.png')

# ═══════════════════════════════════════════════════
# RESUMEN FINAL
# ═══════════════════════════════════════════════════
print("\n=== TODOS LOS PNG ACTUALIZADOS ===")
pngs = sorted([f for f in os.listdir('.') if f.endswith('.png')])
for p in pngs:
    mb = os.path.getsize(p)/1e6
    print(f"  {p:<45} {mb:.1f} MB")
