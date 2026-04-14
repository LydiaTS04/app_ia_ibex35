# -*- coding: utf-8 -*-
"""
plot_comparativo.py -- Graficas comparativas completas (modelos finales)
=========================================================================
Genera DOS figuras:
  1. grafica_comparativa_separada.png  -- 3 filas (una por modelo) con serie
                                          completa + zoom 120 dias
  2. grafica_comparativa_juntos.png    -- los 3 modelos superpuestos en el
                                          mismo eje + metricas
  3. analisis_regresion_final.png      -- matriz completa de regresion
                                          actualizada con los nuevos modelos
"""

import os, warnings, pickle
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyBboxPatch
import yfinance as yf
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import MinMaxScaler

# ─────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────
DEVICE     = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
SEQ_LEN    = 90
HORIZON    = 10
DROPOUT    = 0.05
BATCH      = 512
START_DATE = '1993-07-12'
END_TRAIN  = '2026-04-01'

SNAPSHOTS = {
    "RNN Simple": "snap_rnn_31mar.pt",
    "LSTM":       "snap_lstm_31mar.pt",
    "GRU":        "snap_gru_31mar.pt",
}
COLORS  = {"RNN Simple": "#E74C3C", "LSTM": "#3498DB", "GRU": "#2ECC71"}
BG      = '#0D1117'
GRID_KW = dict(color='#2A2A3A', linewidth=0.5, linestyle='--')
SPINE_C = '#3A3A4A'

def style_ax(ax, title='', xlabel='', ylabel='', fs=9):
    ax.set_facecolor(BG)
    for sp in ax.spines.values():
        sp.set_edgecolor(SPINE_C)
    ax.tick_params(colors='#AAAAAA', labelsize=7.5)
    ax.grid(**GRID_KW)
    if title:  ax.set_title(title,  color='white',   fontsize=fs, fontweight='bold', pad=6)
    if xlabel: ax.set_xlabel(xlabel, color='#AAAAAA', fontsize=7.5)
    if ylabel: ax.set_ylabel(ylabel, color='#AAAAAA', fontsize=7.5)

# ─────────────────────────────────────────────────────────────
# 1. DATOS
# ─────────────────────────────────────────────────────────────
print("Descargando datos...")
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
df['RSI']   = 100 - (100 / (1 + gain / (loss + 1e-9)))
df['Mom10'] = df['Close'] - df['Close'].shift(10)
df.dropna(inplace=True)

feature_cols = list(df.columns)
n_features   = len(feature_cols)
close_idx    = feature_cols.index('Close')
dates_all    = df.index
close_full   = df['Close'].values

print(f"  {df.index[0].date()} -> {df.index[-1].date()}  ({len(df)} sesiones)")

with open('scaler_final.pkl','rb') as f:      scaler = pickle.load(f)
with open('close_scaler_final.pkl','rb') as f: close_scaler = pickle.load(f)
scaled = scaler.transform(df.values)

# ─────────────────────────────────────────────────────────────
# 2. SECUENCIAS
# ─────────────────────────────────────────────────────────────
def make_seq(arr, sl, hz):
    X, y = [], []
    for i in range(sl, len(arr) - hz + 1):
        X.append(arr[i-sl:i])
        y.append(arr[i:i+hz, close_idx])
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)

X_all, y_all = make_seq(scaled, SEQ_LEN, HORIZON)
n      = len(X_all)
dates_seq = dates_all[SEQ_LEN: SEQ_LEN + n]
val_cut   = int(n * 0.90)
X_val, y_val = X_all[val_cut:], y_all[val_cut:]

loader_all = DataLoader(TensorDataset(torch.tensor(X_all), torch.tensor(y_all)),
                        batch_size=BATCH, shuffle=False)
loader_val = DataLoader(TensorDataset(torch.tensor(X_val), torch.tensor(y_val)),
                        batch_size=BATCH, shuffle=False)

# ─────────────────────────────────────────────────────────────
# 3. ARQUITECTURAS
# ─────────────────────────────────────────────────────────────
class Attention(nn.Module):
    def __init__(self, h):
        super().__init__()
        self.attn = nn.Linear(h, h)
        self.v    = nn.Linear(h, 1, bias=False)
    def forward(self, x):
        return (torch.softmax(self.v(torch.tanh(self.attn(x))), dim=1) * x).sum(dim=1)

class RNNModelFinal(nn.Module):
    def __init__(self):
        super().__init__()
        self.rnn  = nn.RNN(n_features, 256, 4, batch_first=True, dropout=DROPOUT)
        self.attn = Attention(256)
        self.fc   = nn.Sequential(nn.Linear(256,128), nn.GELU(),
                                   nn.Linear(128,64),  nn.GELU(), nn.Linear(64,HORIZON))
    def forward(self, x):
        out, _ = self.rnn(x); return self.fc(self.attn(out))

class LSTMModelFinal(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(n_features, 512, 4, batch_first=True, dropout=DROPOUT)
        self.attn = Attention(512)
        self.bn   = nn.BatchNorm1d(512)
        self.fc   = nn.Sequential(nn.Linear(512,256), nn.GELU(), nn.Linear(256,128),
                                   nn.GELU(), nn.Linear(128,64), nn.GELU(), nn.Linear(64,HORIZON))
    def forward(self, x):
        out, _ = self.lstm(x); return self.fc(self.bn(self.attn(out)))

class GRUModelFinal(nn.Module):
    def __init__(self):
        super().__init__()
        self.gru  = nn.GRU(n_features, 512, 4, batch_first=True, dropout=DROPOUT)
        self.attn = Attention(512)
        self.bn   = nn.BatchNorm1d(512)
        self.fc   = nn.Sequential(nn.Linear(512,256), nn.GELU(), nn.Linear(256,128),
                                   nn.GELU(), nn.Linear(128,64), nn.GELU(), nn.Linear(64,HORIZON))
    def forward(self, x):
        out, _ = self.gru(x); return self.fc(self.bn(self.attn(out)))

BUILDERS = {"RNN Simple": RNNModelFinal, "LSTM": LSTMModelFinal, "GRU": GRUModelFinal}

# ─────────────────────────────────────────────────────────────
# 4. INFERENCIA
# ─────────────────────────────────────────────────────────────
results = {}
for mname, Cls in BUILDERS.items():
    snap = SNAPSHOTS[mname]
    model = Cls().to(DEVICE)
    ckpt  = torch.load(snap, map_location=DEVICE, weights_only=True)
    model.load_state_dict(ckpt['model_state'])
    model.eval()

    preds = []
    with torch.no_grad():
        for xb, _ in loader_all:
            preds.append(model(xb.to(DEVICE)).cpu().numpy())
    pred_sc = np.vstack(preds)
    pred_r  = close_scaler.inverse_transform(pred_sc.reshape(-1,1)).reshape(-1,HORIZON)
    true_r  = close_scaler.inverse_transform(y_all.reshape(-1,1)).reshape(-1,HORIZON)

    rmse_h = [np.sqrt(mean_squared_error(true_r[:,d], pred_r[:,d])) for d in range(HORIZON)]
    mae_h  = [mean_absolute_error(true_r[:,d], pred_r[:,d]) for d in range(HORIZON)]
    r2_h   = [r2_score(true_r[:,d], pred_r[:,d]) for d in range(HORIZON)]
    rmse   = np.sqrt(mean_squared_error(true_r.flatten(), pred_r.flatten()))
    mae    = mean_absolute_error(true_r.flatten(), pred_r.flatten())
    r2     = r2_score(true_r.flatten(), pred_r.flatten())
    mape   = np.mean(np.abs((true_r - pred_r) / (np.abs(true_r) + 1e-6))) * 100
    resid  = (pred_r - true_r).flatten()

    results[mname] = dict(pred=pred_r, true=true_r, dates=dates_seq,
                          rmse=rmse, mae=mae, r2=r2, mape=mape,
                          rmse_h=rmse_h, mae_h=mae_h, r2_h=r2_h,
                          resid=resid, val_cut=val_cut,
                          epoch=ckpt.get('epoch','?'))
    print(f"  {mname}: RMSE={rmse:.1f}  MAE={mae:.1f}  R2={r2:.4f}  MAPE={mape:.2f}%")

model_names = list(results.keys())
ZOOM = 120

# ═════════════════════════════════════════════════════════════
# FIGURA 1 — SEPARADA (3 filas, una por modelo)
# ═════════════════════════════════════════════════════════════
print("\nGenerando figura 1: separada...")

fig1 = plt.figure(figsize=(22, 20))
fig1.patch.set_facecolor(BG)
gs1 = gridspec.GridSpec(3, 2, figure=fig1,
                        hspace=0.45, wspace=0.25,
                        left=0.06, right=0.97, top=0.93, bottom=0.05)

for row, mname in enumerate(model_names):
    r     = results[mname]
    col_c = COLORS[mname]

    # ── Col 0: Serie completa ──────────────────────────────
    ax = fig1.add_subplot(gs1[row, 0])
    ax.plot(dates_all, close_full, color='#4A90D9', lw=0.5, alpha=0.7, label='IBEX Real')
    ax.plot(r['dates'], r['pred'][:,0], color=col_c, lw=0.8, alpha=0.9, label='Pred t+1')
    val_date = r['dates'][r['val_cut']]
    ax.axvline(val_date, color='#F39C12', lw=1.2, ls=':', label='Inicio val')
    ax.fill_betweenx([close_full.min()*0.95, close_full.max()*1.05],
                     val_date, r['dates'][-1],
                     alpha=0.06, color='#F39C12')
    style_ax(ax, f'{mname} — Serie completa 1993-2026  (t+1)',
             ylabel='Puntos IBEX')
    ep = r['epoch']
    ax.text(0.02, 0.96,
            f"RMSE={r['rmse']:.1f}  MAE={r['mae']:.1f}\nR²={r['r2']:.4f}  MAPE={r['mape']:.2f}%\nepoch={ep}",
            transform=ax.transAxes, fontsize=7.5, va='top', color='white',
            bbox=dict(boxstyle='round,pad=0.4', fc='#1A1A2E', ec=col_c, alpha=0.85))
    ax.legend(fontsize=6.5, facecolor='#1A1A2E', edgecolor=SPINE_C,
              labelcolor='#DDDDDD', loc='upper left')

    # ── Col 1: Zoom 120 dias + todos los horizontes ────────
    ax2 = fig1.add_subplot(gs1[row, 1])
    dates_z = r['dates'][-ZOOM:]
    pred_z  = r['pred'][-ZOOM:]
    true_z  = r['true'][-ZOOM:, 0]

    # Banda de incertidumbre: entre t+1 y t+10
    ax2.fill_between(dates_z, pred_z[:,0], pred_z[:,-1],
                     alpha=0.15, color=col_c, label='Banda t+1→t+10')
    for d in range(HORIZON):
        alpha = max(0.15, 0.90 - d*0.08)
        lw    = 1.6 if d == 0 else 0.5
        ax2.plot(dates_z, pred_z[:,d], color=col_c, lw=lw, alpha=alpha,
                 label=f't+{d+1}' if d < 3 else None)
    ax2.plot(dates_z, true_z, color='white', lw=1.8, alpha=0.95, label='Real t+1')

    rmse_z = np.sqrt(mean_squared_error(true_z, pred_z[:,0]))
    style_ax(ax2, f'{mname} — Zoom ultimos {ZOOM} dias (todos los horizontes)',
             ylabel='Puntos IBEX')
    ax2.text(0.02, 0.96, f'RMSE t+1 (zoom)={rmse_z:.1f}',
             transform=ax2.transAxes, fontsize=7.5, va='top', color='white',
             bbox=dict(boxstyle='round,pad=0.3', fc='#1A1A2E', ec=col_c, alpha=0.85))
    ax2.legend(fontsize=6, facecolor='#1A1A2E', edgecolor=SPINE_C,
               labelcolor='#DDDDDD', ncol=5, loc='upper right')

fig1.suptitle('Analisis Retrospectivo — Modelos Separados  |  Entrenados hasta 31/03/2026',
              color='white', fontsize=13, fontweight='bold', y=0.97)
fig1.savefig('grafica_comparativa_separada.png', dpi=150, bbox_inches='tight',
             facecolor=BG)
plt.close(fig1)
print("  -> grafica_comparativa_separada.png")

# ═════════════════════════════════════════════════════════════
# FIGURA 2 — JUNTOS (3 modelos superpuestos)
# ═════════════════════════════════════════════════════════════
print("Generando figura 2: juntos...")

fig2 = plt.figure(figsize=(22, 18))
fig2.patch.set_facecolor(BG)
gs2 = gridspec.GridSpec(3, 2, figure=fig2,
                        hspace=0.45, wspace=0.25,
                        left=0.06, right=0.97, top=0.93, bottom=0.05)

# ── Panel A: Serie completa los 3 juntos ──────────────────
ax_full = fig2.add_subplot(gs2[0, :])
ax_full.plot(dates_all, close_full, color='white', lw=0.8, alpha=0.85,
             label='IBEX Real', zorder=5)
for mname in model_names:
    r = results[mname]
    ax_full.plot(r['dates'], r['pred'][:,0], color=COLORS[mname], lw=0.8, alpha=0.75,
                 label=f'{mname}  R²={r["r2"]:.4f}  MAPE={r["mape"]:.2f}%')
val_date = results[model_names[0]]['dates'][val_cut]
ax_full.axvline(val_date, color='#F39C12', lw=1.5, ls=':', alpha=0.8)
ax_full.fill_betweenx([close_full.min()*0.95, close_full.max()*1.05],
                      val_date, results[model_names[0]]['dates'][-1],
                      alpha=0.05, color='#F39C12')
style_ax(ax_full, 'Serie completa 1993-2026 — 3 modelos superpuestos (t+1)',
         ylabel='Puntos IBEX', fs=10)
ax_full.legend(fontsize=8, facecolor='#1A1A2E', edgecolor=SPINE_C,
               labelcolor='#DDDDDD', loc='upper left')

# ── Panel B: Zoom 120 dias los 3 juntos ───────────────────
ax_zoom = fig2.add_subplot(gs2[1, :])
true_z = results[model_names[0]]['true'][-ZOOM:, 0]
dates_z = results[model_names[0]]['dates'][-ZOOM:]
ax_zoom.plot(dates_z, true_z, color='white', lw=2, alpha=0.95,
             label='IBEX Real', zorder=5)
for mname in model_names:
    r = results[mname]
    pred_z = r['pred'][-ZOOM:, 0]
    rmse_z = np.sqrt(mean_squared_error(true_z, pred_z))
    ax_zoom.plot(dates_z, pred_z, color=COLORS[mname], lw=1.5, alpha=0.85,
                 label=f'{mname}  RMSE={rmse_z:.1f}')
style_ax(ax_zoom, f'Zoom ultimos {ZOOM} dias — 3 modelos superpuestos (t+1)',
         ylabel='Puntos IBEX', fs=10)
ax_zoom.legend(fontsize=8.5, facecolor='#1A1A2E', edgecolor=SPINE_C,
               labelcolor='#DDDDDD', loc='upper left')

# ── Panel C: RMSE por horizonte ───────────────────────────
ax_rmse = fig2.add_subplot(gs2[2, 0])
dias = list(range(1, HORIZON+1))
for mname in model_names:
    r = results[mname]
    ax_rmse.plot(dias, r['rmse_h'], 'o-', color=COLORS[mname], lw=2.5, ms=7,
                 label=f'{mname}  (avg {r["rmse"]:.0f})')
    for d, v in enumerate(r['rmse_h']):
        ax_rmse.annotate(f'{v:.0f}', (dias[d], v), textcoords='offset points',
                         xytext=(0,7), fontsize=6.5, ha='center', color=COLORS[mname])
ax_rmse.set_xticks(dias)
ax_rmse.set_xticklabels([f't+{d}' for d in dias], fontsize=8)
style_ax(ax_rmse, 'RMSE por dia de horizonte', xlabel='Dia predicho', ylabel='RMSE (pts)')
ax_rmse.legend(fontsize=8, facecolor='#1A1A2E', edgecolor=SPINE_C, labelcolor='#DDDDDD')

# ── Panel D: Barras comparativas de metricas ─────────────
ax_bar = fig2.add_subplot(gs2[2, 1])
metrics_labels = ['RMSE', 'MAE', 'MAPE×100', 'R²×10000']
x = np.arange(len(metrics_labels))
width = 0.25
for i, mname in enumerate(model_names):
    r = results[mname]
    vals = [r['rmse'], r['mae'], r['mape']*100, r['r2']*10000]
    bars = ax_bar.bar(x + i*width, vals, width, label=mname,
                      color=COLORS[mname], alpha=0.85, edgecolor='#1A1A2E')
    for bar, v in zip(bars, vals):
        ax_bar.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 20,
                    f'{v:.0f}', ha='center', va='bottom', fontsize=6.5, color='white')
ax_bar.set_xticks(x + width)
ax_bar.set_xticklabels(metrics_labels, fontsize=8, color='#AAAAAA')
style_ax(ax_bar, 'Comparativa de metricas (escaladas para visualizacion)')
ax_bar.legend(fontsize=8, facecolor='#1A1A2E', edgecolor=SPINE_C, labelcolor='#DDDDDD')

fig2.suptitle('Comparativa 3 Modelos Superpuestos — IBEX 35  |  Entrenados hasta 31/03/2026',
              color='white', fontsize=13, fontweight='bold', y=0.97)
fig2.savefig('grafica_comparativa_juntos.png', dpi=150, bbox_inches='tight',
             facecolor=BG)
plt.close(fig2)
print("  -> grafica_comparativa_juntos.png")

# ═════════════════════════════════════════════════════════════
# FIGURA 3 — REGRESION COMPLETA (matriz actualizada)
# ═════════════════════════════════════════════════════════════
print("Generando figura 3: regresion completa...")

fig3 = plt.figure(figsize=(22, 32))
fig3.patch.set_facecolor(BG)
gs3 = gridspec.GridSpec(8, 3, figure=fig3,
                        hspace=0.55, wspace=0.30,
                        left=0.06, right=0.97, top=0.96, bottom=0.03)

# Fila 0: Scatter real vs predicho (t+1) ── completo
for col, mname in enumerate(model_names):
    r    = results[mname]
    ax   = fig3.add_subplot(gs3[0, col])
    yt   = r['true'][:,0].flatten()
    yp   = r['pred'][:,0].flatten()
    ax.scatter(yt, yp, s=0.8, alpha=0.2, color=COLORS[mname], rasterized=True)
    vmin, vmax = min(yt.min(),yp.min()), max(yt.max(),yp.max())
    ax.plot([vmin,vmax],[vmin,vmax],'--', color='#F39C12', lw=1.2, label='Ideal')
    coef = np.polyfit(yt, yp, 1)
    xl   = np.linspace(vmin, vmax, 200)
    ax.plot(xl, np.poly1d(coef)(xl), '-', color='white', lw=1.0, alpha=0.8)
    ax.text(0.04, 0.93,
            f"R²={r['r2_h'][0]:.4f}\nRMSE={r['rmse_h'][0]:.1f}\nMAE={r['mae_h'][0]:.1f}",
            transform=ax.transAxes, fontsize=7.5, va='top', color='white',
            bbox=dict(boxstyle='round,pad=0.3', fc='#1A1A2E', ec=COLORS[mname], alpha=0.85))
    style_ax(ax, f'{mname} — Scatter t+1', xlabel='Real', ylabel='Predicho')
    ax.legend(fontsize=6.5, facecolor='#1A1A2E', edgecolor=SPINE_C, labelcolor='#DDDDDD')

# Fila 1: Scatter t+5
for col, mname in enumerate(model_names):
    r  = results[mname]
    ax = fig3.add_subplot(gs3[1, col])
    yt = r['true'][:,4].flatten(); yp = r['pred'][:,4].flatten()
    ax.scatter(yt, yp, s=0.8, alpha=0.2, color=COLORS[mname], rasterized=True)
    vmin, vmax = min(yt.min(),yp.min()), max(yt.max(),yp.max())
    ax.plot([vmin,vmax],[vmin,vmax],'--', color='#F39C12', lw=1.2)
    coef = np.polyfit(yt, yp, 1)
    xl   = np.linspace(vmin, vmax, 200)
    ax.plot(xl, np.poly1d(coef)(xl), '-', color='white', lw=1.0, alpha=0.8)
    ax.text(0.04, 0.93,
            f"R²={r['r2_h'][4]:.4f}\nRMSE={r['rmse_h'][4]:.1f}",
            transform=ax.transAxes, fontsize=7.5, va='top', color='white',
            bbox=dict(boxstyle='round,pad=0.3', fc='#1A1A2E', ec=COLORS[mname], alpha=0.85))
    style_ax(ax, f'{mname} — Scatter t+5', xlabel='Real', ylabel='Predicho')

# Fila 2: Scatter t+10
for col, mname in enumerate(model_names):
    r  = results[mname]
    ax = fig3.add_subplot(gs3[2, col])
    yt = r['true'][:,9].flatten(); yp = r['pred'][:,9].flatten()
    ax.scatter(yt, yp, s=0.8, alpha=0.2, color=COLORS[mname], rasterized=True)
    vmin, vmax = min(yt.min(),yp.min()), max(yt.max(),yp.max())
    ax.plot([vmin,vmax],[vmin,vmax],'--', color='#F39C12', lw=1.2)
    coef = np.polyfit(yt, yp, 1)
    xl   = np.linspace(vmin, vmax, 200)
    ax.plot(xl, np.poly1d(coef)(xl), '-', color='white', lw=1.0, alpha=0.8)
    ax.text(0.04, 0.93,
            f"R²={r['r2_h'][9]:.4f}\nRMSE={r['rmse_h'][9]:.1f}",
            transform=ax.transAxes, fontsize=7.5, va='top', color='white',
            bbox=dict(boxstyle='round,pad=0.3', fc='#1A1A2E', ec=COLORS[mname], alpha=0.85))
    style_ax(ax, f'{mname} — Scatter t+10', xlabel='Real', ylabel='Predicho')

# Fila 3: Distribucion de residuos
for col, mname in enumerate(model_names):
    r    = results[mname]
    ax   = fig3.add_subplot(gs3[3, col])
    resid = r['resid']
    ax.hist(resid, bins=80, color=COLORS[mname], alpha=0.75, edgecolor='none', density=True)
    mu, sigma = resid.mean(), resid.std()
    from scipy.stats import norm
    x_norm = np.linspace(resid.min(), resid.max(), 300)
    ax.plot(x_norm, norm.pdf(x_norm, mu, sigma), '--', color='white', lw=1.5)
    ax.axvline(0, color='#F39C12', lw=1.2, ls=':')
    ax.text(0.04, 0.93, f'mu={mu:.1f}\nsigma={sigma:.1f}',
            transform=ax.transAxes, fontsize=7.5, va='top', color='white',
            bbox=dict(boxstyle='round,pad=0.3', fc='#1A1A2E', ec=COLORS[mname], alpha=0.85))
    style_ax(ax, f'{mname} — Distribucion de residuos', xlabel='Error (pts)', ylabel='Densidad')

# Fila 4: Residuos en el tiempo
for col, mname in enumerate(model_names):
    r    = results[mname]
    ax   = fig3.add_subplot(gs3[4, col])
    resid_t1 = (r['pred'][:,0] - r['true'][:,0])
    ax.plot(r['dates'], resid_t1, color=COLORS[mname], lw=0.5, alpha=0.7)
    ax.axhline(0, color='white', lw=1.0, ls='--', alpha=0.5)
    ax.fill_between(r['dates'], resid_t1, 0,
                    where=resid_t1 > 0, alpha=0.3, color='#E74C3C')
    ax.fill_between(r['dates'], resid_t1, 0,
                    where=resid_t1 < 0, alpha=0.3, color='#3498DB')
    style_ax(ax, f'{mname} — Residuos en el tiempo (t+1)', ylabel='Error (pts)')

# Fila 5: MAPE por periodo (ventana 252 dias = 1 año)
for col, mname in enumerate(model_names):
    r    = results[mname]
    ax   = fig3.add_subplot(gs3[5, col])
    mape_t = np.abs((r['pred'][:,0] - r['true'][:,0]) / (np.abs(r['true'][:,0]) + 1e-6)) * 100
    window = 252
    mape_roll = pd.Series(mape_t).rolling(window).mean().values
    ax.plot(r['dates'], mape_roll, color=COLORS[mname], lw=1.2)
    ax.axhline(r['mape'], color='#F39C12', lw=1.0, ls='--',
               label=f'MAPE global={r["mape"]:.2f}%')
    ax.fill_between(r['dates'], mape_roll, alpha=0.2, color=COLORS[mname])
    style_ax(ax, f'{mname} — MAPE rodante 1 año', ylabel='MAPE (%)')
    ax.legend(fontsize=7, facecolor='#1A1A2E', edgecolor=SPINE_C, labelcolor='#DDDDDD')

# Fila 6: R² por horizonte
ax_r2 = fig3.add_subplot(gs3[6, :])
dias = list(range(1, HORIZON+1))
for mname in model_names:
    r = results[mname]
    ax_r2.plot(dias, r['r2_h'], 'o-', color=COLORS[mname], lw=2.5, ms=7,
               label=f'{mname}  (R² global {r["r2"]:.4f})')
    for d, v in enumerate(r['r2_h']):
        ax_r2.annotate(f'{v:.4f}', (dias[d], v), textcoords='offset points',
                       xytext=(0, 8), fontsize=6, ha='center', color=COLORS[mname])
ax_r2.set_xticks(dias)
ax_r2.set_xticklabels([f't+{d}' for d in dias], fontsize=9)
ax_r2.set_ylim(0.97, 1.001)
style_ax(ax_r2, 'R² por dia de horizonte — comparacion 3 modelos',
         xlabel='Dia predicho', ylabel='R²', fs=10)
ax_r2.legend(fontsize=9, facecolor='#1A1A2E', edgecolor=SPINE_C, labelcolor='#DDDDDD')

# Fila 7: Tabla resumen
ax_tbl = fig3.add_subplot(gs3[7, :])
ax_tbl.axis('off')
headers = ['Modelo', 'RMSE', 'MAE', 'MAPE (%)', 'R²',
           'RMSE t+1', 'RMSE t+5', 'RMSE t+10', 'R² t+1', 'R² t+10', 'Epochs']
rows = []
for mname in model_names:
    r = results[mname]
    rows.append([mname,
                 f"{r['rmse']:.1f}", f"{r['mae']:.1f}", f"{r['mape']:.2f}", f"{r['r2']:.4f}",
                 f"{r['rmse_h'][0]:.1f}", f"{r['rmse_h'][4]:.1f}", f"{r['rmse_h'][9]:.1f}",
                 f"{r['r2_h'][0]:.4f}", f"{r['r2_h'][9]:.4f}", str(r['epoch'])])
tbl = ax_tbl.table(cellText=rows, colLabels=headers, loc='center', cellLoc='center')
tbl.auto_set_font_size(False)
tbl.set_fontsize(8.5)
tbl.scale(1, 1.8)
for (ri, ci), cell in tbl.get_celld().items():
    if ri == 0:
        cell.set_facecolor('#1A3A5C'); cell.set_text_props(color='white', fontweight='bold')
    else:
        mn = rows[ri-1][0] if ri-1 < len(rows) else ''
        bg = {'RNN Simple':'#2A1A1A','LSTM':'#1A1A2A','GRU':'#1A2A1A'}.get(mn, '#111827')
        cell.set_facecolor(bg); cell.set_text_props(color='#DDDDDD')
    cell.set_edgecolor('#2A2A3A')

fig3.suptitle('Analisis de Regresion Completo — Modelos Finales IBEX 35  |  31/03/2026',
              color='white', fontsize=13, fontweight='bold', y=0.975)
fig3.savefig('analisis_regresion_final.png', dpi=150, bbox_inches='tight', facecolor=BG)
plt.close(fig3)
print("  -> analisis_regresion_final.png")

print("\nFicheros generados:")
for f in ['grafica_comparativa_separada.png',
          'grafica_comparativa_juntos.png',
          'analisis_regresion_final.png']:
    mb = os.path.getsize(f)/1e6 if os.path.exists(f) else 0
    print(f"  {f}  ({mb:.1f} MB)")
