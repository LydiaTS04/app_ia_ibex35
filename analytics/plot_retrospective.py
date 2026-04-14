# -*- coding: utf-8 -*-
"""
plot_retrospective.py -- Grafica retrospectiva completa (nuevos modelos finales)
=================================================================================
Carga los snapshots finales (snap_*_31mar.pt) entrenados con train_final.py
y genera una figura completa con:
  - Fila 1: Serie historica completa + prediccion in-sample (todos los modelos)
  - Fila 2-4: Zoom en los ultimos 120 dias con prediccion multi-step por modelo
  - Fila 5-7: Scatter real vs predicho (regresion) por modelo
  - Fila 8:   RMSE acumulado por dia de horizonte (los 3 modelos juntos)

EJECUTAR:
  py plot_retrospective.py
"""

import os, warnings, pickle
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D
from datetime import date

import yfinance as yf
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import MinMaxScaler

# ─────────────────────────────────────────────────────────────────
# CONFIG (debe coincidir con train_final.py)
# ─────────────────────────────────────────────────────────────────
DEVICE    = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
SEQ_LEN   = 90
HORIZON   = 10
DROPOUT   = 0.05
BATCH     = 256
START_DATE = '1993-07-12'
END_TRAIN  = '2026-04-01'

SNAPSHOTS = {
    "RNN Simple": "snap_rnn_31mar.pt",
    "LSTM":       "snap_lstm_31mar.pt",
    "GRU":        "snap_gru_31mar.pt",
}
COLORS = {"RNN Simple": "#E74C3C", "LSTM": "#2980B9", "GRU": "#27AE60"}

OUTPUT_PNG = "grafica_retrospectiva_final.png"

# ─────────────────────────────────────────────────────────────────
# 1. DATOS
# ─────────────────────────────────────────────────────────────────
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
df['RSI']   = 100 - (100 / (1 + gain / (loss + 1e-9)))
df['Mom10'] = df['Close'] - df['Close'].shift(10)
df.dropna(inplace=True)

feature_cols = list(df.columns)
n_features   = len(feature_cols)
close_idx    = feature_cols.index('Close')
dates_all    = df.index

print(f"  {df.index[0].date()} -> {df.index[-1].date()}  ({len(df)} sesiones)")

# Scalers (usar los guardados si existen)
scaler       = MinMaxScaler()
close_scaler = MinMaxScaler()

if os.path.exists('scaler_final.pkl'):
    with open('scaler_final.pkl', 'rb') as f:
        scaler = pickle.load(f)
    with open('close_scaler_final.pkl', 'rb') as f:
        close_scaler = pickle.load(f)
    scaled = scaler.transform(df.values)
    print("  Scalers cargados desde archivos.")
else:
    scaled = scaler.fit_transform(df.values)
    close_scaler.fit(df[['Close']].values)
    print("  Scalers recalculados.")

# ─────────────────────────────────────────────────────────────────
# 2. SECUENCIAS
# ─────────────────────────────────────────────────────────────────
def make_sequences_ms(arr, seq_len, horizon):
    X, y = [], []
    for i in range(seq_len, len(arr) - horizon + 1):
        X.append(arr[i - seq_len:i])
        y.append(arr[i:i + horizon, close_idx])
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)

X_all, y_all = make_sequences_ms(scaled, SEQ_LEN, HORIZON)
n = len(X_all)

# Fechas correspondientes al primer dia predicho de cada secuencia
dates_seq = dates_all[SEQ_LEN : SEQ_LEN + n]

val_cut   = int(n * 0.90)
X_val, y_val = X_all[val_cut:], y_all[val_cut:]

loader_all = DataLoader(TensorDataset(torch.tensor(X_all), torch.tensor(y_all)),
                        batch_size=BATCH, shuffle=False, pin_memory=(DEVICE.type=='cuda'))

# ─────────────────────────────────────────────────────────────────
# 3. MODELOS (misma arquitectura que train_final.py)
# ─────────────────────────────────────────────────────────────────
class Attention(nn.Module):
    def __init__(self, h):
        super().__init__()
        self.attn = nn.Linear(h, h)
        self.v    = nn.Linear(h, 1, bias=False)
    def forward(self, x):
        scores  = self.v(torch.tanh(self.attn(x)))
        weights = torch.softmax(scores, dim=1)
        return (weights * x).sum(dim=1)

class RNNModelFinal(nn.Module):
    def __init__(self):
        super().__init__()
        self.rnn  = nn.RNN(n_features, 256, num_layers=4,
                           batch_first=True, dropout=DROPOUT, nonlinearity='tanh')
        self.attn = Attention(256)
        self.fc   = nn.Sequential(nn.Linear(256,128), nn.GELU(),
                                   nn.Linear(128,64),  nn.GELU(),
                                   nn.Linear(64, HORIZON))
    def forward(self, x):
        out, _ = self.rnn(x)
        return self.fc(self.attn(out))

class LSTMModelFinal(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(n_features, 512, num_layers=4,
                            batch_first=True, dropout=DROPOUT)
        self.attn = Attention(512)
        self.bn   = nn.BatchNorm1d(512)
        self.fc   = nn.Sequential(nn.Linear(512,256), nn.GELU(),
                                   nn.Linear(256,128), nn.GELU(),
                                   nn.Linear(128,64),  nn.GELU(),
                                   nn.Linear(64, HORIZON))
    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(self.bn(self.attn(out)))

class GRUModelFinal(nn.Module):
    def __init__(self):
        super().__init__()
        self.gru  = nn.GRU(n_features, 512, num_layers=4,
                           batch_first=True, dropout=DROPOUT)
        self.attn = Attention(512)
        self.bn   = nn.BatchNorm1d(512)
        self.fc   = nn.Sequential(nn.Linear(512,256), nn.GELU(),
                                   nn.Linear(256,128), nn.GELU(),
                                   nn.Linear(128,64),  nn.GELU(),
                                   nn.Linear(64, HORIZON))
    def forward(self, x):
        out, _ = self.gru(x)
        return self.fc(self.bn(self.attn(out)))

BUILDERS = {"RNN Simple": RNNModelFinal, "LSTM": LSTMModelFinal, "GRU": GRUModelFinal}

# ─────────────────────────────────────────────────────────────────
# 4. INFERENCIA
# ─────────────────────────────────────────────────────────────────
results = {}

for mname, ModelClass in BUILDERS.items():
    snap = SNAPSHOTS[mname]
    if not os.path.exists(snap):
        print(f"  [SKIP] {mname}: snapshot no encontrado ({snap})")
        continue

    model = ModelClass().to(DEVICE)
    ckpt  = torch.load(snap, map_location=DEVICE, weights_only=True)
    model.load_state_dict(ckpt['model_state'])
    model.eval()
    print(f"  [OK] {mname} cargado  (epoch {ckpt.get('epoch','?')})")

    preds = []
    with torch.no_grad():
        for xb, _ in loader_all:
            preds.append(model(xb.to(DEVICE)).cpu().numpy())
    pred_sc = np.vstack(preds)       # (N, HORIZON)

    pred_real = close_scaler.inverse_transform(pred_sc.reshape(-1,1)).reshape(-1, HORIZON)
    true_real = close_scaler.inverse_transform(y_all.reshape(-1,1)).reshape(-1, HORIZON)

    # Metricas por horizonte
    rmse_h = [np.sqrt(mean_squared_error(true_real[:,d], pred_real[:,d])) for d in range(HORIZON)]
    mae_h  = [mean_absolute_error(true_real[:,d], pred_real[:,d]) for d in range(HORIZON)]
    r2_h   = [r2_score(true_real[:,d], pred_real[:,d]) for d in range(HORIZON)]

    # Metricas globales
    rmse  = np.sqrt(mean_squared_error(true_real.flatten(), pred_real.flatten()))
    mae   = mean_absolute_error(true_real.flatten(), pred_real.flatten())
    r2    = r2_score(true_real.flatten(), pred_real.flatten())
    mape  = np.mean(np.abs((true_real - pred_real) / (np.abs(true_real) + 1e-6))) * 100

    results[mname] = {
        'pred':   pred_real,   # (N, HORIZON)
        'true':   true_real,
        'dates':  dates_seq,
        'rmse':   rmse,   'mae': mae,   'r2': r2,   'mape': mape,
        'rmse_h': rmse_h, 'mae_h': mae_h, 'r2_h': r2_h,
        'val_cut': val_cut,
    }
    print(f"      RMSE={rmse:.1f}  MAE={mae:.1f}  R2={r2:.4f}  MAPE={mape:.2f}%")

if not results:
    print("ERROR: No se encontro ningun snapshot. Ejecuta train_final.py primero.")
    exit(1)

# Precios reales full (dia a dia)
close_real_full = df['Close'].values         # (total_dias,)
dates_full      = dates_all                  # DatetimeIndex completo

# ─────────────────────────────────────────────────────────────────
# 5. FIGURA
# ─────────────────────────────────────────────────────────────────
print("\nGenerando grafica retrospectiva...")

n_models = len(results)
fig = plt.figure(figsize=(22, 30))
fig.patch.set_facecolor('#0D1117')

gs_main = gridspec.GridSpec(
    8, n_models,
    figure=fig,
    hspace=0.55, wspace=0.30,
    left=0.06, right=0.97, top=0.95, bottom=0.04
)

GRID_KW  = dict(color='#2A2A3A', linewidth=0.5, linestyle='--')
SPINE_C  = '#3A3A4A'

def style_ax(ax, title='', xlabel='', ylabel=''):
    ax.set_facecolor('#0D1117')
    for sp in ax.spines.values():
        sp.set_edgecolor(SPINE_C)
    ax.tick_params(colors='#AAAAAA', labelsize=7.5)
    ax.grid(**GRID_KW)
    if title:  ax.set_title(title,  color='white',   fontsize=9,  fontweight='bold')
    if xlabel: ax.set_xlabel(xlabel, color='#AAAAAA', fontsize=7.5)
    if ylabel: ax.set_ylabel(ylabel, color='#AAAAAA', fontsize=7.5)

model_names = list(results.keys())

# ── FILA 0: Serie completa in-sample (toda la historia) ───────────
for col, mname in enumerate(model_names):
    r   = results[mname]
    ax  = fig.add_subplot(gs_main[0, col])
    col_c = COLORS[mname]

    # Real
    ax.plot(dates_full, close_real_full, color='#4A90D9', lw=0.6,
            alpha=0.7, label='Real IBEX')

    # Predicho dia+1 (primer horizonte) para toda la historia
    pred_d1 = r['pred'][:, 0]
    ax.plot(r['dates'], pred_d1, color=col_c, lw=0.7, alpha=0.85,
            label=f'{mname} t+1')

    # Sombrear zona val
    val_date = r['dates'][r['val_cut']]
    ax.axvline(val_date, color='#F39C12', lw=1.2, ls=':', label='Val inicio')

    style_ax(ax, f'{mname} — Serie completa (t+1)', ylabel='Puntos IBEX')
    ax.legend(fontsize=6.5, facecolor='#1A1A2E', edgecolor=SPINE_C,
              labelcolor='#DDDDDD', loc='upper left')

# ── FILA 1: Zoom ultimos 120 dias ─────────────────────────────────
ZOOM = 120
for col, mname in enumerate(model_names):
    r   = results[mname]
    ax  = fig.add_subplot(gs_main[1, col])
    col_c = COLORS[mname]

    dates_z = r['dates'][-ZOOM:]
    pred_z  = r['pred'][-ZOOM:, :]    # (120, 10)
    true_z  = r['true'][-ZOOM:, 0]   # t+1 real

    real_dates_z = dates_full[-ZOOM - SEQ_LEN : -SEQ_LEN] if ZOOM + SEQ_LEN < len(dates_full) else dates_full[-ZOOM:]

    # Dibujar cada horizonte con alpha decreciente
    for d in range(HORIZON):
        alpha = max(0.15, 0.85 - d * 0.07)
        lw    = 1.4 if d == 0 else 0.6
        ax.plot(dates_z, pred_z[:, d], color=col_c, lw=lw, alpha=alpha,
                label=f't+{d+1}' if d < 3 else None)

    ax.plot(dates_z, true_z, color='white', lw=1.4, alpha=0.9, label='Real t+1')

    style_ax(ax, f'{mname} — Zoom 120 dias (todos los horizontes)', ylabel='Puntos IBEX')
    ax.legend(fontsize=6, facecolor='#1A1A2E', edgecolor=SPINE_C,
              labelcolor='#DDDDDD', ncol=4, loc='upper left')

# ── FILAS 2-4: Scatter real vs predicho por horizonte seleccionado ──
HORIZONS_SCATTER = [0, 4, 9]   # t+1, t+5, t+10
HTITLES = ['t+1 (1 dia)', 't+5 (5 dias)', 't+10 (10 dias)']

for row_offset, (h_idx, htitle) in enumerate(zip(HORIZONS_SCATTER, HTITLES)):
    for col, mname in enumerate(model_names):
        r    = results[mname]
        ax   = fig.add_subplot(gs_main[2 + row_offset, col])
        col_c = COLORS[mname]

        y_true = r['true'][:, h_idx].flatten()
        y_pred = r['pred'][:, h_idx].flatten()

        # Scatter con densidad de color
        ax.scatter(y_true, y_pred, s=1.2, alpha=0.25, color=col_c, rasterized=True)

        # Linea identidad
        vmin = min(y_true.min(), y_pred.min())
        vmax = max(y_true.max(), y_pred.max())
        ax.plot([vmin, vmax], [vmin, vmax], '--', color='#F39C12', lw=1.2, label='Ideal')

        # Linea regresion
        coef = np.polyfit(y_true, y_pred, 1)
        poly = np.poly1d(coef)
        x_line = np.linspace(vmin, vmax, 200)
        ax.plot(x_line, poly(x_line), '-', color='white', lw=1.0, alpha=0.8, label='Regresion')

        r2   = r['r2_h'][h_idx]
        rmse = r['rmse_h'][h_idx]
        ax.text(0.04, 0.92, f'R²={r2:.4f}\nRMSE={rmse:.0f}',
                transform=ax.transAxes, fontsize=7.5,
                color='white', va='top',
                bbox=dict(boxstyle='round,pad=0.3', fc='#1A1A2E', ec=SPINE_C, alpha=0.8))

        style_ax(ax,
                 f'{mname} — Scatter {htitle}',
                 xlabel='Real (pts)', ylabel='Predicho (pts)')
        ax.legend(fontsize=6, facecolor='#1A1A2E', edgecolor=SPINE_C, labelcolor='#DDDDDD')

# ── FILA 7: RMSE por dia de horizonte (los 3 modelos) ─────────────
ax_rmse = fig.add_subplot(gs_main[6, :])
dias = list(range(1, HORIZON + 1))
for mname in model_names:
    r = results[mname]
    ax_rmse.plot(dias, r['rmse_h'], 'o-', color=COLORS[mname],
                 lw=2, ms=6, label=f"{mname}  (RMSE global {r['rmse']:.0f})")
    # Anotar valor maximo
    for d, v in enumerate(r['rmse_h']):
        ax_rmse.annotate(f"{v:.0f}", (dias[d], v), textcoords='offset points',
                         xytext=(0, 6), fontsize=6.5, ha='center',
                         color=COLORS[mname])

ax_rmse.set_xticks(dias)
ax_rmse.set_xticklabels([f't+{d}' for d in dias], fontsize=8)
style_ax(ax_rmse,
         'RMSE por dia de horizonte — comparacion 3 modelos',
         xlabel='Dia predicho', ylabel='RMSE (puntos IBEX)')
ax_rmse.legend(fontsize=8, facecolor='#1A1A2E', edgecolor=SPINE_C, labelcolor='#DDDDDD')

# ── FILA 8: Tabla resumen metricas ────────────────────────────────
ax_tbl = fig.add_subplot(gs_main[7, :])
ax_tbl.axis('off')

headers = ['Modelo', 'RMSE', 'MAE', 'MAPE (%)', 'R²',
           'RMSE t+1', 'RMSE t+5', 'RMSE t+10', 'Epocas', 'Params']

rows = []
for mname in model_names:
    r    = results[mname]
    snap = torch.load(SNAPSHOTS[mname], map_location='cpu', weights_only=True)
    ep   = snap.get('epoch', '?')
    # Contar parametros
    model_tmp = BUILDERS[mname]().cpu()
    n_p = sum(p.numel() for p in model_tmp.parameters())
    rows.append([
        mname,
        f"{r['rmse']:.1f}",
        f"{r['mae']:.1f}",
        f"{r['mape']:.2f}",
        f"{r['r2']:.4f}",
        f"{r['rmse_h'][0]:.1f}",
        f"{r['rmse_h'][4]:.1f}",
        f"{r['rmse_h'][9]:.1f}",
        str(ep),
        f"{n_p/1e6:.2f}M",
    ])

tbl = ax_tbl.table(
    cellText=rows, colLabels=headers,
    loc='center', cellLoc='center'
)
tbl.auto_set_font_size(False)
tbl.set_fontsize(8.5)
tbl.scale(1, 1.7)

for (r_i, c_i), cell in tbl.get_celld().items():
    if r_i == 0:
        cell.set_facecolor('#1A3A5C')
        cell.set_text_props(color='white', fontweight='bold')
    else:
        mname_row = rows[r_i - 1][0] if r_i - 1 < len(rows) else ''
        cell.set_facecolor('#0D2035' if r_i % 2 == 0 else '#111827')
        cell.set_text_props(color='#DDDDDD')
    cell.set_edgecolor('#2A2A3A')

# ── TITULO GLOBAL ─────────────────────────────────────────────────
fig.suptitle(
    "Analisis Retrospectivo — Modelos Finales IBEX 35  (Entrenados hasta 31/03/2026)",
    color='white', fontsize=13, fontweight='bold', y=0.975
)

# ─────────────────────────────────────────────────────────────────
# 6. GUARDAR
# ─────────────────────────────────────────────────────────────────
fig.savefig(OUTPUT_PNG, dpi=160, bbox_inches='tight',
            facecolor=fig.get_facecolor())
plt.close(fig)
print(f"\nGrafica guardada: {OUTPUT_PNG}")

# Resumen consola
print("\nMETRICAS FINALES:")
print(f"{'Modelo':<12} {'RMSE':>8} {'MAE':>8} {'MAPE%':>8} {'R2':>8}")
print("-" * 50)
for mname in model_names:
    r = results[mname]
    print(f"{mname:<12} {r['rmse']:8.1f} {r['mae']:8.1f} {r['mape']:8.2f} {r['r2']:8.4f}")
