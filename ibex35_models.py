# -*- coding: utf-8 -*-
"""
Prediccion del IBEX 35 — RNN Simple, LSTM, GRU
PyTorch + CUDA (GPU NVIDIA RTX 5050)
Universidad Alfonso X el Sabio (UAX)

MEJORAS APLICADAS:
  - PyTorch con CUDA: entrenamiento en GPU NVIDIA RTX 5050
  - Arquitecturas mas profundas y con mayor capacidad
  - Early Stopping (patience=15): para cuando no mejora
  - ReduceLROnPlateau: baja el learning rate si se estanca
  - Checkpoint: guarda siempre el mejor epoch (no el ultimo)
  - Gradient clipping: evita gradiente explosivo
  - Indicadores tecnicos como features adicionales (MA, RSI, Volatilidad)
  - Reentrenamiento incremental: si existe checkpoint, continua desde ahi

PARA REENTRENAR MANANA:
  py ibex35_models.py
  -> descarga datos frescos del IBEX automaticamente
  -> carga los mejores pesos guardados y hace fine-tuning
"""

import os, warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from datetime import datetime
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from statsmodels.tsa.seasonal import seasonal_decompose

import yfinance as yf
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# ─────────────────────────────────────────────────────────────
# CONFIGURACION
# ─────────────────────────────────────────────────────────────
DEVICE     = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
SEQ_LEN    = 60
EPOCHS     = 100      # Early Stopping corta antes si no mejora
BATCH      = 256      # Batch grande para GPU
LR         = 0.001
PATIENCE   = 15       # Epocas sin mejora antes de parar
START_DATE = '1993-07-12'

CKPT = {
    "RNN Simple": "ckpt_rnn.pt",
    "LSTM":       "ckpt_lstm.pt",
    "GRU":        "ckpt_gru.pt",
}
COLORS = {"RNN Simple": "tomato", "LSTM": "steelblue", "GRU": "seagreen"}

print("=" * 60)
print(f"DISPOSITIVO: {DEVICE}")
if DEVICE.type == 'cuda':
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"VRAM total: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
print("=" * 60)


# ─────────────────────────────────────────────────────────────
# 1. DATOS + INDICADORES TECNICOS
# ─────────────────────────────────────────────────────────────
print("\n1. DESCARGANDO DATOS DEL IBEX 35")
end_date = datetime.now().strftime('%Y-%m-%d')
raw = yf.download('^IBEX', start=START_DATE, end=end_date, progress=True)
raw.dropna(inplace=True)

# Aplanar MultiIndex si existe
raw.columns = raw.columns.get_level_values(0)
df = raw[['Open','High','Low','Close','Volume']].copy()

# Indicadores tecnicos
df['MA10']   = df['Close'].rolling(10).mean()
df['MA30']   = df['Close'].rolling(30).mean()
df['MA60']   = df['Close'].rolling(60).mean()
df['Std10']  = df['Close'].rolling(10).std()      # Volatilidad 10d

# RSI (14 periodos)
delta = df['Close'].diff()
gain  = delta.clip(lower=0).rolling(14).mean()
loss  = (-delta.clip(upper=0)).rolling(14).mean()
rs    = gain / (loss + 1e-9)
df['RSI'] = 100 - (100 / (1 + rs))

# Momentum
df['Mom10'] = df['Close'] - df['Close'].shift(10)

df.dropna(inplace=True)
print(f"Registros: {len(df)}  |  {df.index[0].date()} -> {df.index[-1].date()}")
print(f"Features usados: {list(df.columns)}")


# ─────────────────────────────────────────────────────────────
# 2. PREPROCESAMIENTO
# ─────────────────────────────────────────────────────────────
print("\n2. PREPROCESAMIENTO")

# Descomposicion
print("  Descomposicion estacional...")
decomp = seasonal_decompose(df['Close'].squeeze(), model='multiplicative', period=252)
fig, axes = plt.subplots(4, 1, figsize=(14, 10))
decomp.observed.plot( ax=axes[0], title='Observada',       color='steelblue')
decomp.trend.plot(    ax=axes[1], title='Tendencia',       color='darkorange')
decomp.seasonal.plot( ax=axes[2], title='Estacionalidad',  color='green')
decomp.resid.plot(    ax=axes[3], title='Residuo',         color='red')
plt.suptitle('Descomposicion Serie IBEX 35', fontsize=14)
plt.tight_layout()
plt.savefig('descomposicion_serie.png', dpi=150, bbox_inches='tight')
plt.close()
print("  Guardado: descomposicion_serie.png")

# Normalizacion por columna
feature_cols = list(df.columns)
n_features   = len(feature_cols)
scaler       = MinMaxScaler(feature_range=(0, 1))
scaled       = scaler.fit_transform(df.values)   # (N, n_features)

# Solo necesitamos el scaler de Close para desnormalizar predicciones
close_idx    = feature_cols.index('Close')
close_scaler = MinMaxScaler(feature_range=(0, 1))
close_scaler.fit(df[['Close']].values)

# Secuencias multivariable
def make_sequences(arr, seq_len):
    X, y = [], []
    for i in range(seq_len, len(arr)):
        X.append(arr[i - seq_len:i])         # (SEQ_LEN, n_features)
        y.append(arr[i, close_idx])           # precio Close del dia siguiente
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)

X_all, y_all = make_sequences(scaled, SEQ_LEN)
dates_all    = df.index[SEQ_LEN:]

n      = len(X_all)
va_end = int(n * 0.85)   # 85% train | 15% val — sin test separado

X_train, y_train = X_all[:va_end], y_all[:va_end]
X_val,   y_val   = X_all[va_end:], y_all[va_end:]

# Para la grafica retrospectiva usamos TODO el historico
X_full, y_full  = X_all, y_all
dates_full      = dates_all

# DataLoaders
def make_loader(X, y, shuffle=False):
    ds = TensorDataset(torch.tensor(X), torch.tensor(y).unsqueeze(1))
    return DataLoader(ds, batch_size=BATCH, shuffle=shuffle, pin_memory=(DEVICE.type=='cuda'))

train_loader = make_loader(X_train, y_train, shuffle=True)
val_loader   = make_loader(X_val,   y_val)

print(f"  Train: {len(X_train)} ({va_end/n*100:.0f}%)  |  Val: {len(X_val)} ({(n-va_end)/n*100:.0f}%)")
print(f"  Historico completo para grafica: {len(X_full)} puntos")
print(f"  Features por paso: {n_features}  |  Ventana: {SEQ_LEN} dias")
print(f"  Periodo completo: {dates_full[0].date()} -> {dates_full[-1].date()}")


# ─────────────────────────────────────────────────────────────
# 3. ARQUITECTURAS PyTorch (OPTIMIZADAS)
# ─────────────────────────────────────────────────────────────
class RNNModel(nn.Module):
    """RNN Simple mejorada: 3 capas, 128 unidades, dropout."""
    def __init__(self):
        super().__init__()
        self.rnn = nn.RNN(n_features, 128, num_layers=3,
                          batch_first=True, dropout=0.3, nonlinearity='tanh')
        self.fc  = nn.Sequential(nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, 1))

    def forward(self, x):
        out, _ = self.rnn(x)
        return self.fc(out[:, -1, :])


class LSTMModel(nn.Module):
    """LSTM optimizada: 3 capas, 256 unidades, dropout."""
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(n_features, 256, num_layers=3,
                            batch_first=True, dropout=0.3)
        self.fc   = nn.Sequential(
            nn.Linear(256, 128), nn.ReLU(),
            nn.Linear(128, 64),  nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])


class GRUModel(nn.Module):
    """GRU optimizada: 3 capas, 256 unidades, dropout."""
    def __init__(self):
        super().__init__()
        self.gru = nn.GRU(n_features, 256, num_layers=3,
                          batch_first=True, dropout=0.3)
        self.fc  = nn.Sequential(
            nn.Linear(256, 128), nn.ReLU(),
            nn.Linear(128, 64),  nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        out, _ = self.gru(x)
        return self.fc(out[:, -1, :])


BUILDERS = {
    "RNN Simple": RNNModel,
    "LSTM":       LSTMModel,
    "GRU":        GRUModel,
}


# ─────────────────────────────────────────────────────────────
# 4. ENTRENAMIENTO INCREMENTAL CON GPU
# ─────────────────────────────────────────────────────────────
def train_model(name, ModelClass, ckpt_path):
    model = ModelClass().to(DEVICE)

    start_epoch = 0
    best_val    = float('inf')

    if os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location=DEVICE)
        model.load_state_dict(ckpt['model_state'])
        best_val    = ckpt.get('best_val', float('inf'))
        start_epoch = ckpt.get('epoch', 0)
        print(f"  [OK] Checkpoint cargado: {ckpt_path}  (epoch {start_epoch}, best_val={best_val:.6f})")
    else:
        print(f"  [NEW] Sin checkpoint previo. Entrenando desde cero.")

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=7
    )

    train_losses, val_losses = [], []
    no_improve = 0

    for epoch in range(start_epoch, start_epoch + EPOCHS):
        # ── TRAIN ──
        model.train()
        epoch_loss = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            optimizer.zero_grad()
            pred = model(xb)
            loss = criterion(pred, yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            epoch_loss += loss.item() * len(xb)
        train_loss = epoch_loss / len(X_train)

        # ── VALIDATION ──
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                val_loss += criterion(model(xb), yb).item() * len(xb)
        val_loss /= len(X_val)

        scheduler.step(val_loss)
        train_losses.append(train_loss)
        val_losses.append(val_loss)

        # ── CHECKPOINT si mejora ──
        if val_loss < best_val:
            best_val = val_loss
            no_improve = 0
            torch.save({
                'model_state': model.state_dict(),
                'best_val':    best_val,
                'epoch':       epoch + 1,
            }, ckpt_path)
        else:
            no_improve += 1

        if (epoch - start_epoch) % 10 == 0 or no_improve == 0:
            lr_now = optimizer.param_groups[0]['lr']
            tag    = " [MEJOR]" if no_improve == 0 else ""
            print(f"    Epoch {epoch+1:3d}/{start_epoch+EPOCHS}  "
                  f"train={train_loss:.6f}  val={val_loss:.6f}  "
                  f"lr={lr_now:.6f}{tag}")

        # ── EARLY STOPPING ──
        if no_improve >= PATIENCE:
            print(f"    Early Stopping en epoch {epoch+1} (sin mejora en {PATIENCE} epocas)")
            break

    # Cargar los mejores pesos para evaluacion
    best_ckpt = torch.load(ckpt_path, map_location=DEVICE)
    model.load_state_dict(best_ckpt['model_state'])
    print(f"  Mejor val_loss: {best_val:.6f}")
    return model, train_losses, val_losses


def evaluate(model, X_arr, y_arr):
    """Pasa X_arr por el modelo en batches y devuelve pred/real desnormalizados."""
    model.eval()
    all_preds = []
    loader = make_loader(X_arr, y_arr, shuffle=False)
    with torch.no_grad():
        for xb, _ in loader:
            all_preds.append(model(xb.to(DEVICE)).cpu().numpy())
    pred_sc = np.vstack(all_preds)
    y_pred  = close_scaler.inverse_transform(pred_sc).flatten()
    y_true  = close_scaler.inverse_transform(y_arr.reshape(-1, 1)).flatten()
    mse     = mean_squared_error(y_true, y_pred)
    return y_pred, y_true, {"MSE": mse, "RMSE": np.sqrt(mse), "MAE": mean_absolute_error(y_true, y_pred)}


# ─────────────────────────────────────────────────────────────
# ENTRENAR LOS 3 MODELOS
# ─────────────────────────────────────────────────────────────
print("\n3. ENTRENAMIENTO EN GPU (100 epocas max, Early Stopping patience=15)")
print("=" * 60)

results = {}
for name, ModelClass in BUILDERS.items():
    print(f"\n{'='*60}")
    print(f"  Modelo: {name}")
    print(f"{'='*60}")
    model, tr_loss, va_loss = train_model(name, ModelClass, CKPT[name])
    # Evaluar sobre el historico COMPLETO para la grafica retrospectiva
    y_pred, y_true, metrics = evaluate(model, X_full, y_full)
    # Metricas sobre val (los ultimos 15%) para comparar modelos honestamente
    val_start = len(X_train)
    _, _, metrics_val = evaluate(model, X_val, y_val)
    results[name] = {
        "model":       model,
        "tr_loss":     tr_loss,
        "va_loss":     va_loss,
        "y_pred":      y_pred,       # historico completo
        "y_true":      y_true,       # historico completo
        "dates":       dates_full,   # historico completo
        "val_start":   val_start,    # indice donde empieza val (para marcar en grafica)
        "metrics":     metrics_val,  # metricas sobre val (zona no vista durante train)
    }
    m = metrics
    print(f"  TEST -> MSE={m['MSE']:,.1f}  RMSE={m['RMSE']:,.2f}  MAE={m['MAE']:,.2f}")


# ─────────────────────────────────────────────────────────────
# 5. TABLA COMPARATIVA
# ─────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("4. METRICAS COMPARATIVAS")
print("=" * 60)
metrics_df = pd.DataFrame({n: r["metrics"] for n, r in results.items()}).T
print(metrics_df.to_string())
best = metrics_df["RMSE"].idxmin()
print(f"\n  Mejor modelo (RMSE minimo): {best}")


# ─────────────────────────────────────────────────────────────
# 6. GRAFICOS — ANALISIS COMPLETO DE REGRESION
# ─────────────────────────────────────────────────────────────
from scipy import stats
from statsmodels.graphics.tsaplots import plot_acf
from sklearn.metrics import r2_score

print("\n" + "=" * 60)
print("5. GENERANDO GRAFICOS DE REGRESION")
print("=" * 60)

# Anadir R2 a las metricas
for name, res in results.items():
    r2 = r2_score(res["y_true"], res["y_pred"])
    res["metrics"]["R2"] = r2

model_names = list(results.keys())

# ─────────────────────────────────────────────────────────────
# MATRIZ UNICA: todas las graficas de regresion en un solo PNG
#
# Layout (10 filas x 3 columnas):
#   Fila 0  : Serie temporal RNN | LSTM | GRU
#   Fila 1  : Residuos en el tiempo RNN | LSTM | GRU
#   Fila 2  : Scatter pred vs real RNN | LSTM | GRU
#   Fila 3  : Distribucion residuos RNN | LSTM | GRU
#   Fila 4  : Q-Q plot RNN | LSTM | GRU
#   Fila 5  : ACF residuos RNN | LSTM | GRU
#   Fila 6  : Curvas de perdida RNN | LSTM | GRU
#   Fila 7  [span 3 cols]: Error absoluto acumulado (3 modelos)
#   Fila 8  [span 3 cols]: Comparativa series (3 modelos apilados) — 3 sub-filas internas
#   Fila 9  : Barras MSE | RMSE | MAE | R2  (4 cols -> 3+1 merge)
# ─────────────────────────────────────────────────────────────

import matplotlib.gridspec as gridspec

FIG_W  = 36    # ancho total en pulgadas
ROW_H  = 3.8   # altura por fila estandar
N_ROWS = 10

fig = plt.figure(figsize=(FIG_W, ROW_H * N_ROWS), facecolor='white')
fig.suptitle(
    "Analisis Completo de Regresion — IBEX 35 | RNN / LSTM / GRU | GPU RTX 5050",
    fontsize=18, fontweight='bold', y=1.002
)

gs = gridspec.GridSpec(
    N_ROWS, 3,
    figure=fig,
    hspace=0.55,
    wspace=0.30,
    left=0.04, right=0.98, top=0.985, bottom=0.03
)

def label(ax, txt, fs=9):
    ax.set_title(txt, fontsize=fs, pad=4)

# ── FILA 0: Serie temporal completa ──────────────────────────
for col, (name, res) in enumerate(results.items()):
    ax = fig.add_subplot(gs[0, col])
    vd = res["dates"][res["val_start"]]
    ax.axvspan(vd, res["dates"][-1], alpha=0.08, color='gray')
    ax.axvline(vd, color='gray', linestyle='--', lw=0.8, alpha=0.6)
    ax.plot(res["dates"], res["y_true"], color='black',      lw=0.7, label='Real')
    ax.plot(res["dates"], res["y_pred"], color=COLORS[name], lw=0.7, alpha=0.85, label='Pred')
    ax.fill_between(res["dates"], res["y_true"], res["y_pred"], alpha=0.15, color=COLORS[name])
    m = res["metrics"]
    label(ax, f"Serie: {name}  RMSE={m['RMSE']:,.0f}  R2={m['R2']:.4f}")
    ax.set_ylabel("pts"); ax.legend(fontsize=7, loc='upper left'); ax.grid(True, alpha=0.2)
    ax.tick_params(axis='x', labelsize=7, rotation=20)

# ── FILA 1: Residuos en el tiempo ────────────────────────────
for col, (name, res) in enumerate(results.items()):
    ax  = fig.add_subplot(gs[1, col])
    res_v = res["y_true"] - res["y_pred"]
    vd  = res["dates"][res["val_start"]]
    ax.axvspan(vd, res["dates"][-1], alpha=0.07, color='gray')
    ax.axhline(0, color='black', lw=0.8, linestyle='--')
    ax.plot(res["dates"], res_v, color=COLORS[name], lw=0.6, alpha=0.8)
    ax.fill_between(res["dates"], res_v, 0, where=(res_v > 0), alpha=0.25, color='green')
    ax.fill_between(res["dates"], res_v, 0, where=(res_v < 0), alpha=0.25, color='red')
    label(ax, f"Residuos: {name}")
    ax.set_ylabel("Real - Pred (pts)"); ax.grid(True, alpha=0.2)
    ax.tick_params(axis='x', labelsize=7, rotation=20)

# ── FILA 2: Scatter pred vs real ─────────────────────────────
for col, (name, res) in enumerate(results.items()):
    ax = fig.add_subplot(gs[2, col])
    yt, yp = res["y_true"], res["y_pred"]
    lo, hi = min(yt.min(), yp.min()), max(yt.max(), yp.max())
    ax.scatter(yt, yp, alpha=0.1, s=2, color=COLORS[name])
    ax.plot([lo, hi], [lo, hi], 'k--', lw=1.2, label='y=x')
    sl, ic, r, *_ = stats.linregress(yt, yp)
    xl = np.linspace(lo, hi, 200)
    ax.plot(xl, sl*xl + ic, 'r-', lw=1.2, label=f'r={r:.3f}')
    m = res["metrics"]
    label(ax, f"Scatter: {name}  R2={m['R2']:.4f}")
    ax.set_xlabel("Real"); ax.set_ylabel("Pred")
    ax.legend(fontsize=7); ax.grid(True, alpha=0.3)

# ── FILA 3: Distribucion de residuos ─────────────────────────
for col, (name, res) in enumerate(results.items()):
    ax    = fig.add_subplot(gs[3, col])
    res_v = res["y_true"] - res["y_pred"]
    mu, sg = res_v.mean(), res_v.std()
    ax.hist(res_v, bins=80, density=True, color=COLORS[name], alpha=0.6, edgecolor='white', lw=0.2)
    x = np.linspace(res_v.min(), res_v.max(), 300)
    ax.plot(x, stats.norm.pdf(x, mu, sg), 'k-', lw=1.4)
    ax.axvline(0, color='red', linestyle='--', lw=1)
    label(ax, f"Dist. residuos: {name}  mu={mu:.0f}  std={sg:.0f}")
    ax.set_xlabel("Residuo"); ax.set_ylabel("Densidad"); ax.grid(True, alpha=0.3)

# ── FILA 4: Q-Q plot ─────────────────────────────────────────
for col, (name, res) in enumerate(results.items()):
    ax    = fig.add_subplot(gs[4, col])
    res_v = res["y_true"] - res["y_pred"]
    (osm, osr), (sl, ic, r) = stats.probplot(res_v, dist="norm")
    ax.scatter(osm, osr, alpha=0.15, s=3, color=COLORS[name])
    xl = np.array([min(osm), max(osm)])
    ax.plot(xl, sl*xl + ic, 'k-', lw=1.4, label=f'r={r:.4f}')
    label(ax, f"Q-Q Plot: {name}")
    ax.set_xlabel("Cuantiles teoricos"); ax.set_ylabel("Cuantiles muestrales")
    ax.legend(fontsize=7); ax.grid(True, alpha=0.3)

# ── FILA 5: ACF residuos ─────────────────────────────────────
for col, (name, res) in enumerate(results.items()):
    ax    = fig.add_subplot(gs[5, col])
    res_v = res["y_true"] - res["y_pred"]
    plot_acf(res_v, ax=ax, lags=40, color=COLORS[name], alpha=0.05, title="")
    label(ax, f"ACF Residuos: {name}")
    ax.set_xlabel("Lag"); ax.set_ylabel("ACF"); ax.grid(True, alpha=0.3)

# ── FILA 6: Curvas de perdida ─────────────────────────────────
for col, (name, res) in enumerate(results.items()):
    ax = fig.add_subplot(gs[6, col])
    ep = range(1, len(res["tr_loss"]) + 1)
    ax.plot(ep, res["tr_loss"], color=COLORS[name], lw=1.2, label='Train')
    ax.plot(ep, res["va_loss"], color=COLORS[name], lw=1.2, linestyle='--', label='Val')
    label(ax, f"Loss: {name}")
    ax.set_xlabel("Epoca"); ax.set_ylabel("MSE")
    ax.legend(fontsize=7); ax.grid(True, alpha=0.3)

# ── FILA 7: Error absoluto acumulado (span 3 cols) ───────────
ax8 = fig.add_subplot(gs[7, :])
for name, res in results.items():
    ae = np.abs(res["y_true"] - res["y_pred"]).cumsum()
    ax8.plot(res["dates"], ae, color=COLORS[name], lw=1.3, label=name)
label(ax8, "Error Absoluto Acumulado — 3 modelos", fs=11)
ax8.set_xlabel("Fecha"); ax8.set_ylabel("Error acum. (pts)")
ax8.legend(fontsize=9); ax8.grid(True, alpha=0.25)
ax8.tick_params(axis='x', labelsize=8, rotation=15)

# ── FILA 8: Comparativa series 3 modelos (span 3 cols, 3 sub-filas) ──
gs_inner = gridspec.GridSpecFromSubplotSpec(3, 1, subplot_spec=gs[8, :], hspace=0.45)
for row_i, (name, res) in enumerate(results.items()):
    ax = fig.add_subplot(gs_inner[row_i])
    vd = res["dates"][res["val_start"]]
    m  = res["metrics"]
    ax.axvspan(vd, res["dates"][-1], alpha=0.07, color='gray')
    ax.axvline(vd, color='gray', linestyle='--', lw=0.7, alpha=0.5)
    ax.plot(res["dates"], res["y_true"], color='black',      lw=0.7, label='Real')
    ax.plot(res["dates"], res["y_pred"], color=COLORS[name], lw=0.7, alpha=0.85, label=name)
    ax.fill_between(res["dates"], res["y_true"], res["y_pred"], alpha=0.12, color=COLORS[name])
    label(ax, f"{name}  RMSE={m['RMSE']:,.0f}  MAE={m['MAE']:,.0f}  R2={m['R2']:.4f}", fs=10)
    ax.set_ylabel("pts"); ax.legend(fontsize=7, loc='upper left'); ax.grid(True, alpha=0.2)
    ax.tick_params(axis='x', labelsize=7, rotation=15)
    if row_i == 0:
        ax.set_title("Comparativa Retrospectiva Completa (1993-2026)", fontsize=12, pad=6)

# ── FILA 9: Barras de metricas (MSE / RMSE / MAE / R2) ───────
gs_bar = gridspec.GridSpecFromSubplotSpec(1, 4, subplot_spec=gs[9, :], wspace=0.35)
bar_colors = [COLORS[n] for n in model_names]
for bi, metric in enumerate(["MSE", "RMSE", "MAE", "R2"]):
    ax = fig.add_subplot(gs_bar[bi])
    vals = [results[n]["metrics"][metric] for n in model_names]
    bars = ax.bar(model_names, vals, color=bar_colors, edgecolor='black', width=0.5)
    fmt = ".4f" if metric == "R2" else ",.0f"
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() * 1.01,
                f"{v:{fmt}}", ha='center', va='bottom', fontsize=8)
    label(ax, metric, fs=11)
    ax.set_ylabel("Valor"); ax.grid(True, axis='y', alpha=0.3)
    ax.tick_params(axis='x', labelsize=9)

plt.savefig("analisis_regresion_completo.png", dpi=130, bbox_inches='tight')
plt.close()
print("  Guardado: analisis_regresion_completo.png")

# ── TABLA RESUMEN IMPRESA ──
metrics_df = pd.DataFrame({n: r["metrics"] for n, r in results.items()}).T
print("\n  Tabla resumen:")
print(metrics_df.to_string())
best = metrics_df["RMSE"].idxmin()

# ─────────────────────────────────────────────────────────────
# 7. RESUMEN
# ─────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("RESUMEN FINAL")
print("=" * 60)
print(f"  GPU usada : {torch.cuda.get_device_name(0) if DEVICE.type=='cuda' else 'CPU'}")
print(f"  Datos     : {START_DATE} -> {end_date}  ({len(df)} sesiones, 100% usados)")
print(f"  Features  : {n_features}  ({feature_cols})")
print(f"  Mejor modelo (RMSE): {best}\n")
print(metrics_df.to_string())
print("""
  Logica de reentrenamiento incremental:
  - Manana: py ibex35_models.py
  -> Descarga datos frescos del IBEX automaticamente
  -> Carga checkpoint (ckpt_*.pt) con los mejores pesos
  -> Continua entrenando (fine-tuning) con los datos nuevos
  -> Guarda nuevo checkpoint solo si mejora el val_loss
""")
print("Archivos guardados:")
for f in ["analisis_regresion_completo.png", "descomposicion_serie.png",
          "ckpt_rnn.pt", "ckpt_lstm.pt", "ckpt_gru.pt"]:
    print(f"  - {f}")
print("=" * 60)
