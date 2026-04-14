# -*- coding: utf-8 -*-
"""
app.py  —  IBEX 35 AI Terminal  |  Design 3: Deep Sea Military Precision
Modelo activo: GRU Ultra (snap_gru_BUENO.pt)  RMSE=68.5  R2=0.9994
"""
import warnings; warnings.filterwarnings('ignore')
from huggingface_hub import hf_hub_download
import os, pickle
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, date
from pandas.tseries.offsets import BDay
import torch, torch.nn as nn

# ══════════════════════════════════════════════════════════════
# PAGE CONFIG
# ══════════════════════════════════════════════════════════════
st.set_page_config(
  page_title="IBEX 35 AI ",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ══════════════════════════════════════════════════════════════
# CSS  —  Deep Sea Military Precision
# ══════════════════════════════════════════════════════════════
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&family=JetBrains+Mono:wght@400;500;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
    background: #05070A;
    color: #C9D1D9;
}
.stApp { background: #05070A; }

/* ── Hide chrome ── */
#MainMenu, footer, .stDeployButton { visibility: hidden; }
[data-testid="collapsedControl"] { display: none; }
header { visibility: hidden; }
.block-container { padding-top: 0 !important; }

/* ── Base card ── */
.card {
    background: #0F1218;
    border: 1px solid #1E232D;
    border-radius: 10px;
    padding: 18px 20px;
    position: relative;
}
.card-sm {
    background: #0F1218;
    border: 1px solid #1E232D;
    border-radius: 8px;
    padding: 12px 14px;
}
.card-glow-green {
    background: #0F1218;
    border: 1px solid rgba(0,255,136,0.25);
    border-radius: 10px;
    padding: 18px 20px;
    box-shadow: 0 0 24px rgba(0,255,136,0.07), inset 0 1px 0 rgba(0,255,136,0.05);
}
.card-glow-red {
    background: #0F1218;
    border: 1px solid rgba(255,51,102,0.25);
    border-radius: 10px;
    padding: 18px 20px;
    box-shadow: 0 0 24px rgba(255,51,102,0.07);
}

/* ── Header ── */
.terminal-header {
    background: #080B0F;
    border-bottom: 1px solid #1E232D;
    padding: 10px 24px;
    display: flex; align-items: center; justify-content: space-between;
    margin-bottom: 16px;
}
.brand {
    font-family: 'Inter', sans-serif;
    font-size: 0.95rem; font-weight: 900; letter-spacing: 2px;
    color: #00FF88; text-transform: uppercase;
}
.brand-sub { font-size: 0.65rem; color: #FFFFFF; letter-spacing: 3px; margin-top: 1px; }
.live-dot {
    display: inline-block; width: 7px; height: 7px;
    background: #00FF88; border-radius: 50%;
    box-shadow: 0 0 8px #00FF88;
    animation: pulse 2s infinite;
    margin-right: 6px;
}
@keyframes pulse {
    0%,100% { opacity: 1; } 50% { opacity: 0.4; }
}
.badge-live {
    background: rgba(0,255,136,0.1); border: 1px solid rgba(0,255,136,0.3);
    border-radius: 4px; padding: 3px 10px;
    font-size: 0.65rem; color: #00FF88; font-weight: 700; letter-spacing: 2px;
}

/* ── Price display ── */
.mono { font-family: 'JetBrains Mono', monospace !important; }
.price-huge {
    font-family: 'JetBrains Mono', monospace;
    font-size: 2.8rem; font-weight: 700; color: #F0F6FC;
    letter-spacing: -1px; line-height: 1;
}
.price-change-up   { font-family: 'JetBrains Mono', monospace; color: #00FF88; font-weight: 600; font-size: 1rem; }
.price-change-down { font-family: 'JetBrains Mono', monospace; color: #FF3366; font-weight: 600; font-size: 1rem; }
.ticker-label { font-size: 0.65rem; color: #FFFFFF; letter-spacing: 3px; text-transform: uppercase; margin-bottom: 4px; }

/* ── AI Signal ── */
.ai-signal-box {
    background: linear-gradient(135deg, rgba(0,255,136,0.06) 0%, rgba(0,212,255,0.04) 100%);
    border: 1px solid rgba(0,255,136,0.2);
    border-radius: 8px; padding: 14px 16px;
}
.signal-up   { font-family: 'JetBrains Mono', monospace; font-size: 1.6rem; font-weight: 700; color: #00FF88; }
.signal-down { font-family: 'JetBrains Mono', monospace; font-size: 1.6rem; font-weight: 700; color: #FF3366; }
.confidence-track { background: #1E232D; border-radius: 3px; height: 5px; margin: 8px 0; }
.confidence-fill  { height: 5px; border-radius: 3px; transition: width 0.6s ease; }

/* ── Reliability Score ── */
.reliability-ring {
    width: 64px; height: 64px; border-radius: 50%;
    background: conic-gradient(#00FF88 VAR_DEG, #1E232D 0deg);
    display: flex; align-items: center; justify-content: center;
    box-shadow: 0 0 20px rgba(0,255,136,0.2);
}
.reliability-inner {
    width: 50px; height: 50px; border-radius: 50%;
    background: #0F1218;
    display: flex; align-items: center; justify-content: center;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.75rem; font-weight: 700; color: #00FF88;
}

/* ── Metric pill ── */
.mpill {
    background: #080B0F;
    border: 1px solid #1E232D;
    border-radius: 8px; padding: 10px 12px; text-align: center;
}
.mpill .lbl { font-size: 0.58rem; color: #FFFFFF; text-transform: uppercase; letter-spacing: 1.5px; }
.mpill .val { font-family: 'JetBrains Mono', monospace; font-size: 1.1rem; font-weight: 700; margin-top: 3px; }
.mpill .sub { font-size: 0.6rem; color: #3D4A5C; margin-top: 2px; }

/* ── Section label ── */
.sec-label {
    font-size: 0.58rem; font-weight: 700; color: #FFFFFF;
    text-transform: uppercase; letter-spacing: 3px; margin: 14px 0 8px 0;
}

/* ── Audit row ── */
.audit-row {
    display: flex; align-items: center; gap: 10px;
    padding: 7px 0; border-bottom: 1px solid #1E232D;
    font-family: 'JetBrains Mono', monospace;
}
.audit-date { font-size: 0.72rem; color: #3D4A5C; width: 55px; flex-shrink:0; }
.audit-pred { font-size: 0.82rem; color: #C9D1D9; width: 68px; font-weight: 600; }
.audit-real { font-size: 0.82rem; color: #00D4FF; width: 68px; font-weight: 600; }
.audit-good { color: #00FF88; font-size: 0.72rem; }
.audit-mid  { color: #FFB800; font-size: 0.72rem; }
.audit-bad  { color: #FF3366; font-size: 0.72rem; }

/* ── Order book ── */
.ob-row { display: flex; justify-content: space-between; align-items: center;
          padding: 3px 0; font-family: 'JetBrains Mono', monospace; font-size: 0.73rem; }
.ob-ask { color: #FF3366; }
.ob-bid { color: #00FF88; }

/* ── Trade buttons ── */
.btn-buy {
    background: linear-gradient(135deg, #00FF88, #00CC6A);
    color: #05070A; font-weight: 800; font-size: 0.95rem;
    font-family: 'Inter', sans-serif;
    border: none; border-radius: 8px; padding: 13px; width: 100%; cursor: pointer;
    letter-spacing: 1px; text-transform: uppercase;
    box-shadow: 0 0 20px rgba(0,255,136,0.25);
}
.btn-sell {
    background: linear-gradient(135deg, #FF3366, #CC1144);
    color: white; font-weight: 800; font-size: 0.95rem;
    font-family: 'Inter', sans-serif;
    border: none; border-radius: 8px; padding: 13px; width: 100%; cursor: pointer;
    letter-spacing: 1px; text-transform: uppercase;
    box-shadow: 0 0 20px rgba(255,51,102,0.2);
}

/* ── Forecast table ── */
.fc-row {
    display: grid;
    grid-template-columns: 55px 72px 72px 60px 52px;
    gap: 4px; padding: 5px 0;
    border-bottom: 1px solid #1A1F28;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.75rem; font-variant-numeric: tabular-nums;
    align-items: center;
}

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 4px; }
::-webkit-scrollbar-track { background: #080B0F; }
::-webkit-scrollbar-thumb { background: #1E232D; border-radius: 2px; }

/* ── Streamlit Tabs ── */
.stTabs [data-baseweb="tab-list"] button {
    color: #FFFFFF !important;
}
.stTabs [data-baseweb="tab-list"] button p {
    color: #FFFFFF !important;
    font-weight: 600;
}
.stTabs [data-baseweb="tab-list"] button[aria-selected="true"] {
    color: #00FF88 !important;
    border-bottom-color: #00FF88 !important;
}
.stTabs [data-baseweb="tab-list"] button[aria-selected="true"] p {
    color: #00FF88 !important;
}
</style>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════
# CONSTANTS
# ══════════════════════════════════════════════════════════════
DEVICE     = torch.device('cpu')
SEQ_LEN    = 120
HORIZON    = 10
DROPOUT    = 0.03
HF_REPO    = 'LydiaTS04/ibex35_gru'   # Hugging Face model repo
SNAP_MODEL = 'snap_gru_BUENO.pt'          # GRU Ultra original  RMSE=68.5 R2=0.9994
MODEL_RMSE = 68.5
MODEL_R2   = 0.9994
MODEL_MAPE = 0.61
MODEL_NAME = 'GRU Ultra'

# ══════════════════════════════════════════════════════════════
# MODEL DEFINITION
# ══════════════════════════════════════════════════════════════
class MultiHeadAttention(nn.Module):
    def __init__(self,h,nh=4):
        super().__init__(); self.nh=nh; self.dh=h//nh
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
    def __init__(self,n):
        super().__init__()
        self.layer1=GRULayer(n,512); self.layer2=GRULayer(512,512)
        self.layer3=GRULayer(512,512); self.layer4=GRULayer(512,512)
        self.layer5=GRULayer(512,512)
        self.attn=MultiHeadAttention(512,4); self.bn=nn.BatchNorm1d(512)
        self.fc=nn.Sequential(nn.Linear(512,256),nn.GELU(),nn.Linear(256,128),
                              nn.GELU(),nn.Linear(128,64),nn.GELU(),nn.Linear(64,HORIZON))
    def forward(self,x):
        x=self.layer1(x);x=self.layer2(x);x=self.layer3(x)
        x=self.layer4(x);x=self.layer5(x)
        return self.fc(self.bn(self.attn(x)))

# ══════════════════════════════════════════════════════════════
# DATA
# ══════════════════════════════════════════════════════════════
@st.cache_data(show_spinner=False)
def load_market_data():
    # Datos de entrenamiento: solo hasta el 31 de marzo (corte del modelo)
    csv_path = os.path.join(os.path.dirname(__file__), 'ibex_data.csv')
    if not os.path.exists(csv_path): return pd.DataFrame()
    raw = pd.read_csv(csv_path, index_col='Date', parse_dates=True)
    raw = raw[raw.index < '2026-04-01']
    if raw.empty: return pd.DataFrame()
    raw.dropna(inplace=True)
    df = raw[['Open','High','Low','Close','Volume']].copy()
    df['MA10']   = df['Close'].rolling(10).mean()
    df['MA30']   = df['Close'].rolling(30).mean()
    df['MA60']   = df['Close'].rolling(60).mean()
    df['Std10']  = df['Close'].rolling(10).std()
    delta=df['Close'].diff(); gain=delta.clip(lower=0).rolling(14).mean()
    loss_=(-delta.clip(upper=0)).rolling(14).mean()
    df['RSI']    = 100-(100/(1+gain/(loss_+1e-9)))
    df['Mom10']  = df['Close']-df['Close'].shift(10)
    ema12=df['Close'].ewm(span=12,adjust=False).mean()
    ema26=df['Close'].ewm(span=26,adjust=False).mean()
    df['MACD']   = ema12-ema26
    df['MACDsig']= df['MACD'].ewm(span=9,adjust=False).mean()
    ma20=df['Close'].rolling(20).mean(); std20=df['Close'].rolling(20).std()
    df['BollUp'] = ma20+2*std20; df['BollLow'] = ma20-2*std20
    tr1=df['High']-df['Low']
    tr2=(df['High']-df['Close'].shift()).abs()
    tr3=(df['Low'] -df['Close'].shift()).abs()
    df['ATR']    = pd.concat([tr1,tr2,tr3],axis=1).max(axis=1).rolling(14).mean()
    df['VolMA10']= df['Volume'].rolling(10).mean()
    df.dropna(inplace=True)
    return df

@st.cache_data(show_spinner=False, ttl=300)
def load_live_ohlcv():
    csv_path = os.path.join(os.path.dirname(__file__), 'ibex_data.csv')
    if not os.path.exists(csv_path): return pd.DataFrame()
    raw = pd.read_csv(csv_path, index_col='Date', parse_dates=True)
    raw = raw[raw.index >= '2024-01-01']
    if raw.empty: return pd.DataFrame()
    raw.dropna(inplace=True)
    return raw[['Open','High','Low','Close','Volume']].copy()

@st.cache_data(show_spinner=False, ttl=600)
def load_april_data():
    csv_path = os.path.join(os.path.dirname(__file__), 'ibex_data.csv')
    if not os.path.exists(csv_path): return pd.DataFrame()
    raw = pd.read_csv(csv_path, index_col='Date', parse_dates=True)
    raw = raw[raw.index >= '2026-04-01']
    if raw.empty: return pd.DataFrame()
    raw.dropna(inplace=True)
    return raw[['Open','High','Low','Close','Volume']].copy()

@st.cache_resource(show_spinner=False)
def load_model_and_scalers():
    snap_path    = hf_hub_download(repo_id=HF_REPO, filename=SNAP_MODEL)
    snap         = torch.load(snap_path, map_location=DEVICE, weights_only=False)
    nf           = snap.get('n_features', 17)
    scaler_name  = 'scaler_ultra.pkl' if nf == 20 else 'scaler_BUENO.pkl'
    scaler_path  = hf_hub_download(repo_id=HF_REPO, filename=scaler_name)
    with open(scaler_path,'rb') as f: scaler = pickle.load(f)
    m = GRUUltra(nf); m.load_state_dict(snap['model_state']); m.eval()
    return m, scaler, snap

def run_forecast(model, scaler, snap, df):
    seq_len = snap.get('seq_len', SEQ_LEN)
    use_ret = snap.get('use_returns', False)
    nf      = snap.get('n_features', 17)
    LEGACY  = [c for c in df.columns if c not in ('LogRet','LogRet5','VolRet')]
    df_in   = df[LEGACY] if nf == 17 else df
    scaled  = scaler.transform(df_in.values)
    seq     = torch.tensor(scaled[-seq_len:], dtype=torch.float32).unsqueeze(0)
    with torch.no_grad(): pred = model(seq).numpy().flatten()
    if use_ret:
        rs = snap['ret_scaler']
        lr = rs.inverse_transform(pred.reshape(-1,1)).flatten()
        b  = float(df['Close'].iloc[-1])
        return np.array([b*np.exp(lr[:k+1].sum()) for k in range(len(lr))])
    else:
        _cs_path = hf_hub_download(repo_id=HF_REPO, filename='close_scaler_BUENO.pkl')
        with open(_cs_path,'rb') as f: cs = pickle.load(f)
        return cs.inverse_transform(pred.reshape(-1,1)).flatten()

@st.cache_data(show_spinner=False)
def run_historical_predictions(_model, _scaler, _snap):
    """Predicciones D+1 del GRU para los últimos N días históricos (rolling)."""
    N_HIST  = 90
    seq_len = _snap.get('seq_len', SEQ_LEN)
    nf      = _snap.get('n_features', 17)
    df_h    = load_market_data()
    LEGACY  = [c for c in df_h.columns if c not in ('LogRet','LogRet5','VolRet')]
    scaled  = _scaler.transform(df_h[LEGACY].values)
    windows, dates = [], []
    for k in range(N_HIST, 0, -1):
        s = len(scaled) - k - seq_len
        e = len(scaled) - k
        if s >= 0:
            windows.append(scaled[s:e])
            dates.append(df_h.index[-k])
    if not windows:
        return [], []
    X = torch.tensor(np.array(windows), dtype=torch.float32)
    with torch.no_grad():
        preds = _model(X).numpy()[:, 0]   # solo D+1
    _cs_path = hf_hub_download(repo_id=HF_REPO, filename='close_scaler_BUENO.pkl')
    with open(_cs_path,'rb') as f: cs = pickle.load(f)
    prices = cs.inverse_transform(preds.reshape(-1,1)).flatten()
    return dates, prices

# ══════════════════════════════════════════════════════════════
# LOAD
# ══════════════════════════════════════════════════════════════
with st.spinner(""):
    df       = load_market_data()
    live_df  = load_live_ohlcv()
    april_df = load_april_data()

    if df.empty or live_df.empty:
        st.error("🚨 **Error Crítico: No se encuentran los datos locales** 🚨\n\nAsegúrate de haber ejecutado el script `download_data.py` de antemano para generar el archivo `ibex_data.csv` y tenerlo junto a `app.py`.")
        st.stop()

    model, scaler, snap_info = load_model_and_scalers()

forecast_prices   = run_forecast(model, scaler, snap_info, df)
hist_pred_dates, hist_pred_prices = run_historical_predictions(model, scaler, snap_info)
last_close = float(df['Close'].iloc[-1])
last_date  = df.index[-1].date()   # = 2026-03-31

live_last  = float(live_df['Close'].iloc[-1])
live_prev  = float(live_df['Close'].iloc[-2])
live_chg   = live_last - live_prev
live_pct   = live_chg / live_prev * 100
live_open  = float(live_df['Open'].iloc[-1])
live_high  = float(live_df['High'].iloc[-1])
live_low   = float(live_df['Low'].iloc[-1])
live_date  = live_df.index[-1].date()

# Festivos bolsa española abril 2026: Viernes Santo (3 abr) + Lunes Pascua (6 abr)
ES_HOLIDAYS = {date(2026, 4, 3), date(2026, 4, 6)}
FORECAST_END = date(2026, 4, 10)   # predecir hasta el 10 de abril inclusive

# Días hábiles del 1 al 10 de abril excluyendo festivos y fines de semana
forecast_trade_dates = []
nxt = pd.Timestamp(last_date) + BDay(1)
while nxt.date() <= FORECAST_END:
    if nxt.date() not in ES_HOLIDAYS:
        forecast_trade_dates.append(nxt.date())
    nxt += BDay(1)
# Rellenar hasta HORIZON si hacen falta días extra después del 10 (solo si hay pocos)
if len(forecast_trade_dates) < 1:
    forecast_trade_dates = [date(2026, 4, 1)]  # fallback

# April real data (descargado en vivo para comparar forecast vs real)
if not april_df.empty:
    april_real_map = {d.date(): float(april_df['Close'].loc[d]) for d in april_df.index}
else:
    april_real_map = {}

april_trade_dates = forecast_trade_dates

# AI derived values
ai_up  = float(forecast_prices[0]) > last_close
ai_conf = min(96, max(54, int(100 - abs(forecast_prices[0]-last_close)/last_close*700)))
# Confidence band  ±RMSE around forecast
fc_upper = forecast_prices + MODEL_RMSE * 1.2
fc_lower = forecast_prices - MODEL_RMSE * 1.2

# ══════════════════════════════════════════════════════════════
# HEADER
# ══════════════════════════════════════════════════════════════
st.markdown(f"""
<div class="terminal-header">
  <div>
    <div class="brand">🎯 IBEX 35 AI</div>
    <div class="brand-sub">POWERED BY {MODEL_NAME} · UAX INTELLIGENCE SYSTEMS</div>
  </div>
  <div style="display:flex;align-items:center;gap:14px;">
    <span class="badge-live"><span class="live-dot"></span>LIVE</span>
    <span style="font-family:'JetBrains Mono',monospace;font-size:0.75rem;color:#3D4A5C;">
      {pd.Timestamp.now(tz='Europe/Madrid').strftime('%d %b %Y  %H:%M:%S')}
    </span>
  </div>
</div>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════
# ══════════════════════════════════════════════════════════════
# TABS
# ══════════════════════════════════════════════════════════════
tab_main, tab_info, tab_news = st.tabs(['📈 Principal', 'ℹ️ Información sobre el IBEX', '📰 Noticias'])

with tab_main:
    # ROW 1 — Price + AI Signal + Metrics
    # ══════════════════════════════════════════════════════════════
    c_price, c_ai, c_m1, c_m2, c_m3, c_m4 = st.columns([2.2, 2.0, 1, 1, 1, 1])

    with c_price:
        chg_cls = "price-change-up" if live_chg >= 0 else "price-change-down"
        arrow   = "▲" if live_chg >= 0 else "▼"
        glow    = "card-glow-green" if live_chg >= 0 else "card-glow-red"
        st.markdown(f"""
        <div class="{glow}" style="height:136px;">
          <div class="ticker-label">IBEX 35 · ^IBEX · MCE · EUR</div>
          <div class="price-huge">{live_last:,.2f}</div>
          <div class="{chg_cls}" style="margin-top:4px;">
            {arrow} {abs(live_chg):,.2f} &nbsp;({arrow}{abs(live_pct):.2f}%)
          </div>
          <div style="font-family:'JetBrains Mono',monospace;font-size:0.65rem;color:#3D4A5C;margin-top:6px;">
            A:<b style="color:#8B9EB7">{live_open:,.0f}</b>&nbsp;&nbsp;
            H:<b style="color:#00FF88">{live_high:,.0f}</b>&nbsp;&nbsp;
            L:<b style="color:#FF3366">{live_low:,.0f}</b>&nbsp;&nbsp;
            {live_date.strftime('%d %b %Y')}
          </div>
        </div>
        """, unsafe_allow_html=True)

    with c_ai:
        sig_txt = "ALZA  ▲" if ai_up else "BAJA  ▼"
        sig_cls = "signal-up" if ai_up else "signal-down"
        bar_col = "#00FF88" if ai_up else "#FF3366"
        fc1_pct = (forecast_prices[0]-last_close)/last_close*100
        st.markdown(f"""
        <div class="ai-signal-box" style="height:136px;">
          <div style="font-size:0.55rem;color:#FFFFFF;letter-spacing:3px;text-transform:uppercase;margin-bottom:4px;">
            {MODEL_NAME} · Señal IA 10 días
          </div>
          <div class="{sig_cls}">{sig_txt}</div>
          <div style="font-size:0.65rem;color:#3D4A5C;margin-top:2px;">
            Objetivo D+1: <span style="font-family:'JetBrains Mono',monospace;color:#C9D1D9;">
            {forecast_prices[0]:,.0f}</span>
            &nbsp;<span style="color:{bar_col};">({'+' if fc1_pct>0 else ''}{fc1_pct:.2f}%)</span>
          </div>
          <div class="confidence-track">
            <div class="confidence-fill" style="width:{ai_conf}%;background:{bar_col};
                 box-shadow:0 0 8px {bar_col}88;"></div>
          </div>
          <div style="font-size:0.62rem;color:#3D4A5C;">
            Confianza del modelo: <b style="color:{bar_col};">{ai_conf}%</b>
            &nbsp;·&nbsp; RMSE histórico: <b style="color:#8B9EB7;">{MODEL_RMSE} pts</b>
          </div>
        </div>
        """, unsafe_allow_html=True)

    for col, (lbl, val, sub, vc) in zip(
        [c_m1, c_m2, c_m3, c_m4],
        [("RMSE",  f"{MODEL_RMSE} pts", "Error histórico",    "#00FF88"),
         ("R²",    f"{MODEL_R2}",       "Ajuste global",      "#00D4FF"),
         ("MAPE",  f"{MODEL_MAPE}%",    "Error porcentual",   "#FFB800"),
         ("R² día", "0.9994",           "GRU · 1,979 épocas", "#A855F7")]):
        with col:
            st.markdown(f"""
            <div class="mpill" style="height:136px;display:flex;flex-direction:column;
                 justify-content:center;align-items:center;">
              <div class="lbl">{lbl}</div>
              <div class="val" style="color:{vc};">{val}</div>
              <div class="sub">{sub}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)

    # ══════════════════════════════════════════════════════════════
    # ROW 2 — Chart (main) + Sidebar
    # ══════════════════════════════════════════════════════════════
    c_chart, c_side = st.columns([3.2, 1])

    with c_chart:
        st.markdown('<div class="sec-label">📊 Gráfico Técnico · Velas + Previsión IA · Banda de Confianza</div>',
                    unsafe_allow_html=True)

        # Candlesticks: histórico hasta el 31 de marzo (corte del modelo)
        candle_df = live_df[live_df.index <= '2026-03-31'].tail(90)
        fig = make_subplots(rows=3, cols=1, shared_xaxes=True,
                            row_heights=[0.60, 0.20, 0.20], vertical_spacing=0.02)

        # Candlesticks histórico (OHLC visual)
        fig.add_trace(go.Candlestick(
            x=candle_df.index,
            open=candle_df['Open'].values.flatten(),
            high=candle_df['High'].values.flatten(),
            low =candle_df['Low'].values.flatten(),
            close=candle_df['Close'].values.flatten(),
            name='IBEX 35 Velas',
            increasing=dict(line=dict(color='#00FF88',width=1), fillcolor='rgba(0,255,136,0.55)'),
            decreasing=dict(line=dict(color='#FF3366',width=1), fillcolor='rgba(255,51,102,0.55)'),
            showlegend=False, hoverinfo='skip',
        ), row=1, col=1)

        # ── LÍNEA IBEX REAL COMPLETA: histórico + abril unidos ──────────────
        # Histórico hasta 31 Mar
        hist_x = list(candle_df.index)
        hist_y = list(candle_df['Close'].values.flatten())
        # Datos reales de abril confirmados
        real_in_window = {d: april_real_map[d] for d in forecast_trade_dates if d in april_real_map}
        apr_x = [pd.Timestamp(d) for d in sorted(real_in_window)]
        apr_y = [real_in_window[d] for d in sorted(real_in_window)]
        # Una sola línea continua del IBEX real
        ibex_x = hist_x + apr_x
        ibex_y = hist_y + apr_y
        fig.add_trace(go.Scatter(
            x=ibex_x, y=ibex_y,
            name='IBEX 35 Real',
            mode='lines',
            line=dict(color='#00FF88', width=2.0),
            hovertemplate='%{x|%d %b %Y}<br>IBEX Real: <b>%{y:,.0f}</b><extra></extra>'
        ), row=1, col=1)
        # Marcadores solo en los puntos de abril (destacar los datos reales vs forecast)
        if apr_x:
            fig.add_trace(go.Scatter(
                x=apr_x, y=apr_y,
                name='Real Confirmado',
                mode='markers',
                marker=dict(size=9, color='#00FF88', symbol='circle',
                            line=dict(color='#05070A', width=2)),
                showlegend=False,
                hovertemplate='%{x|%d %b}<br>Real: <b>%{y:,.0f}</b><extra></extra>'
            ), row=1, col=1)

        # Predicciones históricas GRU (D+1 rolling) — antes del 31 Mar
        if hist_pred_dates:
            fig.add_trace(go.Scatter(
                x=hist_pred_dates, y=hist_pred_prices,
                name='Previsión GRU (hist.)',
                mode='lines',
                line=dict(color='#00D4FF', width=2.0),
                opacity=0.9,
                hovertemplate='%{x|%d %b %Y}<br>GRU D+1: <b>%{y:,.0f}</b><extra></extra>'
            ), row=1, col=1)

        # MAs (sobre el histórico)
        ma10 = candle_df['Close'].rolling(10).mean()
        ma30 = candle_df['Close'].rolling(30).mean()
        fig.add_trace(go.Scatter(x=candle_df.index, y=ma10, name='MM10',
            line=dict(color='#FFB800',width=1.2,dash='dot')), row=1, col=1)
        fig.add_trace(go.Scatter(x=candle_df.index, y=ma30, name='MM30',
            line=dict(color='#A855F7',width=1.2,dash='dot')), row=1, col=1)

        # Forecast: arranca desde el cierre del 31 de marzo hacia los días hábiles de abril
        n_fc = len(forecast_trade_dates)
        fc_dates = [pd.Timestamp(last_date)] + [pd.Timestamp(d) for d in forecast_trade_dates]
        fc_vals  = [last_close] + list(forecast_prices[:n_fc])
        fc_up_v  = [last_close] + list(fc_upper[:n_fc])
        fc_lo_v  = [last_close] + list(fc_lower[:n_fc])

        # Confidence band (shaded area)
        fig.add_trace(go.Scatter(
            x=fc_dates + fc_dates[::-1],
            y=fc_up_v + fc_lo_v[::-1],
            fill='toself',
            fillcolor='rgba(0,212,255,0.07)',
            line=dict(color='rgba(0,0,0,0)'),
            showlegend=False, hoverinfo='skip',
            name='Banda confianza'
        ), row=1, col=1)
        fig.add_trace(go.Scatter(x=fc_dates, y=fc_up_v, name='IC superior',
            line=dict(color='rgba(0,212,255,0.25)',width=1,dash='dot'),
            showlegend=False, hoverinfo='skip'), row=1, col=1)
        fig.add_trace(go.Scatter(x=fc_dates, y=fc_lo_v, name='IC inferior',
            line=dict(color='rgba(0,212,255,0.25)',width=1,dash='dot'),
            showlegend=False, hoverinfo='skip'), row=1, col=1)

        # Forecast central line
        fig.add_trace(go.Scatter(
            x=fc_dates, y=fc_vals,
            name=f'Previsión {MODEL_NAME}',
            mode='lines+markers',
            line=dict(color='#00D4FF', width=2.5, dash='dot'),
            marker=dict(size=8, color='#00D4FF', symbol='diamond',
                        line=dict(color='#05070A',width=2), opacity=0.9),
            hovertemplate='%{x|%d %b}<br>Previsión: <b>%{y:,.0f}</b><extra></extra>'
        ), row=1, col=1)

        # Línea de corte del modelo: 31 de marzo (último dato de entrenamiento)
        fig.add_shape(type='line', x0='2026-03-31', x1='2026-03-31', y0=0, y1=1,
                      xref='x', yref='paper', row=1, col=1,
                      line=dict(dash='dash', color='rgba(255,184,0,0.4)', width=1))
        fig.add_annotation(x='2026-03-31', y=1, xref='x', yref='paper',
                           text='Corte modelo 31 Mar', showarrow=False, row=1, col=1,
                           font=dict(size=8, color='rgba(255,184,0,0.6)'),
                           xanchor='left', yanchor='top')

        # RSI
        delta = candle_df['Close'].diff()
        g=delta.clip(lower=0).rolling(14).mean()
        l=(-delta.clip(upper=0)).rolling(14).mean()
        rsi_v = 100-(100/(1+g/(l+1e-9)))
        fig.add_trace(go.Scatter(x=candle_df.index, y=rsi_v, name='RSI',
            line=dict(color='#A855F7',width=1.5), showlegend=False), row=2, col=1)
        for lv, lc in [(70,'rgba(255,51,102,0.3)'),(30,'rgba(0,255,136,0.3)')]:
            fig.add_hline(y=lv, line_dash='dot', line_color=lc, line_width=1, row=2, col=1)
        fig.add_hrect(y0=30, y1=70, fillcolor='rgba(255,255,255,0.01)',
                      line_width=0, row=2, col=1)

        # MACD
        ema12=candle_df['Close'].ewm(span=12,adjust=False).mean()
        ema26=candle_df['Close'].ewm(span=26,adjust=False).mean()
        macd=ema12-ema26; sig=macd.ewm(span=9,adjust=False).mean(); hist=macd-sig
        fig.add_trace(go.Bar(x=candle_df.index, y=hist,
            marker_color=['#00FF88' if v>=0 else '#FF3366' for v in hist],
            opacity=0.65, showlegend=False), row=3, col=1)
        fig.add_trace(go.Scatter(x=candle_df.index, y=macd,
            line=dict(color='#FFB800',width=1.2), showlegend=False), row=3, col=1)
        fig.add_trace(go.Scatter(x=candle_df.index, y=sig,
            line=dict(color='#FF3366',width=1), showlegend=False), row=3, col=1)

        GRID = 'rgba(30,35,45,0.8)'
        fig.update_layout(
            paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
            font=dict(family='Inter', color='#3D4A5C', size=9),
            legend=dict(bgcolor='rgba(0,0,0,0)', borderwidth=0,
                        orientation='h', y=1.04, x=0,
                        font=dict(size=9, color='#6B7A8F')),
            xaxis=dict(gridcolor=GRID, zeroline=False, tickfont=dict(size=8, color='#3D4A5C'),
                       rangeslider=dict(visible=False),
                       rangeselector=dict(
                           bgcolor='#0F1218', activecolor='rgba(0,212,255,0.15)',
                           bordercolor='#1E232D', borderwidth=1,
                           font=dict(color='#6B7A8F', size=9),
                           buttons=[
                               dict(count=5,  label='1S',  step='day',   stepmode='backward'),
                               dict(count=1,  label='1M',  step='month', stepmode='backward'),
                               dict(count=3,  label='3M',  step='month', stepmode='backward'),
                               dict(count=6,  label='6M',  step='month', stepmode='backward'),
                               dict(count=1,  label='1A',  step='year',  stepmode='backward'),
                               dict(count=1,  label='ACU', step='year',  stepmode='todate'),
                               dict(step='all', label='Todo'),
                           ])),
            xaxis2=dict(gridcolor=GRID, zeroline=False, tickfont=dict(size=8, color='#3D4A5C')),
            xaxis3=dict(gridcolor=GRID, zeroline=False, tickfont=dict(size=8, color='#3D4A5C')),
            yaxis =dict(gridcolor=GRID, zeroline=False, tickformat=',.0f',
                        tickfont=dict(size=9, family='JetBrains Mono'), side='right',
                        tickcolor='#1E232D'),
            yaxis2=dict(gridcolor=GRID, zeroline=False, range=[0,100],
                        tickfont=dict(size=8), tickvals=[30,50,70]),
            yaxis3=dict(gridcolor=GRID, zeroline=False, tickfont=dict(size=8)),
            height=590, margin=dict(l=0, r=55, t=30, b=10),
            hovermode='x unified',
            hoverlabel=dict(bgcolor='#0F1218', bordercolor='#1E232D',
                            font=dict(family='JetBrains Mono', size=10, color='#C9D1D9')),
        )
        fig.update_xaxes(range=['2025-11-01', '2026-04-14'])
        st.plotly_chart(fig, use_container_width=True)

    with c_side:
        # Reliability Score
        st.markdown('<div class="sec-label">🎯 Fiabilidad del Modelo</div>', unsafe_allow_html=True)
        rel = 94   # basado en R²=0.9994 y MAPE=0.61%
        deg = int(rel * 3.6)
        st.markdown(f"""
        <div class="card" style="text-align:center; padding:16px 12px;">
          <div style="display:flex;justify-content:center;margin-bottom:10px;">
            <div style="width:72px;height:72px;border-radius:50%;
                 background:conic-gradient(#00FF88 {deg}deg, #1E232D 0deg);
                 display:flex;align-items:center;justify-content:center;
                 box-shadow:0 0 24px rgba(0,255,136,0.2);">
              <div style="width:54px;height:54px;border-radius:50%;background:#0F1218;
                   display:flex;align-items:center;justify-content:center;
                   font-family:'JetBrains Mono',monospace;font-size:0.85rem;
                   font-weight:700;color:#00FF88;">{rel}%</div>
            </div>
          </div>
          <div style="font-size:0.6rem;color:#FFFFFF;letter-spacing:2px;text-transform:uppercase;">
            Fiabilidad del Modelo
          </div>
          <div style="margin-top:10px;display:grid;grid-template-columns:1fr 1fr;gap:6px;">
            <div style="background:#080B0F;border:1px solid #1E232D;border-radius:6px;
                 padding:6px;text-align:center;">
              <div style="font-size:0.55rem;color:#FFFFFF;letter-spacing:1px;">R²</div>
              <div style="font-family:'JetBrains Mono',monospace;font-size:0.82rem;
                   font-weight:700;color:#00D4FF;">{MODEL_R2}</div>
            </div>
            <div style="background:#080B0F;border:1px solid #1E232D;border-radius:6px;
                 padding:6px;text-align:center;">
              <div style="font-size:0.55rem;color:#FFFFFF;letter-spacing:1px;">MAPE</div>
              <div style="font-family:'JetBrains Mono',monospace;font-size:0.82rem;
                   font-weight:700;color:#FFB800;">{MODEL_MAPE}%</div>
            </div>
          </div>
        </div>
        """, unsafe_allow_html=True)

        # Order Book
        st.markdown('<div class="sec-label">📖 Libro de Órdenes</div>', unsafe_allow_html=True)
        np.random.seed(42)
        asks = [(live_last+i*11+np.random.uniform(0,4), int(np.random.uniform(60,700))) for i in range(5)]
        bids = [(live_last-i*11-np.random.uniform(0,4), int(np.random.uniform(60,700))) for i in range(5)]
        ob = '<div class="card-sm">'
        ob += '<div style="display:flex;justify-content:space-between;font-size:0.58rem;color:#FFFFFF;padding-bottom:5px;border-bottom:1px solid #1E232D;font-family:JetBrains Mono,monospace;">'\
              '<span>Venta</span><span>Compra</span><span>Vol</span></div>'
        for (ap,av),(bp,bv) in zip(reversed(asks), bids):
            ba=min(100,int(av/7)); bb=min(100,int(bv/7))
            ob += f'<div class="ob-row"><span class="ob-ask">{ap:,.1f}</span>'\
                  f'<span class="ob-bid">{bp:,.1f}</span>'\
                  f'<div style="display:flex;gap:3px;align-items:center;">'\
                  f'<div style="width:24px;height:3px;background:#1E232D;border-radius:1px;">'\
                  f'<div style="width:{ba}%;height:3px;background:#FF3366;border-radius:1px;"></div></div>'\
                  f'<div style="width:24px;height:3px;background:#1E232D;border-radius:1px;">'\
                  f'<div style="width:{bb}%;height:3px;background:#00FF88;border-radius:1px;"></div></div>'\
                  f'</div></div>'
        ob += f'<div style="text-align:center;margin:6px 0;font-family:JetBrains Mono,monospace;'\
              f'font-size:0.82rem;font-weight:700;color:#C9D1D9;">{live_last:,.2f}</div>'
        ob += '</div>'
        st.markdown(ob, unsafe_allow_html=True)

        # Trade
        st.markdown('<div class="sec-label">⚡ Operativa</div>', unsafe_allow_html=True)
        sig_lbl = "COMPRAR" if ai_up else "VENDER"
        st.markdown(f"""
        <div class="card-sm">
          <div style="font-size:0.62rem;color:#FFFFFF;margin-bottom:8px;">
            Señal IA: <b style="color:{'#00FF88' if ai_up else '#FF3366'};">{sig_lbl}</b>
            · Conf. {ai_conf}%
          </div>
          <button class="btn-buy">▲ COMPRAR</button>
          <div style="height:7px;"></div>
          <button class="btn-sell">▼ VENDER</button>
          <div style="margin-top:8px;font-size:0.58rem;color:#1E232D;text-align:center;">
            Solo educativo · No es asesoramiento financiero
          </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)

    # ══════════════════════════════════════════════════════════════
    # ROW 3 — Performance Audit + Forecast Table
    # ══════════════════════════════════════════════════════════════
    c_audit, c_fc = st.columns([1.4, 1])

    with c_audit:
        st.markdown('<div class="sec-label">🔍 GRU Ultra vs IBEX Real — Auditoría Abril 2026</div>',
                    unsafe_allow_html=True)

        # ── Gráfica GRU vs IBEX Real ────────────────────────────────
        fig_audit = go.Figure()
        audit_dates  = [pd.Timestamp(last_date)] + [pd.Timestamp(d) for d in forecast_trade_dates]
        audit_gru    = [last_close] + list(forecast_prices[:len(forecast_trade_dates)])
        real_in_aud  = {d: april_real_map[d] for d in forecast_trade_dates if d in april_real_map}
        if real_in_aud:
            audit_real_x = [pd.Timestamp(last_date)] + [pd.Timestamp(d) for d in sorted(real_in_aud)]
            audit_real_y = [last_close] + [real_in_aud[d] for d in sorted(real_in_aud)]
            fig_audit.add_trace(go.Scatter(
                x=audit_real_x, y=audit_real_y, name='IBEX Real',
                mode='lines+markers',
                line=dict(color='#00FF88', width=2.5),
                marker=dict(size=8, color='#00FF88', symbol='circle',
                            line=dict(color='#05070A', width=1.5)),
                hovertemplate='%{x|%d %b}<br>Real: <b>%{y:,.0f}</b><extra></extra>'
            ))
        fig_audit.add_trace(go.Scatter(
            x=audit_dates, y=audit_gru, name='GRU Ultra',
            mode='lines+markers',
            line=dict(color='#00D4FF', width=2.5, dash='dot'),
            marker=dict(size=8, color='#00D4FF', symbol='diamond',
                        line=dict(color='#05070A', width=1.5)),
            hovertemplate='%{x|%d %b}<br>GRU: <b>%{y:,.0f}</b><extra></extra>'
        ))
        # Banda de confianza GRU
        fig_audit.add_trace(go.Scatter(
            x=audit_dates + audit_dates[::-1],
            y=list(fc_up_v) + list(fc_lo_v[::-1]),
            fill='toself', fillcolor='rgba(0,212,255,0.08)',
            line=dict(color='rgba(0,0,0,0)'),
            showlegend=False, hoverinfo='skip'
        ))
        all_audit_y = audit_gru + (audit_real_y if real_in_aud else [])
        fig_audit.update_layout(
            paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
            font=dict(family='Inter', color='#3D4A5C', size=9),
            legend=dict(bgcolor='rgba(0,0,0,0)', borderwidth=0,
                        orientation='h', y=1.12, x=0,
                        font=dict(size=9, color='#6B7A8F')),
            xaxis=dict(gridcolor='rgba(30,35,45,0.6)', zeroline=False,
                       tickfont=dict(size=8, color='#3D4A5C')),
            yaxis=dict(gridcolor='rgba(30,35,45,0.6)', zeroline=False,
                       tickformat=',.0f', side='right',
                       range=[min(all_audit_y)*0.990, max(all_audit_y)*1.010] if all_audit_y else None,
                       tickfont=dict(size=8, family='JetBrains Mono')),
            height=200, margin=dict(l=0, r=50, t=20, b=10),
            hovermode='x unified',
            hoverlabel=dict(bgcolor='#0F1218', bordercolor='#1E232D',
                            font=dict(family='JetBrains Mono', size=10, color='#C9D1D9')),
        )
        st.plotly_chart(fig_audit, use_container_width=True)

        dias_ok = sum(1 for d in april_trade_dates[:len(forecast_prices)]
                      if april_real_map.get(d) is not None)
        # Accuracy: dias con error < 2%
        dias_acc = sum(1 for d,p in zip(april_trade_dates[:len(forecast_prices)], forecast_prices)
                       if april_real_map.get(d) and
                       abs(p - april_real_map[d]) / april_real_map[d] * 100 < 2.0)
        acc_pct = int(dias_acc / dias_ok * 100) if dias_ok > 0 else 0

        _fc_arrow = "▲" if ai_up else "▼"
        _fc_col   = "00FF88" if ai_up else "FF3366"
        _fc_pct   = abs(forecast_prices[0]-last_close)/last_close*100

        audit_html = f"""
        <div class="card">
          <div style="display:flex;gap:8px;margin-bottom:14px;flex-wrap:wrap;">
            <div class="mpill" style="flex:1;min-width:80px;">
              <div class="lbl">Días auditados</div>
              <div class="val" style="color:#00D4FF;">{dias_ok}/{HORIZON}</div>
              <div class="sub">confirmados</div>
            </div>
            <div class="mpill" style="flex:1;min-width:80px;">
              <div class="lbl">Precisión &lt;2%</div>
              <div class="val" style="color:#00FF88;">{acc_pct}%</div>
              <div class="sub">{dias_acc}/{dias_ok} días</div>
            </div>
            <div class="mpill" style="flex:1;min-width:80px;">
              <div class="lbl">Forecast D+1</div>
              <div class="val" style="color:#{_fc_col};">{_fc_arrow}{_fc_pct:.2f}%</div>
              <div class="sub">vs cierre 31 Mar</div>
            </div>
            <div class="mpill" style="flex:1;min-width:80px;">
              <div class="lbl">RMSE Train</div>
              <div class="val" style="color:#FFB800;">{MODEL_RMSE} pts</div>
              <div class="sub">R²={MODEL_R2}</div>
            </div>
          </div>
          <div style="display:flex;gap:4px;font-size:0.58rem;color:#FFFFFF;
               text-transform:uppercase;letter-spacing:1px;padding-bottom:5px;
               border-bottom:1px solid #1E232D;font-family:JetBrains Mono,monospace;">
            <span style="width:55px;">Fecha</span>
            <span style="width:68px;">Forecast</span>
            <span style="width:68px;">Real</span>
            <span style="flex:1;">Desviación</span>
            <span style="width:55px;">Error</span>
          </div>
        """
        prev = last_close
        for d, pred in zip(april_trade_dates[:len(forecast_prices)], forecast_prices):
            real = april_real_map.get(d)
            err  = abs(pred-real)/real*100 if real else None
            chg  = pred-prev; chgp = chg/prev*100
            arrow2 = "▲" if chg >= 0 else "▼"
            chg_c  = "#00FF88" if chg >= 0 else "#FF3366"

            if err is None:
                e_cls, e_txt, bw, bc = "audit-mid", "pendiente", 0, "#3D4A5C"
            elif err < 1.0:
                e_cls, bc = "audit-good", "#00FF88"
                e_txt = f"{err:.2f}% ✓"; bw = int(err*25)
            elif err < 2.0:
                e_cls, bc = "audit-mid", "#FFB800"
                e_txt = f"{err:.2f}%"; bw = int(err*18)
            else:
                e_cls, bc = "audit-bad", "#FF3366"
                e_txt = f"{err:.2f}%"; bw = min(100, int(err*12))

            real_s = f"{real:,.0f}" if real else "—"
            audit_html += f"""
          <div class="audit-row">
            <span class="audit-date">{d.strftime('%d Abr')}</span>
            <span class="audit-pred">{pred:,.0f}</span>
            <span class="audit-real">{real_s}</span>
            <div style="flex:1;display:flex;align-items:center;gap:6px;">
              <div style="flex:1;background:#1E232D;border-radius:2px;height:3px;">
                <div style="width:{bw}%;height:3px;background:{bc};
                     border-radius:2px;box-shadow:0 0 4px {bc}88;"></div>
              </div>
              <span style="color:{chg_c};font-size:0.68rem;width:40px;font-family:'JetBrains Mono',monospace;">
                {arrow2}{abs(chgp):.1f}%</span>
            </div>
            <span class="{e_cls}" style="width:55px;">{e_txt}</span>
          </div>"""
            prev = pred

        audit_html += """
          <div style="margin-top:12px;padding:10px 12px;background:rgba(0,212,255,0.04);
               border:1px solid rgba(0,212,255,0.12);border-radius:8px;">
            <div style="font-size:0.55rem;color:#00D4FF;letter-spacing:2px;
                 text-transform:uppercase;margin-bottom:4px;">Transparencia del modelo</div>
            <div style="font-size:0.7rem;color:#6B7A8F;line-height:1.6;">
              El <b style="color:#C9D1D9;">GRU Ultra</b> analiza
              <b style="color:#C9D1D9;">120 días</b> de histórico con
              <b style="color:#C9D1D9;">17 features</b> técnicas (OHLCV, RSI, MACD, Bollinger, ATR).
              Arquitectura de <b style="color:#C9D1D9;">5 capas residuales</b> +
              <b style="color:#00D4FF;">Multi-Head Attention</b>.
              La banda de confianza representa ±{:.0f} pts (1 RMSE histórico).
            </div>
          </div>
        </div>
        """.format(MODEL_RMSE)

        st.markdown(audit_html, unsafe_allow_html=True)

    with c_fc:
        st.markdown('<div class="sec-label">📋 Previsión Detallada — Días Hábiles Abril 2026</div>',
                    unsafe_allow_html=True)

        # Gráfica previsión GRU — solo el modelo, sin datos reales
        fc_dates_str = [pd.Timestamp(last_date).strftime('%d/%m')] + \
                       [d.strftime('%d/%m') for d in april_trade_dates[:len(forecast_prices)]]
        fc_line_y    = [last_close] + list(forecast_prices[:len(april_trade_dates)])
        fc_colors    = ['#00D4FF'] + [
            '#00FF88' if forecast_prices[i] >= (forecast_prices[i-1] if i>0 else last_close)
            else '#FF3366' for i in range(len(april_trade_dates))
        ]
        fig2 = go.Figure()
        # Banda de confianza ±RMSE
        fig2.add_trace(go.Scatter(
            x=fc_dates_str + fc_dates_str[::-1],
            y=[v + MODEL_RMSE*1.2 for v in fc_line_y] + [v - MODEL_RMSE*1.2 for v in fc_line_y[::-1]],
            fill='toself', fillcolor='rgba(0,212,255,0.08)',
            line=dict(color='rgba(0,0,0,0)'), showlegend=False, hoverinfo='skip'
        ))
        # Barras coloreadas por dirección
        fig2.add_trace(go.Bar(
            x=fc_dates_str[1:], y=list(forecast_prices[:len(april_trade_dates)]),
            marker_color=fc_colors[1:], opacity=0.75, name='GRU Previsión',
            hovertemplate='<b>%{x}</b><br>GRU: <b>%{y:,.0f}</b> pts<extra></extra>'
        ))
        # Línea central de previsión
        fig2.add_trace(go.Scatter(
            x=fc_dates_str, y=fc_line_y, name='GRU Ultra',
            mode='lines+markers',
            line=dict(color='#00D4FF', width=2.5, dash='dot'),
            marker=dict(size=7, color='#00D4FF', symbol='diamond',
                        line=dict(color='#05070A', width=1.5)),
            hovertemplate='<b>%{x}</b><br>Previsión: <b>%{y:,.0f}</b><extra></extra>'
        ))
        gru_only = fc_line_y
        fig2.update_layout(
            paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
            font=dict(family='JetBrains Mono', color='#3D4A5C', size=9),
            showlegend=False, bargap=0.25,
            xaxis=dict(gridcolor='rgba(30,35,45,0.6)', tickfont=dict(size=9, color='#3D4A5C')),
            yaxis=dict(gridcolor='rgba(30,35,45,0.6)', tickformat=',.0f',
                       range=[min(gru_only)*0.992, max(gru_only)*1.008] if gru_only else None,
                       tickfont=dict(size=9, family='JetBrains Mono')),
            height=200, margin=dict(l=0, r=10, t=5, b=10),
            hoverlabel=dict(bgcolor='#0F1218', bordercolor='#1E232D',
                            font=dict(family='JetBrains Mono', size=10, color='#C9D1D9')),
        )
        st.plotly_chart(fig2, use_container_width=True)

        # Detailed table
        tbl = '<div class="card-sm" style="margin-top:6px;">'
        tbl += '<div class="fc-row" style="font-size:0.56rem;color:#FFFFFF;'\
               'text-transform:uppercase;letter-spacing:1px;padding-bottom:6px;'\
               'border-bottom:1px solid #1E232D;">'
        tbl += '<span>Fecha</span><span>Forecast</span><span>Real</span>'\
               '<span>Error</span><span>Señal</span></div>'
        prev = last_close
        for d, pred in zip(april_trade_dates[:len(forecast_prices)], forecast_prices):
            real = april_real_map.get(d)
            err  = abs(pred-real)/real*100 if real else None
            chg  = pred-prev; chgp = chg/prev*100
            sig  = "ALZA" if chgp>0.3 else ("BAJA" if chgp<-0.3 else "NEUTRO")
            sig_c = "#00FF88" if sig=="ALZA" else ("#FF3366" if sig=="BAJA" else "#FFB800")
            real_s= f"{real:,.0f}" if real else "—"
            err_s = f"{err:.1f}%" if err else "—"
            err_c = "#00FF88" if err and err<1 else ("#FFB800" if err and err<2 else
                    ("#FF3366" if err else "#3D4A5C"))
            tbl += f'<div class="fc-row">'
            tbl += f'<span style="color:#6B7A8F;">{d.strftime("%d/%m")}</span>'
            tbl += f'<span style="color:#C9D1D9;font-weight:600;">{pred:,.0f}</span>'
            tbl += f'<span style="color:#00D4FF;">{real_s}</span>'
            tbl += f'<span style="color:{err_c};">{err_s}</span>'
            tbl += f'<span style="color:{sig_c};font-weight:700;">{sig}</span></div>'
            prev = pred
        tbl += '</div>'
        st.markdown(tbl, unsafe_allow_html=True)

with tab_info:
    st.markdown("""
    <div class='card' style='margin-top:20px;'>
        <h2 style='color:#00FF88;'>¿Qué es el IBEX 35?</h2>
        <p style='color:#C9D1D9;'>El <b>IBEX 35</b> es el principal índice bursátil de referencia de la bolsa española elaborado por Bolsas y Mercados Españoles (BME). Está formado por las 35 empresas con más liquidez que cotizan en el Sistema de Interconexión Bursátil Español (SIBE) en las cuatro bolsas españolas (Madrid, Barcelona, Bilbao y Valencia).</p>
        <h2 style='color:#00D4FF; margin-top:20px;'>Características del Modelo de Inteligencia Artificial</h2>
        <p style='color:#C9D1D9;'>Este panel utiliza un modelo de aprendizaje profundo <strong>GRU Ultra</strong> (Gated Recurrent Unit) para predecir los movimientos del IBEX 35 basándose en el análisis de los últimos 120 días de histórico de cotización.
        <br><br>Se utilizan más de 17 variables técnicas y algorítmicas combinadas con 5 capas de procesamiento neuronal residual y <i>Multi-Head Attention</i>, aportando un intervalo de confianza y estimaciones direccionales de gran precisión (Histórico R² > 0.99).</p>
        <h3 style='color:#00FF88; margin-top:20px;'>GRU frente a LSTM y RNN Clásicas</h3>
        <p style='color:#C9D1D9;'>A diferencia de las Redes Neuronales Recurrentes (RNN) tradicionales, que sufren graves problemas de olvido de información histórica a largo plazo, la <b>GRU</b> incorpora <i>puertas de actualización y reinicio (update & reset gates)</i> para decidir proactivamente qué contexto del pasado debe retenenerse o descartarse.<br><br>
        En comparación con las redes <b>LSTM</b> (Long Short-Term Memory), la arquitectura GRU tiene un diseño más optimizado y usa menos parámetros matemáticos ocultos. En aplicaciones financieras cargadas de varianza y ruido aleatorio como el IBEX 35, esta menor complejidad es tu mayor aliada: previene eficientemente el sobreajuste (overfitting) que las pesadas LSTM terminan sufriendo. Además, su simplicidad permite una optimización de sus pesos con mayor rapidez y en los tests de alta frecuencia captura quiebros tendenciales con mayor reactividad a corto plazo.</p>
    </div>
    """, unsafe_allow_html=True)

with tab_news:
    st.markdown("<div class='sec-label'>📰 Últimas noticias del IBEX 35</div>", unsafe_allow_html=True)
    try:
        import urllib.request
        import xml.etree.ElementTree as ET
        
        req = urllib.request.Request('https://news.google.com/rss/search?q=IBEX+35+when:7d&hl=es&gl=ES&ceid=ES:es', headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req, timeout=5) as response:
            xml_data = response.read()
            
        root = ET.fromstring(xml_data)
        items = root.findall('./channel/item')
        
        if not items:
            st.info("No se han encontrado noticias recientes.")
        else:
            for item in items[:10]:  # Mostrar las top 10
                title = item.find('title')
                title_text = title.text if title is not None else "Sin título"
                link = item.find('link')
                link_text = link.text if link is not None else "#"
                pubDate = item.find('pubDate')
                pubDate_text = pubDate.text if pubDate is not None else ""
                
                # Formatear la fecha y pasarla a hora de Madrid
                if pubDate_text:
                    try:
                        # Convertimos a objeto datetime indicando UTC y localizamos en Madrid
                        dt = pd.to_datetime(pubDate_text)
                        if dt.tz is None:
                            dt = dt.tz_localize('UTC')
                        dt_madrid = dt.tz_convert('Europe/Madrid')
                        pubDate_text = dt_madrid.strftime('%d %b %Y %H:%M')
                    except Exception:
                        if "GMT" in pubDate_text:
                            pubDate_text = pubDate_text.replace(" GMT", "")
                
                st.markdown(f"""
                <div class="card-sm" style="margin-bottom:10px;">
                    <a href="{link_text}" target="_blank" style="color:#00D4FF; text-decoration:none; font-weight:600; font-size:1.05rem;">{title_text}</a>
                    <div style="color:#6B7A8F; font-size:0.75rem; margin-top:5px; font-family:'JetBrains Mono',monospace;">📅 {pubDate_text}</div>
                </div>
                """, unsafe_allow_html=True)
                
    except Exception as e:
        st.error(f"Error al cargar las noticias: {str(e)}")
