# -*- coding: utf-8 -*-
"""
monitor_progress.py -- Monitor en tiempo real del entrenamiento final
Ejecutar en una terminal separada mientras train_final.py corre en otra.
  py monitor_progress.py
"""
import os, sys, time
from datetime import datetime, timedelta

WORK_DIR = os.path.dirname(os.path.abspath(__file__))

MODELS = ["RNN Simple", "LSTM", "GRU"]
SNAPS  = {
    "RNN Simple": os.path.join(WORK_DIR, "snap_rnn_31mar.pt"),
    "LSTM":       os.path.join(WORK_DIR, "snap_lstm_31mar.pt"),
    "GRU":        os.path.join(WORK_DIR, "snap_gru_31mar.pt"),
}

PROCESS_START_FILE = os.path.join(WORK_DIR, "training_log.csv")

BAR_W  = 40
TICK   = 3   # segundos entre refresco

# ── Colores ANSI (funciona en Windows 10+) ──────────────────────
GRN  = "\033[92m"
YLW  = "\033[93m"
RED  = "\033[91m"
CYN  = "\033[96m"
WHT  = "\033[97m"
DIM  = "\033[2m"
RST  = "\033[0m"
BOLD = "\033[1m"

def clear():
    os.system('cls' if os.name == 'nt' else 'clear')

def bar(pct, width=BAR_W, color=GRN):
    filled = int(width * pct / 100)
    empty  = width - filled
    return f"{color}{'█' * filled}{DIM}{'░' * empty}{RST}"

def file_age_s(path):
    """Segundos desde la ultima modificacion del fichero."""
    return time.time() - os.path.getmtime(path)

def fmt_time(seconds):
    if seconds < 0 or seconds > 86400:
        return "--:--"
    seconds = int(seconds)
    h, rem = divmod(seconds, 3600)
    m, s   = divmod(rem, 60)
    if h:
        return f"{h}h {m:02d}m"
    return f"{m}m {s:02d}s"

# Guarda cuando empezamos a monitorizar (no cuando empezo el proceso)
monitor_start = time.time()
last_sizes    = {m: 0 for m in MODELS}
model_start_t = {}   # cuando detecto que un modelo empieza a generar su snap

try:
    while True:
        now = time.time()

        # ── Estado de cada modelo ───────────────────────────────
        done    = []
        active  = None
        pending = []

        for m in MODELS:
            snap = SNAPS[m]
            if os.path.exists(snap):
                age = file_age_s(snap)
                # Si el fichero fue modificado hace menos de 30s -> activo
                if age < 30:
                    active = m
                else:
                    done.append(m)
            else:
                if active is None and not done and not pending:
                    active = m   # todavia no tiene snap -> puede estar en 1er epoch
                else:
                    pending.append(m)

        # Correccion: si todos los snaps existen y todos tienen >30s de antiguedad
        # -> entrenamiento completado
        all_exist = all(os.path.exists(SNAPS[m]) for m in MODELS)

        # ── Tiempo por modelo ───────────────────────────────────
        times = {}
        for m in done:
            snap = SNAPS[m]
            mtime = os.path.getmtime(snap)
            # Calculamos cuando empezo ese modelo (heuristico: snap del anterior + 1s)
            times[m] = None   # no tenemos inicio exacto, lo aproximamos abajo

        # ── Estimar duracion de cada modelo ─────────────────────
        # RNN termino = hora de creacion de su snap (primera escritura = 10:38)
        rnn_snap = SNAPS["RNN Simple"]
        lstm_snap = SNAPS["LSTM"]

        # Tiempo que tardo el RNN: desde inicio del proceso hasta su snap
        # Como no tenemos el inicio exacto, usamos la diferencia RNN->LSTM
        rnn_mtime  = os.path.getmtime(rnn_snap)  if os.path.exists(rnn_snap)  else None
        lstm_mtime = os.path.getmtime(lstm_snap) if os.path.exists(lstm_snap) else None

        # Estimacion de duracion por modelo
        rnn_dur = None
        if rnn_mtime and lstm_mtime:
            # LSTM empezo justo cuando RNN termino, con offset de unos segundos
            # RNN dur = lstm_snap_first_write - process_start
            # Como no sabemos el inicio exacto, usamos rnn_mtime como referencia
            pass

        # ── Calcular %  ─────────────────────────────────────────
        n_done = len(done)
        if active and active in SNAPS and os.path.exists(SNAPS[active]):
            # El modelo activo tiene snap -> mide por edad de ultimo checkpoint
            age_active = file_age_s(SNAPS[active])
            # Tamanyo actual vs ultima vez
            cur_size = os.path.getsize(SNAPS[active])
            # Velocidad de progreso: estima epoch por cambios en el fichero
            # Sin datos de epoch, usamos tiempo transcurrido vs RNN como escala
            pct_models = (n_done / 3) * 100
            # Dentro del modelo activo: usamos tiempo sin mejora como proxy de avance
            # Si lleva mucho sin mejora, esta cerca del early stop
            # PATIENCE=300 epocas sin mejora = fin
        else:
            pct_models = (n_done / 3) * 100

        overall_pct = (n_done / 3) * 100

        # ── Estimacion de tiempo restante ───────────────────────
        elapsed_total = now - monitor_start

        # Duraciones reales si podemos calcularlas
        dur_rnn  = None
        dur_lstm = None

        if rnn_mtime and lstm_mtime:
            dur_rnn = lstm_mtime - rnn_mtime   # tiempo que tardo el LSTM en arrancar = tiempo RNN

        eta_str = "calculando..."
        if all_exist:
            # Todos listos
            eta_str = f"{GRN}COMPLETADO{RST}"
            overall_pct = 100
        elif n_done == 2 and active == "GRU":
            # LSTM ya termino, GRU activo
            # Dur LSTM = now - lstm_first_snap (aproximacion)
            if rnn_mtime and lstm_mtime:
                dur_lstm_real = now - lstm_mtime   # tiempo desde que LSTM hizo su ultimo checkpoint
                # GRU similar en tamanyo a LSTM -> similar duracion
                # Usamos duracion LSTM como estimacion para GRU
                lstm_total_dur = lstm_mtime - rnn_mtime  # tiempo entre RNN y LSTM terminaron
                gru_eta = max(0, lstm_total_dur - dur_lstm_real)
                eta_str = f"~{fmt_time(gru_eta)} para GRU"
        elif n_done == 1 and active == "LSTM":
            if lstm_mtime:
                age_lstm = now - lstm_mtime
                eta_str  = f"LSTM activo (ultimo ckpt hace {fmt_time(age_lstm)})"

        # ── Render ───────────────────────────────────────────────
        clear()
        print(f"\n{BOLD}{CYN}  MONITOR DE ENTRENAMIENTO IBEX 35 — train_final.py{RST}")
        print(f"  {DIM}Actualizando cada {TICK}s  |  Ctrl+C para salir{RST}\n")
        print(f"  {'─'*60}")

        # Progreso global
        print(f"  {BOLD}PROGRESO GLOBAL{RST}")
        print(f"  {bar(overall_pct)}  {WHT}{overall_pct:.0f}%{RST}")
        print(f"  Modelos: {WHT}{n_done}/3 completados{RST}")
        print(f"  ETA: {YLW}{eta_str}{RST}\n")
        print(f"  {'─'*60}")

        # Estado por modelo
        print(f"  {BOLD}ESTADO POR MODELO{RST}\n")
        for m in MODELS:
            snap = SNAPS[m]
            if m in done:
                size_mb = os.path.getsize(snap) / 1e6
                mtime   = datetime.fromtimestamp(os.path.getmtime(snap))
                print(f"  {GRN}[DONE]{RST}  {BOLD}{m:<12}{RST}  "
                      f"snap={size_mb:.1f}MB  "
                      f"guardado {mtime.strftime('%H:%M:%S')}")

            elif m == active:
                if os.path.exists(snap):
                    age_s   = file_age_s(snap)
                    size_mb = os.path.getsize(snap) / 1e6
                    age_str = fmt_time(age_s)
                    # Barra de "sin mejora": de 0 a 300 epocas
                    # Cada actualizacion del snap reinicia el contador
                    # Cuanto mas tiempo sin actualizarse, mas cerca del early stop
                    # Estimamos 1 epoch ~= segs basandonos en observacion
                    # Proxy: si no actualizo en X min, progreso alto
                    max_idle = 300   # aprox max segs sin mejora antes de EarlyStop
                    idle_pct = min(100, (age_s / max_idle) * 100)
                    print(f"  {YLW}[ACTIVO]{RST} {BOLD}{m:<12}{RST}  "
                          f"snap={size_mb:.1f}MB")
                    print(f"           Ultimo checkpoint hace: {YLW}{age_str}{RST}")
                    print(f"           Early-stop progreso:    "
                          f"{bar(idle_pct, 25, YLW)}  "
                          f"{YLW}{idle_pct:.0f}% de paciencia usada{RST}")
                    print(f"           {DIM}(llega al 100% -> modelo termina){RST}")
                else:
                    # Primer modelo, snap aun no creado (epoch 1)
                    print(f"  {YLW}[ACTIVO]{RST} {BOLD}{m:<12}{RST}  "
                          f"iniciando... (epoch 1-10, snap pendiente)")

            else:
                print(f"  {DIM}[ESPERA]{RST}  {m:<12}  (en cola)")

        print(f"\n  {'─'*60}")

        # Info de ficheros
        print(f"  {BOLD}ARCHIVOS{RST}")
        for m in MODELS:
            snap = SNAPS[m]
            if os.path.exists(snap):
                size_mb = os.path.getsize(snap) / 1e6
                age_s   = file_age_s(snap)
                mtime   = datetime.fromtimestamp(os.path.getmtime(snap))
                status  = f"{GRN}OK{RST}" if m in done else f"{YLW}actualizando{RST}"
                print(f"  {snap.split(os.sep)[-1]:<25}  "
                      f"{size_mb:5.1f} MB  "
                      f"mod: {mtime.strftime('%H:%M:%S')}  [{status}]")
            else:
                print(f"  {SNAPS[m].split(os.sep)[-1]:<25}  {DIM}pendiente{RST}")

        print(f"\n  {DIM}Hora: {datetime.now().strftime('%H:%M:%S')}  |  "
              f"Monitor activo desde: {fmt_time(elapsed_total)}{RST}\n")

        if all_exist and all(file_age_s(SNAPS[m]) > 30 for m in MODELS):
            print(f"\n  {GRN}{BOLD}ENTRENAMIENTO COMPLETADO.{RST}")
            print(f"  Ejecuta: {CYN}py plot_retrospective.py{RST}\n")
            break

        time.sleep(TICK)

except KeyboardInterrupt:
    print(f"\n  Monitor detenido.")
