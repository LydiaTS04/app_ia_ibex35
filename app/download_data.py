import yfinance as yf
import pandas as pd
from datetime import datetime
import os

def main():
    end = (datetime.now() + pd.Timedelta(days=1)).strftime('%Y-%m-%d')
    print(f"Descargando datos del ^IBEX hasta {end}...")
    raw = yf.download('^IBEX', start='1993-07-12', end=end, progress=False)
    
    if raw.empty:
        print("Error: No se ha podido descargar los datos.")
        return
        
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)
        
    # Nos aseguramos de guardar solo las columnas necesarias
    df = raw[['Open', 'High', 'Low', 'Close', 'Volume']]
    
    # Guarda en el mismo directorio donde está el script
    csv_path = os.path.join(os.path.dirname(__file__), 'ibex_data.csv')
    df.to_csv(csv_path)
    print(f"Archivo guardado correctamente en: {csv_path}")

if __name__ == "__main__":
    main()
