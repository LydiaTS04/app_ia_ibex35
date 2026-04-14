# 🎯 IBEX 35 AI Terminal

![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?logo=PyTorch&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B.svg?logo=Streamlit&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green.svg)

> **Terminal de Inteligencia Artificial para la predicción financiera del índice IBEX 35.** 
> Basado en un modelo de predicción profunda con arquitecturas GRU Ultra y Multi-Head Attention.

---

## 📖 Descripción General

Este proyecto consiste en una plataforma integral de análisis y predicción financiera orientada exclusivamente al índice bursátil español **IBEX 35**. 
A través del uso de Redes Neuronales Recurrentes (RNN), específicamente arquitecturas GRU (*Gated Recurrent Unit*) avanzadas combinadas con mecanismos de atención, la aplicación es capaz de inferir tendencias y proyectar escenarios de cotización a corto plazo.

El sistema principal consta de un **Dashboard Interactivo en tiempo real**, diseñado con la estética de un terminal militar-financiero, donde se monitorizan en directo:
- Señales de dirección de la Inteligencia Artificial.
- Gráficas interactivas con bandas de confianza.
- Backtesting e histórico de error (Auditoría del Modelo).

## ✨ Características Principales

- 🤖 **Predicción IA Avanzada**: Implementación y despliegue del modelo propietario "GRU Ultra", entrenado para lidiar con el ruido de las cotizaciones (altos valores de *R²* en validación).
- 📊 **Dashboard Interactivo**: Visualización fluida mediante Streamlit (estilo *Deep Sea*) que incluye gráficos profesionales con `Plotly`.
- 📡 **Datos Desacoplados**: Extracción de precios (OHLCV) mediante un script independiente, salvaguardando las ejecuciones en plataformas cloud de los bloqueos (Rate Limits) muy comunes en proveedores financieros gratuitos.
- 📈 **Análisis Técnico**: Gráficas dinámicas tipo vela unidas al *forecast*, acompañadas de los indicadores RSI, MACD, Medias Móviles y Bandas de Confianza Dinámicas.
- 🧠 **Arquitectura Escalable y Completa**: Diferentes scripts para el diseño de la red `ibex35_models.py`, entrenamiento progresivo `train_daily.py` y profundo `train_ultra.py`.

## 📁 Estructura del Proyecto

```text
📦 ibex35-cloud
 ┣ 📂 analytics/           # Scripts y notebooks de análisis predictivo extra
 ┣ 📂 app/                 # Lógica Web Frontend (El Dashboard)
 ┃ ┣ 📜 app.py             # Aplicación principal de Streamlit
 ┃ ┣ 📜 download_data.py   # Actualizador del histórico (evita Rate Limits Cloud)
 ┃ ┗ 📜 ibex_data.csv      # Base de datos tabular persistente
 ┣ 📂 docs/                # Documentación exhaustiva y metodologías
 ┣ 📂 models/              # Almacenamiento de pesos del modelo (.pt) y scalers
 ┣ 📜 ibex35_models.py     # Arquitecturas PyTorch (MultiHeadAttention, GRULayers)
 ┣ 📜 requirements.txt     # Dependencias oficiales (Simplificado para la nube)
 ┣ 📜 train_ultra.py       # Script matriz de entrenamiento base del GRU Ultra
 ┗ 📜 train_daily.py       # Rutinas de re-entrenamiento diario / fine-tuning
```

## 🚀 Instalación y Despliegue Local

### 1. Clonar el repositorio
```bash
git clone https://github.com/usuario/ibex35-cloud.git
cd ibex35-cloud
```

### 2. Entorno virtual (Recomendado)
```bash
python -m venv .env
# Activar en Windows:
.env\Scripts\activate
# Activar en macOS/Linux:
source .env/bin/activate
```

### 3. Instalar dependencias
```bash
pip install -r requirements.txt
```

## 💻 Uso Práctico

La aplicación extrae los históricos operables desde un origen local para así evadir limitaciones de Rate Limit si se ejecuta en modo Cloud Público (Streamlit Community).

1. **Extraer el mercado actual (Actualización de Base de Datos):**
   *Solo necesario ejecutarse puntualmente, de manera local, antes de realizar un push de actualización de datos a la nube.*
   ```bash
   cd app
   python download_data.py
   cd ..
   ```
2. **Levantar el Terminal:**
   ```bash
   streamlit run app/app.py
   ```
   *La web se desplegará en http://localhost:8501.*

## 🧠 Arquitectura de la Red Neural

El módulo de Inferencia IA utiliza un procesamiento secuencial optimizado:
* **Entradas (Features)**: Procesa *look-back windows* de los últimos 120 días observando 17 características (*features*) en paralelo (OHLCV estándar, y variables generadas como RSI, MACD, Bollinger, y Momentum).
* **Capas Residuales**: 5 bloques basados en arquitecturas RNN **GRU** fuertemente unidas con proyecciones lineales, superposición y regularización mediante `LayerNorm` y Dropout del 3%.
* **Atención Dirigida (Attention)**: Componente lógico `MultiHeadAttention` que destaca qué periodos pasados dentro de la ventana de 120 días son matemáticamente más relevantes en la predicción final.
* **Proyección (Dense Head)**: Regresión a través de subcapas `GELU` proyectando un horizonte estricto hacia el futuro (t+1, t+2... t+h).

---

## 👩‍💻 Autor

🔹 **Lydia Tomas Sanz**  

---

> ⚠️ **Aviso de Responsabilidad (Disclaimer):**  
> *Este proyecto ha sido desarrollado con orientación puramente académica, pedagógica y de investigación algorítmica. Bajo ningún concepto representa asesoramiento ni recomendaciones financieras. La autora no se responsabiliza de estrategias de mercado reales construidas basándose en las salidas de este modelo computacional.*