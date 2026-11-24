# Pronóstico de Demanda Eléctrica Horaria sobre la Red PJM mediante Arquitectura Híbrida CNN1D-LSTM

**Asignatura:** Inteligencia Artificial 2 — Universidad de las Américas (UDLA)  
**Docente:** Ph.D. Enrique Vinicio Carrera  
**Entregable Actual:** Entregable 2 — *Métodos y Resultados Preliminares* (Semanas 6–10)  
**Integrantes:**
- **Juan José Villalba** (`juan.villalba@udla.edu.ec`) — Ingeniería de datos causal, limpieza DST, EDA y verificación anti-*leakage*.
- **Sebastián Cadena** (`sebastian.cadena@udla.edu.ec`) — Líneas base (Persistencia 24h/168h, Perfil Condicional, SARIMA, LSTM Pura) y framework de evaluación.
- **Anthonny Mosquera** (`anthonny.mosquera@udla.edu.ec`) — Arquitectura híbrida `CNN1D-LSTM`, entrenamiento en GPU CUDA, métricas de latencia y diagnóstico de convergencia.

---

## 1. Descripción General y Política Anti-Fuga de Datos

Este repositorio implementa de manera reproducible el pipeline metodológico completo del proyecto semestral sobre la serie horaria real **PJME** (*PJM Interconnection East*, 2002–2018):

- **Limpieza causal y continuidad horaria:** resolución determinista de marcas de tiempo duplicadas en transiciones de horario de verano (*Daylight Saving Time*, DST) e imputación lineal acotada sobre una grilla horaria continua (`freq="h"`).
- **Partición cronológica estricta (70% / 15% / 15%):**
  - **Entrenamiento (`train`, 70%):** ajuste exclusivo del escalador `MinMaxScaler` y cálculo de umbrales de demanda pico (percentil 90).
  - **Validación (`val`, 15%):** evaluación y comparativa preliminar oficial del **Entregable 2**.
  - **Prueba ciega (`test`, 15%):** mantenida **sellada** hasta el Entregable 3 para evitar fuga de información durante la fase de optimización y ablación (Semanas 11–14).
- **Comparativa controlada:** `PureLSTMForecaster` y `HybridCNN1DLSTM` comparten exactamente el mismo bloque recurrente (`LSTM 128`) y cabeza densa (`FC 64 -> 24`), aislando el aporte de las convoluciones 1D (`Conv1D k=3, k=5 + MaxPool1D + Dropout1d`).

---

## 2. Estructura del Repositorio

```text
Proyecto Parte 2/
├── proyecto.pdf                 # Lineamientos oficiales del curso
├── README.md                    # Instrucciones de instalación y reproducción
├── requirements.txt             # Dependencias Python del proyecto
├── data/
│   ├── raw/PJME_hourly.csv      # Dataset real PJM East (descargado localmente, excluido por .gitignore si aplica)
│   └── processed/               # Auditoría JSON de limpieza DST y splits
├── src/
│   ├── config.py                # Hiperparámetros, rutas y semillas reproducibles
│   ├── data_pipeline.py         # Carga, limpieza DST, features cíclicas/lag-168, ventanas W=72 -> H=24
│   ├── metrics.py               # RMSE, MAE, MAPE, R2 en MW reales, error por horizonte y latencia
│   ├── models/
│   │   ├── base.py              # Bloque común LSTM + proyección densa (paridad arquitectónica)
│   │   ├── baselines.py         # Persistencia 24h/168h, Perfil Condicional y SARIMA
│   │   ├── lstm_pure.py         # Línea base Deep Learning (LSTM Pura)
│   │   └── hybrid_cnn_lstm.py   # Modelo propuesto (CNN1D-LSTM)
│   ├── train.py                 # Entrenamiento unificado GPU/CPU (EarlyStopping, ReduceLROnPlateau)
│   ├── evaluate.py              # Benchmark homogéneo en Validación y análisis por regímenes
│   ├── plots.py                 # Generación de figuras científicas (PNG 300 DPI + PDF vectorial)
│   └── run_pipeline.py          # Orquestador completo (tests -> train -> evaluate -> plots)
├── tests/
│   ├── test_data_leakage.py     # Tests automáticos de causalidad y cero data leakage
│   └── test_models.py           # Tests de formas, gradientes y métricas
├── results/
│   ├── checkpoints/             # Pesos óptimos (.pt)
│   ├── logs/                    # Curvas por época y predicciones serializadas
│   ├── tables/                  # Tablas CSV y diagnósticos JSON
│   └── figures/                 # Gráficas generadas
├── entregable_2/
│   ├── main.tex                 # Artículo científico IEEEtran (5–7 páginas, IMRaD)
│   ├── referencias.bib          # Bibliografía verificada en formato BibTeX IEEE
│   └── figures/                 # Figuras listas para compilar en Overleaf
└── docs/
    ├── bitacora_experimentos.md # Registro de pruebas, fallos e iteraciones
    └── guion_presentacion_e2.md # Guion oral (7–10 min) y banco de preguntas técnicas
```

---

## 3. Instalación y Configuración del Entorno

### 3.1 Clonar el repositorio y crear entorno virtual
```bash
git clone https://github.com/AnthonnyM31/Proyecto-IA-2-Parte-2.git
cd Proyecto-IA-2-Parte-2
python -m venv .venv
```

Activar el entorno virtual:
- **Windows (PowerShell):** `.\.venv\Scripts\Activate.ps1`
- **Linux/macOS:** `source .venv/bin/activate`

### 3.2 Instalar dependencias (con PyTorch CUDA)
```bash
pip install --upgrade pip
pip install torch --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
```

Verificar aceleración GPU (opcional, funciona también en CPU):
```bash
python -c "import torch; print('PyTorch:', torch.__version__, '| CUDA:', torch.cuda.is_available())"
```

### 3.3 Descargar el dataset real (`PJME_hourly.csv`)
Colocar el archivo `PJME_hourly.csv` (PJM East, 2002–2018, 145,366 registros) en `data/raw/PJME_hourly.csv`. Puede descargarse desde Kaggle (`robikscube/hourly-energy-consumption`) o vía espejo abierto de GitHub en PowerShell:
```powershell
New-Item -ItemType Directory -Force -Path data\raw | Out-Null
Invoke-WebRequest -Uri "https://raw.githubusercontent.com/archd3sai/Hourly-Energy-Consumption-Prediction/master/PJME_hourly.csv" -OutFile "data/raw/PJME_hourly.csv"
```

---

## 4. Instrucciones de Ejecución Reproducible

### Ejecución completa de extremo a extremo
```bash
python -m src.run_pipeline
```
Este comando ejecuta secuencialmente:
1. `pytest -v`: verificación de limpieza DST, splits cronológicos y cero *data leakage*.
2. `python -m src.train`: entrenamiento multi-semilla (`42, 123, 456`) de `PureLSTM` y `CNN1D-LSTM`.
3. `python -m src.evaluate`: evaluación homogénea en Validación de todas las líneas base (incluyendo `SARIMA(1,1,1)x(1,1,1)_24` estratificada), desagregación por horizonte $h=1\dots 24$ y horas pico, y latencia de inferencia.
4. `python -m src.plots`: generación de figuras en `results/figures/` y `entregable_2/figures/`.

### Ejecución por etapas individuales
```bash
pytest -v                       # Solo pruebas unitarias
python -m src.train             # Solo entrenamiento neuronal
python -m src.evaluate          # Solo evaluación y exportación de tablas
python -m src.plots             # Solo regeneración de figuras
```

---

## 5. Compilación del Artículo Científico en Overleaf

1. Subir el contenido de la carpeta `entregable_2/` (`main.tex`, `referencias.bib` y la subcarpeta `figures/`) a un proyecto en **Overleaf**.
2. Seleccionar compilador **pdfLaTeX**.
3. Compilar para obtener el informe evolutivo de **5 a 7 páginas** bajo plantilla oficial **IEEE Conference (`IEEEtran`)** y estructura **IMRaD**.
