# Bitácora de Experimentos, Hallazgos y Dificultades (Parte 2)

Este registro documenta de forma transparente las iteraciones técnicas, decisiones de diseño, fallos iniciales y dificultades reales encontradas durante la ejecución de la Parte 2, sirviendo como respaldo directo para la sección de **Discusión Preliminar** del Entregable 2.

---

## 1. Hallazgos en Datos y Preprocesamiento Causal (Fases 0–1)
- **Migración de datos sintéticos a serie real:** En la Parte 1, `PJME_hourly.csv` contenía 26,280 filas sintéticas (`2015–2017`). Se descargó el registro real completo (`145,366` registros entre `2002-01-01 01:00` y `2018-08-03 00:00`).
- **Auditoría DST real:** Al reindexar a frecuencia horaria continua (`freq="h"`), se detectaron y promediaron **4 timestamps duplicados** (transiciones DST de otoño) y se insertaron e interpolaron linealmente **30 horas faltantes**, con una brecha consecutiva máxima de **1 hora**, alcanzando **145,392 registros horarios limpios**.
- **Fallo corregido en pruebas unitarias iniciales:** En la primera corrida de `pytest`, dos tests en `tests/test_data_leakage.py` fallaron porque la serie sintética de prueba (`1,000` y `1,500` horas) dejaba menos de `168 + 72 + 24 = 264` horas dentro del 15% de validación tras aislar cada split. Se ajustó la serie de test a `3,000` horas y se actualizó `.dt.day_of_week` para compatibilidad sin advertencias con `pandas 3.1.1`.
- **Diferenciación estacional en diagnóstico ADF:** Inicialmente `np.diff(sub, n=24)` aplicó la 24.ª diferencia ordinaria ($\Delta^{24}$) generando `SingularMatrixWarning` por colinealidad. Se corrigió por la diferencia estacional real $(1 - B^{24})y_t = y_t - y_{t-24}$, confirmando estacionariedad ($ADF = -11.66, p = 1.98 \times 10^{-21}$) frente a la serie original (KPSS = $1.66, p \le 0.01$).

## 2. Dificultades y Limitaciones Observadas en Líneas Base (Fase 2)
- **Costo computacional de SARIMA:** Ajustar `SARIMA(1,1,1)x(1,1,1)_24` sobre una ventana reciente de 60 días (`1,440` horas) tomó en promedio **10.11 s por origen** (`283.21 s` totales para 28 días estratificados), haciendo computacionalmente inviable re-estimarlo sobre las `21,545` ventanas de validación o sobre las `101,774` horas de entrenamiento.
- **Rigidez en episodios de carga extrema:** En el episodio pico del `2015-07-20` (~55 GW), SARIMA subestimó la cresta en ~3 GW (prediciendo ~52 GW), obteniendo `RMSE = 2,269.12 MW` (`MAPE = 5.17%`, $R^2 = 0.8639$) frente a `1,608.88 MW` (`MAPE = 3.72%`, $R^2 = 0.9316$) de `CNN1D-LSTM` en los mismos 28 días.

## 3. Comportamiento y Dificultades en Redes Neuronales (Fase 3 — RTX 3050 CUDA)
- **Variabilidad entre semillas y Early Stopping:**
  - `LSTM_Pura` (80,984 parámetros): `seed=42` detuvo tempranamente en época 21 (`best_val_mse = 0.001620`), mientras `seed=123` (`0.001512`) y `seed=456` (`0.001538`) convergieron mejor hacia épocas 38–42. Promedio: **RMSE $1,872.38 \pm 33.68$ MW**, **MAE $1,375.92 \pm 30.81$ MW**, **MAPE $4.36 \pm 0.10\%$**.
  - `CNN1D-LSTM` (131,480 parámetros): `seed=42` logró el mejor mínimo global (`best_val_mse = 0.001448`, época 39) y `seed=456` (`0.001467`, época 34), mientras `seed=123` detuvo en época 28 (`0.001565`). Promedio: **RMSE $1,834.00 \pm 38.41$ MW**, **MAE $1,340.91 \pm 49.11$ MW**, **MAPE $4.24 \pm 0.18\%$**.
- **Velocidad de inferencia:** Pese a tener ~62% más parámetros (`131,480` vs. `80,984`), `CNN1D-LSTM` infirió más rápido por muestra (**0.0131 ms** vs. **0.0156 ms**) porque `MaxPool1D(2)` reduce la secuencia recurrente de `72` a `36` pasos.
- **Hallazgo crítico en regímenes extremos (`Pico_Demanda_P90 >= 40,850.7 MW`):** Aunque `CNN1D-LSTM` supera a `LSTM_Pura` en desempeño global (`1,806.45` vs. `1,845.75 MW`), en la franja horaria `14–20h` (`2,108.24` vs. `2,138.31 MW`) y en días laborales/fines de semana, en el decil superior extremo (`Pico_Demanda_P90`) presenta un RMSE ligeramente mayor (`2,801.73` vs. `2,728.90 MW`). Hipótesis: `MaxPool1D(2)` + `Dropout1d(0.2)` suavizan las crestas rápidas cuando no hay variables meteorológicas exógenas (temperatura/humedad), motivando directamente los estudios de ablación (con/sin pooling, `W=168`, kernels, `GRU`/`TCN`) y ajuste con Optuna planificados para las Semanas 11–14.
