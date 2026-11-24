# Bitácora de Experimentos, Hallazgos y Dificultades (Parte 2)

Este registro documenta de forma transparente las iteraciones técnicas, decisiones de diseño y dificultades reales encontradas durante la implementación de la Parte 2, sirviendo como respaldo directo para la sección de **Discusión Preliminar** del Entregable 2.

---

## 1. Hallazgos en Migración de Datos y Preprocesamiento (Fase 0–1)
- **Brecha inicial heredada de Parte 1:** El archivo inicial `PJME_hourly.csv` de la primera etapa contenía 26,280 registros sintéticos (`2015–2017`). Se migró al registro histórico real completo de PJM East (`2002–2018`).
- *(Se completará con estadísticas reales exactas de limpieza DST, duplicados y huecos tras ejecutar el pipeline.)*

## 2. Dificultades y Limitaciones Observadas en Líneas Base (Fase 2)
- **Escalabilidad de SARIMA:** Ajustar modelos SARIMA$(p,d,q)\times(P,D,Q)_{24}$ sobre >100,000 registros horarios continuos resulta computacionalmente intratable en memoria/tiempo con `SARIMAX`. Se adoptó un protocolo causal estratificado sobre ventana móvil reciente (`60 días`).
- *(Se completarán tiempos de ajuste por origen, pruebas ADF/KPSS y errores ante picos abruptos.)*

## 3. Iteraciones en Redes Neuronales (`PureLSTM` vs. `CNN1D-LSTM`) (Fase 3)
- *(Se completarán observaciones de convergencia por semilla, comportamiento ante picos de demanda sin temperatura exógena, impacto de `Dropout1d` y latencia en GPU RTX 3050.)*
