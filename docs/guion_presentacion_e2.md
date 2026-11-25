# Guión y Estructura de Presentación Oral — Entregable 2 (Avance Técnico)

**Asignatura:** Inteligencia Artificial 2 — Ph.D. Enrique Vinicio Carrera (UDLA)  
**Duración Objetivo:** 8 minutos 30 segundos (rango oficial: **7 a 10 minutos** + ronda de preguntas)  
**Repositorio GitHub:** `https://github.com/AnthonnyM31/Proyecto-IA-2-Parte-2`  
**Equipo:** Juan José Villalba, Sebastián Cadena y Anthonny Mosquera  

---

## Estructura Recomendada de Diapositivas (9 Diapositivas)

| Diapositiva | Contenido Visual Principal | Tiempo | Orador |
| :--- | :--- | :---: | :--- |
| **D1** | Portada, título, link al repo GitHub y objetivos de E2 | 0:30 | Juan José Villalba |
| **D2** | Continuidad desde E1: mejoras de rúbrica y política `val` vs `test` sellado | 0:50 | Juan José Villalba |
| **D3** | Datos reales PJME (145{,}392 h), limpieza DST y **Fig. 2** (EDA + ACF) | 1:15 | Juan José Villalba |
| **D4** | Arquitectura controlada (`PureLSTM` vs `CNN1D-LSTM`, **Fig. 1**) y `pytest` | 1:15 | Anthonny Mosquera |
| **D5** | Líneas base implementadas: Persistencia 24h/168h, Perfil, ADF/KPSS y SARIMA | 1:05 | Sebastián Cadena |
| **D6** | Resultados en Validación Completa (**Tabla Global** + **Fig. 3** Curvas) | 1:15 | Sebastián Cadena |
| **D7** | Comparativa emparejada con SARIMA (28 días) y episodio pico (**Fig. 4**) | 1:00 | Sebastián Cadena |
| **D8** | Desagregación por horizonte y regímenes críticos (**Fig. 5** — respuesta PI-2) | 0:50 | Anthonny Mosquera |
| **D9** | Dificultades encontradas, hallazgo en $P_{90}$ y Plan Ajustado (Semanas 11–15) | 0:50 | Anthonny Mosquera |

---

## Guión Detallado por Bloque

### Bloque 1 — Juan José Villalba (0:00 – 2:35 | Diapositivas D1–D3)
- **D1–D2:** *"Buenas tardes profesor Enrique Carrera y compañeros. Presentamos el avance técnico del Entregable 2 de nuestro proyecto de pronóstico de carga horaria en la red PJM. Partiendo de la retroalimentación del Entregable 1, consolidamos un repositorio reproducible en GitHub, ampliamos la revisión bibliográfica a 18 referencias verificadas y adoptamos una regla estricta anti-fuga de datos: toda evaluación preliminar de este entregable se realizó exclusivamente sobre Validación (2013–2016), manteniendo el conjunto de Prueba ciega (2016–2018) completamente sellado hasta el Entregable 3."*
- **D3 (Fig. 2 y Tabla II):** *"Migramos al conjunto real completo de PJME con 145,366 registros horarios entre 2002 y 2018. Nuestro pipeline causal resolvió 4 marcas de tiempo duplicadas por transiciones DST de otoño e interpoló 30 horas faltantes con brecha máxima de 1 hora, obteniendo 145,392 registros continuos. Como muestra la Figura 2, la autocorrelación presenta picos fuertes en 24h, 72h y 168h, justificando nuestra ventana $W=72$ horas y la variable causal `lag_168`. Todo el escalamiento Min-Max se ajustó únicamente sobre entrenamiento y está blindado por 6 tests unitarios en `pytest`."*

### Bloque 2 — Sebastián Cadena (2:35 – 5:55 | Diapositivas D5–D7)
- **D5:** *"Para evaluar rigurosamente la arquitectura propuesta, implementamos cuatro líneas base en MW reales desnormalizados: Persistencia 24h, Persistencia Semanal 168h, Perfil Horario Condicional y SARIMA$(1,1,1)\times(1,1,1)_{24}$, cuya diferenciación estacional fue validada con pruebas KPSS y ADF ($p = 1.98 \times 10^{-21}$ tras diferenciar en $s=24$). También implementamos una LSTM Pura que comparte exactamente el mismo bloque recurrente y cabeza densa para aislar el efecto de las convoluciones 1D."*
- **D6 (Tabla Completa + Fig. 3):** *"Entrenando sobre GPU RTX 3050 con tres semillas aleatorias (`42, 123, 456`) sobre las 21,545 ventanas de validación, CNN1D-LSTM alcanzó un RMSE promedio de $1,834.00 \pm 38.41$ MW ($\text{MAPE}=4.24\%$, $R^2=0.9121$), superando a Persistencia 24h ($3,062.79$ MW, reducción del 40.1%) y superando en promedio a la LSTM Pura ($1,872.38 \pm 33.68$ MW). Además, gracias al submuestreo temporal del MaxPool1D, CNN1D-LSTM infiere más rápido ($0.0131$ ms vs. $0.0156$ ms por muestra) pese a tener más parámetros."*
- **D7 (Tabla SARIMA + Fig. 4):** *"En el protocolo estratificado de 28 días para comparar de forma justa con SARIMA (ya que ajustar SARIMA toma 10.1 s por origen), CNN1D-LSTM alcanzó $1,608.88$ MW ($\text{MAPE}=3.72\%$) frente a $2,269.12$ MW de SARIMA (una mejora del 29.1%). En la Figura 4(b) se observa cómo durante el pico estival de ~55 GW, SARIMA subestima la cresta en ~3 GW mientras las arquitecturas profundas siguen de cerca la curva real."*

### Bloque 3 — Anthonny Mosquera (5:55 – 8:30 | Diapositivas D4, D8–D9)
- **D4 y D8 (Fig. 1 y Fig. 5):** *"En el análisis desagregado por horizonte $h=1\dots 24$ y regímenes críticos, CNN1D-LSTM supera consistentemente a LSTM Pura en error global ($1,806.45$ vs. $1,845.75$ MW), en la franja pico diaria de 14:00 a 20:00 ($2,108.24$ vs. $2,138.31$ MW), en días laborales y en fines de semana."*
- **D9 (Dificultades y Plan Semanas 11–15):** *"Con espíritu crítico, detectamos tres hallazgos y dificultades importantes: primero, en el decil extremo superior ($\ge P_{90} = 40,850.7$ MW), CNN1D-LSTM obtuvo ligeramente mayor error ($2,801.7$ vs. $2,728.9$ MW de LSTM Pura), lo que atribuimos al efecto suavizador de `MaxPool1D(2)` y `Dropout1d` ante picos térmicos sin variables meteorológicas exógenas. Segundo, observamos variabilidad entre semillas comparable a la ganancia media ($\sim 38$ MW). Tercero, ajustar SARIMA sobre toda la serie es computacionalmente prohibitivo. Por ello, en las Semanas 11 a 15 ejecutaremos optimización bayesiana con Optuna sobre validación, un estudio de ablación eliminando pooling y comparando variantes CNN-GRU y TCN, y finalmente liberaremos el conjunto `test` sellado con pruebas estadísticas Diebold-Mariano. Muchas gracias, quedamos atentos a sus preguntas."*

---

## Banco de Preguntas Técnicas Esperables

1. **¿Por qué reportaron resultados preliminares sobre Validación y no sobre Prueba (`test`)?**
   - *Respuesta:* Porque las Semanas 11–14 contemplan optimización de hiperparámetros (Optuna) y ablación arquitectónica. Si evaluáramos y observáramos errores sobre `test` en la Semana 10, contaminaríamos el conjunto de prueba ciega. Reportar sobre `val` garantiza evaluación final insesgada en el Entregable 3.
2. **¿Por qué `CNN1D-LSTM` tuvo un RMSE ligeramente mayor que `LSTM Pura` justo en el decil extremo $\ge P_{90}$?**
   - *Respuesta:* La combinación de `MaxPool1D(2)` y `Dropout1d(0.2)` actúa como filtro paso-bajo reduciendo ruido en condiciones normales y franjas pico diarias (`14–20h`), pero atenúa ligeramente crestas extremas provocadas por olas de calor al carecer de temperatura como entrada. Precisamente incluimos ablación sin pooling en la Semana 12 para verificarlo.
3. **¿Cómo garantizan que no existe fuga de datos en el cálculo de `lag_168` o del escalador?**
   - *Respuesta:* Primero se divide la serie limpia cronológicamente (70/15/15). Luego el `MinMaxScaler` y el umbral $P_{90}$ se ajustan únicamente sobre `train`. Cada split construye su propio `lag_168` y ventanas deslizantes por separado, verificado automáticamente mediante `pytest`.
4. **¿Por qué `CNN1D-LSTM` infiere más rápido en GPU si tiene más parámetros ($131\text{k}$ vs. $81\text{k}$)?**
   - *Respuesta:* La capa LSTM es secuencial en el tiempo; como `MaxPool1D(2)` comprime la longitud temporal de 72 a 36 pasos antes de entrar a la LSTM, se reduce a la mitad la cantidad de pasos recurrentes secuenciales, superando el mínimo costo paralelo de las dos capas `Conv1D`.
