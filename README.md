# ⚡ Demanda residual eléctrica en España

Predicción diaria de la demanda eléctrica y de la producción solar en España, combinadas en un panel de control que estima cuánta energía debe cubrirse cada día con fuentes distintas a la solar (hidráulica, eólica, nuclear, ciclo combinado, etc.).

El proyecto cubre el ciclo completo de un problema de forecasting: desde los datos en bruto hasta una aplicación interactiva que consume los modelos entrenados.

## Contenido

- [Motivación](#motivación)
- [Datos](#datos)
- [Metodología](#metodología)
- [Resultados](#resultados)
- [Estructura del repositorio](#estructura-del-repositorio)
- [Instalación](#instalación)
- [Uso](#uso)
- [Limitaciones y próximos pasos](#limitaciones-y-próximos-pasos)

## Motivación

Un operador del sistema eléctrico necesita anticipar, día a día, cuánta energía tendrá que aportar el resto del mix eléctrico una vez descontada la producción solar. Ese hueco —la **demanda residual**— es la magnitud que se busca estimar:

```
demanda residual = demanda prevista − producción solar prevista
```

El proyecto entrena un modelo independiente para cada magnitud (demanda y solar) y combina sus predicciones en un panel de control, en lugar de modelar la resta directamente. Esto permite evaluar, guardar y monitorizar cada modelo por separado, y usar en cada uno la variable climática que realmente le afecta (temperatura para la demanda, radiación solar para la producción fotovoltaica).

## Datos

Dos series diarias con origen en Red Eléctrica de España (REE):

| Serie | Periodo | Variables |
|---|---|---|
| Demanda eléctrica | 2021-01-01 → presente | `demanda_gwh`, `temperatura_madrid` |
| Producción solar | 2023-01-01 → presente | `solar_gwh`, `radiacion_madrid` |

Ambas series se combinan por fecha para calcular la demanda residual histórica, usada como referencia en el panel de control.

## Metodología

**1. Limpieza.** Verificación de nulos, duplicados, huecos en el calendario y valores imposibles (ninguno detectado). Se detectó y corrigió un único valor anómalo: la demanda del 28 de abril de 2025 recoge el impacto del apagón ibérico de esa fecha, un evento excepcional no representativo de un lunes normal. Se sustituyó por la mediana de lunes de semanas adyacentes, sin festivos.

**2. Análisis exploratorio.** La demanda presenta un marcado ciclo semanal (el consumo cae un 10-15 % en fin de semana) y una estacionalidad anual bimodal, con picos en invierno y verano y una relación no lineal con la temperatura. La producción solar es estacional y muestra una tendencia estructural al alza explicada por el crecimiento de la potencia fotovoltaica instalada, no por un aumento de la radiación disponible (verificado comparando la radiación media anual, estable, frente a la producción por unidad de radiación, que crece de forma sostenida).

**3. Modelado.** Para cada serie se separan cronológicamente entrenamiento y test (los últimos 60 días), sin barajar el tiempo. Se define un baseline ingenuo (predecir el mismo valor que hace 7 días) y se entrenan dos modelos de regresión —`LinearRegression` y `RandomForestRegressor`— sobre las mismas variables: calendario (día de la semana, mes, festivo nacional), valores pasados de la propia serie (lags de 1 y 7 días, media móvil de 7 días) y la variable climática del día objetivo, asumida como una previsión meteorológica disponible.

**4. Evaluación.** Los modelos se comparan por error absoluto medio (MAE) en las unidades del problema (GWh). Ambos modelos superan claramente al baseline en las dos series.

**5. Despliegue.** El modelo con mejor MAE de cada serie se reentrena con el histórico completo y se guarda en disco. Una aplicación construida con Streamlit carga ambos modelos y ofrece un panel de control interactivo.

## Resultados

MAE en GWh sobre el conjunto de test (últimos 60 días):

| Modelo | Demanda | Solar |
|---|---|---|
| Baseline (mismo valor que hace 7 días) | 39,71 | 25,21 |
| `RandomForestRegressor` | 17,31 | 23,67 |
| **`LinearRegression`** | **15,19** | **19,31** |

Se seleccionó `LinearRegression` para ambas series, por ofrecer el menor error y por ser más simple e interpretable. No se aplicó normalización a las variables: al tratarse de mínimos cuadrados ordinarios, el escalado no afecta a las predicciones, solo a la magnitud de los coeficientes.

## Estructura del repositorio

```
├── data/
│   ├── demanda_electrica_espana.csv
│   └── solar_espana.csv
├── modelos/
│   ├── modelo_demanda.pkl        # modelo + lista de features
│   ├── modelo_solar.pkl
│   ├── ultimos_demanda.csv       # últimos días reales, para calcular lags en producción
│   ├── ultimos_solar.csv
│   ├── backtest_demanda.csv      # real / predicción / baseline de los últimos 60 días
│   └── backtest_solar.csv
├── electricity.ipynb             # limpieza, EDA, modelado y exportación
├── app.py                        # panel de control en Streamlit
├── requirements.txt              # dependencias con versión fijada
└── README.md
```

## Instalación

Requiere Python 3.10+.

```bash
git clone <url-del-repositorio>
cd <carpeta-del-proyecto>
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Las versiones quedan fijadas en `requirements.txt` para que el entorno local y el de despliegue (Streamlit Community Cloud) coincidan exactamente; una discrepancia de versión entre `scikit-learn`/`joblib` locales y las del servidor es la causa más común de que un modelo `.pkl` no cargue correctamente.

## Uso

**Notebook** — reproduce el análisis completo y regenera los modelos y ficheros de `modelos/`:

```bash
jupyter notebook electricity.ipynb
```

**Panel de control** — requiere que `modelos/` ya contenga los ficheros generados por el notebook:

```bash
streamlit run app.py
```

El panel permite introducir la temperatura y la radiación previstas para el día siguiente y muestra al instante la demanda prevista, la producción solar prevista, el hueco a cubrir con otras fuentes y una pestaña de validación histórica (backtest) de los últimos 60 días.

**Despliegue en Streamlit Community Cloud** — el repositorio debe incluir `requirements.txt` en la raíz (Streamlit Cloud no trae preinstaladas dependencias como `pandas` o `scikit-learn`) y la carpeta `modelos/` debe estar comprometida en el repositorio, no excluida por `.gitignore`.

## Limitaciones y próximos pasos

- `LinearRegression` extrapola linealmente fuera del rango histórico observado (2021-2026); sus predicciones son fiables dentro de ese rango, pero no están garantizadas ante eventos extremos.
- El calendario de festivos considerado es nacional y no incluye festivos autonómicos.
- Posibles ampliaciones: ajuste de hiperparámetros del `RandomForestRegressor` con validación cruzada temporal, incorporación de modelos específicos de series temporales (Holt-Winters, SARIMA) y despliegue continuo de la aplicación.

---

**Fuente de datos:** Red Eléctrica de España (REE).
