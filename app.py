import streamlit as st
import pandas as pd
import joblib
import holidays
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error

st.set_page_config(
    page_title="Panel eléctrico España",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- Paleta del proyecto: azul = red eléctrica (demanda), ámbar = fotovoltaica (solar) ---
COLOR_DEMANDA = "#1f4e8c"
COLOR_SOLAR = "#e8a13a"
COLOR_RESIDUAL = "#6b7280"
COLOR_BASELINE = "#9ca3af"

CSS = f"""
<style>
.hero {{
    background: linear-gradient(90deg, {COLOR_DEMANDA} 0%, {COLOR_SOLAR} 100%);
    padding: 1.4rem 1.8rem;
    border-radius: 0.6rem;
    color: white;
    margin-bottom: 1.2rem;
}}
.hero h1 {{ margin: 0; font-size: 1.6rem; }}
.hero p {{ margin: 0.2rem 0 0 0; opacity: 0.9; }}
div[data-testid="stMetric"] {{
    background-color: rgba(127, 127, 127, 0.06);
    border: 1px solid rgba(127, 127, 127, 0.18);
    border-radius: 0.5rem;
    padding: 0.8rem 1rem;
}}
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)


@st.cache_resource
def cargar_modelos():
    """Carga los dos modelos entrenados (demanda y solar) junto con su lista de features, una sola vez por sesión."""
    demanda = joblib.load("modelos/modelo_demanda.pkl")
    solar = joblib.load("modelos/modelo_solar.pkl")
    return demanda, solar


@st.cache_data
def cargar_ultimos():
    """Carga los últimos días reales de cada serie, necesarios para calcular los lags (ayer, hace 7 días, media 7 días) de la predicción de mañana."""
    dem = pd.read_csv("modelos/ultimos_demanda.csv", parse_dates=["fecha"])
    sol = pd.read_csv("modelos/ultimos_solar.csv", parse_dates=["fecha"])
    return dem, sol


@st.cache_data
def cargar_backtest():
    """Carga el backtest de los últimos 60 días (real, predicción y baseline) exportado desde el notebook."""
    dem = pd.read_csv("modelos/backtest_demanda.csv", parse_dates=["fecha"])
    sol = pd.read_csv("modelos/backtest_solar.csv", parse_dates=["fecha"])
    return dem, sol


def construir_vector(fecha_pred, serie, col_obj, col_clima, valor_clima, info):
    """Arma, en el mismo orden de columnas que vio el modelo al entrenar, el vector de features de un único día a predecir."""
    fila = {
        "festivo": int(fecha_pred in festivos_es),
        "lag1": serie[col_obj].iloc[-1],
        "lag7": serie[col_obj].iloc[-7],
        "media7": serie[col_obj].iloc[-7:].mean(),
        col_clima: valor_clima,
    }
    dow, mes = fecha_pred.dayofweek, fecha_pred.month
    for f in info["features"]:
        if f.startswith("dow_"):
            fila[f] = int(f == f"dow_{dow}")
        elif f.startswith("mes_"):
            fila[f] = int(f == f"mes_{mes}")
    return pd.DataFrame([fila])[info["features"]]


def dibujar_contexto(ultimos_dem, ultimos_sol, fecha_manana, pred_demanda, pred_solar):
    """Dibuja los últimos días reales de demanda y solar, la predicción de mañana y sombrea el hueco entre ambas series (la demanda residual)."""
    fig, ax = plt.subplots(figsize=(9, 4))

    ax.plot(ultimos_dem["fecha"], ultimos_dem["demanda_gwh"], "o-",
            color=COLOR_DEMANDA, lw=2, label="Demanda real")
    ax.plot(ultimos_sol["fecha"], ultimos_sol["solar_gwh"], "o-",
            color=COLOR_SOLAR, lw=2, label="Solar real")
    ax.fill_between(ultimos_dem["fecha"], ultimos_dem["demanda_gwh"],
                     ultimos_sol["solar_gwh"], color=COLOR_RESIDUAL, alpha=0.12,
                     label="A cubrir con otras fuentes")

    ax.plot(fecha_manana, pred_demanda, "D", color=COLOR_DEMANDA, markersize=11,
            markeredgecolor="white", label="Demanda prevista")
    ax.plot(fecha_manana, pred_solar, "D", color=COLOR_SOLAR, markersize=11,
            markeredgecolor="white", label="Solar prevista")

    ax.set_ylabel("GWh")
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False, ncol=2, loc="upper center", bbox_to_anchor=(0.5, -0.15))
    fig.tight_layout()
    fig.subplots_adjust(bottom=0.28)
    return fig


def dibujar_backtest(bt, titulo, color_serie):
    """Dibuja el real, la predicción del modelo y el baseline ('hace 7 días') de una serie, día a día, en el periodo de test."""
    fig, ax = plt.subplots(figsize=(9, 3.2))
    ax.plot(bt["fecha"], bt["real"], color="#222831", lw=2, label="Real")
    ax.plot(bt["fecha"], bt["prediccion"], color=color_serie, lw=2, label="Predicción")
    ax.plot(bt["fecha"], bt["baseline"], color=COLOR_BASELINE, lw=1.2, ls="--",
            label="Baseline (hace 7 días)")
    ax.set_title(titulo, loc="left", fontsize=11)
    ax.set_ylabel("GWh")
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False, ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.22), fontsize=8)
    fig.tight_layout()
    fig.subplots_adjust(bottom=0.32)
    return fig


# --- Carga de datos y modelos ---
try:
    modelo_demanda, modelo_solar = cargar_modelos()
    ultimos_dem, ultimos_sol = cargar_ultimos()
except FileNotFoundError as e:
    st.error(f"No se han encontrado los ficheros del modelo ({e}). "
             "Comprueba que la carpeta `modelos/` está desplegada junto a la app.")
    st.stop()

festivos_es = holidays.Spain(years=range(2021, 2028))
fecha_manana = ultimos_dem["fecha"].max() + pd.Timedelta(days=1)

# --- Cabecera ---
st.markdown(f"""
<div class="hero">
    <h1>⚡ Panel de demanda y producción solar — España</h1>
    <p>Predicción para el {fecha_manana.date()} ({fecha_manana.day_name()}), a partir del histórico de Red Eléctrica</p>
</div>
""", unsafe_allow_html=True)

# --- Barra lateral: entradas de previsión ---
st.sidebar.header("Previsión meteorológica")
st.sidebar.caption("Introduce la temperatura y radiación previstas para mañana en Madrid.")
temp = st.sidebar.slider("🌡️ Temperatura prevista (ºC)", -5.0, 40.0, 18.0, 0.1)
rad = st.sidebar.slider("☀️ Radiación prevista", 0.0, 32.0, 15.0, 0.1)

with st.sidebar.expander("ℹ️ Acerca de este panel"):
    st.write(
        "Modelos `LinearRegression` entrenados con calendario, lags de la propia "
        "serie y la variable climática asociada (temperatura para la demanda, "
        "radiación para la solar). El resultado se recalcula al mover los sliders."
    )

tab_prediccion, tab_backtest = st.tabs(["📈 Predicción de mañana", "🧪 Backtest (60 días)"])

# ============================== PESTAÑA 1 ==============================
with tab_prediccion:
    X_dem = construir_vector(fecha_manana, ultimos_dem, "demanda_gwh",
                              "temperatura_madrid", temp, modelo_demanda)
    X_sol = construir_vector(fecha_manana, ultimos_sol, "solar_gwh",
                              "radiacion_madrid", rad, modelo_solar)

    pred_demanda = modelo_demanda["modelo"].predict(X_dem)[0]
    pred_solar = modelo_solar["modelo"].predict(X_sol)[0]
    residual = pred_demanda - pred_solar
    cobertura = pred_solar / pred_demanda * 100 if pred_demanda > 0 else 0.0

    demanda_ayer = ultimos_dem["demanda_gwh"].iloc[-1]
    solar_ayer = ultimos_sol["solar_gwh"].iloc[-1]
    residual_ayer = demanda_ayer - solar_ayer

    # Los tres deltas comparten un mismo criterio: menos GWh que cubrir con otras
    # fuentes es bueno para el sistema, así que "bajar" se colorea siempre en verde.
    c1, c2, c3 = st.columns(3)
    c1.metric("🏭 Demanda prevista", f"{pred_demanda:.1f} GWh",
              f"{pred_demanda - demanda_ayer:+.1f} vs. ayer", delta_color="inverse")
    c2.metric("☀️ Solar prevista", f"{pred_solar:.1f} GWh",
              f"{pred_solar - solar_ayer:+.1f} vs. ayer", delta_color="normal")
    c3.metric("🔌 A cubrir con otras fuentes", f"{residual:.1f} GWh",
              f"{residual - residual_ayer:+.1f} vs. ayer", delta_color="inverse")

    st.progress(min(int(cobertura), 100), text=f"Cobertura solar de la demanda: {cobertura:.1f} %")

    st.divider()
    st.pyplot(dibujar_contexto(ultimos_dem, ultimos_sol, fecha_manana, pred_demanda, pred_solar))

    st.caption("Fuente: Red Eléctrica de España (REE) · Modelos entrenados en `electricity.ipynb`")

# ============================== PESTAÑA 2 ==============================
with tab_backtest:
    try:
        bt_dem, bt_sol = cargar_backtest()
    except FileNotFoundError as e:
        st.warning(f"No se han encontrado los ficheros de backtest ({e}). "
                   "Ejecuta la celda `exportar_backtest` del notebook para generarlos.")
        st.stop()

    mae_dem = mean_absolute_error(bt_dem["real"], bt_dem["prediccion"])
    mae_sol = mean_absolute_error(bt_sol["real"], bt_sol["prediccion"])
    mae_dem_base = mean_absolute_error(bt_dem["real"], bt_dem["baseline"])
    mae_sol_base = mean_absolute_error(bt_sol["real"], bt_sol["baseline"])

    st.caption(
        f"Comparación día a día entre el valor real y el predicho por el modelo "
        f"en los últimos {len(bt_dem)} días, frente al baseline ('mismo valor que hace 7 días')."
    )

    m1, m2 = st.columns(2)
    m1.metric("MAE demanda (modelo)", f"{mae_dem:.2f} GWh",
              f"{mae_dem - mae_dem_base:+.2f} vs. baseline", delta_color="inverse")
    m2.metric("MAE solar (modelo)", f"{mae_sol:.2f} GWh",
              f"{mae_sol - mae_sol_base:+.2f} vs. baseline", delta_color="inverse")

    st.pyplot(dibujar_backtest(bt_dem, "Demanda: real vs. predicho", COLOR_DEMANDA))
    st.pyplot(dibujar_backtest(bt_sol, "Solar: real vs. predicho", COLOR_SOLAR))