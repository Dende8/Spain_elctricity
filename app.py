import streamlit as st
import pandas as pd
import joblib
import holidays
import matplotlib.pyplot as plt

st.set_page_config(page_title="Panel eléctrico España", layout="centered")

# --- Carga de modelos y últimos datos ---
@st.cache_resource
def cargar_modelos():
    demanda = joblib.load("modelos/modelo_demanda.pkl")
    solar = joblib.load("modelos/modelo_solar.pkl")
    return demanda, solar

@st.cache_data
def cargar_ultimos():
    dem = pd.read_csv("modelos/ultimos_demanda.csv", parse_dates=["fecha"])
    sol = pd.read_csv("modelos/ultimos_solar.csv", parse_dates=["fecha"])
    return dem, sol

modelo_demanda, modelo_solar = cargar_modelos()
ultimos_dem, ultimos_sol = cargar_ultimos()
festivos_es = holidays.Spain(years=range(2021, 2028))

def construir_vector(fecha_pred, serie, col_obj, col_clima, valor_clima, info):
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
    X = pd.DataFrame([fila])[info["features"]]
    return X

# --- Interfaz ---
st.title("Panel de demanda y producción solar")
fecha_manana = ultimos_dem["fecha"].max() + pd.Timedelta(days=1)
st.caption(f"Predicción para el **{fecha_manana.date()}** "
           f"({fecha_manana.day_name()})")

col1, col2 = st.columns(2)
temp = col1.slider("Temperatura prevista en Madrid (ºC)", -5.0, 40.0, 18.0, 0.1)
rad = col2.slider("Radiación prevista en Madrid", 0.0, 32.0, 15.0, 0.1)

if st.button("Calcular predicción"):
    X_dem = construir_vector(fecha_manana, ultimos_dem, "demanda_gwh",
                              "temperatura_madrid", temp, modelo_demanda)
    X_sol = construir_vector(fecha_manana, ultimos_sol, "solar_gwh",
                              "radiacion_madrid", rad, modelo_solar)

    pred_demanda = modelo_demanda["modelo"].predict(X_dem)[0]
    pred_solar = modelo_solar["modelo"].predict(X_sol)[0]
    residual = pred_demanda - pred_solar
    cobertura = pred_solar / pred_demanda * 100

    c1, c2, c3 = st.columns(3)
    c1.metric("Demanda prevista", f"{pred_demanda:.1f} GWh")
    c2.metric("Solar prevista", f"{pred_solar:.1f} GWh")
    c3.metric("A cubrir con otras fuentes", f"{residual:.1f} GWh")
    st.progress(min(int(cobertura), 100), text=f"Cobertura solar: {cobertura:.1f} %")

    # Gráfico: últimos 7 días reales + predicción de mañana
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(ultimos_dem["fecha"], ultimos_dem["demanda_gwh"], "o-", label="Demanda real")
    ax.plot(ultimos_sol["fecha"], ultimos_sol["solar_gwh"], "o-", label="Solar real")
    ax.plot(fecha_manana, pred_demanda, "D", color="tab:blue", markersize=10, label="Demanda prevista")
    ax.plot(fecha_manana, pred_solar, "D", color="tab:orange", markersize=10, label="Solar prevista")
    ax.legend(); ax.set_ylabel("GWh")
    st.pyplot(fig)