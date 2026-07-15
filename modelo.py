# Modelo de Predicción de Ventas HORECA
# Pipeline completo de Machine Learning para predecir el volumen de ventas
# mensual en el canal HORECA (Hostelería, Restauración y Catering) de España.
# Se siguen las fases de la metodología CRISP-DM:
#   1. Carga y unificación de datos
#   2. Limpieza y tratamiento de anomalías
#   3. Análisis exploratorio (EDA)
#   4. Ingeniería de variables (feature engineering)
#   5. Modelos baseline
#   6. Modelos de Machine Learning (XGBoost, LightGBM)
#   7. Deep Learning (LSTM)
#   8. Comparativa final de resultados
# =============================================================================

# Realizamos todos los imports necesarios

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import requests
import xgboost as xgb
import lightgbm as lgb
import optuna
import shap
import json
import warnings
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import MinMaxScaler

warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)

# Ruta a los datos de y donde se guardarán los resultados
RUTA = "C:/Users/PaulaMartinMerino/OneDrive - UVE/Documentos/Paula/Master/TFE/"

# Columnas que NO son features del modelo (identificadores, fechas, target)
COLS_EXCLUIR = [
    "FechaMes", "producto", "Categoria", "Marca", "Provincia",
    "volumen_total", "importe_total", "año"
]

# =============================================================================

# Funciones auxiliares -> mensual agregado
# El MAPE calculado fila a fila se dispara cuando hay productos con volumen
# casi 0. La métrica correcta para el negocio es el error sobre el volumen
# mensual total de la categoría, que es lo que el fabricante usa para planificar.

def calcular_metricas_mensual(df_test, col_real, col_pred, nombre=""):
    real_mes = df_test.groupby("FechaMes")[col_real].sum()
    pred_mes = df_test.groupby("FechaMes")[col_pred].sum()

    mae  = mean_absolute_error(real_mes.values, pred_mes.values)
    rmse = np.sqrt(mean_squared_error(real_mes.values, pred_mes.values))
    mape = np.mean(
        np.abs((real_mes.values - pred_mes.values) / real_mes.values)
    ) * 100

    print(f"\n{'='*55}")
    print(f"MÉTRICAS — {nombre} (nivel mensual agregado)")
    print(f"{'='*55}")
    print(f"  {'Mes':<10} {'Real (L)':>14} {'Predicho (L)':>14} {'Error %':>8}")
    print(f"  {'-'*50}")
    for fecha, r, p in zip(real_mes.index, real_mes.values, pred_mes.values):
        err = abs(r - p) / r * 100
        print(f"  {fecha.strftime('%Y-%m'):<10} {r:>14,.0f} {p:>14,.0f} {err:>7.1f}%")
    print(f"  {'-'*50}")
    print(f"  MAE:  {mae:>14,.0f} litros")
    print(f"  RMSE: {rmse:>14,.0f} litros")
    print(f"  MAPE: {mape:>14.2f}%")

    return {"modelo": nombre, "MAE": mae, "RMSE": rmse, "MAPE": mape}

# =============================================================================

# BLOQUE 1 — CARGA DE DATOS
# Se cargan los tres maestros exportados desde el sistema de gestión de la empresa para el fabricante X:
#
#   - MaestroFacturas:  cada línea de factura con fecha, producto,
#                       cliente, cantidad en litros y precio unitario.
#   - MaestroClientes:  información de cada cliente HORECA (bar, restaurante,
#                       hotel) con su localización geográfica.
#   - MaestroProductos: catálogo de productos con categoría y marca.

facturas = pd.read_csv(
    RUTA + "MaestroFacturas.csv",
    sep=";", low_memory=False, dtype={"numfactura": str}
)
clientes = pd.read_csv(
    RUTA + "MaestroClientes.csv",
    sep=";", low_memory=False
)
productos = pd.read_csv(
    RUTA + "MaestroProductos.csv",
    sep=";"
)

print("--- DATOS CARGADOS ---")
print(f"Facturas:  {facturas.shape}")
print(f"Clientes:  {clientes.shape}")
print(f"Productos: {productos.shape}")

# =============================================================================

# BLOQUE 2 — LIMPIEZA Y PREPARACIÓN DE DATOS
# Al exportar desde Excel se arrastran columnas vacías con nombre "Unnamed" debemos limpiar 
clientes = clientes.drop(
    columns=[c for c in clientes.columns if "Unnamed" in c]
)

# Homogeneizamos los tipos de las claves de unión entre tablas
# Usamos Int64 (con mayúscula) porque acepta valores nulos
facturas["Idcproducto"]     = facturas["Idcproducto"].astype("Int64")
productos["Idcproducto"]    = productos["Idcproducto"].astype("Int64")
facturas["Idcclientefinal"] = facturas["Idcclientefinal"].astype("Int64")
clientes["Idcclientefinal"] = clientes["Idcclientefinal"].astype("Int64")

# Combinamos año, mes y día en una columna de fecha real
# FechaMes trunca al primer día del mes porque predecimos a nivel mensual
facturas["Fecha"] = pd.to_datetime(
    facturas[["AñoFact", "MesFact", "DiaFact"]].rename(
        columns={"AñoFact": "year", "MesFact": "month", "DiaFact": "day"}
    ),
    errors="coerce"
)
facturas["FechaMes"] = facturas["Fecha"].dt.to_period("M").dt.to_timestamp()

# Los dos primeros dígitos del código postal identifican la provincia en España
clientes["CodigoPostal"]  = (
    clientes["CodigoPostal"].astype(str).str.strip().str.zfill(5)
)
clientes["Provincia_cod"] = clientes["CodigoPostal"].str[:2]

provincias_map = {
    "01": "Álava",        "02": "Albacete",      "03": "Alicante",
    "04": "Almería",      "05": "Ávila",         "06": "Badajoz",
    "07": "Baleares",     "08": "Barcelona",     "09": "Burgos",
    "10": "Cáceres",      "11": "Cádiz",         "12": "Castellón",
    "13": "Ciudad Real",  "14": "Córdoba",       "15": "A Coruña",
    "16": "Cuenca",       "17": "Girona",        "18": "Granada",
    "19": "Guadalajara",  "20": "Gipuzkoa",      "21": "Huelva",
    "22": "Huesca",       "23": "Jaén",          "24": "León",
    "25": "Lleida",       "26": "La Rioja",      "27": "Lugo",
    "28": "Madrid",       "29": "Málaga",        "30": "Murcia",
    "31": "Navarra",      "32": "Ourense",       "33": "Asturias",
    "34": "Palencia",     "35": "Las Palmas",    "36": "Pontevedra",
    "37": "Salamanca",    "38": "S.C. Tenerife", "39": "Cantabria",
    "40": "Segovia",      "41": "Sevilla",       "42": "Soria",
    "43": "Tarragona",    "44": "Teruel",        "45": "Toledo",
    "46": "Valencia",     "47": "Valladolid",    "48": "Bizkaia",
    "49": "Zamora",       "50": "Zaragoza",      "51": "Ceuta",
    "52": "Melilla"
}
clientes["Provincia"] = (
    clientes["Provincia_cod"].map(provincias_map).fillna("Desconocida")
)

# =============================================================================

# BLOQUE 3 — UNIFICACIÓN DE TABLAS (MERGE)
# Unimos los tres maestros usando left join para conservar todas las facturas
# aunque algún producto o cliente no este en el maestro correspondiente

df = facturas.merge(productos, on="Idcproducto",     how="left")
df = df.merge(clientes,        on="Idcclientefinal", how="left")

print(f"\n=== DATASET UNIFICADO ===")
print(f"Shape: {df.shape}")

# =============================================================================

# BLOQUE 4 — AGREGACIÓN MENSUAL
# Agrupamos a nivel mensual por producto y provincia.
# Calculamos volumen, importe, clientes activos y número de pedidos.
# Eliminamos filas sin producto o provincia identificados.

df_mensual = df.groupby(
    ["FechaMes", "Idcproducto", "producto", "Categoria", "Marca", "Provincia"],
    as_index=False
).agg(
    volumen_total        = ("CantLTR",         "sum"),
    importe_total        = ("Precio",          "sum"),
    num_clientes_activos = ("Idcclientefinal", "nunique"),
    num_pedidos          = ("numfactura",      "nunique")
).dropna(subset=["producto", "Provincia"])

print(f"\n=== AGREGACIÓN MENSUAL ===")
print(f"Shape:             {df_mensual.shape}")
print(f"Rango fechas:      {df_mensual['FechaMes'].min()} → {df_mensual['FechaMes'].max()}")
print(f"Productos únicos:  {df_mensual['Idcproducto'].nunique()}")
print(f"Provincias únicas: {df_mensual['Provincia'].nunique()}")
print(f"Categorías:        {df_mensual['Categoria'].value_counts().to_dict()}")

df_mensual.to_csv(RUTA + "dataset_modelo.csv", index=False, encoding="utf-8-sig")
print("\n Dataset mensual guardado como dataset_modelo.csv")

# =============================================================================

# BLOQUE 5 — ANÁLISIS EXPLORATORIO DE DATOS (EDA)
# Buscamos tendencias, estacionalidad, diferencias entre categorías y anomalías.

df = pd.read_csv(RUTA + "dataset_modelo.csv", parse_dates=["FechaMes"])
df["mes"] = df["FechaMes"].dt.month
df["año"] = df["FechaMes"].dt.year
meses     = ["Ene","Feb","Mar","Abr","May","Jun","Jul","Ago","Sep","Oct","Nov","Dic"]

# Visión general 
vol_mes  = df.groupby("FechaMes")["volumen_total"].sum().reset_index().sort_values("FechaMes")
vol_cat  = df.groupby(["FechaMes","Categoria"])["volumen_total"].sum().reset_index().sort_values("FechaMes")
top10    = df.groupby("producto")["volumen_total"].sum().nlargest(10).sort_values()
top_prov = df.groupby("Provincia")["volumen_total"].sum().nlargest(15).sort_values()

fig, axes = plt.subplots(2, 2, figsize=(16, 10))
fig.suptitle("EDA — Visión General Ventas Danone HORECA", fontsize=14)

axes[0,0].plot(vol_mes["FechaMes"], vol_mes["volumen_total"],
               marker="o", linewidth=2, color="#1f77b4")
axes[0,0].set_title("Volumen total mensual")
axes[0,0].set_ylabel("Litros")
axes[0,0].xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
axes[0,0].xaxis.set_major_locator(mdates.MonthLocator(interval=3))
plt.setp(axes[0,0].xaxis.get_majorticklabels(), rotation=45, ha="right")

for cat, grp in vol_cat.groupby("Categoria"):
    axes[0,1].plot(grp["FechaMes"], grp["volumen_total"],
                   marker="o", label=cat, linewidth=2)
axes[0,1].set_title("Volumen mensual por Categoría")
axes[0,1].legend()
axes[0,1].xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
axes[0,1].xaxis.set_major_locator(mdates.MonthLocator(interval=3))
plt.setp(axes[0,1].xaxis.get_majorticklabels(), rotation=45, ha="right")

axes[1,0].barh(top10.index, top10.values, color="#2ca02c")
axes[1,0].set_title("Top 10 productos por volumen total")
axes[1,0].set_xlabel("Litros totales")

axes[1,1].barh(top_prov.index, top_prov.values, color="#ff7f0e")
axes[1,1].set_title("Top 15 provincias por volumen total")
axes[1,1].set_xlabel("Litros totales")

plt.tight_layout()
plt.savefig(RUTA + "eda_general.png", dpi=150, bbox_inches="tight")
plt.show()

# Estacionalidad global 
# En hostelería esperamos picos en verano y caídas en invierno
vol_mes_anio = df.groupby("mes")["volumen_total"].mean()
vol_anio     = df.groupby("año")["volumen_total"].sum()

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle("EDA — Estacionalidad Global", fontsize=14)

axes[0].bar(range(1,13), vol_mes_anio.values, color="#9467bd")
axes[0].set_xticks(range(1,13))
axes[0].set_xticklabels(meses)
axes[0].set_title("Volumen medio por mes (todos los años)")
axes[0].set_ylabel("Volumen medio")

axes[1].bar(vol_anio.index.astype(str), vol_anio.values, color="#8c564b")
axes[1].set_title("Volumen total por año")
axes[1].set_ylabel("Litros totales")

plt.tight_layout()
plt.savefig(RUTA + "eda_estacionalidad.png", dpi=150, bbox_inches="tight")
plt.show()

# EDA por categoría 
# Waters y Alpro tienen comportamientos muy distintos —> las analizamos por separado
for cat in df["Categoria"].unique():
    df_cat = df[df["Categoria"] == cat].copy()

    vol_t        = df_cat.groupby("FechaMes")["volumen_total"].sum().reset_index().sort_values("FechaMes")
    vol_mes_cat  = df_cat.groupby("mes")["volumen_total"].mean()
    vol_anio_cat = df_cat.groupby("año")["volumen_total"].sum()
    top10_cat    = df_cat.groupby("producto")["volumen_total"].sum().nlargest(10).sort_values()
    top_prov_cat = df_cat.groupby("Provincia")["volumen_total"].sum().nlargest(10).sort_values()

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle(f"EDA — Categoría: {cat}", fontsize=15, fontweight="bold")

    axes[0,0].plot(vol_t["FechaMes"], vol_t["volumen_total"],
                   marker="o", linewidth=2, color="#1f77b4")
    axes[0,0].set_title("Volumen mensual total")
    axes[0,0].set_ylabel("Litros")
    axes[0,0].xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    axes[0,0].xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    plt.setp(axes[0,0].xaxis.get_majorticklabels(), rotation=45, ha="right")

    axes[0,1].bar(range(1,13), vol_mes_cat.values, color="#9467bd")
    axes[0,1].set_xticks(range(1,13))
    axes[0,1].set_xticklabels(meses)
    axes[0,1].set_title("Estacionalidad media por mes")
    axes[0,1].set_ylabel("Volumen medio")

    axes[0,2].bar(vol_anio_cat.index.astype(str), vol_anio_cat.values, color="#8c564b")
    axes[0,2].set_title("Volumen total por año")
    axes[0,2].set_ylabel("Litros totales")

    axes[1,0].barh(top10_cat.index, top10_cat.values, color="#2ca02c")
    axes[1,0].set_title("Top 10 productos")
    axes[1,0].set_xlabel("Litros totales")
    axes[1,0].tick_params(axis="y", labelsize=8)

    axes[1,1].barh(top_prov_cat.index, top_prov_cat.values, color="#ff7f0e")
    axes[1,1].set_title("Top 10 provincias")
    axes[1,1].set_xlabel("Litros totales")

    for anio, grp in df_cat.groupby("año"):
        grp_mes = grp.groupby("mes")["volumen_total"].sum()
        axes[1,2].plot(grp_mes.index, grp_mes.values,
                       marker="o", label=str(anio), linewidth=2)
    axes[1,2].set_xticks(range(1,13))
    axes[1,2].set_xticklabels(meses)
    axes[1,2].set_title("Comparativa estacional por año")
    axes[1,2].set_ylabel("Volumen")
    axes[1,2].legend()

    plt.tight_layout()
    plt.savefig(RUTA + f"eda_{cat.lower()}.png", dpi=150, bbox_inches="tight")
    plt.show()

    print(f"\n{'='*50}")
    print(f"RESUMEN ESTADÍSTICO — {cat}")
    print(f"{'='*50}")
    print(f"  Productos únicos:      {df_cat['producto'].nunique()}")
    print(f"  Provincias activas:    {df_cat['Provincia'].nunique()}")
    print(f"  Volumen total:         {df_cat['volumen_total'].sum():,.0f} litros")
    print(f"  Volumen medio mensual: {vol_t['volumen_total'].mean():,.0f} litros")
    print(f"  Mes pico:              {meses[vol_mes_cat.idxmax()-1]} ({vol_mes_cat.max():,.0f} litros)")
    print(f"  Mes valle:             {meses[vol_mes_cat.idxmin()-1]} ({vol_mes_cat.min():,.0f} litros)")

# =============================================================================

# BLOQUE 6 — TRATAMIENTO DE ANOMALÍAS Y FILTRADO TEMPORAL
# En el EDA se ha detectado un pico anómalo en 2023: volúmenes hasta 6 veces
# superiores a 2024. La causa es que los datos de 2023 se reportada de mala manera
# haciendo los datos no comparables.
# Para solucionar se han realizado distantas acciones:
#   1. Eliminar filas con volumen por pedido > 3x el percentil 99 de 2024
#   2. Usar solo datos de 2024-2025 (cobertura homogénea de distribuidores)
#   3. Excluir 2026 porque solo tiene datos hasta abril (serie incompleta)

df["vol_por_pedido"] = df["volumen_total"] / df["num_pedidos"]

df_2024 = df[df["año"] == 2024].copy()
df_2024["vol_por_pedido"] = df_2024["volumen_total"] / df_2024["num_pedidos"]
umbral_normal = df_2024["vol_por_pedido"].quantile(0.99)

anomalos  = df["vol_por_pedido"] > umbral_normal * 3
df_limpio = df[~anomalos].copy().reset_index(drop=True)
df_limpio = df_limpio.drop(columns=["vol_por_pedido"])

print(f"\n=== TRATAMIENTO DE ANOMALÍAS ===")
print(f"Umbral de normalidad (p99 x 3): {umbral_normal * 3:.0f} litros/pedido")
print(f"Filas eliminadas:  {anomalos.sum():,} ({anomalos.mean()*100:.1f}%)")
print(f"Filas conservadas: {len(df_limpio):,}")

df_limpio.to_csv(RUTA + "dataset_limpio.csv", index=False, encoding="utf-8-sig")

# Filtramos solo 2024-2025
df_limpio["año"] = df_limpio["FechaMes"].dt.year
df_limpio["mes"] = df_limpio["FechaMes"].dt.month
df_modelo = df_limpio[df_limpio["año"].isin([2024, 2025])].copy().reset_index(drop=True)

print(f"\n=== DATASET FINAL PARA EL MODELO ===")
print(f"Filas:             {len(df_modelo):,}")
print(f"Rango fechas:      {df_modelo['FechaMes'].min()} → {df_modelo['FechaMes'].max()}")
print(f"Productos únicos:  {df_modelo['producto'].nunique()}")
print(f"Provincias únicas: {df_modelo['Provincia'].nunique()}")
print(f"Categorías:        {df_modelo['Categoria'].value_counts().to_dict()}")

vol_orig  = df_limpio.groupby("FechaMes")["volumen_total"].sum().reset_index()
vol_final = df_modelo.groupby("FechaMes")["volumen_total"].sum().reset_index()

fig, axes = plt.subplots(1, 2, figsize=(16, 5))
fig.suptitle("Serie temporal: completa vs dataset final del modelo", fontsize=13)

axes[0].plot(vol_orig["FechaMes"], vol_orig["volumen_total"],
             marker="o", linewidth=2, color="#d62728")
axes[0].set_title("Serie completa (incluye 2023 anomalo)")
axes[0].set_ylabel("Litros")
axes[0].xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
axes[0].xaxis.set_major_locator(mdates.MonthLocator(interval=3))
plt.setp(axes[0].xaxis.get_majorticklabels(), rotation=45, ha="right")

axes[1].plot(vol_final["FechaMes"], vol_final["volumen_total"],
             marker="o", linewidth=2, color="#2ca02c")
axes[1].set_title("Dataset final del modelo (2024-2025)")
axes[1].set_ylabel("Litros")
axes[1].xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
axes[1].xaxis.set_major_locator(mdates.MonthLocator(interval=1))
plt.setp(axes[1].xaxis.get_majorticklabels(), rotation=45, ha="right")

plt.tight_layout()
plt.savefig(RUTA + "dataset_final_serie.png", dpi=150, bbox_inches="tight")
plt.show()

df_modelo.to_csv(RUTA + "dataset_modelo_final.csv", index=False, encoding="utf-8-sig")
print("\n Dataset final guardado como dataset_modelo_final.csv")

# =============================================================================

# BLOQUE 7 — INGENIERÍA DE VARIABLES (FEATURE ENGINEERING)
# Creamos cuatro tipos de variables:
#   A. Variables temporales: capturan la estacionalidad del calendario
#   B. Variables externas:   factores del entorno (temperatura, festivos...)
#   C. Variables históricas: memoria del comportamiento pasado (lags)
#   D. Variables binarias:   características cualitativas del negocio (0/1)

df = pd.read_csv(RUTA + "dataset_modelo_final.csv", parse_dates=["FechaMes"])
df["mes"] = df["FechaMes"].dt.month
df["año"] = df["FechaMes"].dt.year

# A. Variables temporales 

# Seno y coseno del mes para que diciembre y enero queden como meses consecutivos
df["mes_sin"] = np.sin(2 * np.pi * df["mes"] / 12)
df["mes_cos"] = np.cos(2 * np.pi * df["mes"] / 12)

df["trimestre"] = df["FechaMes"].dt.quarter

df["es_verano"]    = df["mes"].isin([6, 7, 8]).astype(int)
df["es_invierno"]  = df["mes"].isin([12, 1, 2]).astype(int)
df["es_primavera"] = df["mes"].isin([3, 4, 5]).astype(int)
df["es_otonio"]    = df["mes"].isin([9, 10, 11]).astype(int)

# Febrero tiene menos días —> esto reduce ventas de forma estructural
dias_habiles_map   = {1:23,2:20,3:21,4:22,5:22,6:21,7:23,8:22,9:21,10:23,11:21,12:21}
df["dias_habiles"] = df["mes"].map(dias_habiles_map)

print(" Variables temporales creadas")

# B. Variables externas

# Festivos nacionales — usamos la API pública Nager.Date (gratuita)
festivos_list = []
for year in [2024, 2025]:
    url  = f"https://date.nager.at/api/v3/PublicHolidays/{year}/ES"
    data = requests.get(url).json()
    for f in data:
        festivos_list.append({"fecha": pd.to_datetime(f["date"])})

festivos_df             = pd.DataFrame(festivos_list)
festivos_df["FechaMes"] = festivos_df["fecha"].dt.to_period("M").dt.to_timestamp()
festivos_mes            = festivos_df.groupby("FechaMes").size().reset_index(name="num_festivos")

df = df.merge(festivos_mes, on="FechaMes", how="left")
df["num_festivos"] = df["num_festivos"].fillna(0).astype(int)
print(f" Festivos añadidos — media: {df['num_festivos'].mean():.1f} festivos/mes")

# Eventos especiales — los marcamos manualmente porque son eventos conocidos
eventos = pd.DataFrame([
    {"FechaMes": "2024-03-01", "semana_santa": 1},
    {"FechaMes": "2025-04-01", "semana_santa": 1},
    {"FechaMes": "2024-12-01", "navidad": 1},
    {"FechaMes": "2025-12-01", "navidad": 1},
    {"FechaMes": "2024-06-01", "evento_deportivo": 1},
    {"FechaMes": "2024-07-01", "evento_deportivo": 1},
])
eventos["FechaMes"] = pd.to_datetime(eventos["FechaMes"])
eventos = eventos.fillna(0).groupby("FechaMes", as_index=False).max()

df = df.merge(eventos, on="FechaMes", how="left")
df[["semana_santa","navidad","evento_deportivo"]] = \
    df[["semana_santa","navidad","evento_deportivo"]].fillna(0).astype(int)
print("✓ Eventos especiales añadidos")

# Temperatura — usamos Open-Meteo (API gratuita, sin registro)
# Devuelve datos diarios que agrupamos a nivel mensual
coords_provincia = {
    "Barcelona":     (41.38,   2.18), "Valencia":      (39.47,  -0.38),
    "Madrid":        (40.42,  -3.70), "Malaga":        (36.72,  -4.42),
    "Baleares":      (39.57,   2.65), "Alicante":      (38.35,  -0.48),
    "Girona":        (41.98,   2.82), "Murcia":        (37.98,  -1.13),
    "S.C. Tenerife": (28.46, -16.25), "Sevilla":       (37.39,  -5.99),
    "Granada":       (37.18,  -3.60), "Tarragona":     (41.12,   1.25),
    "Castellon":     (39.99,  -0.05), "Almeria":       (36.84,  -2.46),
    "Las Palmas":    (28.12, -15.43), "Cadiz":         (36.53,  -6.30),
    "Zaragoza":      (41.65,  -0.88), "A Coruna":      (43.37,  -8.40),
    "Asturias":      (43.36,  -5.85), "Bizkaia":       (43.26,  -2.93),
}

temp_list = []
for provincia, (lat, lon) in coords_provincia.items():
    try:
        r = requests.get(
            "https://archive-api.open-meteo.com/v1/archive",
            params={"latitude": lat, "longitude": lon,
                    "start_date": "2024-01-01", "end_date": "2025-12-31",
                    "daily": "temperature_2m_mean", "timezone": "Europe/Madrid"},
            timeout=15
        ).json()
        temp_diaria = pd.DataFrame({
            "fecha":    pd.to_datetime(r["daily"]["time"]),
            "temp_dia": r["daily"]["temperature_2m_mean"]
        })
        temp_diaria["FechaMes"] = temp_diaria["fecha"].dt.to_period("M").dt.to_timestamp()
        temp_mensual            = temp_diaria.groupby("FechaMes")["temp_dia"].mean().reset_index()
        temp_mensual.columns    = ["FechaMes", "temp_media"]
        temp_mensual["Provincia"] = provincia
        temp_list.append(temp_mensual)
        print(f"   {provincia}")
    except Exception as e:
        print(f"  ⚠ Error en {provincia}: {e}")

if temp_list:
    temp_df       = pd.concat(temp_list, ignore_index=True)
    temp_nacional = temp_df.groupby("FechaMes")["temp_media"].mean().reset_index(
        name="temp_nacional"
    )
    df = df.merge(temp_df,       on=["FechaMes", "Provincia"], how="left")
    df = df.merge(temp_nacional, on="FechaMes",                how="left")
    df["temp_media"] = df["temp_media"].fillna(df["temp_nacional"])
    df = df.drop(columns=["temp_nacional"])
    print(f"\n Temperatura añadida para {len(temp_list)} provincias")
else:
    print("⚠ API no disponible — usando temperatura aproximada por mes")
    temp_aprox     = {1:8,2:9,3:12,4:15,5:19,6:23,7:26,8:26,9:22,10:17,11:12,12:9}
    df["temp_media"] = df["mes"].map(temp_aprox)

df["temp_alta"]     = (df["temp_media"] > 20).astype(int)
df["temp_muy_alta"] = (df["temp_media"] > 27).astype(int)
print(" Variables de temperatura creadas")

# C. Variables históricas — Lags y medias móviles 
# Los lags le dan al modelo información sobre qué pasó en meses anteriores
#   lag_1: ventas del mes anterior (la señal más fuerte)
#   lag_2: hace 2 meses
#   lag_3: hace 3 meses
#   lag_6: hace 6 meses
#   lag_12: hace 12 meses (mismo mes del año anterior)

df = df.sort_values(["Idcproducto", "Provincia", "FechaMes"]).reset_index(drop=True)

for lag in [1, 2, 3, 6, 12]:
    df[f"lag_{lag}"] = df.groupby(
        ["Idcproducto", "Provincia"]
    )["volumen_total"].shift(lag)

df["media_movil_3m"] = df.groupby(["Idcproducto","Provincia"])["volumen_total"]\
                         .transform(lambda x: x.shift(1).rolling(3, min_periods=1).mean())
df["media_movil_6m"] = df.groupby(["Idcproducto","Provincia"])["volumen_total"]\
                         .transform(lambda x: x.shift(1).rolling(6, min_periods=1).mean())

print(" Lags y medias móviles creados")
print(f"  Nulos en lag_1: {df['lag_1'].isna().sum():,} (primeros meses sin histórico)")

# D. Variables categóricas binarias (0/1) ---

# Cliente regular — compra más del 33% de los meses disponibles
total_meses   = df["FechaMes"].nunique()
meses_activos = df.groupby(["Idcproducto","Provincia"])["FechaMes"]\
                  .nunique().reset_index(name="meses_con_venta")
meses_activos["pct_presencia"]      = meses_activos["meses_con_venta"] / total_meses
meses_activos["es_cliente_regular"] = (meses_activos["pct_presencia"] >= 0.33).astype(int)

df = df.merge(
    meses_activos[["Idcproducto","Provincia","meses_con_venta","es_cliente_regular"]],
    on=["Idcproducto","Provincia"], how="left"
)
print(f" Clientes regulares: {df['es_cliente_regular'].mean()*100:.1f}%")

# Producto de alto volumen — mueve más que la mediana del catálogo
vol_medio_prod     = df.groupby("producto")["volumen_total"].mean()
productos_alto_vol = vol_medio_prod[vol_medio_prod >= vol_medio_prod.median()].index
df["es_producto_alto_volumen"] = df["producto"].isin(productos_alto_vol).astype(int)
print(f" Productos alto volumen: {df['es_producto_alto_volumen'].mean()*100:.1f}%")

# Julio y agosto — máxima afluencia turística en España
df["es_mes_turistico"] = df["mes"].isin([7, 8]).astype(int)

# Provincia costera — mayor hostelería turística y estacionalidad veraniega
provincias_costeras = [
    "Barcelona","Valencia","Malaga","Alicante","Baleares","Girona","Murcia",
    "Tarragona","Almeria","Cadiz","Las Palmas","S.C. Tenerife","Huelva",
    "Granada","Castellon","A Coruna","Asturias","Cantabria","Gipuzkoa"
]
df["es_provincia_costera"] = df["Provincia"].isin(provincias_costeras).astype(int)
print(f" Provincias costeras: {df['es_provincia_costera'].mean()*100:.1f}%")

# Enero, febrero y noviembre — meses de menor actividad HORECA
df["es_mes_bajo"] = df["mes"].isin([1, 2, 11]).astype(int)

# Producto con tendencia creciente — vende más en 2º semestre que en 1º
vol_s1 = df[df["mes"].isin([1,2,3,4,5,6])].groupby("producto")["volumen_total"].mean()
vol_s2 = df[df["mes"].isin([7,8,9,10,11,12])].groupby("producto")["volumen_total"].mean()
vol_s1, vol_s2 = vol_s1.align(vol_s2, join="inner")
productos_crecientes        = vol_s2[vol_s2 > vol_s1 * 1.05].index
df["es_producto_creciente"] = df["producto"].isin(productos_crecientes).astype(int)
print(f" Productos crecientes: {df['es_producto_creciente'].mean()*100:.1f}%")

# Diciembre y enero — ajustes de inventario a cierre fiscal
df["es_fin_inicio_anyo"] = df["mes"].isin([12, 1]).astype(int)

vars_binarias = [
    "es_cliente_regular","es_producto_alto_volumen","es_mes_turistico",
    "es_provincia_costera","es_mes_bajo","es_producto_creciente",
    "temp_alta","temp_muy_alta","semana_santa","navidad",
    "evento_deportivo","es_fin_inicio_anyo"
]

print(f"\n{'='*65}")
print(f"RESUMEN — VARIABLES BINARIAS Y CORRELACION CON VOLUMEN")
print(f"{'='*65}")
print(f"{'Variable':<30} {'% positivos':>12} {'Corr. volumen':>15}")
print(f"{'-'*60}")
for v in vars_binarias:
    pct  = df[v].mean() * 100
    corr = df[v].corr(df["volumen_total"])
    print(f"{v:<30} {pct:>11.1f}% {corr:>15.3f}")

print(f"\nShape final: {df.shape}")
nulos = df.isnull().sum()
print(f"\nNulos por columna:\n{nulos[nulos > 0]}")

df.to_csv(RUTA + "dataset_features.csv", index=False, encoding="utf-8-sig")
print(f"\n Dataset features guardado como dataset_features.csv")

# =============================================================================

# BLOQUE 8 — ANÁLISIS DE CORRELACIONES
# Validamos que las variables tienen sentido estadístico y detectamos redundancias.

df = pd.read_csv(RUTA + "dataset_features.csv", parse_dates=["FechaMes"])

vars_numericas = [
    "volumen_total","mes","trimestre","dias_habiles","num_festivos",
    "temp_media","num_clientes_activos","num_pedidos","mes_sin","mes_cos",
    "es_verano","es_invierno","es_primavera","es_otonio",
    "es_cliente_regular","es_producto_alto_volumen","es_mes_turistico",
    "es_provincia_costera","es_mes_bajo","es_producto_creciente",
    "temp_alta","temp_muy_alta","semana_santa","navidad",
    "evento_deportivo","es_fin_inicio_anyo",
    "lag_1","lag_2","lag_3","lag_6","media_movil_3m","media_movil_6m"
]
vars_existentes = [v for v in vars_numericas if v in df.columns]
corr_target     = df[vars_existentes].corr()["volumen_total"].drop("volumen_total").sort_values()

fig, ax = plt.subplots(figsize=(10, 12))
colores = ["#d62728" if c < 0 else "#2ca02c" for c in corr_target.values]
ax.barh(corr_target.index, corr_target.values, color=colores)
ax.axvline(x=0,    color="black", linewidth=0.8)
ax.axvline(x=0.1,  color="gray",  linewidth=0.5, linestyle="--", alpha=0.5)
ax.axvline(x=-0.1, color="gray",  linewidth=0.5, linestyle="--", alpha=0.5)
ax.set_title("Correlacion de cada variable con el Volumen de Ventas", fontsize=13)
ax.set_xlabel("Correlacion de Pearson")
plt.tight_layout()
plt.savefig(RUTA + "correlacion_target.png", dpi=150, bbox_inches="tight")
plt.show()

print("=== CORRELACION CON VOLUMEN_TOTAL ===")
print(corr_target.round(3).to_string())

# Mapa de calor — detecta variables redundantes (correlacion > 0.9)
vars_heatmap = [
    "volumen_total","temp_media","mes","dias_habiles","num_festivos",
    "es_verano","es_invierno","es_mes_turistico","es_mes_bajo",
    "es_provincia_costera","es_cliente_regular","temp_alta","temp_muy_alta",
    "lag_1","lag_2","lag_3","media_movil_3m","media_movil_6m",
    "num_clientes_activos","num_pedidos"
]
vars_heatmap = [v for v in vars_heatmap if v in df.columns]

fig, ax = plt.subplots(figsize=(14, 12))
sns.heatmap(
    df[vars_heatmap].corr(),
    annot=True, fmt=".2f", cmap="RdYlGn",
    center=0, vmin=-1, vmax=1,
    linewidths=0.5, ax=ax, annot_kws={"size": 7}
)
ax.set_title("Mapa de correlaciones entre variables", fontsize=13)
plt.tight_layout()
plt.savefig(RUTA + "heatmap_correlaciones.png", dpi=150, bbox_inches="tight")
plt.show()

# Temperatura vs volumen
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle("Relacion temperatura — volumen de ventas", fontsize=13)
df_temp = df[["temp_media","volumen_total"]].dropna().copy()
axes[0].scatter(df_temp["temp_media"], df_temp["volumen_total"],
                alpha=0.3, color="#1f77b4", s=10)
axes[0].set_xlabel("Temperatura media (C)")
axes[0].set_ylabel("Volumen vendido (litros)")
axes[0].set_title("Temperatura vs Volumen")
df_temp["temp_grupo"] = (df_temp["temp_media"] // 2 * 2).astype(int)
vol_temp = df_temp.groupby("temp_grupo")["volumen_total"].mean().reset_index()
axes[1].bar(vol_temp["temp_grupo"], vol_temp["volumen_total"],
            width=1.8, color="#ff7f0e", edgecolor="white")
axes[1].set_xlabel("Temperatura media (C)")
axes[1].set_ylabel("Volumen medio (litros)")
axes[1].set_title("Volumen medio por rango de temperatura")
plt.tight_layout()
plt.savefig(RUTA + "temperatura_vs_volumen.png", dpi=150, bbox_inches="tight")
plt.show()

# Lags vs volumen — lag_1 deberia tener la correlacion mas alta
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.suptitle("Relacion entre lags historicos y volumen actual", fontsize=13)
for i, lag in enumerate(["lag_1","lag_2","lag_3"]):
    if lag in df.columns:
        df_lag = df[[lag,"volumen_total"]].dropna()
        axes[i].scatter(df_lag[lag], df_lag["volumen_total"],
                        alpha=0.3, color="#9467bd", s=10)
        corr = df_lag[lag].corr(df_lag["volumen_total"])
        axes[i].set_xlabel(f"Volumen hace {i+1} mes(es)")
        axes[i].set_ylabel("Volumen actual")
        axes[i].set_title(f"{lag} — correlacion: {corr:.3f}")
        print(f"{lag}: {len(df_lag):,} filas | correlacion: {corr:.3f}")
plt.tight_layout()
plt.savefig(RUTA + "lags_vs_volumen.png", dpi=150, bbox_inches="tight")
plt.show()

print("\n Analisis de correlacion completado")

# =============================================================================

# BLOQUE 9 — MODELOS BASELINE
# Referencia minima que debe superar cualquier modelo de ML.
# Tres baseline por categoria:
#   Naive estacional: predice el mismo mes del año anterior
#   Media movil 3m: promedio de los ultimos 3 meses del train
#   Media movil 6m: promedio de los ultimos 6 meses del train

df = pd.read_csv(RUTA + "dataset_features.csv", parse_dates=["FechaMes"])
resultados = []

for categoria in ["Waters", "Alpro"]:
    print(f"\n{'='*60}")
    print(f"BASELINE — {categoria}")
    print(f"{'='*60}")

    serie = df[df["Categoria"] == categoria]\
              .groupby("FechaMes")["volumen_total"].sum()\
              .reset_index().sort_values("FechaMes")
    serie.columns = ["ds", "y"]

    train = serie.iloc[:-3]
    test  = serie.iloc[-3:]

    print(f"Train: {train['ds'].min().date()} → {train['ds'].max().date()} ({len(train)} meses)")
    print(f"Test:  {test['ds'].min().date()} → {test['ds'].max().date()} ({len(test)} meses)")

    pred_naive = []
    for _, row in test.iterrows():
        mes_anterior = row["ds"] - pd.DateOffset(months=12)
        match = train[train["ds"] == mes_anterior]["y"]
        pred_naive.append(match.values[0] if len(match) > 0 else train["y"].tail(3).mean())
    pred_naive = np.array(pred_naive)

    pred_ma3 = np.full(3, train["y"].tail(3).mean())
    pred_ma6 = np.full(3, train["y"].tail(6).mean())
    y_real   = test["y"].values

    for pred, nombre in [
        (pred_naive, f"Naive_estacional_{categoria}"),
        (pred_ma3,   f"Media_movil_3m_{categoria}"),
        (pred_ma6,   f"Media_movil_6m_{categoria}"),
    ]:
        mae  = mean_absolute_error(y_real, pred)
        rmse = np.sqrt(mean_squared_error(y_real, pred))
        mask = y_real > 0
        mape = np.mean(np.abs((y_real[mask] - pred[mask]) / y_real[mask])) * 100
        print(f"  {nombre}: MAE={mae:,.0f} | MAPE={mape:.2f}%")
        resultados.append({"modelo": nombre, "MAE": mae, "RMSE": rmse, "MAPE": mape})

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(serie["ds"], serie["y"], marker="o", linewidth=2,
            color="#1f77b4", label="Real", zorder=3)
    ax.plot(test["ds"], pred_naive, marker="s", linewidth=2,
            linestyle="--", color="#d62728", label="Naive estacional")
    ax.plot(test["ds"], pred_ma3,   marker="^", linewidth=2,
            linestyle="--", color="#ff7f0e", label="Media movil 3m")
    ax.plot(test["ds"], pred_ma6,   marker="D", linewidth=2,
            linestyle="--", color="#9467bd", label="Media movil 6m")
    ax.axvline(x=train["ds"].max(), color="gray", linestyle=":",
               linewidth=1.5, label="Inicio test")
    ax.set_title(f"Baseline — {categoria}: Prediccion vs Real", fontsize=13)
    ax.set_ylabel("Volumen (litros)")
    ax.legend()
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(RUTA + f"baseline_{categoria.lower()}.png", dpi=150, bbox_inches="tight")
    plt.show()

with open(RUTA + "resultados_baseline.json", "w") as f:
    json.dump(resultados, f)
print("\n Resultados baseline guardados")


# =============================================================================
# BLOQUE 10 — XGBOOST CON OPTIMIZACION DE HIPERPARAMETROS
# =============================================================================
# XGBoost construye arboles de decision de forma secuencial: cada arbol
# corrige los errores del anterior. Hacemos tres experimentos por categoria:
#   1. Parametros por defecto → referencia sin optimizar
#   2. Optimizacion con Optuna (50 pruebas) → busqueda automatica del mejor
#   3. Solo las 15 features mas relevantes → ver si menos es mas

df = pd.read_csv(RUTA + "dataset_features.csv", parse_dates=["FechaMes"])

with open(RUTA + "resultados_baseline.json") as f:
    resultados = json.load(f)

mejores_params   = {}
mejores_modelos  = {}
mejores_features = {}

for categoria in ["Waters", "Alpro"]:
    print(f"\n{'='*65}")
    print(f"XGBOOST — {categoria}")
    print(f"{'='*65}")

    df_cat = df[df["Categoria"] == categoria].copy().sort_values(
        ["Idcproducto", "Provincia", "FechaMes"]
    )
    features = [c for c in df_cat.columns if c not in COLS_EXCLUIR]

    fecha_corte = pd.Timestamp("2025-10-01")
    train_df = df_cat[df_cat["FechaMes"] < fecha_corte].copy()
    test_df  = df_cat[df_cat["FechaMes"] >= fecha_corte].copy()
    train_df = train_df.dropna(subset=features + ["volumen_total"])
    test_df  = test_df.dropna(subset=features)

    X_train = train_df[features]
    y_train = train_df["volumen_total"]
    X_test  = test_df[features]
    y_test  = test_df["volumen_total"]

    print(f"Train: {len(train_df):,} filas | Test: {len(test_df):,} filas")

    # Experimento 1: parametros por defecto
    modelo_default = xgb.XGBRegressor(
        n_estimators=300, learning_rate=0.1, max_depth=6,
        random_state=42, n_jobs=-1, verbosity=0
    )
    modelo_default.fit(X_train, y_train)
    test_df["pred_default"] = np.maximum(modelo_default.predict(X_test), 0)
    r_default = calcular_metricas_mensual(
        test_df, "volumen_total", "pred_default", f"XGB_default_{categoria}"
    )
    resultados.append(r_default)

    # Experimento 2: optimizacion con Optuna
    # Prueba 50 combinaciones de parametros y se queda con la que minimiza el MAE
    print(f"\nOptimizando XGBoost con Optuna (50 trials)")

    def objective(trial):
        params = {
            "n_estimators":     trial.suggest_int("n_estimators", 100, 800),
            "max_depth":        trial.suggest_int("max_depth", 3, 9),
            "learning_rate":    trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "subsample":        trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
            "reg_alpha":        trial.suggest_float("reg_alpha", 0.0, 1.0),
            "reg_lambda":       trial.suggest_float("reg_lambda", 0.5, 2.0),
            "random_state": 42, "n_jobs": -1, "verbosity": 0
        }
        fecha_val = train_df["FechaMes"].max() - pd.DateOffset(months=2)
        X_tr  = train_df[train_df["FechaMes"] < fecha_val][features].dropna()
        y_tr  = train_df[train_df["FechaMes"] < fecha_val].loc[X_tr.index, "volumen_total"]
        X_val = train_df[train_df["FechaMes"] >= fecha_val][features].dropna()
        y_val = train_df[train_df["FechaMes"] >= fecha_val].loc[X_val.index, "volumen_total"]
        if len(X_tr) == 0 or len(X_val) == 0:
            return float("inf")
        m = xgb.XGBRegressor(**params)
        m.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)
        val_df = train_df[train_df["FechaMes"] >= fecha_val].copy()
        val_df = val_df.dropna(subset=features)
        val_df["pred"] = np.maximum(m.predict(val_df[features]), 0)
        real_mes_val = val_df.groupby("FechaMes")["volumen_total"].sum()
        pred_mes_val = val_df.groupby("FechaMes")["pred"].sum()
        return mean_absolute_error(real_mes_val.values, pred_mes_val.values)

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=50, show_progress_bar=True)

    best_params = study.best_params
    best_params.update({"random_state": 42, "n_jobs": -1, "verbosity": 0})
    print(f"\nMejores parametros encontrados:")
    for k, v in best_params.items():
        if k not in ["random_state", "n_jobs", "verbosity"]:
            print(f"  {k:<25} {v}")

    modelo_opt = xgb.XGBRegressor(**best_params)
    modelo_opt.fit(X_train, y_train)
    test_df["pred_opt"] = np.maximum(modelo_opt.predict(X_test), 0)
    r_opt = calcular_metricas_mensual(
        test_df, "volumen_total", "pred_opt", f"XGB_optimizado_{categoria}"
    )
    resultados.append(r_opt)
    mejores_modelos[categoria]  = modelo_opt
    mejores_params[categoria]   = best_params
    mejores_features[categoria] = features

    # Experimento 3: solo las 15 variables mas correlacionadas
    features_top = [
        "lag_1", "lag_2", "lag_3", "lag_6", "media_movil_3m", "media_movil_6m",
        "num_clientes_activos", "num_pedidos", "temp_media",
        "es_verano", "es_provincia_costera", "mes_sin", "mes_cos",
        "es_producto_alto_volumen", "dias_habiles", "Idcproducto"
    ]
    features_top = [f for f in features_top if f in df_cat.columns]
    train_top    = train_df.dropna(subset=features_top + ["volumen_total"])
    test_top     = test_df.dropna(subset=features_top).copy()
    modelo_top   = xgb.XGBRegressor(**best_params)
    modelo_top.fit(train_top[features_top], train_top["volumen_total"])
    test_top["pred_top"] = np.maximum(modelo_top.predict(test_top[features_top]), 0)
    r_top = calcular_metricas_mensual(
        test_top, "volumen_total", "pred_top", f"XGB_top_features_{categoria}"
    )
    resultados.append(r_top)

    baseline_mape   = 2.43  if categoria == "Waters" else 5.36
    baseline_nombre = "Naive estacional" if categoria == "Waters" else "Media movil 6m"
    mejora = (baseline_mape - r_opt["MAPE"]) / baseline_mape * 100
    print(f"\n{'─'*55}")
    print(f"  Baseline ({baseline_nombre}): MAPE = {baseline_mape:.2f}%")
    print(f"  XGB optimizado:            MAPE = {r_opt['MAPE']:.2f}%")
    print(f"  Mejora relativa:           {mejora:+.1f}%")
    if mejora >= 20:
        print(f" OBJETIVO CONSEGUIDO (reduccion >= 20%)")
    print(f"{'─'*55}")

    # Graficas de resultados
    real_mes     = test_df.groupby("FechaMes")["volumen_total"].sum()
    pred_def_mes = test_df.groupby("FechaMes")["pred_default"].sum()
    pred_opt_mes = test_df.groupby("FechaMes")["pred_opt"].sum()
    hist_mes     = df_cat.groupby("FechaMes")["volumen_total"].sum()

    fig, axes = plt.subplots(1, 2, figsize=(16, 5))
    fig.suptitle(f"XGBoost — {categoria}: Prediccion vs Real", fontsize=13)
    axes[0].plot(hist_mes.index, hist_mes.values, marker="o", linewidth=2,
                 color="#1f77b4", label="Real", zorder=3)
    axes[0].plot(pred_def_mes.index, pred_def_mes.values, marker="s",
                 linewidth=2, linestyle="--", color="#ff7f0e",
                 label=f"XGB default (MAPE {r_default['MAPE']:.1f}%)")
    axes[0].plot(pred_opt_mes.index, pred_opt_mes.values, marker="^",
                 linewidth=2, linestyle="--", color="#2ca02c",
                 label=f"XGB optimizado (MAPE {r_opt['MAPE']:.1f}%)")
    axes[0].axvline(x=pd.Timestamp("2025-10-01"), color="gray",
                    linestyle=":", linewidth=1.5, label="Inicio test")
    axes[0].set_title("Volumen mensual total")
    axes[0].set_ylabel("Litros")
    axes[0].legend(fontsize=9)
    axes[0].xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    plt.setp(axes[0].xaxis.get_majorticklabels(), rotation=45, ha="right")
    axes[1].scatter(y_test, test_df["pred_opt"], alpha=0.3, color="#2ca02c", s=10)
    lim = max(y_test.max(), test_df["pred_opt"].max()) * 1.05
    axes[1].plot([0, lim], [0, lim], "k--", linewidth=1, label="Prediccion perfecta")
    axes[1].set_xlabel("Volumen real (litros)")
    axes[1].set_ylabel("Volumen predicho (litros)")
    axes[1].set_title("Real vs Predicho — XGB optimizado")
    axes[1].legend()
    plt.tight_layout()
    plt.savefig(RUTA + f"xgboost_{categoria.lower()}.png", dpi=150, bbox_inches="tight")
    plt.show()

    fig, ax = plt.subplots(figsize=(10, 5))
    meses_test = real_mes.index
    x = np.arange(len(meses_test))
    width = 0.3
    ax.bar(x - width, real_mes.values / 1e6,     width, label="Real",           color="#1f77b4", alpha=0.85)
    ax.bar(x,          pred_def_mes.values / 1e6, width, label="XGB default",    color="#ff7f0e", alpha=0.85)
    ax.bar(x + width,  pred_opt_mes.values / 1e6, width, label="XGB optimizado", color="#2ca02c", alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels([m.strftime("%Y-%m") for m in meses_test])
    ax.set_title(f"Detalle periodo test — {categoria} (Oct-Dic 2025)")
    ax.set_ylabel("Volumen (millones de litros)")
    ax.legend()
    plt.tight_layout()
    plt.savefig(RUTA + f"xgboost_{categoria.lower()}_test_detalle.png", dpi=150, bbox_inches="tight")
    plt.show()

    # SHAP — explica que variables influyen mas y en que direccion
    print(f"\nCalculando SHAP values para {categoria}...")
    explainer    = shap.TreeExplainer(modelo_opt)
    X_test_clean = X_test.dropna()
    shap_values  = explainer.shap_values(X_test_clean)

    fig, ax = plt.subplots(figsize=(10, 8))
    shap.summary_plot(shap_values, X_test_clean,
                      plot_type="bar", show=False, max_display=15)
    plt.title(f"SHAP — Importancia de variables: {categoria}", fontsize=12)
    plt.tight_layout()
    plt.savefig(RUTA + f"shap_{categoria.lower()}.png", dpi=150, bbox_inches="tight")
    plt.show()

    fig, ax = plt.subplots(figsize=(10, 8))
    shap.summary_plot(shap_values, X_test_clean, show=False, max_display=15)
    plt.title(f"SHAP — Direccion del efecto: {categoria}", fontsize=12)
    plt.tight_layout()
    plt.savefig(RUTA + f"shap_{categoria.lower()}_beeswarm.png", dpi=150, bbox_inches="tight")
    plt.show()
    print(f" SHAP guardado")

resultados_limpios = [
    {k: v for k, v in r.items() if k not in ["real_mes","pred_mes"]}
    for r in resultados
]
with open(RUTA + "resultados_todos.json", "w") as f:
    json.dump(resultados_limpios, f, indent=2)
print("\n Resultados XGBoost guardados")


# =============================================================================
# BLOQUE 11 — LIGHTGBM CON OPTIMIZACION DE HIPERPARAMETROS
# =============================================================================
# LightGBM crece los arboles hoja a hoja (leaf-wise) en lugar de nivel a nivel.
# Esto puede dar ventajas en datasets de tamaño moderado como el nuestro.
# Dos experimentos: parametros por defecto y optimizacion con Optuna.

with open(RUTA + "resultados_todos.json") as f:
    resultados = json.load(f)

for categoria in ["Waters", "Alpro"]:
    print(f"\n{'='*65}")
    print(f"LIGHTGBM — {categoria}")
    print(f"{'='*65}")

    df_cat = df[df["Categoria"] == categoria].copy().sort_values(
        ["Idcproducto", "Provincia", "FechaMes"]
    )
    features = [c for c in df_cat.columns if c not in COLS_EXCLUIR]

    fecha_corte = pd.Timestamp("2025-10-01")
    train_df = df_cat[df_cat["FechaMes"] < fecha_corte].dropna(
        subset=features + ["volumen_total"]
    )
    test_df = df_cat[df_cat["FechaMes"] >= fecha_corte].dropna(
        subset=features
    ).copy()

    print(f"Train: {len(train_df):,} filas | Test: {len(test_df):,} filas")

    # Experimento 1: parametros por defecto
    modelo_lgb = lgb.LGBMRegressor(
        n_estimators=300, learning_rate=0.1, max_depth=6,
        random_state=42, n_jobs=-1, verbose=-1
    )
    modelo_lgb.fit(train_df[features], train_df["volumen_total"])
    test_df["pred_lgb"] = np.maximum(modelo_lgb.predict(test_df[features]), 0)
    r_lgb = calcular_metricas_mensual(
        test_df, "volumen_total", "pred_lgb", f"LightGBM_default_{categoria}"
    )
    resultados.append(r_lgb)

    # Experimento 2: optimizacion con Optuna
    print(f"\nOptimizando LightGBM con Optuna (50 trials)...")

    def objective_lgb(trial):
        params = {
            "n_estimators":      trial.suggest_int("n_estimators", 100, 800),
            "max_depth":         trial.suggest_int("max_depth", 3, 9),
            "learning_rate":     trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "num_leaves":        trial.suggest_int("num_leaves", 15, 127),
            "subsample":         trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree":  trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "min_child_samples": trial.suggest_int("min_child_samples", 5, 50),
            "reg_alpha":         trial.suggest_float("reg_alpha", 0.0, 1.0),
            "reg_lambda":        trial.suggest_float("reg_lambda", 0.0, 2.0),
            "random_state": 42, "n_jobs": -1, "verbose": -1
        }
        fecha_val = train_df["FechaMes"].max() - pd.DateOffset(months=2)
        X_tr  = train_df[train_df["FechaMes"] < fecha_val][features]
        y_tr  = train_df[train_df["FechaMes"] < fecha_val]["volumen_total"]
        X_val = train_df[train_df["FechaMes"] >= fecha_val][features]
        y_val = train_df[train_df["FechaMes"] >= fecha_val]["volumen_total"]
        if len(X_tr) == 0 or len(X_val) == 0:
            return float("inf")
        m = lgb.LGBMRegressor(**params)
        m.fit(X_tr, y_tr,
              eval_set=[(X_val, y_val)],
              callbacks=[lgb.early_stopping(20, verbose=False),
                         lgb.log_evaluation(-1)])
        val_df = train_df[train_df["FechaMes"] >= fecha_val].copy()
        val_df["pred"] = np.maximum(m.predict(val_df[features]), 0)
        real_mes_val = val_df.groupby("FechaMes")["volumen_total"].sum()
        pred_mes_val = val_df.groupby("FechaMes")["pred"].sum()
        return mean_absolute_error(real_mes_val.values, pred_mes_val.values)

    study_lgb = optuna.create_study(direction="minimize")
    study_lgb.optimize(objective_lgb, n_trials=50, show_progress_bar=True)

    best_lgb = study_lgb.best_params
    best_lgb.update({"random_state": 42, "n_jobs": -1, "verbose": -1})
    print(f"\nMejores parametros LightGBM:")
    for k, v in best_lgb.items():
        if k not in ["random_state", "n_jobs", "verbose"]:
            print(f"  {k:<25} {v}")

    modelo_lgb_opt = lgb.LGBMRegressor(**best_lgb)
    modelo_lgb_opt.fit(train_df[features], train_df["volumen_total"])
    test_df["pred_lgb_opt"] = np.maximum(
        modelo_lgb_opt.predict(test_df[features]), 0
    )
    r_lgb_opt = calcular_metricas_mensual(
        test_df, "volumen_total", "pred_lgb_opt", f"LightGBM_optimizado_{categoria}"
    )
    resultados.append(r_lgb_opt)

    baseline_mape   = 2.43  if categoria == "Waters" else 5.36
    baseline_nombre = "Naive estacional" if categoria == "Waters" else "Media movil 6m"
    mejor_mape = min(r_lgb["MAPE"], r_lgb_opt["MAPE"])
    mejora = (baseline_mape - mejor_mape) / baseline_mape * 100
    print(f"\n{'─'*55}")
    print(f"  Baseline ({baseline_nombre}): MAPE = {baseline_mape:.2f}%")
    print(f"  LightGBM default:          MAPE = {r_lgb['MAPE']:.2f}%")
    print(f"  LightGBM optimizado:       MAPE = {r_lgb_opt['MAPE']:.2f}%")
    print(f"  Mejor mejora relativa:     {mejora:+.1f}%")
    if mejora >= 20:
        print(f"  OBJETIVO CONSEGUIDO (reduccion >= 20%)")
    print(f"{'─'*55}")

with open(RUTA + "resultados_todos.json", "w") as f:
    json.dump(resultados, f, indent=2)
print("\n Resultados LightGBM guardados")

# =============================================================================

# BLOQUE 12 — RED NEURONAL LSTM (Deep Learning)
# Arquitectura: 2 capas LSTM + capa lineal de salida
# Entrada: secuencias de SEQ_LEN meses → prediccion del mes siguiente

SEQ_LEN = 6  # usa los ultimos 6 meses como contexto


class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size=64, num_layers=2, dropout=0.2):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            batch_first=True
        )
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])
        return out.squeeze()


def crear_secuencias(grupo_df, X_scaled_all, y_scaled_all, seq_len):
    """Para cada mes T crea una secuencia con los seq_len meses anteriores."""
    seqs, targets, fechas = [], [], []
    indices = grupo_df["idx"].values
    for i in range(seq_len, len(indices)):
        idx_seq = indices[i-seq_len:i]
        idx_tgt = indices[i]
        seqs.append(X_scaled_all[idx_seq])
        targets.append(y_scaled_all[idx_tgt])
        fechas.append(grupo_df.iloc[i]["FechaMes"])
    return seqs, targets, fechas


for categoria in ["Waters", "Alpro"]:
    print(f"\n{'='*65}")
    print(f"LSTM — {categoria}")
    print(f"{'='*65}")

    df_cat = df[df["Categoria"] == categoria].copy().sort_values(
        ["Idcproducto", "Provincia", "FechaMes"]
    )
    features = [c for c in df_cat.columns if c not in COLS_EXCLUIR]
    df_cat   = df_cat.dropna(subset=features + ["volumen_total"])

    fecha_corte = pd.Timestamp("2025-10-01")

    # Escalamos los datos — la LSTM es sensible a la magnitud de los valores
    scaler_X = MinMaxScaler()
    scaler_y = MinMaxScaler()
    X_all    = df_cat[features].values
    y_all    = df_cat["volumen_total"].values.reshape(-1, 1)
    X_scaled = scaler_X.fit_transform(X_all)
    y_scaled = scaler_y.fit_transform(y_all).flatten()

    df_cat["idx"] = range(len(df_cat))

    X_train_seqs, y_train_seqs = [], []
    X_test_seqs,  y_test_seqs  = [], []
    fechas_test = []

    for _, grupo in df_cat.groupby(["Idcproducto", "Provincia"]):
        if len(grupo) <= SEQ_LEN:
            continue
        seqs, targets, fechas = crear_secuencias(
            grupo, X_scaled, y_scaled, SEQ_LEN
        )
        for s, t, f in zip(seqs, targets, fechas):
            if f < fecha_corte:
                X_train_seqs.append(s)
                y_train_seqs.append(t)
            else:
                X_test_seqs.append(s)
                y_test_seqs.append(t)
                fechas_test.append(f)

    if len(X_train_seqs) == 0 or len(X_test_seqs) == 0:
        print("⚠ No hay suficientes secuencias — saltando LSTM")
        continue

    X_train_t = torch.FloatTensor(np.array(X_train_seqs))
    y_train_t = torch.FloatTensor(np.array(y_train_seqs))
    X_test_t  = torch.FloatTensor(np.array(X_test_seqs))
    y_test_t  = torch.FloatTensor(np.array(y_test_seqs))

    print(f"Secuencias train: {len(X_train_t):,} | test: {len(X_test_t):,}")
    print(f"Shape entrada LSTM: {X_train_t.shape}")

    dataset_train = TensorDataset(X_train_t, y_train_t)
    loader_train  = DataLoader(dataset_train, batch_size=64, shuffle=True)

    # Definimos el modelo, la funcion de perdida y el optimizador
    input_size  = X_train_t.shape[2]
    modelo_lstm = LSTMModel(input_size=input_size, hidden_size=64,
                            num_layers=2, dropout=0.2)
    criterion   = nn.MSELoss()
    optimizer   = torch.optim.Adam(modelo_lstm.parameters(), lr=0.001)
    scheduler   = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=5, factor=0.5
    )

    EPOCHS       = 50
    best_loss    = float("inf")
    train_losses = []

    print(f"\nEntrenando LSTM ({EPOCHS} epocas)...")
    for epoch in range(EPOCHS):
        modelo_lstm.train()
        epoch_loss = 0
        for X_batch, y_batch in loader_train:
            optimizer.zero_grad()
            pred  = modelo_lstm(X_batch)
            loss  = criterion(pred, y_batch)
            loss.backward()
            # Clip de gradientes para evitar explosion durante el entrenamiento
            torch.nn.utils.clip_grad_norm_(modelo_lstm.parameters(), 1.0)
            optimizer.step()
            epoch_loss += loss.item()

        epoch_loss /= len(loader_train)
        train_losses.append(epoch_loss)
        scheduler.step(epoch_loss)

        if epoch_loss < best_loss:
            best_loss = epoch_loss
            torch.save(modelo_lstm.state_dict(),
                       RUTA + f"lstm_{categoria.lower()}_best.pt")

        if (epoch + 1) % 10 == 0:
            print(f"  Epoca {epoch+1:3d}/{EPOCHS} | Loss: {epoch_loss:.6f}")

    # Cargamos el mejor modelo (menor loss durante el entrenamiento)
    modelo_lstm.load_state_dict(
        torch.load(RUTA + f"lstm_{categoria.lower()}_best.pt")
    )

    # Prediccion y desescalado al rango original de litros
    modelo_lstm.eval()
    with torch.no_grad():
        preds_scaled = modelo_lstm(X_test_t).numpy()

    preds_orig  = scaler_y.inverse_transform(
        preds_scaled.reshape(-1, 1)
    ).flatten()
    preds_orig  = np.maximum(preds_orig, 0)
    y_test_orig = scaler_y.inverse_transform(
        y_test_t.numpy().reshape(-1, 1)
    ).flatten()

    test_df_lstm = pd.DataFrame({
        "FechaMes":      fechas_test,
        "volumen_total": y_test_orig,
        "pred_lstm":     preds_orig
    })

    r_lstm = calcular_metricas_mensual(
        test_df_lstm, "volumen_total", "pred_lstm", f"LSTM_{categoria}"
    )
    resultados.append(r_lstm)

    baseline_mape   = 2.43  if categoria == "Waters" else 5.36
    baseline_nombre = "Naive estacional" if categoria == "Waters" else "Media movil 6m"
    mejora = (baseline_mape - r_lstm["MAPE"]) / baseline_mape * 100
    print(f"\n{'─'*55}")
    print(f"  Baseline ({baseline_nombre}): MAPE = {baseline_mape:.2f}%")
    print(f"  LSTM:                      MAPE = {r_lstm['MAPE']:.2f}%")
    print(f"  Mejora relativa:           {mejora:+.1f}%")
    if mejora >= 20:
        print(f"  ✓ OBJETIVO TFM CONSEGUIDO (reduccion >= 20%)")
    print(f"{'─'*55}")

    # Grafica: curva de perdida + prediccion vs real
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(f"LSTM — {categoria}", fontsize=13)

    axes[0].plot(train_losses, color="#9467bd", linewidth=2)
    axes[0].set_title("Curva de perdida (entrenamiento)")
    axes[0].set_xlabel("Epoca")
    axes[0].set_ylabel("MSE Loss")

    real_mes = test_df_lstm.groupby("FechaMes")["volumen_total"].sum()
    pred_mes = test_df_lstm.groupby("FechaMes")["pred_lstm"].sum()
    x     = np.arange(len(real_mes))
    width = 0.35
    axes[1].bar(x - width/2, real_mes.values, width,
                label="Real", color="#1f77b4", alpha=0.85)
    axes[1].bar(x + width/2, pred_mes.values, width,
                label="LSTM", color="#9467bd", alpha=0.85)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels([m.strftime("%Y-%m") for m in real_mes.index])
    axes[1].set_title("Prediccion vs Real — Oct-Dic 2025")
    axes[1].set_ylabel("Litros")
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(RUTA + f"lstm_{categoria.lower()}.png", dpi=150, bbox_inches="tight")
    plt.show()
    print(f"✓ Grafica LSTM guardada")

# =============================================================================
# BLOQUE 13 — COMPARATIVA FINAL DE TODOS LOS MODELOS
# =============================================================================

# Funcion para convertir tipos numpy/torch a tipos Python estandar
# JSON no sabe serializar float32 de PyTorch ni int64 de numpy
def convertir_a_python(obj):
    if isinstance(obj, dict):
        return {k: convertir_a_python(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convertir_a_python(i) for i in obj]
    elif hasattr(obj, "item"):  # numpy/torch scalar → Python nativo
        return obj.item()
    else:
        return obj

print(f"\n{'='*80}")
print(f"COMPARATIVA FINAL COMPLETA — TODOS LOS MODELOS")
print(f"{'='*80}")
print(f"{'Modelo':<42} {'MAE':>12} {'RMSE':>12} {'MAPE':>8}")
print(f"{'-'*80}")

for cat in ["Waters", "Alpro"]:
    print(f"\n  --- {cat} ---")
    cat_res = [r for r in resultados if cat in r["modelo"]]
    cat_res_sorted = sorted(cat_res, key=lambda x: x["MAPE"])
    for i, r in enumerate(cat_res_sorted):
        marca = " *" if i == 0 else ""
        print(f"  {r['modelo']:<40}{marca} {r['MAE']:>12,.0f} "
              f"{r['RMSE']:>12,.0f} {r['MAPE']:>7.2f}%")

# Grafica de barras horizontales comparando el MAPE de todos los modelos
fig, axes = plt.subplots(1, 2, figsize=(16, 6))
fig.suptitle("Comparativa MAPE — todos los modelos", fontsize=14)

for i, cat in enumerate(["Waters", "Alpro"]):
    cat_res = [r for r in resultados if cat in r["modelo"]]
    cat_res = sorted(cat_res, key=lambda x: x["MAPE"], reverse=True)
    nombres = [r["modelo"].replace(f"_{cat}", "").replace("_", " ") for r in cat_res]
    mapes   = [r["MAPE"] for r in cat_res]

    colores = []
    for m, n in zip(mapes, nombres):
        if m == min(mapes):
            colores.append("#2ca02c")    # verde → mejor modelo
        elif "LSTM" in n:
            colores.append("#9467bd")    # morado → deep learning
        elif "LightGBM" in n:
            colores.append("#1f77b4")    # azul → LightGBM
        elif "XGB" in n:
            colores.append("#ff7f0e")    # naranja → XGBoost
        else:
            colores.append("#d62728")    # rojo → baseline

    axes[i].barh(nombres, mapes, color=colores, alpha=0.85)
    axes[i].axvline(x=20, color="red", linestyle="--", linewidth=1.5,
                    label="Objetivo TFM (20% mejora)")
    axes[i].set_title(f"MAPE por modelo — {cat}")
    axes[i].set_xlabel("MAPE (%)")
    axes[i].legend(fontsize=9)

plt.tight_layout()
plt.savefig(RUTA + "comparativa_final_mape.png", dpi=150, bbox_inches="tight")
plt.show()
print("✓ Grafica comparativa final guardada")

# Convertimos y guardamos todos los resultados finales
resultados_serializables = convertir_a_python(resultados)

with open(RUTA + "resultados_finales_completos.json", "w") as f:
    json.dump(resultados_serializables, f, indent=2)

print(f"\n{'='*65}")
print("PIPELINE COMPLETADO")
print(f"{'='*65}")
print("Ficheros generados:")
print("  dataset_modelo.csv               → agregacion mensual")
print("  dataset_limpio.csv               → tras tratamiento de anomalias")
print("  dataset_modelo_final.csv         → filtrado 2024-2025")
print("  dataset_features.csv             → con todas las variables del modelo")
print("  resultados_finales_completos.json → metricas de todos los modelos")
print("  *.png                            → graficas del EDA y los modelos")
print("  lstm_*_best.pt                   → pesos de las redes LSTM entrenadas")
