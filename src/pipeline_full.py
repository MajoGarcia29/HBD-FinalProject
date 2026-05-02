"""
pipeline_full.py — Pasos 7, 8 y 9 del proyecto HBD-FinalProject
-----------------------------------------------------------------
Paso 7: Carga del dataset limpio en PostgreSQL con copy_expert
Paso 8: Clustering KMeans de municipios por perfil de violencia
Paso 9: Todo orquestado con Prefect (@task + @flow)

Prerequisito: correr primero src/pipeline.py para generar
              staging/processed/homicidios_clean.csv
"""

import os
import io
import time
import logging
import warnings

import numpy as np
import pandas as pd
import psycopg2
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from prefect import flow, task, get_run_logger
from prefect.cache_policies import NO_CACHE

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# CONFIGURACIÓN DE BASE DE DATOS
# ─────────────────────────────────────────────

DB_CONFIG = {
    "host": os.getenv("DB_HOST", "localhost"),
    "port": int(os.getenv("DB_PORT", 5432)),
    "dbname": os.getenv("DB_NAME", "homicidios_db"),
    "user": os.getenv("DB_USER", "homicidios_user"),
    "password": os.getenv("DB_PASS", "homicidios_pass"),
}

CLEAN_CSV = "staging/processed/homicidios_clean.csv"
DASHBOARD_DIR = "dashboard"
TABLE_NAME = "homicidios_clean"


# ─────────────────────────────────────────────
# PASO 7: POSTGRESQL
# ─────────────────────────────────────────────

@task(name="7a - Conectar a PostgreSQL", retries=3, retry_delay_seconds=5)
def conectar_postgres() -> psycopg2.extensions.connection:
    logger = get_run_logger()
    logger.info(f"Conectando a PostgreSQL en {DB_CONFIG['host']}:{DB_CONFIG['port']}")
    conn = psycopg2.connect(**DB_CONFIG)
    logger.info("Conexión establecida ✓")
    return conn


@task(name="7b - Crear tabla homicidios_clean", cache_policy=NO_CACHE)
def crear_tabla(conn: psycopg2.extensions.connection, df: pd.DataFrame) -> None:
    logger = get_run_logger()

    # Mapear dtypes de pandas a tipos SQL
    tipo_map = {
        "int64": "INTEGER",
        "float64": "DOUBLE PRECISION",
        "object": "TEXT",
        "bool": "BOOLEAN",
    }

    columnas_sql = []
    for col, dtype in df.dtypes.items():
        sql_type = tipo_map.get(str(dtype), "TEXT")
        col_safe = col.replace(" ", "_").replace("(", "").replace(")", "").lower()
        columnas_sql.append(f'"{col_safe}" {sql_type}')

    sep = ",\n        "
    ddl = f"""
    DROP TABLE IF EXISTS {TABLE_NAME};
    CREATE TABLE {TABLE_NAME} (
        {sep.join(columnas_sql)}
    );
    """

    with conn.cursor() as cur:
        cur.execute(ddl)
    conn.commit()
    logger.info(f"Tabla '{TABLE_NAME}' creada con {len(columnas_sql)} columnas ✓")


@task(name="7c - Cargar datos con copy_expert", cache_policy=NO_CACHE)
def cargar_datos_postgres(conn: psycopg2.extensions.connection, df: pd.DataFrame) -> int:
    logger = get_run_logger()

    # Normalizar nombres de columnas igual que en la tabla
    df_copy = df.copy()
    df_copy.columns = [
        c.replace(" ", "_").replace("(", "").replace(")", "").lower()
        for c in df_copy.columns
    ]

    # copy_expert: más rápido que INSERT fila por fila
    buffer = io.StringIO()
    df_copy.to_csv(buffer, index=False, header=True)
    buffer.seek(0)

    cols = ", ".join([f'"{c}"' for c in df_copy.columns])
    copy_sql = f"COPY {TABLE_NAME} ({cols}) FROM STDIN WITH CSV HEADER NULL ''"

    t0 = time.time()
    with conn.cursor() as cur:
        cur.copy_expert(copy_sql, buffer)
    conn.commit()
    elapsed = time.time() - t0

    logger.info(f"{len(df_copy):,} filas cargadas en {elapsed:.2f}s con copy_expert ✓")
    return len(df_copy)


@task(name="7d - Queries SQL de validación", cache_policy=NO_CACHE)
def queries_validacion(conn: psycopg2.extensions.connection) -> dict:
    logger = get_run_logger()

    queries = {
        "total_registros": f"SELECT COUNT(*) FROM {TABLE_NAME};",
        "por_año": f"""
            SELECT año_del_hecho, COUNT(*) as casos
            FROM {TABLE_NAME}
            GROUP BY año_del_hecho
            ORDER BY año_del_hecho;
        """,
        "top5_departamentos": f"""
            SELECT departamento_del_hecho_dane, COUNT(*) as casos
            FROM {TABLE_NAME}
            GROUP BY departamento_del_hecho_dane
            ORDER BY casos DESC
            LIMIT 5;
        """,
        "por_sexo": f"""
            SELECT sexo_de_la_victima, COUNT(*) as casos
            FROM {TABLE_NAME}
            GROUP BY sexo_de_la_victima;
        """,
    }

    resultados = {}
    with conn.cursor() as cur:
        for nombre, sql in queries.items():
            cur.execute(sql)
            resultados[nombre] = cur.fetchall()
            logger.info(f"Query '{nombre}': {resultados[nombre][:3]}...")

    return resultados


# ─────────────────────────────────────────────
# PASO 8: CLUSTERING KMEANS
# ─────────────────────────────────────────────

@task(name="8a - Preparar features para clustering")
def preparar_features_clustering(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    logger = get_run_logger()

    # Detectar columna de departamento/municipio con nombres flexibles
    col_municipio = next(
        (c for c in df.columns if "municipio" in c.lower() and "hecho" in c.lower()),
        next((c for c in df.columns if "municipio" in c.lower()), None)
    )
    col_depto = next(
        (c for c in df.columns if "departamento" in c.lower() and "hecho" in c.lower()),
        next((c for c in df.columns if "departamento" in c.lower()), None)
    )
    col_año = next((c for c in df.columns if "año" in c.lower()), None)
    col_sexo = next((c for c in df.columns if "sexo" in c.lower()), None)
    col_mecanismo = next((c for c in df.columns if "mecanismo" in c.lower()), None)

    logger.info(f"Columnas detectadas → municipio: {col_municipio}, depto: {col_depto}, año: {col_año}")

    if col_municipio is None:
        raise ValueError("No se encontró columna de municipio en el dataset")

    group_cols = [col_municipio]
    if col_depto:
        group_cols.append(col_depto)

    # Construir features por municipio una por una (evita MultiIndex)
    municipio_df = df.groupby(group_cols).size().reset_index(name="total_casos")

    # Feature: ratio de víctimas masculinas
    if col_sexo:
        df["_es_masculino"] = (df[col_sexo].str.lower() == "masculino").astype(int)
        ratio = df.groupby(group_cols)["_es_masculino"].mean().reset_index()
        ratio.rename(columns={"_es_masculino": "ratio_masculino"}, inplace=True)
        municipio_df = municipio_df.merge(ratio, on=group_cols, how="left")

    # Feature: años activos (qué tan sostenida es la violencia en ese municipio)
    if col_año:
        años_activos = df.groupby(group_cols)[col_año].nunique().reset_index()
        años_activos.rename(columns={col_año: "años_activos"}, inplace=True)
        municipio_df = municipio_df.merge(años_activos, on=group_cols, how="left")

    # Filtrar municipios con >= 5 casos para clustering estable
    municipio_df = municipio_df[municipio_df["total_casos"] >= 5].copy()

    # Seleccionar features numéricas
    excluir = set(group_cols)
    feature_cols = [
        c for c in municipio_df.columns
        if municipio_df[c].dtype in [np.float64, np.int64]
        and c not in excluir
    ]

    logger.info(f"Features para clustering: {feature_cols}")
    logger.info(f"Municipios con >= 5 casos: {len(municipio_df):,}")

    features = municipio_df[feature_cols].fillna(0)

    return municipio_df, features


@task(name="8b - Determinar K óptimo (método del codo)")
def encontrar_k_optimo(features: pd.DataFrame, k_max: int = 8) -> int:
    logger = get_run_logger()

    scaler = StandardScaler()
    X = scaler.fit_transform(features)

    inercias = []
    silhouettes = []
    ks = range(2, k_max + 1)

    for k in ks:
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = km.fit_predict(X)
        inercias.append(km.inertia_)
        sil = silhouette_score(X, labels)
        silhouettes.append(sil)
        logger.info(f"  K={k} → inercia={km.inertia_:.0f}, silhouette={sil:.3f}")

    # K óptimo = mejor silhouette score
    k_optimo = list(ks)[int(np.argmax(silhouettes))]
    logger.info(f"K óptimo seleccionado: {k_optimo} (silhouette={max(silhouettes):.3f}) ✓")

    return k_optimo


@task(name="8c - Entrenar KMeans y asignar clusters")
def entrenar_kmeans(municipio_df: pd.DataFrame, features: pd.DataFrame, k: int) -> pd.DataFrame:
    logger = get_run_logger()

    scaler = StandardScaler()
    X = scaler.fit_transform(features)

    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    municipio_df = municipio_df.copy()
    municipio_df["cluster"] = km.fit_predict(X)

    # Describir cada cluster
    feature_cols = list(features.columns)
    resumen = municipio_df.groupby("cluster")[feature_cols + ["total_casos"]].mean().round(2)

    logger.info("Resumen de clusters:\n" + resumen.to_string())

    # Guardar resultado
    os.makedirs("staging/processed", exist_ok=True)
    municipio_df.to_csv("staging/processed/municipios_clustered.csv", index=False)
    logger.info("municipios_clustered.csv guardado ✓")

    return municipio_df


@task(name="8d - Visualizar clusters")
def visualizar_clusters(municipio_df: pd.DataFrame) -> None:
    logger = get_run_logger()
    os.makedirs(DASHBOARD_DIR, exist_ok=True)

    col_municipio = next(
        (c for c in municipio_df.columns if "municipio" in c.lower()), "municipio"
    )
    col_depto = next(
        (c for c in municipio_df.columns if "departamento" in c.lower()), None
    )

    # ── Gráfica 1: Distribución de casos por cluster ──
    fig_box = px.box(
        municipio_df,
        x="cluster",
        y="total_casos",
        color="cluster",
        title="Distribución de homicidios por cluster de municipio",
        labels={"total_casos": "Total de casos", "cluster": "Cluster"},
        color_discrete_sequence=px.colors.qualitative.Bold,
    )
    fig_box.update_layout(showlegend=False)

    # ── Gráfica 2: Top 10 municipios por cluster ──
    top_municipios = (
        municipio_df.sort_values("total_casos", ascending=False)
        .head(40)
    )

    hover_cols = [col_municipio, "total_casos", "cluster"]
    if col_depto:
        hover_cols.insert(1, col_depto)

    fig_scatter = px.scatter(
        top_municipios,
        x="total_casos",
        y="cluster",
        color="cluster",
        hover_data=hover_cols,
        size="total_casos",
        title="Top 40 municipios más violentos coloreados por cluster",
        labels={"total_casos": "Casos totales", "cluster": "Cluster"},
        color_discrete_sequence=px.colors.qualitative.Bold,
    )

    # ── Gráfica 3: Heatmap cluster × departamento (si existe) ──
    figures = [fig_box, fig_scatter]
    if col_depto:
        heatmap_data = (
            municipio_df.groupby([col_depto, "cluster"])["total_casos"]
            .sum()
            .unstack(fill_value=0)
        )
        fig_heat = px.imshow(
            heatmap_data,
            title="Casos por departamento y cluster",
            labels={"x": "Cluster", "y": "Departamento", "color": "Casos"},
            color_continuous_scale="Reds",
            aspect="auto",
        )
        figures.append(fig_heat)

    # ── Combinar en HTML ──
    html_parts = [f.to_html(full_html=False, include_plotlyjs="cdn" if i == 0 else False)
                  for i, f in enumerate(figures)]

    html_output = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="utf-8">
        <title>Clustering Municipios — Homicidios Colombia</title>
        <style>
            body {{ font-family: Arial, sans-serif; background: #f8f9fa; padding: 20px; }}
            h1 {{ color: #2c3e50; }}
            .chart {{ background: white; border-radius: 8px; padding: 15px;
                      margin-bottom: 20px; box-shadow: 0 2px 6px rgba(0,0,0,0.1); }}
        </style>
    </head>
    <body>
        <h1>Clustering de Municipios por Perfil de Violencia</h1>
        <p>Análisis KMeans sobre homicidios Colombia 2015–2024 (INMLCF)</p>
        {"".join(f'<div class="chart">{p}</div>' for p in html_parts)}
    </body>
    </html>
    """

    out_path = f"{DASHBOARD_DIR}/clustering_municipios.html"
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(html_output)

    logger.info(f"Dashboard de clustering guardado en {out_path} ✓")


# ─────────────────────────────────────────────
# PASO 9: FLOW PRINCIPAL DE PREFECT
# ─────────────────────────────────────────────

@flow(name="Pipeline Homicidios Colombia — Pasos 7-8-9")
def pipeline_homicidios():
    logger = get_run_logger()

    # ── Cargar dataset limpio ──────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("Cargando staging/processed/homicidios_clean.csv")
    logger.info("=" * 60)

    if not os.path.exists(CLEAN_CSV):
        raise FileNotFoundError(
            f"No se encontró {CLEAN_CSV}. "
            "Ejecuta primero src/pipeline.py para generar el dataset limpio."
        )

    df = pd.read_csv(CLEAN_CSV, low_memory=False)
    logger.info(f"Dataset cargado: {df.shape[0]:,} filas × {df.shape[1]} columnas")

    # ── PASO 7: PostgreSQL ─────────────────────────────────────────────
    logger.info("\n[PASO 7] Carga en PostgreSQL")
    conn = conectar_postgres()
    crear_tabla(conn, df)
    n_filas = cargar_datos_postgres(conn, df)
    resultados_sql = queries_validacion(conn)
    conn.close()
    logger.info(f"Paso 7 completado: {n_filas:,} filas en DB ✓")

    # ── PASO 8: KMeans ─────────────────────────────────────────────────
    logger.info("\n[PASO 8] Clustering KMeans de municipios")
    municipio_df, features = preparar_features_clustering(df)
    k_optimo = encontrar_k_optimo(features)
    municipio_clustered = entrenar_kmeans(municipio_df, features, k_optimo)
    visualizar_clusters(municipio_clustered)
    logger.info("Paso 8 completado: clusters generados y visualizados ✓")

    # ── Resumen final ──────────────────────────────────────────────────
    logger.info("\n" + "=" * 60)
    logger.info("PIPELINE COMPLETADO")
    logger.info(f"  Registros en DB: {n_filas:,}")
    logger.info(f"  Clusters generados: {k_optimo}")
    logger.info(f"  Dashboard: {DASHBOARD_DIR}/clustering_municipios.html")
    logger.info("=" * 60)

    return {
        "registros_db": n_filas,
        "k_clusters": k_optimo,
        "municipios_procesados": len(municipio_clustered),
    }


if __name__ == "__main__":
    resultado = pipeline_homicidios()
    print("\nResultado final:", resultado)