# Homicidios Colombia 2015–2024
### Pipeline de datos forenses | HBD Final Project

Pipeline de almacenamiento, análisis con machine learning y orquestación sobre el dataset de homicidios del Instituto Nacional de Medicina Legal y Ciencias Forenses (INMLCF).

---


### Paso 7 — Carga en PostgreSQL

El dataset limpio (`staging/processed/homicidios_clean.csv`) se carga en una base de datos PostgreSQL corriendo en Docker. La tabla se crea dinámicamente con los tipos de datos correctos inferidos del DataFrame, y la carga se hace con `copy_expert` de psycopg2 — un método bulk que cargó 125,284 registros en menos de 2 segundos. Se ejecutan cuatro queries de validación para confirmar integridad de los datos.

### Paso 8 — Clustering KMeans de municipios

Se agruparon 911 municipios (con 5 o más casos registrados) según su perfil de violencia usando KMeans. Las features usadas fueron `total_casos`, `ratio_masculino` y `años_activos`. El número óptimo de clusters (K=4) se determinó con el coeficiente de silueta. Los resultados se visualizan en un dashboard interactivo en `dashboard/clustering_municipios.html`.

### Paso 9 — Orquestación con Prefect

Los pasos 7 y 8 están orquestados como un `@flow` de Prefect con 8 `@task` individuales. Cada task tiene su propio log de estado, tiempo de ejecución y manejo de errores con reintentos automáticos en la conexión a base de datos.

---

## Requisitos

- Python 3.11 (recomendado via Miniconda)
- Docker Desktop

```bash
conda create -n hbd-project python=3.11 -y
conda activate hbd-project
pip install -r requirements.txt
```

---

## Cómo correr

```bash
# 1. Levantar PostgreSQL
docker-compose up -d postgres

# 2. Correr el pipeline (PostgreSQL + KMeans + Prefect)
python src/pipeline_full.py
```

---

## Dataset

- **Fuente:** INMLCF vía datos.gov.co
- **Período:** 2015–2024
- **Registros:** 125,284 homicidios
- **Variables:** 30 columnas — departamento, municipio, sexo, edad, mecanismo de muerte, escolaridad, ciclo vital, entre otras

---

## Tecnologías usadas en mi parte

| Herramienta | Uso |
|---|---|
| PostgreSQL 15 + Docker | Almacenamiento estructurado |
| psycopg2 `copy_expert` | Carga eficiente en bulk |
| Scikit-learn KMeans | Clustering de municipios |
| Plotly | Dashboard interactivo |
| Prefect 2 | Orquestación del pipeline |
