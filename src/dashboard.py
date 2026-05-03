

import os
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots


# Paleta y estilo compartido
COLORES_ZONA = {
    "cabecera municipal":             "#e63946",
    "parte rural (vereda y campo)":   "#2a9d8f",
    "centro poblado":                 "#e9c46a",
}

COLOR_PRINCIPAL = "#e63946"
COLOR_SECUNDARIO = "#264653"
FUENTE = "Inter, Arial, sans-serif"

LAYOUT_BASE = dict(
    font=dict(family=FUENTE, size=13),
    paper_bgcolor="#121724",
    plot_bgcolor="#0f1117",
    font_color="#e0e0e0",
    margin=dict(l=60, r=40, t=60, b=60),
)



# Gráfica 1 — Serie de tiempo: homicidios por año


def fig_serie_temporal(df_anio: pd.DataFrame) -> go.Figure:
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=df_anio["Año del hecho"],
        y=df_anio["total"],
        mode="lines+markers+text",
        text=df_anio["total"],
        textposition="top center",
        textfont=dict(size=11),
        line=dict(color=COLOR_PRINCIPAL, width=3),
        marker=dict(size=8, color=COLOR_PRINCIPAL),
        fill="tozeroy",
        fillcolor="rgba(230,57,70,0.15)",
        name="Homicidios",
    ))

    fig.update_layout(
        **LAYOUT_BASE,
        title="Homicidios por año · Colombia 2015–2024",
        xaxis=dict(title="Año", tickmode="linear", dtick=1, gridcolor="#2a2a3a"),
        yaxis=dict(title="Total de casos", gridcolor="#2a2a3a"),
    )
    return fig



# Gráfica 2 — Top 10 departamentos (barras horizontales)


def fig_top_departamentos(df_deptos: pd.DataFrame) -> go.Figure:
    df_sorted = df_deptos.sort_values("total", ascending=True)

    fig = go.Figure(go.Bar(
        x=df_sorted["total"],
        y=df_sorted["Departamento del hecho DANE"].str.title(),
        orientation="h",
        marker=dict(
            color=df_sorted["total"],
            colorscale="Reds",
            showscale=False,
        ),
        text=df_sorted["total"],
        textposition="outside",
    ))

    fig.update_layout(
        **LAYOUT_BASE,
        title="Top 10 departamentos con más homicidios (2015–2024)",
        xaxis=dict(title="Total de casos", gridcolor="#2a2a3a"),
        yaxis=dict(title=""),
        height=420,
    )
    return fig



# Gráfica 3 — Distribución por sexo (pie)

def fig_distribucion_sexo(df_sexo: pd.DataFrame) -> go.Figure:
    etiquetas = df_sexo["Sexo de la victima"].str.title()
    colores = ["#e63946", "#457b9d", "#a8dadc"]

    fig = go.Figure(go.Pie(
        labels=etiquetas,
        values=df_sexo["total"],
        hole=0.45,
        marker=dict(colors=colores),
        textinfo="label+percent",
        textfont=dict(size=13),
    ))

    fig.update_layout(
        **LAYOUT_BASE,
        title="Víctimas por sexo · 2015–2024",
        showlegend=True,
        legend=dict(orientation="h", y=-0.1),
    )
    return fig



# Gráfica 4 — Mecanismo causal (barras verticales, top 8)


def fig_mecanismo_causal(df_mec: pd.DataFrame) -> go.Figure:
    df_top = df_mec.head(8)

    fig = go.Figure(go.Bar(
        x=df_top["mecanismo"].str.title(),
        y=df_top["total"],
        marker=dict(
            color=df_top["total"],
            colorscale="OrRd",
            showscale=False,
        ),
        text=df_top["total"],
        textposition="outside",
    ))

    fig.update_layout(
        **LAYOUT_BASE,
        title="Mecanismo causal de muerte · Top 8",
        xaxis=dict(title="", tickangle=-30, gridcolor="#2a2a3a"),
        yaxis=dict(title="Casos", gridcolor="#2a2a3a"),
        height=420,
    )
    return fig



# Gráfica 5 — Heatmap: departamento × circunstancia

def fig_heatmap(pivot: pd.DataFrame) -> go.Figure:
    fig = go.Figure(go.Heatmap(
        z=pivot.values,
        x=[c.title() for c in pivot.columns],
        y=[r.title() for r in pivot.index],
        colorscale="Reds",
        text=pivot.values,
        texttemplate="%{text}",
        textfont=dict(size=10),
        hovertemplate="Depto: %{y}<br>Circunstancia: %{x}<br>Casos: %{z}<extra></extra>",
    ))

    layout = {**LAYOUT_BASE, "margin": dict(l=180, r=40, t=60, b=130)}
    fig.update_layout(
        **layout,
        title="Heatmap: Departamento × Circunstancia del hecho",
        xaxis=dict(tickangle=-35),
        height=500,
    )
    return fig



# Gráfica 6 — Líneas por zona a lo largo del tiempo


def fig_zona_anio(df_zona: pd.DataFrame) -> go.Figure:
    fig = go.Figure()

    for zona in df_zona["Zona del Hecho"].unique():
        subset = df_zona[df_zona["Zona del Hecho"] == zona]
        color = COLORES_ZONA.get(zona, "#cccccc")

        fig.add_trace(go.Scatter(
            x=subset["Año del hecho"],
            y=subset["total"],
            mode="lines+markers",
            name=zona.title(),
            line=dict(color=color, width=2.5),
            marker=dict(size=7),
        ))

    fig.update_layout(
        **LAYOUT_BASE,
        title="Homicidios por zona del hecho · 2015–2024",
        xaxis=dict(title="Año", tickmode="linear", dtick=1, gridcolor="#2a2a3a"),
        yaxis=dict(title="Casos", gridcolor="#2a2a3a"),
        legend=dict(orientation="h", y=-0.15),
    )
    return fig



# Gráfica 7 — Feminicidios desde 2018


def fig_feminicidios(df_femi: pd.DataFrame) -> go.Figure:
    fig = go.Figure()

    fig.add_trace(go.Bar(
        x=df_femi["Año del hecho"],
        y=df_femi["total_feminicidios"],
        marker=dict(color="#c77dff"),
        text=df_femi["total_feminicidios"],
        textposition="outside",
        name="Feminicidios",
    ))

    fig.update_layout(
        **LAYOUT_BASE,
        title="Feminicidios registrados por el INMLCF · 2018–2024",
        xaxis=dict(title="Año", tickmode="linear", dtick=1, gridcolor="#2a2a3a"),
        yaxis=dict(title="Casos registrados", gridcolor="#2a2a3a"),
        showlegend=False,
    )
    return fig



# Función principal: recibe los resultados de analysis.py y genera el HTML


def generar_dashboard(resultados: dict, output_path: str = "src/dashboard_homicidios.html"):
    """
    Recibe el diccionario de DataFrames de ejecutar_analisis()
    y exporta un único archivo HTML con todas las visualizaciones.
    """
    print("\n====== PASO 6: GENERANDO DASHBOARD ======\n")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Generar cada figura
    figs = {
        "serie_temporal":    fig_serie_temporal(resultados["por_anio"]),
        "top_departamentos": fig_top_departamentos(resultados["top_deptos"]),
        "sexo":              fig_distribucion_sexo(resultados["por_sexo"]),
        "mecanismo":         fig_mecanismo_causal(resultados["mecanismo"]),
        "heatmap":           fig_heatmap(resultados["heatmap"]),
        "zona_anio":         fig_zona_anio(resultados["zona_anio"]),
        "feminicidios":      fig_feminicidios(resultados["feminicidios"]),
    }

    # Convertir cada figura a HTML embebido (sin JS repetido)
    bloques_html = []
    for i, (nombre, fig) in enumerate(figs.items()):
        # include_plotlyjs solo en la primera figura
        include_js = "cdn" if i == 0 else False
        html_fig = fig.to_html(
            full_html=False,
            include_plotlyjs=include_js,
            div_id=f"fig_{nombre}",
        )
        bloques_html.append(f'<div class="chart-card">{html_fig}</div>')

    # Armar el HTML completo
    html_final = f"""<!DOCTYPE html>
<html lang="es">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0"/>
  <title>Presuntos Homicidios · Colombia 2015–2024</title>
  <style>
    *, *::before, *::after {{ box-sizing: border-box; margin: 0; padding: 0; }}

    body {{
      background: #0f1117;
      color: #e0e0e0;
      font-family: Inter, Arial, sans-serif;
      padding: 2rem;
    }}

    header {{
      text-align: center;
      margin-bottom: 2.5rem;
    }}

    header h1 {{
      font-size: 2rem;
      color: #e63946;
      letter-spacing: 0.03em;
    }}

    header p {{
      font-size: 0.95rem;
      color: #999;
      margin-top: 0.4rem;
    }}

    .grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(560px, 1fr));
      gap: 1.5rem;
    }}

    .chart-card {{
      background: #1a1d27;
      border-radius: 12px;
      padding: 1.2rem;
      border: 1px solid #2a2a3a;
      box-shadow: 0 4px 20px rgba(0,0,0,0.4);
    }}

    /* Heatmap y serie temporal ocupan el ancho completo */
    .chart-card:nth-child(1),
    .chart-card:nth-child(5) {{
      grid-column: 1 / -1;
    }}

    footer {{
      text-align: center;
      margin-top: 2.5rem;
      color: #555;
      font-size: 0.82rem;
    }}
  </style>
</head>
<body>
  <header>
    <h1>Presuntos Homicidios · Colombia 2015–2024</h1>
    <p>Fuente: Instituto Nacional de Medicina Legal y Ciencias Forenses (INMLCF) · datos.gov.co</p>
  </header>

  <div class="grid">
    {''.join(bloques_html)}
  </div>

  <footer>
    Proyecto Big Data Tools · Pipeline orquestado con Prefect · Visualización con Plotly
  </footer>
</body>
</html>"""

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html_final)

    print(f"[6] Dashboard exportado → {output_path}")
    print(f"    Tamaño: {os.path.getsize(output_path) / 1024:.1f} KB")
    print("\n====== DASHBOARD COMPLETADO ======\n")


if __name__ == "__main__":
    import pandas as pd
    import sys
    sys.path.insert(0, "src")
    from analisis import ejecutar_analisis

    df = pd.read_csv("staging/processed/homicidios_clean.csv")
    resultados = ejecutar_analisis(df)
    generar_dashboard(resultados)