import pandas as pd

def homicidios_por_anio(df):
    return df.groupby("Año del hecho").size().reset_index(name="total").sort_values("Año del hecho")

def top_departamentos(df, top_n=10):
    return df.dropna(subset=["Departamento del hecho DANE"]).groupby("Departamento del hecho DANE").size().reset_index(name="total").sort_values("total", ascending=False).head(top_n)

def distribucion_sexo(df):
    resultado = df.groupby("Sexo de la victima").size().reset_index(name="total").sort_values("total", ascending=False)
    resultado["porcentaje"] = (resultado["total"] / resultado["total"].sum() * 100).round(2)
    return resultado

def mecanismo_causal(df):
    unificaciones = {"corto punzante": "cortopunzante", "corto contundente": "cortocontundente"}
    df_temp = df.copy()
    df_temp["mecanismo"] = df_temp["Mecanismo Causal de la Lesión Fatal"].replace(unificaciones)
    return df_temp.dropna(subset=["mecanismo"]).groupby("mecanismo").size().reset_index(name="total").sort_values("total", ascending=False)

def heatmap_depto_circunstancia(df, top_deptos=10, top_circ=8):
    df_temp = df.dropna(subset=["Departamento del hecho DANE", "Circunstancia del Hecho Detallada"])
    top_d = df_temp.groupby("Departamento del hecho DANE").size().nlargest(top_deptos).index.tolist()
    top_c = df_temp.groupby("Circunstancia del Hecho Detallada").size().nlargest(top_circ).index.tolist()
    df_f = df_temp[df_temp["Departamento del hecho DANE"].isin(top_d) & df_temp["Circunstancia del Hecho Detallada"].isin(top_c)]
    return df_f.groupby(["Departamento del hecho DANE", "Circunstancia del Hecho Detallada"]).size().reset_index(name="total").pivot(index="Departamento del hecho DANE", columns="Circunstancia del Hecho Detallada", values="total").fillna(0).astype(int)

def homicidios_por_zona_anio(df):
    return df.dropna(subset=["Zona del Hecho"]).groupby(["Zona del Hecho", "Año del hecho"]).size().reset_index(name="total").sort_values(["Zona del Hecho", "Año del hecho"])

def feminicidios_desde_2018(df):
    df_femi = df[(df["Año del hecho"] >= 2018) & (df["Circunstancia del Hecho Detallada"].str.contains("femini", na=False, case=False))]
    return df_femi.groupby("Año del hecho").size().reset_index(name="total_feminicidios").sort_values("Año del hecho")

def ejecutar_analisis(df):
    print("\n====== PASO 5: ANÁLISIS AGREGADO ======\n")
    resultados = {
        "por_anio":     homicidios_por_anio(df),
        "top_deptos":   top_departamentos(df),
        "por_sexo":     distribucion_sexo(df),
        "mecanismo":    mecanismo_causal(df),
        "heatmap":      heatmap_depto_circunstancia(df),
        "zona_anio":    homicidios_por_zona_anio(df),
        "feminicidios": feminicidios_desde_2018(df),
    }
    print("\n====== ANÁLISIS COMPLETADO ======\n")
    return resultados

if __name__ == "__main__":
    df = pd.read_csv("staging/processed/homicidios_clean.csv")
    resultados = ejecutar_analisis(df)
    for nombre, df_res in resultados.items():
        print(f"\n=== {nombre} ===")
        print(df_res)