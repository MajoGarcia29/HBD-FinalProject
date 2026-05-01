import pandas as pd
import os

df = pd.read_csv("data/raw/homicidios.csv")

# NORMALIZAR TEXTO
def clean_text(col):
    return (
        col.str.strip()
        .str.lower()
        .str.replace(r"\s+", " ", regex=True)
    )

cols_str = df.select_dtypes(include="object").columns
for col in cols_str:
    df[col] = clean_text(df[col])


# UNIFICAR CATEGORÍAS

# Estado civil
df["Estado Civil"] = df["Estado Civil"].replace({
    "soltero (a)": "soltero(a)",
    "casado (a)": "casado(a)",
    "viudo (a)": "viudo(a)",
    "unión libre": "union libre",
    "unión libre ": "union libre"
})

# Zona
df["Zona del Hecho"] = df["Zona del Hecho"].replace({
    "centro poblado(corregimiento, inspección de policía y caserío)":
    "centro poblado",
    "centro poblado (corregimiento, inspección de policía y caserío)":
    "centro poblado"
})

# Escenario
df["Escenario del Hecho"] = df["Escenario del Hecho"].replace({
    "vía pública": "via publica"
})


# REEMPLAZAR SIN INFORMACIÓN
df = df.replace("sin información", pd.NA)


# ELIMINAR COLUMNAS INÚTILES
df = df.drop(columns=["Manera de muerte"])


# ELIMINAR COLUMNAS CON MUCHOS NULOS
cols_drop = [
    "Orientación Sexual",
    "Identidad de Género",
    "Transgénero"
]

df = df.drop(columns=cols_drop)


#CREAR FEATURES ÚTILES

# Año como int por seguridad
df["Año del hecho"] = df["Año del hecho"].astype(int)

# Crear columna de fecha 
df["Fecha"] = df["Año del hecho"].astype(str) + "-" + df["Mes del hecho"] + "-" + df["Dia del hecho"]

# 7. GUARDAR
os.makedirs("staging/processed", exist_ok=True)

df.to_csv("staging/processed/homicidios_clean.csv", index=False)

print("Dataset limpio guardado correctamente")