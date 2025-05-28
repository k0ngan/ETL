# utils.py actualizado
import pandas as pd
import numpy as np


def cargar_datos(archivo):
    """Carga datos desde un archivo Excel o CSV y normaliza nombres de columnas."""
    if archivo.name.endswith('.xlsx'):
        df = pd.read_excel(archivo)
    elif archivo.name.endswith('.csv'):
        df = pd.read_csv(archivo)
    else:
        raise ValueError("Formato de archivo no soportado")

    df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_").str.replace(r'[^\w_]', '', regex=True)
    return df


def limpiar_datos(df):
    """Elimina duplicados, transforma fechas, y rellena valores nulos."""
    df = df.drop_duplicates()

    # Conversión de fechas si aplica
    for col in df.select_dtypes(include='object'):
        if df[col].str.match(r'\d{4}[-/]\d{2}[-/]\d{2}').any():
            df[col] = pd.to_datetime(df[col], errors='coerce')

    # Imputación básica de nulos
    num_cols = df.select_dtypes(include='number').columns
    cat_cols = df.select_dtypes(include='object').columns

    df[num_cols] = df[num_cols].fillna(df[num_cols].median())
    df[cat_cols] = df[cat_cols].fillna(df[cat_cols].mode().iloc[0])

    return df