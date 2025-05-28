import streamlit as st
import pandas as pd
import plotly.express as px
from utils import cargar_datos

st.set_page_config(page_title="EDA", layout="wide")
st.title("📊 Análisis Exploratorio")

archivo = st.file_uploader("Sube tu archivo CSV o Excel", type=["csv", "xlsx"])

if archivo:
    df = cargar_datos(archivo)
    
    st.subheader("Vista previa")
    st.dataframe(df.head())

    st.subheader("Estadísticas básicas")
    st.write(df.describe())

    st.subheader("Promedio por columna numérica")
    st.write(df.select_dtypes("number").mean())

    numeric_cols = df.select_dtypes("number").columns
    cat_cols = df.select_dtypes(include=["object", "category"]).columns

    # Histograma
    st.subheader("Histograma")
    col_hist = st.selectbox("Selecciona columna numérica", numeric_cols)
    fig_hist = px.histogram(df, x=col_hist, nbins=30, title=f"Histograma de {col_hist}")
    st.plotly_chart(fig_hist)
    
    # Dispersión
    if len(numeric_cols) >= 2:
        st.subheader("Dispersión")
        x_col = st.selectbox("Eje X", numeric_cols, key="x")
        y_col = st.selectbox("Eje Y", [col for col in numeric_cols if col != x_col], key="y")
        fig_scatter = px.scatter(df, x=x_col, y=y_col, title=f"{y_col} vs {x_col}")
        st.plotly_chart(fig_scatter)