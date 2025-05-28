import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from utils import cargar_datos

st.set_page_config(page_title="Análisis Exploratorio", layout="wide")

st.title("Análisis Exploratorio de Datos")

archivo = st.file_uploader("Sube tu dataset (CSV o Excel)")

if archivo:
    df = cargar_datos(archivo)
    st.subheader("Vista preliminar")
    st.dataframe(df.head())

    st.subheader("Estadísticos básicos")
    st.write(df.describe(include="all").T)

    # Gráfica rápida
    with st.expander("Histograma interactivo"):
        col = st.selectbox("Selecciona variable numérica", df.select_dtypes("number").columns)
        fig, ax = plt.subplots()
        sns.histplot(df[col], kde=True, ax=ax)
        st.pyplot(fig)

st.title("Análisis Libre de Ausentismo")
st.write("Explora los datos de ausentismo de manera interactiva aquí.")
