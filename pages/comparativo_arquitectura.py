import streamlit as st

st.set_page_config(page_title="Comparativo de Arquitecturas", layout="wide")
st.title("Comparativo de Arquitecturas y Justificación Técnica")

st.markdown("""
| Opción | Ventajas | Desventajas | Adecuación al proyecto |
|--------|----------|-------------|------------------------|
| **Local + MongoDB** | Simplicidad, sin costos cloud | Escalabilidad limitada | ✅ Adecuado para demo académica |
| **AWS (S3 + RDS)** | Alta disponibilidad, backups | Requiere cuenta y billing | ⚠️ Excesivo para un POC |
| **Serverless (Glue + Athena)** | Sin gestión de servidores, Cobra por uso | Curva de aprendizaje | ❌ Sobredimensionado |
""")

st.info("""
**Elección**: _Local + MongoDB_ cumple los criterios de reproducibilidad, bajo costo y
facilidad de evaluación por los docentes.
""")

try:
    st.image("static/arquitectura.png", caption="Diagrama de arquitectura técnica", use_column_width=True)
except:
    st.warning("No se encontró el archivo arquitectura.png en la carpeta static/")
