# Evaluación Integradora – Ciencia de Datos

## Estructura

- `app.py`: app principal
- `etl.py`: flujo ETL con conexión a MongoDB
- `utils.py`: funciones de carga y limpieza
- `pages/`: vistas Streamlit (EDA, comparativo)
- `static/`: diagrama de arquitectura
- `data/`: archivo de entrada y salida limpio

## Cómo ejecutar

```bash
python -m venv venv
venv\Scripts\activate       # o source venv/bin/activate
pip install -r requirements.txt
python etl.py -f data/ejemplo.xlsx
streamlit run app.py
