import streamlit as st
import pandas as pd
import numpy as np
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

    # --- Análisis predictivo automático con visualizaciones ---
    st.subheader("🤖 Análisis de predicción y visualización")

    from sklearn.model_selection import train_test_split
    from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
    from sklearn.metrics import accuracy_score, confusion_matrix, r2_score, mean_squared_error, roc_curve, auc

    target_col = st.selectbox("Selecciona la variable objetivo", df.columns, key="target_pred")

    if target_col:
        df_model = df.drop(columns=[target_col, 'rut', 'nombre'], errors='ignore')
        high_card_cols = [col for col in df_model.select_dtypes(include='object') if df_model[col].nunique() > 50]
        df_model = df_model.drop(columns=high_card_cols, errors='ignore')

        X = pd.get_dummies(df_model.select_dtypes(include=['number', 'object']), drop_first=True)
        y = df[target_col]

        if y.dtype == 'object' or y.nunique() < 15:
            y_encoded = pd.factorize(y)[0]
            modelo = RandomForestClassifier(n_estimators=100, random_state=42)
            tipo = "Clasificación"
        else:
            y_encoded = y
            modelo = RandomForestRegressor(n_estimators=100, random_state=42)
            tipo = "Regresión"

        try:
            X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.3, random_state=42)
            modelo.fit(X_train, y_train)
            y_pred = modelo.predict(X_test)

            st.markdown(f"**Tipo de modelo aplicado**: {tipo}")

            importancias = modelo.feature_importances_
            features = X.columns
            fig_imp = px.bar(x=features, y=importancias, title="Importancia de variables", labels={'x': 'Variable', 'y': 'Importancia'})
            fig_imp.update_layout(xaxis_tickangle=45)
            st.plotly_chart(fig_imp)

            if tipo == "Clasificación":
                acc = accuracy_score(y_test, y_pred)
                st.write(f"**Precisión:** {acc:.2f}")

                cm = confusion_matrix(y_test, y_pred)
                cm_df = pd.DataFrame(cm, index=np.unique(y_test), columns=np.unique(y_test))
                fig_cm = px.imshow(cm_df, text_auto=True, title="Matriz de Confusión")
                st.plotly_chart(fig_cm)

                if len(np.unique(y_encoded)) == 2:
                    y_proba = modelo.predict_proba(X_test)[:, 1]
                    fpr, tpr, _ = roc_curve(y_test, y_proba)
                    roc_auc = auc(fpr, tpr)
                    fig_roc = px.area(
                        x=fpr, y=tpr,
                        title=f"Curva ROC (AUC={roc_auc:.2f})",
                        labels=dict(x='Tasa de falsos positivos', y='Tasa de verdaderos positivos')
                    )
                    fig_roc.add_shape(type='line', line=dict(dash='dash'), x0=0, x1=1, y0=0, y1=1)
                    st.plotly_chart(fig_roc)
            else:
                r2 = r2_score(y_test, y_pred)
                rmse = mean_squared_error(y_test, y_pred, squared=False)
                st.write(f"**R² Score:** {r2:.2f}")
                st.write(f"**RMSE:** {rmse:.2f}")

                fig_pred = px.scatter(x=y_test, y=y_pred, labels={'x': 'Real', 'y': 'Predicho'}, title="Predicción de valores")
                st.plotly_chart(fig_pred)

        except Exception as e:
            st.error(f"Error durante la predicción: {e}")
