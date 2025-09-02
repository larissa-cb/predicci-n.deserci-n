# app.py
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
import joblib

# Configuración para evitar problemas con matplotlib en entornos headless
import matplotlib
matplotlib.use('Agg')  # Usar backend no interactivo
plt.style.use('seaborn-v0_8')

# Configuración de la página
st.set_page_config(
    page_title="Sistema de Predicción de Deserción Universitaria",
    page_icon="🎓",
    layout="wide"
)

# Título principal
st.title("🎓 Sistema de Alerta Temprana para Deserción Estudiantil")
st.markdown("---")

# Sidebar para entrada de datos
st.sidebar.header("📋 Información del Estudiante")

# Simulamos un modelo
@st.cache_resource
def load_model():
    # Modelo simulado - en producción cargarías tu modelo real
    return RandomForestClassifier()

model = load_model()

# Formulario de entrada de datos
st.sidebar.subheader("Datos Demográficos")
age = st.sidebar.slider("Edad", 17, 50, 20)
gender = st.sidebar.selectbox("Género", ["Masculino", "Femenino"])
international = st.sidebar.selectbox("Estudiante Internacional", ["Sí", "No"])

st.sidebar.subheader("Datos Académicos")
previous_grade = st.sidebar.slider("Calificación Previa", 0, 200, 120)
scholarship = st.sidebar.selectbox("Beca", ["Sí", "No"])
attendance = st.sidebar.slider("Asistencia (%)", 0, 100, 85)

st.sidebar.subheader("Datos Socioeconómicos")
parent_education = st.sidebar.selectbox("Educación de los Padres", 
                                      ["Primaria", "Secundaria", "Universitaria"])
family_income = st.sidebar.selectbox("Ingreso Familiar", 
                                   ["Bajo", "Medio", "Alto"])

# Botón para predecir
if st.sidebar.button("🔍 Predecir Riesgo de Deserción"):
    # Preprocesar datos
    data = {
        'age': age,
        'previous_grade': previous_grade,
        'attendance': attendance,
        'scholarship': 1 if scholarship == "Sí" else 0,
        'international': 1 if international == "Sí" else 0
    }
    
    # Convertir a DataFrame
    input_df = pd.DataFrame([data])
    
    # Hacer predicción (simulada)
    risk_level = "Alto" if previous_grade < 100 or attendance < 70 else "Moderado" if previous_grade < 120 else "Bajo"
    confidence = 0.85 if risk_level == "Alto" else 0.72 if risk_level == "Moderado" else 0.65
    
    # Mostrar resultados
    st.subheader("📊 Resultados de la Predicción")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Nivel de Riesgo", risk_level)
    
    with col2:
        st.metric("Confianza", f"{confidence*100:.1f}%")
    
    with col3:
        probability = 0.75 if risk_level == "Alto" else 0.45 if risk_level == "Moderado" else 0.15
        st.metric("Probabilidad de Abandono", f"{probability*100:.1f}%")
    
    # Recomendaciones
    st.subheader("🎯 Recomendaciones de Intervención")
    
    if risk_level == "Alto":
        st.error("🚨 Intervención prioritaria requerida")
        st.write("""
        - **Tutoría intensiva** semanal
        - **Evaluación psicológica** recomendada
        - **Beca de apoyo** a considerar
        - **Contacto con familia** inmediato
        """)
    elif risk_level == "Moderado":
        st.warning("⚠️ Monitoreo cercano recomendado")
        st.write("""
        - **Seguimiento académico** quincenal
        - **Talleres de habilidades** de estudio
        - **Mentoría** con estudiante avanzado
        """)
    else:
        st.success("✅ Riesgo bajo - Continuar monitoreo regular")
        st.write("""
        - **Seguimiento** semestral estándar
        - **Participación** en actividades extracurriculares
        """)
    
    # Análisis de factores
    st.subheader("🔍 Factores de Riesgo Identificados")
    
    factors = []
    if previous_grade < 100:
        factors.append(f"Calificación previa baja ({previous_grade}/200)")
    if attendance < 70:
        factors.append(f"Asistencia preocupante ({attendance}%)")
    if scholarship == "No":
        factors.append("Falta de apoyo económico (sin beca)")
    
    if factors:
        for factor in factors:
            st.write(f"• {factor}")
    else:
        st.write("No se identificaron factores de riesgo significativos")
    
    # Gráfico de factores
    st.subheader("📈 Análisis de Impacto de Factores")
    
    fig, ax = plt.subplots(figsize=(10, 6))
    factors_data = {
        'Calificación Previa': max(0, (100 - previous_grade) / 100),
        'Asistencia': max(0, (70 - attendance) / 70),
        'Beca': 0.4 if scholarship == "No" else 0.1,
        'Edad': 0.2 if age > 25 else 0.05
    }
    
    colors = ['red' if x > 0.3 else 'orange' if x > 0.1 else 'green' for x in factors_data.values()]
    bars = ax.bar(factors_data.keys(), factors_data.values(), color=colors)
    ax.set_ylabel('Impacto en Riesgo')
    ax.set_title('Contribución de Factores al Riesgo de Deserción')
    plt.xticks(rotation=45)
    
    # Asegurar que el gráfico se muestre correctamente
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)  # Cerrar la figura para liberar memoria

else:
    st.info("👈 Complete la información del estudiante en la barra lateral y haga clic en 'Predecir Riesgo'")

# Información adicional
st.sidebar.markdown("---")
st.sidebar.info("""
**ℹ️ Acerca del Sistema:**
Este sistema predictivo utiliza machine learning para identificar estudiantes en riesgo de deserción universitaria, permitiendo intervenciones tempranas y personalizadas.
""")
