import streamlit as st
import pandas as pd
import joblib

# Configuración inicial de la página
st.set_page_config(
    page_title="Analytics & ML Suite | Laboratorio Minería",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Estilos CSS profesionales (SaaS UI)
st.markdown("""
    <style>
    .main {
        background-color: #0e1117;
        color: #ffffff;
    }
    .stSidebar {
        background-color: #161b22;
    }
    .stSidebar h1, .stSidebar h2, .stSidebar h3, .stSidebar label, .stSidebar .stSelectbox div, .stSidebar span {
        color: #f3f4f6 !important;
    }
    .card {
        background: linear-gradient(135deg, #1f2937 0%, #111827 100%);
        padding: 25px;
        border-radius: 15px;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.37);
        border: 1px solid rgba(255, 255, 255, 0.18);
        margin-bottom: 20px;
    }
    .metric-title {
        font-size: 16px;
        color: #9ca3af;
        font-weight: 600;
    }
    .metric-value {
        font-size: 32px;
        color: #38bdf8;
        font-weight: 800;
    }
    </style>
""", unsafe_allow_html=True)

# Barra lateral de navegación
st.sidebar.markdown("### 🎛️ Panel de Control")
st.sidebar.markdown("---")
modelo_seleccionado = st.sidebar.selectbox(
    "Seleccione el Modelo Predictivo:",
    ["📊 Resumen General", "💵 Predicción de Dólar", "🩺 Predicción de Glucosa", "⚡ Consumo de Energía"]
)
st.sidebar.markdown("---")
st.sidebar.info("💡 **Metodología CRISP-DM:** Modelos entrenados con Regresión Lineal Múltiple sobre datasets depurados mediante IQR.")

# ==========================================
# VISTA 1: RESUMEN GENERAL / HOME
# ==========================================
if modelo_seleccionado == "📊 Resumen General":
    st.title("🚀 Suite de Modelos Predictivos en Minería de Datos")
    st.markdown("Plataforma interactiva de estimación analítica basada en Machine Learning y la metodología CRISP-DM.")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
            <div class="card">
                <h3>💵 Mercado Cambiario</h3>
                <p>Modelo enfocado en la estimación del precio diario del dólar a partir de variables macroeconómicas clave.</p>
            </div>
        """, unsafe_allow_html=True)
        
    with col2:
        st.markdown("""
            <div class="card">
                <h3>🩺 Salud Clínica</h3>
                <p>Predicción de niveles de glucosa en sangre basada en parámetros antropométricos y actividad física.</p>
            </div>
        """, unsafe_allow_html=True)
        
    with col3:
        st.markdown("""
            <div class="card">
                <h3>⚡ Sector Energético</h3>
                <p>Estimación del consumo eléctrico residencial e industrial condicionado por variables térmicas y temporales.</p>
            </div>
        """, unsafe_allow_html=True)
        
    st.markdown("---")
    st.markdown("👈 **Seleccione un modelo en el menú lateral** para ingresar parámetros e interactuar con el sistema de predicción en tiempo real.")

# ==========================================
# VISTA 2: MODELO DÓLAR
# ==========================================
elif modelo_seleccionado == "💵 Predicción de Dólar":
    st.title("💵 Modelo de Predicción: Precio del Dólar")
    st.markdown("Estime la tendencia futura del valor del dólar introduciendo las variables económicas correspondientes.")
    
    col_form, col_res = st.columns([1, 1], gap="large")
    
    with col_form:
        st.markdown("### 📥 Parámetros de Entrada")
        with st.container():
            dia = st.number_input("Número de Día", min_value=1, max_value=2000, value=501, step=1)
            inflacion = st.number_input("Tasa de Inflación Diaria (ej: 0.02 para 2%)", min_value=0.0, max_value=0.20, value=0.02, format="%.4f")
            tasa_interes = st.number_input("Tasa de Interés Diaria (%)", min_value=1.0, max_value=20.0, value=5.00, format="%.2f")
            
            calcular_dolar = st.button("Ejecutar Predicción del Dólar", type="primary", use_container_width=True)

    with col_res:
        st.markdown("### 📤 Resultado de la Estimación")
        if calcular_dolar:
            try:
                model = joblib.load('modelos_pkl/modelo_dolar.pkl')
                input_data = pd.DataFrame([[dia, inflacion, tasa_interes]], columns=['Dia', 'Inflacion', 'Tasa_interes'])
                pred = model.predict(input_data)[0]
                
                # Tarjeta limpia sin texto técnico de R^2
                st.markdown(f"""
                    <div class="card" style="text-align: center; border-color: #38bdf8; padding: 40px 20px;">
                        <p class="metric-title">PRECIO ESTIMADO DEL DÓLAR</p>
                        <p class="metric-value">${pred:,.2f}</p>
                    </div>
                """, unsafe_allow_html=True)
            except Exception as e:
                st.error(f"Error cargando el modelo: {e}")
        else:
            st.info("Ingrese los valores en el panel izquierdo y haga clic en el botón para calcular.")

# ==========================================
# VISTA 3: MODELO GLUCOSA
# ==========================================
elif modelo_seleccionado == "🩺 Predicción de Glucosa":
    st.title("🩺 Modelo de Predicción: Niveles de Glucosa")
    st.markdown("Evaluación metabólica y estimación del nivel de glucosa en sangre en función del perfil del paciente.")
    
    col_form, col_res = st.columns([1, 1], gap="large")
    
    with col_form:
        st.markdown("### 📥 Parámetros Clínicos")
        with st.container():
            edad = st.number_input("Edad del Paciente (años)", min_value=1, max_value=120, value=45, step=1)
            imc = st.number_input("Índice de Masa Corporal (IMC)", min_value=5.0, max_value=60.0, value=24.5, format="%.2f")
            actividad = st.number_input("Actividad Física (horas semanales)", min_value=0.0, max_value=30.0, value=4.0, step=0.5)
            
            calcular_glu = st.button("Ejecutar Predicción de Glucosa", type="primary", use_container_width=True)

    with col_res:
        st.markdown("### 📤 Resultado Clínico Estimado")
        if calcular_glu:
            try:
                model = joblib.load('modelos_pkl/modelo_glucosa.pkl')
                input_data = pd.DataFrame([[edad, imc, actividad]], columns=['Edad', 'IMC', 'Actividad_Fisica'])
                pred = model.predict(input_data)[0]
                
                # Tarjeta limpia sin texto técnico de R^2
                st.markdown(f"""
                    <div class="card" style="text-align: center; border-color: #a855f7; padding: 40px 20px;">
                        <p class="metric-title">NIVEL DE GLUCOSA ESTIMADO</p>
                        <p class="metric-value" style="color: #c084fc;">{pred:.2f} mg/dL</p>
                    </div>
                """, unsafe_allow_html=True)
            except Exception as e:
                st.error(f"Error cargando el modelo: {e}")
        else:
            st.info("Ingrese los parámetros clínicos en el panel izquierdo para generar el diagnóstico predictivo.")

# ==========================================
# VISTA 4: MODELO ENERGÍA
# ==========================================
elif modelo_seleccionado == "⚡ Consumo de Energía":
    st.title("⚡ Modelo de Predicción: Consumo Eléctrico")
    st.markdown("Pronóstico de la demanda de energía eléctrica con base en factores térmicos y temporales.")
    
    col_form, col_res = st.columns([1, 1], gap="large")
    
    with col_form:
        st.markdown("### 📥 Parámetros Ambientales y Temporales")
        with st.container():
            temp = st.number_input("Temperatura Ambiental (°C)", min_value=-20.0, max_value=50.0, value=25.0, format="%.2f")
            hora = st.slider("Hora del Día (1 a 24)", min_value=1, max_value=24, value=14)
            dia_sem = st.selectbox(
                "Día de la Semana",
                options=[1, 2, 3, 4, 5, 6, 7],
                format_func=lambda x: ["Lunes", "Martes", "Miércoles", "Jueves", "Viernes", "Sábado", "Domingo"][x-1]
            )
            
            calcular_ene = st.button("Ejecutar Predicción de Energía", type="primary", use_container_width=True)

    with col_res:
        st.markdown("### 📤 Consumo Estimado")
        if calcular_ene:
            try:
                model = joblib.load('modelos_pkl/modelo_energia.pkl')
                input_data = pd.DataFrame([[temp, hora, dia_sem]], columns=['Temperatura', 'Hora', 'Dia_Semana'])
                pred = model.predict(input_data)[0]
                
                # Tarjeta limpia sin texto técnico de R^2
                st.markdown(f"""
                    <div class="card" style="text-align: center; border-color: #eab308; padding: 40px 20px;">
                        <p class="metric-title">DEMANDA ELÉCTRICA ESTIMADA</p>
                        <p class="metric-value" style="color: #facc15;">{pred:.2f} kWh</p>
                    </div>
                """, unsafe_allow_html=True)
            except Exception as e:
                st.error(f"Error cargando el modelo: {e}")
        else:
            st.info("Configure las variables ambientales y temporales para estimar el consumo energético.")

# Pie de página institucional
st.markdown("---")
st.markdown(
    "<p style='text-align: center; color: #9ca3af; font-size: 14px; font-weight: 500;'>"
    "Plataforma Desarrollada para Laboratorio de Minería de Datos • CRISP-DM & Machine Learning<br>"
    "<b>Autor:</b> Daniers Solarte | <b>Docente:</b> Cristian Camilo Ordoñez Quintero"
    "</p>", 
    unsafe_allow_html=True
)