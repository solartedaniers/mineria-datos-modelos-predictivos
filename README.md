# 📊 Laboratorio de Minería de Datos: Modelos Predictivos (CRISP-DM)

<div align="center">

| **Estudiante** | Daniers Solarte |
| :--- | :--- |
| **Materia** | Inteligencia de Negocios y Minería de Datos |
| **Docente** | Cristian Camilo Ordoñez Quintero |
| **Metodología** | CRISP-DM |

</div>

---

## 🚀 Descripción General del Proyecto
Este proyecto implementa una solución completa de analítica predictiva bajo la metodología **CRISP-DM** (Cross-Industry Standard Process for Data Mining). Consta de tres modelos de **Regresión Lineal Múltiple** entrenados, evaluados e integrados en una aplicación web interactiva de alto rendimiento desarrollada en Streamlit.

---

## 🛠️ Tecnologías Aplicadas
* **Python**: Lenguaje principal de programación y lógica analítica.
* **Pandas & NumPy**: Manipulación, limpieza y estructuración de datasets.
* **Scikit-Learn**: Construcción, entrenamiento y evaluación de los modelos de Regresión Lineal Múltiple.
* **Joblib**: Serialización y persistencia de modelos entrenados (`.pkl`).
* **Streamlit**: Desarrollo de la interfaz gráfica web y despliegue interactivo.

---

## 🧹 Fase de Preparación y Limpieza de Datos (Auditoría IQR)
Siguiendo un rigor analítico profesional, los datasets originales fueron sometidos a una auditoría estricta de valores atípicos utilizando el **Rango Intercuartílico (IQR)**:
1. **Dólar:** Se evaluaron 500 registros iniciales. Tras aplicar los filtros de control de ruido, se excluyeron los outliers estadísticos, optimizando la estabilidad del modelo ($R^2 \approx 0.9959$).
2. **Glucosa:** Partiendo de 2,000 registros clínicos, se depuraron valores extremos en el Índice de Masa Corporal (IMC) y niveles de glucosa, obteniendo un dataset limpio de 1,981 registros ($R^2 \approx 0.6948$).
3. **Energía:** Sobre un total inicial de 10,000 registros de consumo eléctrico, se aplicó el filtrado IQR en variables térmicas y de consumo, consolidando 9,867 registros altamente confiables ($R^2 \approx 0.8954$).

---

## ⚙️ Guía de Ejecución Local (Paso a Paso)

Si deseas clonar y ejecutar este repositorio en tu equipo local, sigue las instrucciones:

1. **Clonar el repositorio:**
   ```bash
   git clone [https://github.com/solartedaniers/mineria-datos-modelos-predictivos.git](https://github.com/solartedaniers/mineria-datos-modelos-predictivos.git)
   cd mineria-datos-modelos-predictivos