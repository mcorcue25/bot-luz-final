import streamlit as st
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
from langchain_google_genai import ChatGoogleGenerativeAI
from obtener_datos import descargar_datos_streamlit

# --- CONFIGURACIÓN DE PÁGINA ---
st.set_page_config(page_title="Bot Luz ⚡", page_icon="⚡", layout="centered")
st.title("⚡ Asistente del Mercado Eléctrico")

# --- BARRA LATERAL ---
with st.sidebar:
    if st.button("🔄 Actualizar Datos ESIOS"):
        descargar_datos_streamlit()
        st.cache_data.clear()
    st.info("💡 Consejo: Pregunta por precios máximos, mínimos o medias.")

# --- CARGAR DATOS ---
@st.cache_data
def cargar_datos():
    archivo = "datos_luz.csv"
    if not os.path.exists(archivo):
        return None
    try:
        df = pd.read_csv(archivo)
        df['fecha_hora'] = pd.to_datetime(df['fecha_hora'])
        return df
    except Exception:
        return None

# --- NUESTRO PROPIO MOTOR DE IA (El "Mini-PandasAI") ---
class AgenteLuz:
    def __init__(self, df, api_key):
        self.df = df
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-pro", # Usamos el modelo estable
            google_api_key=api_key,
            temperature=0
        )

    def preguntar(self, pregunta):
        # 1. Preparamos la información para Gemini
        dtypes = str(self.df.dtypes)
        columns = str(list(self.df.columns))
        head = str(self.df.head(3).to_markdown())
        
        # 2. El Prompt Maestro (Instrucciones precisas)
        prompt = f"""
        Actúa como un analista de datos experto en Python y Pandas.
        Tienes un dataframe cargado en la variable 'df'.
        
        ESTRUCTURA DEL DATAFRAME:
        Columnas: {columns}
        Tipos: \n{dtypes}
        Ejemplo de datos: \n{head}
        
        PREGUNTA DEL USUARIO: "{pregunta}"
        
        TU TAREA:
        1. Genera código Python ejecutable para responder a la pregunta.
        2. Usa la variable 'df' directamente.
        3. Si la respuesta es un dato (número, texto), guárdalo en la variable 'resultado'.
        4. Si el usuario pide un GRÁFICO:
           - Crea el gráfico con matplotlib/seaborn.
           - Guárdalo en un objeto 'fig' (ej: fig = plt.gcf()).
           - Asigna resultado = "GRÁFICO_GENERADO"
        5. IMPORTANTE: NO uses print().
        6. IMPORTANTE: Devuelve SOLO el código, sin comillas de markdown (```python).
        """
        
        # 3. Llamamos a Gemini
        try:
            codigo_generado = self.llm.invoke(prompt).content
            
            # Limpieza básica por si Gemini pone comillas
            codigo_generado = codigo_generado.replace("```python", "").replace("```", "").strip()
            
            # 4. EJECUCIÓN DEL CÓDIGO (La Magia)
            # Creamos un entorno seguro con las librerías necesarias
            local_vars = {
                "df": self.df, 
                "pd": pd, 
                "plt": plt, 
                "sns": sns,
                "resultado": None,
                "fig": None
            }
            
            # Ejecutamos el código generado por la IA
            exec(codigo_generado, {}, local_vars)
            
            # 5. Recuperamos lo que la IA calculó
            resultado = local_vars.get("resultado")
            figura = local_vars.get("fig")
            
            if resultado == "GRÁFICO_GENERADO" and figura:
                return "IMG", figura
            elif resultado is not None:
                return "TXT", str(resultado)
            else:
                return "ERR", "La IA ejecutó el código pero no guardó nada en la variable 'resultado'."
                
        except Exception as e:
            return "ERR", f"Error de ejecución: {str(e)}\n\nCódigo que falló:\n{codigo_generado}"

# --- INTERFAZ PRINCIPAL ---
df = cargar_datos()

if df is None:
    st.warning("⚠️ No hay datos. Pulsa 'Actualizar Datos' en la izquierda.")
else:
    # Mostramos resumen rápido
    st.success(f"✅ Datos cargados: {len(df)} registros disponibles.")

    # Inicializamos chat
    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            if message.get("type") == "image":
                st.pyplot(message["content"])
            else:
                st.markdown(message["content"])

    # Input del usuario
    if prompt := st.chat_input("Ej: ¿Cuál es el precio medio de hoy?"):
        # Guardar mensaje usuario
        st.session_state.messages.append({"role": "user", "content": prompt, "type": "text"})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Respuesta del Asistente
        with st.chat_message("assistant"):
            with st.spinner("Analizando datos..."):
                try:
                    # Instanciamos nuestro Agente Casero
                    api_key = st.secrets["GEMINI_API_KEY"]
                    bot = AgenteLuz(df, api_key)
                    
                    # Preguntamos
                    tipo, respuesta = bot.preguntar(prompt)
                    
                    if tipo == "IMG":
                        st.pyplot(respuesta)
                        st.session_state.messages.append({"role": "assistant", "content": respuesta, "type": "image"})
                    elif tipo == "TXT":
                        st.write(respuesta)
                        st.session_state.messages.append({"role": "assistant", "content": respuesta, "type": "text"})
                    else: # Error
                        st.error(respuesta)
                        
                except Exception as e:
                    st.error(f"❌ Error general: {e}")
