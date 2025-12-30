import streamlit as st
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
import datetime
import pytz
from groq import Groq
from obtener_datos import descargar_datos_streamlit

# --- CONFIGURACIÓN ---
st.set_page_config(page_title="Bot Llama 3 🦙", page_icon="⚡", layout="centered")
st.title("⚡ Asistente Eléctrico (Motor Groq)")
st.caption("Potenciado por Llama 3-70b a través de Groq Cloud")

# --- BARRA LATERAL ---
with st.sidebar:
    if st.button("🔄 Actualizar Datos ESIOS"):
        descargar_datos_streamlit()
        st.cache_data.clear()

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

# --- CEREBRO GROQ ---
class CerebroGroq:
    def __init__(self, df, api_key):
        self.df = df
        # Inicializamos el cliente con tu clave
        self.client = Groq(api_key=api_key)
        
    def pensar_y_programar(self, pregunta):
        # 1. Definir contexto temporal (ESPAÑA)
        zona_es = pytz.timezone('Europe/Madrid')
        ahora = datetime.datetime.now(zona_es)
        hoy_str = ahora.strftime("%Y-%m-%d")
        
        # 2. Resumen de datos
        info_datos = self.df.head(3).to_markdown(index=False)
        dtypes = str(self.df.dtypes)
        
        # 3. Prompt Optimizado para Llama 3
        prompt_sistema = f"""
        Eres un experto programador en Python y analista de datos energéticos.
        Hoy es: {hoy_str}.
        
        DATOS DISPONIBLES (variable 'df'):
        {dtypes}
        
        MUESTRA:
        {info_datos}
        
        INSTRUCCIONES CRÍTICAS:
        1. Genera código Python ejecutable para responder a la pregunta.
        2. IMPORTANTE: El código DEBE guardar la respuesta final explicada en una variable de texto llamada 'resultado'.
           Ejemplo: resultado = "El precio medio de hoy es 50 euros."
        3. SI PIDEN GRÁFICO: Usa matplotlib, crea la figura y NO definas la variable 'resultado'. El sistema detectará la figura automáticamente.
        4. No uses print().
        5. IMPORTANTE: Devuelve ÚNICAMENTE el bloque de código, sin explicaciones antes ni después.
        """
        
        try:
            # Llamada a la API de Groq
            chat_completion = self.client.chat.completions.create(
                messages=[
                    {
                        "role": "system",
                        "content": prompt_sistema
                    },
                    {
                        "role": "user",
                        "content": pregunta
                    }
                ],
                # Usamos el modelo grande (70b) porque es el más listo para programar
                model="llama-3.3-70b-versatile",
                temperature=0,
                stop=None,
            )
            
            # Limpieza de respuesta (Llama a veces pone texto extra)
            codigo = chat_completion.choices[0].message.content
            codigo = codigo.replace("```python", "").replace("```", "").strip()
            
            # --- EJECUCIÓN DEL CÓDIGO ---
            local_vars = {"df": self.df, "pd": pd, "plt": plt, "sns": sns, "resultado": None}
            exec(codigo, {}, local_vars)
            
            resultado = local_vars.get("resultado")
            fig = plt.gcf()
            
            # Lógica de respuesta
            if len(fig.axes) > 0: 
                return "IMG", fig
            elif resultado:
                return "TXT", str(resultado)
            else:
                return "ERR", "La IA calculó algo pero olvidó guardarlo en la variable 'resultado'."
                
        except Exception as e:
            return "ERR", f"Error Groq: {e}"

# --- INTERFAZ DE CHAT ---
df = cargar_datos()

if df is None:
    st.warning("⚠️ No hay datos. Pulsa 'Actualizar Datos' en la izquierda.")
else:
    # Historial
    if "messages" not in st.session_state:
        st.session_state.messages = []

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            if msg.get("type") == "image":
                st.pyplot(msg["content"])
            else:
                st.markdown(msg["content"])

    # Input Usuario
    if prompt := st.chat_input("Ej: Compárame el precio de hoy con el del año pasado"):
        st.session_state.messages.append({"role": "user", "content": prompt, "type": "text"})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Llama 3 pensando..."):
                try:
                    # Intentamos coger la clave de los secretos
                    if "GROQ_API_KEY" in st.secrets:
                        api_key = st.secrets["GROQ_API_KEY"]
                        bot = CerebroGroq(df, api_key)
                        tipo, respuesta = bot.pensar_y_programar(prompt)
                        
                        if tipo == "IMG":
                            st.pyplot(respuesta)
                            st.session_state.messages.append({"role": "assistant", "content": respuesta, "type": "image"})
                            plt.clf()
                        elif tipo == "TXT":
                            st.write(respuesta)
                            st.session_state.messages.append({"role": "assistant", "content": respuesta, "type": "text"})
                        else:
                            st.error(f"❌ {respuesta}")
                    else:
                        st.error("❌ Falta la GROQ_API_KEY en los Secrets de Streamlit.")
                        
                except Exception as e:
                    st.error(f"Error crítico: {e}")

