import streamlit as st
import pandas as pd
import os
import re
import datetime
from pandasai import SmartDataframe
from langchain_google_genai import ChatGoogleGenerativeAI
from obtener_datos import descargar_datos_streamlit
# Importamos la base necesaria
from pandasai.llm import LLM

st.set_page_config(page_title="Bot Luz ⚡", page_icon="⚡")
st.title("⚡ Asistente del Mercado Eléctrico")

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

# --- ADAPTADOR BLINDADO (SOLUCIÓN DEFINITIVA) ---
class GeminiAdapter(LLM):
    def __init__(self, api_key):
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-1.5-flash",
            google_api_key=api_key,
            temperature=0
        )
    
    def call(self, instruction, value, suffix=""):
        # Prompt diseñado para responder COMO UN CHATBOT
        prompt = (
            f"PREGUNTA DEL USUARIO: {instruction}\n"
            f"DATOS (df): {value}\n"
            "--- INSTRUCCIONES TÉCNICAS ---\n"
            "1. Genera código Python usando pandas.\n"
            "2. NO imprimas (print) el resultado.\n"
            "3. IMPORTANTE: Guarda la respuesta final en una variable llamada 'result'.\n"
            "4. La variable 'result' DEBE SER UN TEXTO (String) explicando el dato amablemente en español.\n"
            "   - Mal: result = 15.4\n"
            "   - Bien: result = 'El precio más bajo fue de 15.4 euros a las 14:00.'\n"
            "5. NO expliques el código. Solo dame el bloque de código.\n"
        )
        
        try:
            # Obtenemos la respuesta cruda de Gemini
            response_text = self.llm.invoke(prompt).content
            
            # --- LIMPIEZA FORZADA (El truco) ---
            # 1. Si ya tiene las etiquetas correctas, lo devolvemos tal cual
            if "```python" in response_text:
                return response_text
            
            # 2. Si tiene etiquetas genéricas (```), las arreglamos
            elif "```" in response_text:
                return response_text.replace("```", "```python", 1)
            
            # 3. Si NO tiene etiquetas (solo código suelto), SE LAS PONEMOS NOSOTROS
            else:
                return f"```python\n{response_text}\n```"
                
        except Exception as e:
            # Si falla la conexión, generamos un código que devuelva el error como texto
            return f"```python\nresult = 'Error de conexión con Google: {str(e)}'\n```"

    @property
    def type(self):
        return "google-gemini"

df = cargar_datos()

if df is None:
    st.warning("⚠️ No hay datos. Pulsa 'Actualizar Datos' en la barra lateral.")
else:
    # --- CONFIGURAR AGENTE ---
    try:
        api_key = st.secrets["GEMINI_API_KEY"]
        llm_propio = GeminiAdapter(api_key)
        hoy = datetime.datetime.now().strftime("%Y-%m-%d")
        
        agent = SmartDataframe(
            df,
            config={
                "llm": llm_propio,
                "verbose": False,
                "enable_cache": False,
                # Descripción de datos para ayudar a la IA
                "field_descriptions": {
                    "fecha_hora": "Fecha y hora del precio. Usar dt.hour o dt.date para filtrar.",
                    "precio_eur_mwh": "Precio de la luz en €/MWh."
                },
            }
        )

        # --- CHAT INTERFACE ---
        if "messages" not in st.session_state:
            st.session_state.messages = []

        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        if prompt := st.chat_input("Ej: ¿Cuál es el precio medio de hoy?"):
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)

            with st.chat_message("assistant"):
                with st.spinner("Analizando precios..."):
                    try:
                        # Contextualizamos la fecha en la pregunta
                        q = f"Hoy es {hoy}. {prompt}"
                        
                        response = agent.chat(q)
                        
                        # Manejo de respuesta (Gráfico o Texto)
                        if isinstance(response, str) and os.path.exists(response) and response.endswith(".png"):
                            st.image(response)
                            st.session_state.messages.append({"role": "assistant", "content": "📊 Gráfico generado."})
                        else:
                            st.write(response)
                            st.session_state.messages.append({"role": "assistant", "content": str(response)})
                            
                    except Exception as e:
                        st.error("❌ Ocurrió un error. Intenta ser más específico.")
                        with st.expander("Ver detalle técnico"):
                            st.write(e)

    except Exception as e:
        st.error(f"❌ Error de configuración: {e}")
