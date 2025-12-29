import streamlit as st
import pandas as pd
import os
import datetime
from pandasai import SmartDataframe
from langchain_google_genai import ChatGoogleGenerativeAI
from obtener_datos import descargar_datos_streamlit
# Traemos la clase padre oficial
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

# --- CLASE ADAPTADOR (Heredando de LLM) ---
class GeminiAdapter(LLM):
    def __init__(self, api_key):
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-1.5-flash",
            google_api_key=api_key,
            temperature=0
        )
    
    def call(self, instruction, value, suffix=""):
        # Convertimos todo a texto para evitar problemas de formato
        prompt = str(instruction) + "\nCONTEXTO DE DATOS:\n" + str(value) + "\n" + suffix
        try:
            response = self.llm.invoke(prompt)
            return response.content
        except Exception as e:
            return f"Error conectando con Google: {e}"

    @property
    def type(self):
        return "google-gemini"

df = cargar_datos()

if df is None:
    st.warning("⚠️ No hay datos. Pulsa 'Actualizar Datos' en la barra lateral.")
else:
    # --- CONFIGURAR GEMINI ---
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
                "custom_prompts": {
                    "system_prompt": (
                        f"Hoy es {hoy}. Eres experto en mercado eléctrico. "
                        "Responde en español."
                    )
                }
            }
        )

        # --- CHAT ---
        if "messages" not in st.session_state:
            st.session_state.messages = []

        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        if prompt := st.chat_input("Ej: ¿A qué hora es más barata la luz hoy?"):
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)

            with st.chat_message("assistant"):
                with st.spinner("Pensando..."):
                    try:
                        response = agent.chat(prompt)
                        
                        # Manejo de respuesta
                        if isinstance(response, str) and response.endswith(".png"):
                            st.image(response)
                            st.session_state.messages.append({"role": "assistant", "content": "📊 [Gráfico]"})
                        else:
                            st.write(response) # Muestra respuesta
                            st.session_state.messages.append({"role": "assistant", "content": str(response)})
                            
                    except Exception as e:
                        # AQUÍ ESTÁ EL CAMBIO: Mostramos el error real
                        st.error(f"❌ Error Técnico Detallado:\n{e}")
                        
    except Exception as e:
        st.error(f"❌ Error de configuración: {e}")
