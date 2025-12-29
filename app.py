import streamlit as st
import pandas as pd
import os
import re
import datetime
from pandasai import SmartDataframe
from langchain_google_genai import ChatGoogleGenerativeAI
from obtener_datos import descargar_datos_streamlit
# Importamos la clase base
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

# --- ADAPTADOR CON "BYPASS" Y MODELO ESTÁNDAR ---
class GeminiAdapter(LLM):
    def __init__(self, api_key):
        # CAMBIO IMPORTANTE: Usamos 'gemini-pro' que es más compatible
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-pro",
            google_api_key=api_key,
            temperature=0
        )
    
    def generate_code(self, instruction, context):
        prompt = (
            f"INSTRUCCIÓN: {instruction}\n"
            f"CONTEXTO: {context}\n"
            "--- REGLAS ABSOLUTAS ---\n"
            "1. Genera SOLO código Python. Sin explicaciones.\n"
            "2. Usa el dataframe 'df'.\n"
            "3. Guarda la respuesta final (texto o número) en la variable 'result'.\n"
            "4. NO uses print().\n"
            "5. Tu código debe asignar un String a 'result' explicando la respuesta.\n"
            "6. Ejemplo: result = 'El precio medio es 50 euros'"
        )
        
        try:
            # 1. Llamamos a Gemini
            response = self.llm.invoke(prompt).content
            
            # 2. LIMPIEZA: Quitamos comillas de markdown
            code = response.replace("```python", "").replace("```", "").strip()
            return code
            
        except Exception as e:
            # En caso de error, devolvemos un mensaje seguro sin comillas conflictivas
            # Usamos comillas dobles fuera y simples dentro para evitar SyntaxError
            return "result = 'Ocurrió un error de conexión con Google Gemini. Inténtalo de nuevo.'"

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
                "field_descriptions": {
                    "fecha_hora": "Fecha y hora. Formato datetime.",
                    "precio_eur_mwh": "Precio electricidad €/MWh."
                },
            }
        )

        # --- CHAT ---
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
                with st.spinner("Consultando precios..."):
                    try:
                        q = f"Hoy es {hoy}. Responde con una frase completa en español. {prompt}"
                        
                        response = agent.chat(q)
                        
                        if isinstance(response, str) and response.endswith(".png"):
                            st.image(response)
                            st.session_state.messages.append({"role": "assistant", "content": "📊 Gráfico generado."})
                        else:
                            st.write(response)
                            st.session_state.messages.append({"role": "assistant", "content": str(response)})
                            
                    except Exception as e:
                        st.error("❌ No pude obtener el dato.")
                        # st.write(e) # Descomentar solo para técnicos

    except Exception as e:
        st.error(f"❌ Error configuración: {e}")
