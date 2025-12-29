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

# --- ADAPTADOR CON "BYPASS" (Saltamos la validación estricta) ---
class GeminiAdapter(LLM):
    def __init__(self, api_key):
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-1.5-flash",
            google_api_key=api_key,
            temperature=0
        )
    
    # Esta función sustituye a la original de PandasAI que daba error.
    # Aquí tomamos el control total.
    def generate_code(self, instruction, context):
        prompt = (
            f"INSTRUCCIÓN: {instruction}\n"
            f"CONTEXTO: {context}\n"
            "--- REGLAS ABSOLUTAS ---\n"
            "1. Genera SOLO código Python. Sin explicaciones.\n"
            "2. Usa el dataframe 'df'.\n"
            "3. Guarda la respuesta final (texto o número) en la variable 'result'.\n"
            "4. NO uses print().\n"
            "5. Ejemplo de formato esperado:\n"
            "result = 'El precio medio es 50 euros'"
        )
        
        try:
            # 1. Llamamos a Gemini
            response = self.llm.invoke(prompt).content
            
            # 2. LIMPIEZA MANUAL (El Bypass)
            # Quitamos las comillas de markdown si existen para dejar solo el código puro
            code = response.replace("```python", "").replace("```", "").strip()
            
            # Devolvemos el código limpio directamente.
            # Al hacerlo aquí, evitamos que PandasAI lance el 'NoCodeFoundError'.
            return code
            
        except Exception as e:
            # En caso de error, devolvemos un código seguro que muestre el fallo
            return f"result = 'Error técnico conectando con la IA: {str(e)}'"

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
        # Usamos nuestro adaptador trucado
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
                        
                        # Ejecutamos el chat (ahora usará nuestro generate_code seguro)
                        response = agent.chat(q)
                        
                        if isinstance(response, str) and response.endswith(".png"):
                            st.image(response)
                            st.session_state.messages.append({"role": "assistant", "content": "📊 Gráfico generado."})
                        else:
                            st.write(response)
                            st.session_state.messages.append({"role": "assistant", "content": str(response)})
                            
                    except Exception as e:
                        st.error("❌ No pude obtener el dato.")
                        with st.expander("Ver detalle"):
                            st.write(e)

    except Exception as e:
        st.error(f"❌ Error configuración: {e}")
