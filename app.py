import streamlit as st
import pandas as pd
import os
import re
import datetime
from pandasai import SmartDataframe
from langchain_google_genai import ChatGoogleGenerativeAI
from obtener_datos import descargar_datos_streamlit
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

# --- CLASE ADAPTADOR CON LIMPIEZA AUTOMÁTICA ---
class GeminiAdapter(LLM):
    def __init__(self, api_key):
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-1.5-flash",
            google_api_key=api_key,
            temperature=0
        )
    
    def call(self, instruction, value, suffix=""):
        # 1. Instrucción reforzada
        prompt = (
            f"{instruction}\n"
            f"DATOS: {value}\n"
            f"{suffix}\n"
            "REGLAS CRÍTICAS:\n"
            "1. NO expliques nada.\n"
            "2. Genera código Python ejecutable.\n"
            "3. El código DEBE estar dentro de bloques ```python ... ```\n"
            "4. Usa la variable 'df'.\n"
            "5. Guarda el resultado final en la variable 'result'."
        )
        
        try:
            # 2. Obtenemos respuesta bruta
            response_text = self.llm.invoke(prompt).content
            
            # 3. "Cirugía": Buscamos si hay código dentro
            match = re.search(r"```python\n(.*?)\n```", response_text, re.DOTALL)
            
            if match:
                # Si encontramos código limpio, devolvemos solo eso
                return match.group(0)
            elif "import" in response_text or "df" in response_text:
                 # Si parece código pero le faltan las comillas, se las ponemos nosotros
                return f"```python\n{response_text}\n```"
            else:
                # Si no parece código, devolvemos tal cual (posiblemente fallará, pero damos info)
                return response_text
                
        except Exception as e:
            return f"Error interno: {e}"

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
                        f"Hoy es {hoy}. Responde siempre escribiendo código Python que analice el dataframe 'df'."
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
                        
                        if isinstance(response, str) and response.endswith(".png"):
                            st.image(response)
                            st.session_state.messages.append({"role": "assistant", "content": "📊 [Gráfico]"})
                        else:
                            st.write(response)
                            st.session_state.messages.append({"role": "assistant", "content": str(response)})
                            
                    except Exception as e:
                        st.error(f"❌ Error al ejecutar el código generado por la IA.\nIntenta reformular la pregunta.")
                        # Debugging: descomentar para ver qué devolvió realmente Gemini si falla
                        # st.warning(f"Detalle técnico: {e}")

    except Exception as e:
        st.error(f"❌ Error de configuración: {e}")
