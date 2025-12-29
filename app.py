import streamlit as st
import pandas as pd
import os
import datetime
from pandasai import SmartDataframe
from langchain_google_genai import ChatGoogleGenerativeAI
from obtener_datos import descargar_datos_streamlit
# IMPORTANTE: Traemos la etiqueta oficial para que PandasAI no se queje
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

# --- CLASE ADAPTADOR (LA SOLUCIÓN AL ERROR) ---
# Esta clase actúa como un "traductor" entre Google y PandasAI
class GeminiAdapter(LLM):
    def __init__(self, api_key):
        # Usamos gemini-pro que es el más fiable
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-pro",
            google_api_key=api_key,
            temperature=0
        )
    
    # Esta función toma el control para asegurar que el código sea válido
    def generate_code(self, instruction, context):
        prompt = (
            f"INSTRUCCIÓN: {instruction}\n"
            f"CONTEXTO: {context}\n"
            "--- REGLAS OBLIGATORIAS ---\n"
            "1. Genera SOLO código Python. Sin explicaciones.\n"
            "2. Usa el dataframe 'df'.\n"
            "3. IMPORTANTE: Para filtrar fechas usa strings. Ej: df['fecha_hora'].dt.strftime('%Y-%m-%d') == '2024-05-20'\n"
            "4. Guarda la respuesta final (frase explicativa) en la variable 'result'.\n"
            "5. NO uses print()."
        )
        
        try:
            response = self.llm.invoke(prompt).content
            # Limpiamos el código para quitar comillas de markdown
            code = response.replace("```python", "").replace("```", "").strip()
            return code
            
        except Exception as e:
            # En caso de error, devolvemos un mensaje seguro (con triple comilla para no romper nada)
            msg = str(e).replace('"', "'")
            return f'result = """Error técnico con Google: {msg}"""'

    @property
    def type(self):
        return "google-gemini"

df = cargar_datos()

if df is None:
    st.warning("⚠️ No hay datos. Pulsa 'Actualizar Datos' en la barra lateral.")
else:
    # --- CONFIGURAR CEREBRO ---
    try:
        api_key = st.secrets["GEMINI_API_KEY"]
        
        # AQUÍ ESTÁ EL CAMBIO: Usamos el adaptador en vez de llm directo
        llm_propio = GeminiAdapter(api_key)
        
        hoy = datetime.datetime.now().strftime("%Y-%m-%d")
        
        agent = SmartDataframe(
            df,
            config={
                "llm": llm_propio, # Pasamos nuestra "caja" compatible
                "verbose": False,
                "enable_cache": False,
                "custom_prompts": {
                    "system_prompt": f"Hoy es {hoy}. Responde en español."
                }
            }
        )

        # --- CHAT WEB ---
        if "messages" not in st.session_state:
            st.session_state.messages = []

        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        if prompt := st.chat_input("👤 Tú: Pregúntame sobre el precio de la luz..."):
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)

            with st.chat_message("assistant"):
                with st.spinner("🤖 Pensando..."):
                    try:
                        res = agent.chat(prompt)
                        st.write(res)
                        st.session_state.messages.append({"role": "assistant", "content": str(res)})
                    except Exception as e:
                        st.error("❌ Hubo un error. Intenta simplificar la pregunta.")

    except Exception as e:
        st.error(f"❌ Error de configuración: {e}")
