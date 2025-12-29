import streamlit as st
import pandas as pd
import os
import datetime
import pytz
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

# --- ADAPTADOR BLINDADO CONTRA ERRORES ---
class GeminiAdapter(LLM):
    def __init__(self, api_key):
        # CAMBIO 1: Usamos 'gemini-pro' que es el modelo más estable en servidores
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-pro",
            google_api_key=api_key,
            temperature=0
        )
    
    def generate_code(self, instruction, context):
        prompt = (
            f"INSTRUCCIÓN: {instruction}\n"
            f"CONTEXTO: {context}\n"
            "--- REGLAS DE ORO ---\n"
            "1. Genera SOLO código Python.\n"
            "2. Usa el dataframe 'df'.\n"
            "3. IMPORTANTE: Para fechas usa Strings. Ej: df['fecha_hora'].dt.strftime('%Y-%m-%d') == '2024-05-20'\n"
            "4. Guarda la respuesta final (frase explicativa) en la variable 'result'.\n"
            "5. NO uses print()."
        )
        
        try:
            response = self.llm.invoke(prompt).content
            # Limpieza del código
            code = response.replace("```python", "").replace("```", "").strip()
            return code
            
        except Exception as e:
            # CAMBIO 2: Usamos TRIPLE COMILLA para que el error no rompa el código si tiene comillas dentro
            mensaje_error = str(e).replace('"', "'") # Limpiamos comillas dobles por si acaso
            return f'result = """Error técnico con Google: {mensaje_error}"""'

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
        
        # Hora de España
        zona_madrid = pytz.timezone('Europe/Madrid')
        hoy = datetime.datetime.now(zona_madrid).strftime("%Y-%m-%d")
        
        agent = SmartDataframe(
            df,
            config={
                "llm": llm_propio,
                "verbose": False,
                "enable_cache": False,
                "field_descriptions": {
                    "fecha_hora": "Fecha y hora completa.",
                    "precio_eur_mwh": "Precio luz."
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
                with st.spinner("Consultando..."):
                    try:
                        q = f"Hoy es {hoy}. Responde en español con una frase completa. {prompt}"
                        response = agent.chat(q)
                        
                        if isinstance(response, str) and response.endswith(".png"):
                            st.image(response)
                            st.session_state.messages.append({"role": "assistant", "content": "📊 Gráfico generado."})
                        else:
                            st.write(response)
                            st.session_state.messages.append({"role": "assistant", "content": str(response)})
                            
                    except Exception as e:
                        st.error("❌ No encontré el dato.")
                        # st.write(e) # Descomentar si falla de nuevo

    except Exception as e:
        st.error(f"❌ Error configuración: {e}")
