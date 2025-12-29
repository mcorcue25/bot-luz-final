import streamlit as st
import pandas as pd
import os
import datetime
from pandasai import SmartDataframe
from langchain_google_genai import ChatGoogleGenerativeAI
from obtener_datos import descargar_datos_streamlit
# Importamos la clase base obligatoria
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

# --- ADAPTADOR UNIVERSAL (MODO DEBUG) ---
class GeminiAdapter(LLM):
    def __init__(self, api_key):
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-pro",
            google_api_key=api_key,
            temperature=0
        )
    
    # Método para versiones modernas de PandasAI
    def generate_code(self, instruction, context):
        return self._logic(instruction, context)

    # Método para versiones antiguas (por si acaso)
    def call(self, instruction, context, suffix=""):
        return self._logic(instruction, context)

    def _logic(self, instruction, context):
        prompt = (
            f"INSTRUCCIÓN: {instruction}\n"
            f"CONTEXTO: {context}\n"
            "--- REGLAS OBLIGATORIAS ---\n"
            "1. Genera SOLO código Python. Sin markdown, sin explicaciones.\n"
            "2. Usa el dataframe 'df'.\n"
            "3. IMPORTANTE: Para fechas usa strings. Ej: df['fecha_hora'].dt.strftime('%Y-%m-%d') == '2024-05-20'\n"
            "4. Guarda la respuesta final (frase explicativa) en la variable 'result'.\n"
            "5. NO uses print()."
        )
        try:
            response = self.llm.invoke(prompt).content
            code = response.replace("```python", "").replace("```", "").strip()
            return code
        except Exception as e:
            return f'result = "Error conexión Google: {str(e)}"'

    @property
    def type(self):
        return "google-gemini"

df = cargar_datos()

if df is None:
    st.warning("⚠️ No hay datos. Pulsa 'Actualizar Datos'.")
else:
    # --- CONFIGURACIÓN ---
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
                "custom_prompts": {"system_prompt": f"Hoy es {hoy}."}
            }
        )

        # --- CHAT ---
        if "messages" not in st.session_state:
            st.session_state.messages = []

        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        if prompt := st.chat_input("Ej: Precio medio hoy"):
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)

            with st.chat_message("assistant"):
                with st.spinner("🤖 Analizando..."):
                    try:
                        # 1. Ejecutamos el chat
                        res = agent.chat(prompt)
                        
                        # 2. Mostramos respuesta
                        st.write(res)
                        st.session_state.messages.append({"role": "assistant", "content": str(res)})
                        
                    except Exception as e:
                        # --- AQUÍ ESTÁ EL CAMBIO IMPORTANTE ---
                        st.error("❌ Error Técnico Detectado:")
                        # Esto imprimirá todo el rastro del error en pantalla
                        st.exception(e)
                        
                        # Intento de mostrar qué código falló (si es posible)
                        try:
                            st.warning("Último código generado (intento):")
                            st.code(agent.last_code_generated)
                        except:
                            pass

    except Exception as e:
        st.error(f"❌ Error Configuración Global: {e}")
