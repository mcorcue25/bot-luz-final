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

# --- ADAPTADOR INTELIGENTE ---
class GeminiAdapter(LLM):
    def __init__(self, api_key):
        # Intentamos usar el modelo Flash (más rápido y listo)
        # Si falla, el requirements nuevo debería arreglarlo
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-1.5-flash",
            google_api_key=api_key,
            temperature=0
        )
    
    def generate_code(self, instruction, context):
        prompt = (
            f"INSTRUCCIÓN: {instruction}\n"
            f"CONTEXTO DE DATOS: {context}\n"
            "--- REGLAS DE ORO PARA PYTHON ---\n"
            "1. Genera SOLO código Python. Nada de texto.\n"
            "2. Usa el dataframe 'df'.\n"
            "3. IMPORTANTE: Para filtrar fechas, usa Strings. \n"
            "   - Ejemplo: df[df['fecha_hora'].dt.strftime('%Y-%m-%d') == '2024-05-20']\n"
            "4. Guarda el resultado final (frase explicativa) en la variable 'result'.\n"
            "5. NO uses print()."
        )
        
        try:
            response = self.llm.invoke(prompt).content
            # Limpieza quirúrgica del código
            code = response.replace("```python", "").replace("```", "").strip()
            return code
        except Exception as e:
            return f"result = 'Error técnico con Google: {str(e)}'"

    @property
    def type(self):
        return "google-gemini"

df = cargar_datos()

if df is None:
    st.warning("⚠️ No hay datos. Pulsa 'Actualizar Datos' en la barra lateral.")
else:
    # --- DIAGNÓSTICO EN SIDEBAR ---
    # Esto te ayudará a ver si realmente hay datos cargados
    with st.sidebar:
        st.write("---")
        st.write("📊 **Estado de Datos:**")
        min_date = df['fecha_hora'].min().strftime('%d/%m/%Y')
        max_date = df['fecha_hora'].max().strftime('%d/%m/%Y')
        st.info(f"Datos desde: {min_date}\nHasta: {max_date}")
        st.write(f"Total registros: {len(df)}")

    # --- CONFIGURAR AGENTE ---
    try:
        api_key = st.secrets["GEMINI_API_KEY"]
        llm_propio = GeminiAdapter(api_key)
        
        # OBTENER HORA REAL DE ESPAÑA
        zona_madrid = pytz.timezone('Europe/Madrid')
        hoy = datetime.datetime.now(zona_madrid).strftime("%Y-%m-%d")
        hora_actual = datetime.datetime.now(zona_madrid).strftime("%H:%M")
        
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
                with st.spinner("Consultando al experto..."):
                    try:
                        # Le damos la fecha masticada a la IA
                        q = (f"Hoy es {hoy} (hora {hora_actual}). "
                             f"Responde con una frase natural en español. "
                             f"Pregunta: {prompt}")
                        
                        response = agent.chat(q)
                        
                        if isinstance(response, str) and response.endswith(".png"):
                            st.image(response)
                            st.session_state.messages.append({"role": "assistant", "content": "📊 Gráfico generado."})
                        else:
                            st.write(response)
                            st.session_state.messages.append({"role": "assistant", "content": str(response)})
                            
                    except Exception as e:
                        st.error("❌ No encontré el dato.")
                        with st.expander("Ver error técnico"):
                            st.write(e)

    except Exception as e:
        st.error(f"❌ Error configuración: {e}")
