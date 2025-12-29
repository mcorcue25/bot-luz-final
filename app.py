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

# --- ADAPTADOR CONVERSACIONAL ---
class GeminiAdapter(LLM):
    def __init__(self, api_key):
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-1.5-flash",
            google_api_key=api_key,
            temperature=0
        )
    
    def call(self, instruction, value, suffix=""):
        # Instrucción Maestra: Le obligamos a responder con TEXTO dentro del código
        prompt = (
            f"{instruction}\n"
            f"INFORMACIÓN DEL DATAFRAME: {value}\n"
            f"{suffix}\n"
            "--- INSTRUCCIONES OBLIGATORIAS ---\n"
            "1. Eres un chatbot amable. NO devuelvas solo un número.\n"
            "2. Genera código Python que calcule la respuesta.\n"
            "3. IMPORTANTE: La última línea del código debe guardar una frase explicativa en español en la variable 'result'.\n"
            "   Ejemplo incorrecto: result = 15.4\n"
            "   Ejemplo CORRECTO: result = 'El precio medio de hoy es de 15.4 euros.'\n"
            "4. Usa siempre el dataframe 'df'.\n"
            "5. Columnas disponibles: 'fecha_hora' (datetime) y 'precio_eur_mwh' (float).\n"
            "6. Envuelve TODO el código entre ```python y ```"
        )
        
        try:
            response_text = self.llm.invoke(prompt).content
            
            # Limpieza: Extraemos el código Python
            match = re.search(r"```python\n(.*?)\n```", response_text, re.DOTALL)
            if match:
                return match.group(0)
            elif "import" in response_text or "df" in response_text:
                return f"```python\n{response_text}\n```"
            else:
                return response_text
        except Exception as e:
            return f"Error: {e}"

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
        llm_propio = GeminiAdapter(api_key)
        hoy = datetime.datetime.now().strftime("%Y-%m-%d")
        
        agent = SmartDataframe(
            df,
            config={
                "llm": llm_propio,
                "verbose": False,
                "enable_cache": False,
                "open_charts": False, 
                # AQUÍ ESTÁ LA CLAVE: Le explicamos sus datos para que no falle
                "field_descriptions": {
                    "fecha_hora": "La fecha y hora del precio. Formato datetime.",
                    "precio_eur_mwh": "El precio de la electricidad en Euros por MWh."
                },
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
                with st.spinner("Analizando mercado..."):
                    try:
                        # Añadimos contexto extra a la pregunta del usuario
                        pregunta_mejorada = f"Hoy es {hoy}. Responde a esto construyendo una frase completa: {prompt}"
                        
                        response = agent.chat(pregunta_mejorada)
                        
                        # Si es gráfico (imagen)
                        if isinstance(response, str) and response.endswith(".png"):
                            st.image(response)
                            st.session_state.messages.append({"role": "assistant", "content": "📊 Aquí tienes el gráfico solicitado."})
                        
                        # Si es texto (respuesta del chatbot)
                        else:
                            st.write(response)
                            st.session_state.messages.append({"role": "assistant", "content": str(response)})
                            
                    except Exception as e:
                        # Si falla, mostramos el error técnico real para poder arreglarlo
                        st.error("❌ No pude calcularlo.")
                        with st.expander("Ver detalle del error (para técnicos)"):
                            st.write(e)

    except Exception as e:
        st.error(f"❌ Error de configuración: {e}")
