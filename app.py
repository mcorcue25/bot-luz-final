import streamlit as st
import pandas as pd
import os
import datetime
from pandasai import SmartDataframe
from langchain_google_genai import ChatGoogleGenerativeAI
# IMPORTANTE: Importamos el "traductor" (Wrapper)
from pandasai.llm import LangChainLLM 
from obtener_datos import descargar_datos_streamlit

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

df = cargar_datos()

if df is None:
    st.warning("⚠️ No hay datos. Pulsa 'Actualizar Datos' en la barra lateral.")
else:
    # --- CONFIGURAR GEMINI (SOLUCIÓN FINAL) ---
    try:
        api_key = st.secrets["GEMINI_API_KEY"]
        
        # 1. Creamos la conexión con LangChain (que sabemos que conecta bien)
        llm_basico = ChatGoogleGenerativeAI(
            model="gemini-1.5-flash",
            google_api_key=api_key,
            temperature=0
        )
        
        # 2. Usamos el "Traductor" para convertirlo a formato PandasAI
        # Esto arregla el error "Input should be an instance of LLM"
        llm_pandas = LangChainLLM(llm_basico)
        
        hoy = datetime.datetime.now().strftime("%Y-%m-%d")
        
        agent = SmartDataframe(
            df,
            config={
                "llm": llm_pandas, # Pasamos el objeto traducido
                "verbose": False,
                "enable_cache": False,
                "custom_prompts": {
                    "system_prompt": (
                        f"Hoy es {hoy}. Eres experto en mercado eléctrico español. "
                        "Responde en español. Si piden gráficos, hazlos."
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
                        # Gestión de gráficos
                        if isinstance(response, str) and response.endswith(".png"):
                            st.image(response)
                            st.session_state.messages.append({"role": "assistant", "content": "📊 [Gráfico]"})
                        else:
                            st.write(response)
                            st.session_state.messages.append({"role": "assistant", "content": str(response)})
                    except Exception as e:
                        st.error("❌ Ocurrió un error. Intenta simplificar la pregunta.")
                        # st.write(e) # Descomentar solo si necesitas ver el error técnico

    except Exception as e:
        st.error(f"❌ Error de configuración: {e}")
