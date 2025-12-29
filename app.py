import streamlit as st
import pandas as pd
import os
import datetime
from pandasai import SmartDataframe
from langchain_google_genai import ChatGoogleGenerativeAI
from obtener_datos import descargar_datos_streamlit

# Configuración de la página web
st.set_page_config(page_title="Bot Luz ⚡", page_icon="⚡")
st.title("⚡ Asistente del Mercado Eléctrico")

# --- BARRA LATERAL (Para descargar datos) ---
with st.sidebar:
    if st.button("🔄 Actualizar Datos ESIOS"):
        descargar_datos_streamlit()
        st.cache_data.clear()

# --- 1. CARGAR DATOS ---
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
    st.warning("⚠️ No encuentro 'datos_luz.csv'. Pulsa el botón de la izquierda para descargarlos.")
else:
    # --- 2. CONFIGURAR CEREBRO ---
    try:
        # Recuperamos la clave de los secretos de la nube
        api_key = st.secrets["GEMINI_API_KEY"]
        
        # CAMBIO NECESARIO: Usamos 'gemini-pro' porque '2.5' no existe y el '1.5' daba error en la nube
        llm = ChatGoogleGenerativeAI(
            model="gemini-pro", 
            google_api_key=api_key,
            temperature=0
        )
        
        # Fecha de hoy
        hoy = datetime.datetime.now().strftime("%Y-%m-%d")
        
        # --- 3. CONFIGURAR AGENTE (TUS PROMPTS EXACTOS) ---
        agent = SmartDataframe(
            df,
            config={
                "llm": llm,
                "verbose": False,
                "enable_cache": False,
                "custom_prompts": {
                    "system_prompt": (
                        f"Hoy es {hoy}. "
                        "Eres un experto analista en el mercado electrico. Responde en español. "
                        "Tienes disponible el dataframe en la variable 'df' y pandas como 'pd'. "
                        "\n\n🛑 REGLA DE SEGURIDAD CRÍTICA (IMPORTANTE): 🛑\n"
                        "1. NO escribas líneas que empiecen por 'import ...' o 'from ...'.\n"
                        "2. El sistema fallará si intentas importar librerías.\n"
                        "3. Usa 'pd.to_datetime()' para fechas en lugar de la librería datetime.\n"
                        "4. Calcula lo pedido y guarda el resultado en la variable 'result' (diccionario type/value)."
                    )
                }
            }
        )

        # --- 4. BUCLE DE CHAT (VERSIÓN WEB) ---
        # Inicializamos historial si no existe
        if "messages" not in st.session_state:
            st.session_state.messages = []

        # Mostramos mensajes anteriores
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        # CAJA DE TEXTO (Sustituye a input())
        if prompt := st.chat_input("👤 Tú: Escribe tu pregunta aquí..."):
            
            # Guardamos lo que escribiste
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)

            # Respuesta del Bot
            with st.chat_message("assistant"):
                with st.spinner("🤖 Pensando..."):
                    try:
                        # Le pasamos la pregunta a tu agente
                        res = agent.chat(prompt)
                        
                        # Mostramos el resultado (Sustituye a print())
                        st.write(res)
                        
                        # Guardamos en historial
                        st.session_state.messages.append({"role": "assistant", "content": str(res)})
                        
                    except Exception as e:
                        st.error("❌ Hubo un error calculando eso. Intenta simplificar la pregunta.")
                        # Si quieres ver el error real si falla:
                        # st.write(f"Error técnico: {e}")

    except Exception as e:
        st.error(f"❌ Error de conexión o configuración: {e}")
