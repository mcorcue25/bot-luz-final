import streamlit as st
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
import datetime
import pytz
from google import genai
from obtener_datos import descargar_datos_streamlit

# --- CONFIGURACIÓN ---
st.set_page_config(page_title="Bot Luz ⚡", page_icon="⚡", layout="wide") # Layout wide para ver mejor las tablas
st.title("⚡ Asistente del Mercado Eléctrico (Modo Diagnóstico)")

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

# --- MOTOR DE IA ---
class BotDirecto:
    def __init__(self, df, api_key):
        self.df = df
        self.client = genai.Client(api_key=api_key)
        
    def preguntar(self, pregunta):
        # 1. Contexto Temporal (CRÍTICO para que no alucine fechas)
        zona_es = pytz.timezone('Europe/Madrid')
        ahora = datetime.datetime.now(zona_es)
        hoy_str = ahora.strftime("%Y-%m-%d")
        hora_str = ahora.strftime("%H:%M")
        
        # 2. Muestra de datos para la IA
        info_datos = self.df.head(5).to_markdown(index=False)
        tipos = str(self.df.dtypes)
        
        # 3. Prompt Maestro
        prompt = f"""
        Eres un analista de datos experto en Python.
        
        CONTEXTO TEMPORAL REAL:
        - Fecha de HOY: {hoy_str}
        - Hora actual: {hora_str}
        
        TUS DATOS (variable 'df'):
        - Columna fecha: 'fecha_hora' (datetime64[ns])
        - Columna precio: 'precio_eur_mwh' (float)
        - Tipos: {tipos}
        - Ejemplo:
        {info_datos}
        
        PREGUNTA DEL USUARIO: "{pregunta}"
        
        TU TAREA:
        1. Escribe código Python para responder.
        2. IMPORTANTE: Si preguntan por "hoy", filtra el df usando la fecha '{hoy_str}'.
           Ejemplo: df_hoy = df[df['fecha_hora'].dt.date == pd.to_datetime('{hoy_str}').date()]
        3. Guarda el resultado final en la variable 'resultado'.
        4. Si piden gráfico, guarda la figura en 'fig'.
        5. Devuelve SOLO el código Python.
        """
        
        try:
            response = self.client.models.generate_content(
                model="gemini-2.0-flash",
                contents=prompt
            )
            
            codigo = response.text.replace("```python", "").replace("```", "").strip()
            
            # Devolvemos el código también para que el usuario lo audite
            return codigo
                
        except Exception as e:
            return f"# Error generando código: {e}"

    def ejecutar(self, codigo):
        try:
            local_vars = {"df": self.df, "pd": pd, "plt": plt, "sns": sns, "resultado": None}
            exec(codigo, {}, local_vars)
            
            resultado = local_vars.get("resultado")
            fig = plt.gcf()
            
            if len(fig.axes) > 0:
                return "IMG", fig
            else:
                return "TXT", str(resultado)
        except Exception as e:
            return "ERR", str(e)

# --- INTERFAZ ---
df = cargar_datos()

if df is None:
    st.warning("⚠️ No hay datos. Pulsa actualizar.")
else:
    # --- ZONA DE DIAGNÓSTICO DE DATOS (AQUÍ VERÁS LA VERDAD) ---
    with st.expander("🕵️ VER DATOS CRUDOS (¿Están bien los datos?)", expanded=False):
        st.write("Primeras 5 filas del archivo CSV:")
        st.dataframe(df.head())
        st.write("Últimas 5 filas (¿Llegan hasta hoy?):")
        st.dataframe(df.tail())
        
        # Chequeo rápido de fechas
        min_date = df['fecha_hora'].min()
        max_date = df['fecha_hora'].max()
        st.info(f"📅 Rango de datos cargados: Desde {min_date} hasta {max_date}")

    # --- CHAT ---
    if "messages" not in st.session_state:
        st.session_state.messages = []

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            if msg.get("type") == "image":
                st.pyplot(msg["content"])
            elif msg.get("type") == "code":
                with st.expander("🛠️ Ver código generado"):
                    st.code(msg["content"], language="python")
            else:
                st.write(msg["content"])

    if prompt := st.chat_input("Pregunta algo..."):
        st.session_state.messages.append({"role": "user", "content": prompt, "type": "text"})
        with st.chat_message("user"):
            st.write(prompt)
            
        with st.chat_message("assistant"):
            with st.spinner("Pensando y programando..."):
                api_key = st.secrets["GEMINI_API_KEY"]
                bot = BotDirecto(df, api_key)
                
                # 1. Generar Código
                codigo = bot.preguntar(prompt)
                
                # Mostramos el código "chivato"
                with st.expander("🛠️ Ver qué código ha pensado la IA"):
                    st.code(codigo, language="python")
                st.session_state.messages.append({"role": "assistant", "content": codigo, "type": "code"})
                
                # 2. Ejecutar Código
                if codigo.startswith("# Error"):
                    st.error(codigo)
                else:
                    tipo, respuesta = bot.ejecutar(codigo)
                    
                    if tipo == "IMG":
                        st.pyplot(respuesta)
                        st.session_state.messages.append({"role": "assistant", "content": respuesta, "type": "image"})
                    elif tipo == "TXT":
                        st.write(respuesta)
                        st.session_state.messages.append({"role": "assistant", "content": respuesta, "type": "text"})
                    else:
                        st.error(f"Error ejecutando código: {respuesta}")

