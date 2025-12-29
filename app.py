import streamlit as st
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
from google import genai # <--- La librería oficial de tu ejemplo
from obtener_datos import descargar_datos_streamlit

# --- CONFIGURACIÓN ---
st.set_page_config(page_title="Bot Luz ⚡", page_icon="⚡", layout="centered")
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

# --- MOTOR DE IA (NATIVO DE GOOGLE) ---
class BotDirecto:
    def __init__(self, df, api_key):
        self.df = df
        self.client = genai.Client(api_key=api_key) # <--- Cliente oficial
        
    def preguntar(self, pregunta):
        # 1. Preparamos el contexto (los datos)
        # Convertimos las primeras filas y la estructura a texto para que Gemini entienda qué datos tiene
        info_datos = self.df.head(5).to_markdown(index=False)
        tipos = str(self.df.dtypes)
        
        # 2. El Prompt (Instrucciones)
        prompt = f"""
        Eres un experto programador en Python y analista de datos.
        
        TIENES ESTOS DATOS (variable 'df'):
        Tipos de columnas:
        {tipos}
        
        Muestra de datos:
        {info_datos}
        
        PREGUNTA DEL USUARIO: "{pregunta}"
        
        TU TAREA:
        1. Escribe código Python que use la variable 'df' para responder.
        2. Guarda el resultado final en la variable 'resultado'.
        3. Si piden un gráfico, usa matplotlib y guarda la figura en 'fig'.
        4. NO uses print().
        5. Devuelve SOLO el código Python, sin explicaciones ni markdown.
        """
        
        try:
            # 3. Llamada directa a Gemini (Modelo Flash)
            response = self.client.models.generate_content(
                model="gemini-1.5-flash",
                contents=prompt
            )
            
            # Limpiamos la respuesta
            codigo = response.text.replace("```python", "").replace("```", "").strip()
            
            # 4. Ejecución segura
            local_vars = {"df": self.df, "pd": pd, "plt": plt, "sns": sns, "resultado": None}
            exec(codigo, {}, local_vars)
            
            resultado = local_vars.get("resultado")
            fig = plt.gcf()
            
            # Comprobamos si hay gráfico
            if len(fig.axes) > 0:
                return "IMG", fig
            elif resultado is not None:
                return "TXT", str(resultado)
            else:
                return "ERR", "El código se ejecutó pero no devolvió ningún resultado."
                
        except Exception as e:
            return "ERR", f"Error: {e}"

# --- INTERFAZ ---
df = cargar_datos()

if df is None:
    st.warning("⚠️ No hay datos. Pulsa actualizar.")
else:
    st.success(f"✅ Datos listos: {len(df)} registros.")

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            if msg.get("type") == "image":
                st.pyplot(msg["content"])
            else:
                st.write(msg["content"])

    if prompt := st.chat_input("Pregunta algo sobre la luz..."):
        st.session_state.messages.append({"role": "user", "content": prompt, "type": "text"})
        with st.chat_message("user"):
            st.write(prompt)
            
        with st.chat_message("assistant"):
            with st.spinner("Pensando..."):
                api_key = st.secrets["GEMINI_API_KEY"]
                bot = BotDirecto(df, api_key)
                tipo, respuesta = bot.preguntar(prompt)
                
                if tipo == "IMG":
                    st.pyplot(respuesta)
                    st.session_state.messages.append({"role": "assistant", "content": respuesta, "type": "image"})
                elif tipo == "TXT":
                    st.write(respuesta)
                    st.session_state.messages.append({"role": "assistant", "content": respuesta, "type": "text"})
                else:
                    st.error(respuesta)
