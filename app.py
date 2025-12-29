import streamlit as st
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
from langchain_google_genai import ChatGoogleGenerativeAI
from obtener_datos import descargar_datos_streamlit

# --- CONFIGURACIÓN DE PÁGINA ---
st.set_page_config(page_title="Bot Luz ⚡", page_icon="⚡", layout="centered")
st.title("⚡ Asistente del Mercado Eléctrico")

# --- BARRA LATERAL ---
with st.sidebar:
    if st.button("🔄 Actualizar Datos ESIOS"):
        descargar_datos_streamlit()
        st.cache_data.clear()
    st.info("💡 Consejo: Pregunta por precios máximos, mínimos o medias.")

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

# --- NUESTRO PROPIO MOTOR DE IA ---
class AgenteLuz:
    def __init__(self, df, api_key):
        self.df = df
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-pro",
            google_api_key=api_key,
            temperature=0
        )

    def preguntar(self, pregunta):
        dtypes = str(self.df.dtypes)
        columns = str(list(self.df.columns))
        # Usamos try/except para evitar errores si tabulate no carga bien, usamos formato simple
        try:
            head = str(self.df.head(3).to_markdown())
        except:
            head = str(self.df.head(3))
        
        prompt = f"""
        Actúa como un analista de datos experto en Python y Pandas.
        Tienes un dataframe cargado en la variable 'df'.
        
        ESTRUCTURA:
        Columnas: {columns}
        Tipos: \n{dtypes}
        Ejemplo: \n{head}
        
        PREGUNTA: "{pregunta}"
        
        TU TAREA:
        1. Genera código Python para responder.
        2. Usa la variable 'df'.
        3. Guarda el resultado final (número o texto) en la variable 'resultado'.
        4. Si piden GRÁFICO: crea figura con matplotlib y asigna resultado = "GRÁFICO".
        5. NO uses print().
        6. Devuelve SOLO el código limpio.
        """
        
        # INICIALIZAMOS LA VARIABLE PARA QUE NO DE ERROR
        codigo_generado = "Error: No se generó código."
        
        try:
            # 1. Llamada a la IA
            respuesta = self.llm.invoke(prompt)
            codigo_generado = respuesta.content
            
            # Limpieza
            codigo_generado = codigo_generado.replace("```python", "").replace("```", "").strip()
            
            # 2. Entorno de ejecución
            local_vars = {
                "df": self.df, 
                "pd": pd, 
                "plt": plt, 
                "sns": sns,
                "resultado": None
            }
            
            # 3. Ejecutar
            exec(codigo_generado, {}, local_vars)
            
            # 4. Obtener resultado
            resultado = local_vars.get("resultado")
            
            # Detectar si se ha pintado algo en matplotlib (gráfico activo)
            fig = plt.gcf()
            hay_grafico = len(fig.axes) > 0
            
            if resultado == "GRÁFICO" or hay_grafico:
                return "IMG", fig
            elif resultado is not None:
                return "TXT", str(resultado)
            else:
                return "ERR", "El código se ejecutó pero no guardó nada en la variable 'resultado'."
                
        except Exception as e:
            return "ERR", f"Error: {str(e)}\n\nIntento de código:\n{codigo_generado}"

# --- INTERFAZ ---
df = cargar_datos()

if df is None:
    st.warning("⚠️ No hay datos. Pulsa 'Actualizar Datos'.")
else:
    st.success(f"✅ Datos cargados: {len(df)} registros.")

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            if message.get("type") == "image":
                st.pyplot(message["content"])
            else:
                st.markdown(message["content"])

    if prompt := st.chat_input("Ej: Precio medio hoy"):
        st.session_state.messages.append({"role": "user", "content": prompt, "type": "text"})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Calculando..."):
                try:
                    # Limpiamos gráficos anteriores para que no se mezclen
                    plt.clf()
                    
                    api_key = st.secrets["GEMINI_API_KEY"]
                    bot = AgenteLuz(df, api_key)
                    tipo, respuesta = bot.preguntar(prompt)
                    
                    if tipo == "IMG":
                        st.pyplot(respuesta)
                        st.session_state.messages.append({"role": "assistant", "content": respuesta, "type": "image"})
                    elif tipo == "TXT":
                        st.write(respuesta)
                        st.session_state.messages.append({"role": "assistant", "content": respuesta, "type": "text"})
                    else:
                        st.error(respuesta)
                except Exception as e:
                    st.error(f"Error crítico: {e}")
