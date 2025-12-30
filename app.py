import streamlit as st
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
import datetime
import pytz
from groq import Groq
from obtener_datos import descargar_datos_streamlit

# --- CONFIGURACIÓN ---
st.set_page_config(page_title="Bot Llama 3.3 🦙", page_icon="⚡", layout="centered")
st.title("⚡ Asistente Eléctrico (Precisión)")
st.caption("Motor: Llama 3.3-70b | Unidades: €/MWh")

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

# --- CEREBRO GROQ MEJORADO ---
class CerebroGroq:
    def __init__(self, df, api_key):
        self.df = df
        self.client = Groq(api_key=api_key)
        
    def pensar_y_programar(self, pregunta):
        # 1. Definir contexto temporal
        zona_es = pytz.timezone('Europe/Madrid')
        ahora = datetime.datetime.now(zona_es)
        hoy_str = ahora.strftime("%Y-%m-%d")
        
        # 2. Resumen de datos
        info_datos = self.df.head(3).to_markdown(index=False)
        dtypes = str(self.df.dtypes)
        
        # 3. Prompt BLINDADO (Unidades y Fechas)
        prompt_sistema = f"""
        Eres un experto analista de datos energéticos en Python.
        
        --- CONTEXTO ---
        HOY ES: {hoy_str}
        DATOS (variable 'df'):
        {dtypes}
        
        --- REGLAS DE ORO (CÚMPLELAS SIEMPRE) ---
        1. UNIDADES: El precio SIEMPRE es "€/MWh". NUNCA digas "euros" a secas.
        2. FECHAS: La columna 'fecha_hora' tiene horas. Para filtrar un día completo, USA SIEMPRE ESTE FORMATO:
           df_filtrado = df[df['fecha_hora'].dt.strftime('%Y-%m-%d') == 'AAAA-MM-DD']
        3. PRECISIÓN: Calcula la media exacta sobre los datos filtrados.
        4. VARIABLE FINAL: Guarda la explicación en la variable texto 'resultado'.
        5. GRÁFICOS: Si piden gráfico, usa matplotlib y no definas 'resultado'.
        6. NO uses print(). Devuelve SOLO el código Python.
        """
        
        try:
            chat_completion = self.client.chat.completions.create(
                messages=[
                    {"role": "system", "content": prompt_sistema},
                    {"role": "user", "content": pregunta}
                ],
                model="llama-3.3-70b-versatile",
                temperature=0,
                stop=None,
            )
            
            codigo = chat_completion.choices[0].message.content
            codigo = codigo.replace("```python", "").replace("```", "").strip()
            
            # Devolvemos el código para mostrarlo al usuario (transparencia)
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
            elif resultado:
                return "TXT", str(resultado)
            else:
                return "ERR", "El código se ejecutó pero no guardó nada en la variable 'resultado'."
        except Exception as e:
            return "ERR", f"Error de ejecución: {e}"

# --- INTERFAZ ---
df = cargar_datos()

if df is None:
    st.warning("⚠️ No hay datos. Pulsa 'Actualizar Datos'.")
else:
    if "messages" not in st.session_state:
        st.session_state.messages = []

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            if msg.get("type") == "image":
                st.pyplot(msg["content"])
            elif msg.get("type") == "code":
                # Ocultamos el código antiguo en un expander cerrado
                with st.expander("🛠️ Ver código técnico"):
                    st.code(msg["content"], language="python")
            else:
                st.markdown(msg["content"])

    if prompt := st.chat_input("Ej: Precio medio de hoy"):
        st.session_state.messages.append({"role": "user", "content": prompt, "type": "text"})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Calculando precios (Llama 3.3)..."):
                try:
                    if "GROQ_API_KEY" in st.secrets:
                        api_key = st.secrets["GROQ_API_KEY"]
                        bot = CerebroGroq(df, api_key)
                        
                        # 1. Generar Código
                        codigo = bot.pensar_y_programar(prompt)
                        
                        # Guardamos el código en el historial (pero oculto en expander)
                        with st.expander("🛠️ Ver código generado (Auditoría)"):
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
                                plt.clf()
                            elif tipo == "TXT":
                                st.write(respuesta)
                                st.session_state.messages.append({"role": "assistant", "content": respuesta, "type": "text"})
                            else:
                                st.error(f"❌ {respuesta}")
                    else:
                        st.error("❌ Falta GROQ_API_KEY en Secrets.")
                        
                except Exception as e:
                    st.error(f"Error crítico: {e}")


