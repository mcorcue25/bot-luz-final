import requests
import pandas as pd
import time
import streamlit as st
import os

# ID 805 = Precio Mercado Spot
INDICATOR_ID = "805"

def descargar_datos_streamlit():
    # Intentamos leer el token de los secretos de la nube
    try:
        token = st.secrets["ESIOS_TOKEN"]
    except Exception:
        st.error("❌ Error: No he encontrado 'ESIOS_TOKEN' en los Secrets.")
        return False

    # Años a descargar (puedes añadir 2023 si quieres más histórico)
    years = [2024, 2025] 
    dfs = []
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for i, year in enumerate(years):
        status_text.text(f"⏳ Descargando datos del año {year}...")
        
        url = f"https://api.esios.ree.es/indicators/{INDICATOR_ID}"
        headers = {
            "x-api-key": token,
            "Content-Type": "application/json"
        }
        params = {
            "start_date": f"{year}-01-01T00:00:00",
            "end_date": f"{year}-12-31T23:59:59",
            "time_trunc": "hour"
        }
        
        try:
            r = requests.get(url, headers=headers, params=params)
            r.raise_for_status()
            data = r.json()
            vals = data['indicator']['values']
            
            if vals:
                df = pd.DataFrame(vals)
                if 'geo_id' in df.columns:
                    df = df[df['geo_id'] == 8741] # Península
                
                df = df.rename(columns={'value': 'precio_eur_mwh', 'datetime': 'fecha_hora'})
                # Limpieza de zona horaria
                df['fecha_hora'] = pd.to_datetime(df['fecha_hora'], utc=True).dt.tz_convert('Europe/Madrid').dt.tz_localize(None)
                
                dfs.append(df[['fecha_hora', 'precio_eur_mwh']])
        except Exception as e:
            st.warning(f"⚠️ Error en {year}: {e}")
        
        progress_bar.progress((i + 1) / len(years))
        time.sleep(0.5)

    status_text.empty()
    progress_bar.empty()

    if dfs:
        full_df = pd.concat(dfs)
        full_df = full_df.sort_values('fecha_hora').reset_index(drop=True)
        full_df.to_csv("datos_luz.csv", index=False)
        st.success(f"✅ ¡Datos actualizados! {len(full_df)} registros.")
        return True
    else:
        st.error("❌ No se pudieron descargar datos.")
        return False
