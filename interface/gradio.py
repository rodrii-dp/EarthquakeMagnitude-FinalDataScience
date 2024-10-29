import gradio as gr
import pandas as pd
from pycaret.regression import load_model, predict_model

# Cargar el modelo entrenado
modelo_terremotos = load_model('best_earthquake_model')

# Definir la función de predicción
def predecir_magnitud(longitud, latitud, profundidad, year, month, day, hour, minute, second):
    # Crear un DataFrame con una sola fila con los valores de entrada
    entrada = pd.DataFrame({
        'longitude': [longitud],
        'latitude': [latitud],
        'depth': [profundidad],
        'year': [year],
        'month': [month],
        'day': [day],
        'hour': [hour],
        'minute': [minute],
        'second': [second]
    })

    # Hacer la predicción
    prediccion = predict_model(modelo_terremotos, data=entrada)
    magnitud_predicha = prediccion['Label'][0]
    return f"La magnitud predicha del terremoto es: {magnitud_predicha:.2f}"

# Crear la interfaz de Gradio
interfaz = gr.Interface(
    fn=predecir_magnitud,
    inputs=[
        gr.inputs.Number(label="Longitud"),
        gr.inputs.Number(label="Latitud"),
        gr.inputs.Number(label="Profundidad (km)"),
        gr.inputs.Number(label="Año"),
        gr.inputs.Number(label="Mes"),
        gr.inputs.Number(label="Día"),
        gr.inputs.Number(label="Hora"),
        gr.inputs.Number(label="Minuto"),
        gr.inputs.Number(label="Segundo"),
    ],
    outputs="text",
    title="Predicción de Magnitud de Terremoto",
    description="Introduce los datos geográficos y temporales para predecir la magnitud de un posible terremoto."
)

# Ejecutar la interfaz
interfaz.launch()
#%%
