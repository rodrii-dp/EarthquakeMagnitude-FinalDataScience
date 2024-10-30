# Proyecto de Machine Learning: Predicción de la Magnitud de Terremotos
Este proyecto utiliza Machine Learning para predecir la magnitud de un terremoto a partir de sus características geográficas, como la latitud, longitud y profundidad. Utilizamos un modelo de Random Forest para realizar las predicciones y la aplicación está implementada en Gradio para proporcionar una interfaz interactiva.

El proyecto está configurado para ejecutarse en Docker y puede ser desplegado fácilmente en cualquier entorno compatible con Docker.

## Descripción del Proyecto
La predicción de terremotos es un problema importante para reducir el impacto de estos eventos naturales. En este proyecto, utilizamos un enfoque basado en Random Forest, un algoritmo de aprendizaje supervisado, para predecir la magnitud de un terremoto. El modelo se entrena con datos de latitud, longitud y profundidad de eventos sísmicos, buscando patrones que permitan estimar su magnitud.

### Objetivos
Predecir la magnitud de un terremoto utilizando un modelo de Random Forest.
Proporcionar una interfaz interactiva mediante Gradio para facilitar el uso de la predicción.
Implementar el proyecto en Docker para asegurar su portabilidad y reproducibilidad.
### Requisitos
- Docker: Asegúrate de tener Docker instalado y ejecutándose en tu sistema.
- DataSpell (opcional): El proyecto está optimizado para desarrollarse y ejecutarse en DataSpell, un IDE orientado a la ciencia de datos.

### Estructura del Proyecto
- app/: Contiene el código principal y el Dockerfile.
- data/: Incluye los conjuntos de datos utilizados en el entrenamiento y validación del modelo.
- notebooks/: Contiene notebooks de Jupyter con análisis de datos y el archivo magnitude.py, que ejecuta el modelo.
- results/: Carpeta donde se guardan los resultados y salidas generadas por el modelo.
- Dockerfile
- El proyecto incluye un archivo Dockerfile que facilita la instalación de dependencias y la ejecución del proyecto en un contenedor Docker. A continuación se explica el contenido del Dockerfile y cómo ejecutarlo.

### Dockerfile
Copiar código
FROM python:3.11-slim
WORKDIR usr/src/app
COPY .. .
RUN mkdir -p results
RUN pip install --no-cache-dir gradio pycaret seaborn matplotlib numpy pandas sweetviz scikit-learn openpyxl requests
EXPOSE 7860
ENV GRADIO_SERVER_NAME="0.0.0.0"
CMD ["python", "/usr/src/app/notebooks/magnitude.py"]
### Explicación del Dockerfile
- FROM python:3.11-slim: Utiliza una imagen base ligera de Python 3.11.
- WORKDIR usr/src/app: Establece el directorio de trabajo en /usr/src/app.
- COPY .. .: Copia todos los archivos del proyecto en el directorio de trabajo del contenedor.
- RUN mkdir -p results: Crea la carpeta results para almacenar las salidas generadas por el modelo.
- RUN pip install: Instala las dependencias necesarias para el proyecto, como Gradio, PyCaret, Seaborn, y Scikit-learn, entre otras.
- EXPOSE 7860: Expone el puerto 7860, que se usa para la interfaz de Gradio.
- ENV GRADIO_SERVER_NAME="0.0.0.0": Configura Gradio para que esté accesible desde cualquier dirección IP.
- CMD: Define el comando para ejecutar el archivo magnitude.py en el contenedor.
### Instrucciones de Uso
- Construcción de la Imagen de Docker
Para construir la imagen de Docker, navega al directorio raíz del proyecto (EarthquakeMagnitude-FinalDataScience2) y ejecuta el siguiente comando en la terminal:
bash
Copiar código
docker build -t earthquake-magnitude-app -f app/Dockerfile .
Este comando construirá una imagen Docker llamada earthquake-magnitude-app utilizando el Dockerfile ubicado en app/.

Ejecución del Contenedor
Después de construir la imagen, ejecuta el contenedor con el siguiente comando:

bash
Copiar código
docker run -p 7860:7860 earthquake-magnitude-app
Este comando mapeará el puerto 7860 de tu máquina local al puerto 7860 del contenedor, permitiéndote acceder a la interfaz de Gradio en http://localhost:7860.

Uso de la Aplicación
Accede a la interfaz de Gradio en tu navegador en http://localhost:7860.
Ingresa los valores de latitud, longitud y profundidad para un evento sísmico.
La aplicación ejecutará el modelo de Random Forest y devolverá la predicción de la magnitud del terremoto.