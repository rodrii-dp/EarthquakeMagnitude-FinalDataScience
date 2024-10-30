FROM python:3.11-slim

WORKDIR usr/src/app
COPY .. .
RUN mkdir -p results
RUN pip install --no-cache-dir gradio pycaret seaborn matplotlib numpy pandas sweetviz scikit-learn openpyxl requests
EXPOSE 7860
ENV GRADIO_SERVER_NAME="0.0.0.0"
CMD ["python", "/usr/src/app/notebooks/magnitude.py"]