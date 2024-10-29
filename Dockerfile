FROM python:3.11-slim

WORKDIR usr/src/app
COPY . .
RUN pip install --no-cache-dir gradio
EXPOSE 7860
ENV GRADIO_SERVER_NAME="0.0.0.0"
CMD ["python", "/usr/src/app/interface/gradio.py"]