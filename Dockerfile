FROM python:3.10.4

WORKDIR /code

COPY requirements.txt .
COPY app.py .
COPY utils.py .

RUN pip install -r requirements.txt

# EXPOSE 8080
EXPOSE 7860

# Hacemos que genere la carpeta "Data"
RUN mkdir data
RUN mkdir data/creds/

# Ejecutamos el panel
CMD ["panel", "serve", "app.py", "--port", "7860", "--address", "0.0.0.0", "--allow-websocket-origin", "*", "--allow-websocket-origin", "localhost:7860"]