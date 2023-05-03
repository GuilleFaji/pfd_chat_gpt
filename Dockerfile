FROM python:3.10.4

WORKDIR /code

COPY requirements.txt .
COPY app.py .
COPY utils.py .

RUN pip install -r requirements.txt

EXPOSE 8080

CMD ["panel", "serve", "app.py", "--port", "8080", "--address", "0.0.0.0", "--allow-websocket-origin", "*"]