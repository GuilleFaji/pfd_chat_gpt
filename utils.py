# FUNCIONES PRINCIPALES
# ############################################################################

# 0 - IMPORTS&SETTINGS
# 1 - Text Handling
# 2 - OpenAI embeddings & queries
# 3 - Orquestador

##################
# Imports&Settings
##################
import numpy as np
import os
import re
import io
import pypdf
import langchain
import openai

# API KEY:
# ToDo: Cambiar a variable de entorno
api_key = json.load(open('./data/creds/gpt_id.json'))['api_key']
openai.api_key = api_key

###############
# Text Handling
###############
# ToDo: Hablar con Ana y Natalia sobre tratamiento de PDFs y OCR.
'''
Funciones para manejo de texto.
Principalmente extractar texto de pdfs y limpiarlo.
En un futuro se podrían añadir funciones para extraer texto de otros formatos
e incluso de imágenes utilizando las últimas habilidades de GPT (se podrían
convertir gráficos a su equivalente en tablas de datos, json, etc.).
'''
def limpieza_texto(texto: str) -> str:
    '''
    Función para limpiar texto de pdfs.
    Cambia saltos de línea, espacios en blanco y caracteres especiales.
    '''
    # Eliminamos espacios en blanco
    texto = re.sub(' +', ' ', texto)
    # Eliminamos caracteres especiales [REVISAR]
    texto = re.sub('[^A-Za-z0-9]+', ' ', texto)
    # Eliminamos saltos múltiples de línea
    texto = re.sub(r"\n\s*\n", "\n\n", texto)
    return texto

def extract_text_from_pdf(pdf_path: str) -> str:
    '''
    Función para extraer texto de un pdf y limpiarlo.
    Devuelve una lista de str, cada una es una página del pdf.
    '''
    # Abrimos el pdf
    with open(pdf_path, 'rb') as f:
        pdf = pypdf.PdfFileReader(f)
        # Obtenemos el número de páginas
        num_pags = pdf.getNumPages()
        count = 0
        text = []
        # Iteramos sobre las páginas
        while count < num_pags:
            pag = pdf.getPage(count)
            count +=1
            texto_pagina = pag.extractText()
            texto_pagina = limpieza_texto(texto_pagina)
            text.append(texto_pagina)
    return text

#############################
# OpenAI embeddings & queries
#############################
'''
Llamadas a la API de OpenAI para obtener embeddings y hacer queries.
Funciones simples que dado un texto devuelven embeddings y dado un input
con contexto te devuelve una respuesta del modelo GPT-3.5.
'''
# EMBEDDINGS
# ToDo:Revisar y optimizar. Contar con límites de la API.
def get_embeddings(text: str, model: str = "ada") -> np.array:
    '''
    Función para obtener los embeddings de un texto.
    '''
    # Uso de API de OpenAI para obtener embeddings
    response = openai.Embedding.create(
        model=model,
        query=text,
    )
    # Devolvemos los embeddings
    return np.array(response['embedding'])

# MENSAJE GPT-3.5:
def send_message(message_log,
                 max_tokens: int = 3800,
                 temp: float = 0.7,
                 stop=None,
                 model: str = "gpt-3.5-turbo",
                 full_output: bool = False):
    '''
    Función para enviar un mensaje a GPT junto con contexto y obtener la
    respuesta del chatbot.

    Parameters
    ----------
    message_log : LIST OF DICTIONARIES
        CONTEXTO: Contexto a proporcionar o historial de conversación,
        como una lista de diccionarios. Cada diccionario debe tener un rol
        y un contenido. El rol puede ser "usuario" o "sistema". El contenido
        es el texto que se envía al modelo.
    max_tokens : INT, optional
        Número tope de tokens. The default is 3800.
    temp : FLOAT, optional
        Temperatura, parámetro que delimita la creatividad del modelo.
        The default is 0.7.
    stop : TYPE, optional
        DESCRIPTION. The default is None.
    model : TYPE, optional
        DESCRIPTION. The default is "gpt-3.5-turbo".
    full_output : BOOL, optional
        Booleano que determina si se devuelve toda la respuesta del server.
        The default is False.

    Returns
    -------
    STRING or DICT
        Devuelve la respuesta del chatbot o el diccionario completo si
        full_output = True.

    '''
    # Uso de API de OpenAI para enviar mensaje y obtener respuesta
    response = openai.ChatCompletion.create(
        model=model,
        messages=message_log,
        max_tokens=max_tokens,
        stop=stop,
        temperature=temp,
    )

    if full_output: return response
    # Si queremos solo la respuesta, filtramos el diccionario
    for choice in response.choices:
        if "text" in choice:
            return choice.text

    # Si no hay respuesta, devolvemos el primer mensaje
    return response.choices[0].message.content

#############
# Orquestador
#############
'''
Organización de las funciones principales.
Principalmente se codifica en embeddings el texto de los pdfs, se hacen
búsquedas por Similitud Coseno y se devuelven N resultados mas cercanos.
Esos resultados (en texto) son los que se le dan como contexto al modelo
gpt-3.5 para que genere la extracción de datos precisos.
'''
