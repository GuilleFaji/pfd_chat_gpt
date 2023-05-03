# -*- coding: utf-8 -*-

# IMPORTS:
import utils

import numpy as np
import pandas as pd
import csv
import os
import re
import io
import json

import pypdf
import tabula

import openai
import pyllamacpp
import tiktoken

import langchain
from langchain.document_loaders.csv_loader import CSVLoader
from langchain.indexes import VectorstoreIndexCreator
from langchain.callbacks.base import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.retrievers import SVMRetriever

import panel as pn
import tempfile
'''
panel serve "D:\tests\pfd_chat_gpt\app.py" --address 0.0.0.0 --port 7680 --allow-websocket-origin 0.0.0.0:7680 --allow-websocket-origin localhost:7680
'''

#############################
# Buscamos variables globales
#############################
try :
    api_key = json.load(open('./data/creds/gpt_id.json'))['api_key']
    openai.api_key = api_key
    os.environ['OPENAI_API_KEY'] = api_key
    text_key = 'Clave OPENAI encontrada. Se usara por defecto.'
except:
    api_key = ''
    text_key = 'Introduzca su clave OPENAI.'

current_vector_store = None    

pn.extension('plotly',
             template='bootstrap',
             sizing_mode='stretch_width',
             theme='dark',)

pn.state.template.param.update(
    main_max_width="2000px",
    header_background="#160020",
    #theme= pn.template.theme.DarkTheme
)



#############################
# Botones y Widgets:
#############################

# PDF
file_input = pn.widgets.FileInput(accept='.pdf' ,width=300)

# Procesador de PDF
pdf_button = pn.widgets.Button(name="Procesar PDF")

# API Key:
openaikey = pn.widgets.PasswordInput(
    name = 'OpenAI API Key',
    value=api_key, placeholder=text_key
)

# Pregunta:
prompt = pn.widgets.TextEditor(
    value="", placeholder="Pregunta cualquier cosa", height=160, toolbar=False
)

# Boton de Pregunta:
run_button = pn.widgets.Button(name="Preguntar")

# Selector de nº de fragmentos:
select_k = pn.widgets.IntSlider(
    name="Fragmentos de texto relevantes", start=1, end=5, step=1, value=2
)

# Selector de tipo de cadena:
select_chain_type = pn.widgets.RadioButtonGroup(
    name='Tipo de Cadena', 
    options=['stuff', 'map_reduce', "refine", "map_rerank"]
)

# Selector de modelo:
select_model_type = pn.widgets.RadioButtonGroup(
    name='Modelo', 
    options=['OpenAI (online, pago)', 'GPT4ALL (local)', "GPT4ALL-J (local)"]
)

# Widget inferior:
widgets = pn.Row(
    pn.Column(prompt, run_button, margin=5),
    pn.Card(
        "**Modelo**:",
        pn.Column(select_model_type, select_k),
        title="Ajustes Especiales", margin=10
    )
)

#############################
# Funciones:
#############################
def contador_tokens(texto, tokenizador= tiktoken.get_encoding('cl100k_base')):
    return len(tokenizador.encode(texto, disallowed_special=()))

def pdf_to_vectorstore(_, svm_retriever=True):
    print(file_input.value)
    if file_input.value is None:
        return None
    
    print('Procesando PDF...')
    archivo = file_input
    out = io.BytesIO()
    archivo.save(out)
    archivo = out
    
    # Extractamos texto y tablas:
    texto = utils.extract_text_from_pdf(archivo)
    
    print(' Creando vector store...')
    # SPLITTER por Tokens - TikToken Tokenizer:
    tk_splitter = langchain.text_splitter.RecursiveCharacterTextSplitter(
        chunk_size=300,
        chunk_overlap=35,
        length_function = contador_tokens,
        separators = ['\n','\r','\t',' ','\n\n'])

    documentos = tk_splitter.create_documents(
        texto, 
        metadatas=[{'source': f'{archivo} pag.{i}'}
                for i in list(range(len(texto)))]
        )
    
    embedding = langchain.embeddings.openai.OpenAIEmbeddings()
    
    vector_store = SVMRetriever.from_texts([i.page_content for i in documentos], embedding)
    
    global current_vector_store 
    current_vector_store = vector_store
    return
     

def pregunta(vector_store, query, chain_type, k):
    
    # Recuperador:
    ''' [DEPRECATED]
    recuperador = current_vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": k})'''
    # Usamos SVM Retriever. Es mas caro computacionalmente pero mas preciso.
    # Karpathy: https://twitter.com/karpathy/status/1647025230546886658
    recuperador = vector_store
    
    openaillm = langchain.llms.openai.OpenAI(
    callback_manager=CallbackManager([StreamingStdOutCallbackHandler()]))
    
    # create a chain to answer questions 
    cadena_openai = langchain.chains.RetrievalQA.from_chain_type(
        chain_type='stuff',
        retriever=recuperador,
        llm=openaillm,
        return_source_documents=True)
    
    result = cadena_openai({"query": query})
    print(result['result'])
    return result

convos = [pn.Row(
            pn.panel("\U0001F916", width=10),
            pn.Column(f'**Aqui apareceran las preguntas, respuestas y sus contextos!**'))]  # store all panel objects in a list

def qa_result(_):
    os.environ["OPENAI_API_KEY"] = openaikey.value
    
    # save pdf file to a temp file 
    if file_input.value is not None:
    
        prompt_text = prompt.value
        if prompt_text:
            result = pregunta(vector_store=current_vector_store,
                              query=prompt_text,
                              chain_type=select_chain_type.value,
                              k=select_k.value)
            
            contexto = '; \n'.join(doc.page_content for doc in result["source_documents"])
            contexto = f'<font size="1">{contexto}</font>'
            convos.extend([
                pn.Row(
                    pn.panel("\U0001F60A", width=10),
                    f'<font size="3">**{prompt_text}**</font>',
                    width=600
                ),
                pn.Row(
                    pn.panel("\U0001F916", width=10),
                    pn.Column(
                        f'<font size="3">**{result["result"]}**</font>',
                        "**Contexto Importante:**",
                        contexto
                    )
                )
            ])
            #return convos
    return pn.Column(*convos,
                     margin=15,
                     height=300,
                     scroll=True,
                     background='black')



################
# Vinculaciones:
################

# Procesado de PDF:

def pdf_to_info(_):
    print(file_input.name)


pdf_button.on_click(pdf_to_vectorstore)

qa_interactive = pn.panel(
    pn.bind(qa_result, run_button),
    loading_indicator=True,
    background='black'
)



output = pn.WidgetBox('**Panel de la IA:**', qa_interactive,
                      scroll=True,
                      background='black')

# layout
pn.Row(
    pn.Column(
        pn.pane.Markdown("""
        ## **Chatbot PDF mediante OpenAI**
        
        **1) Sube o selecciona un archivo PDF y dale a '*Procesar*' (Puede tardar unos minutos).**
        
        **2) Pon tu API Key de OpenAI.**
        
        - Si guardas tu clave en el archivo `data/creds/gpt_id.json` no será necesario introducirla
        
        - Usar OpenAI tiene costes, vigílalos en [**OpenAI**](https://platform.openai.com/account).
        
        **3) Realiza preguntas en el apartado inferior.**
        
        """),
        file_input,
        pn.Row(pdf_button,''),
        openaikey,
        loading_indicator=True, width=300),
    pn.Column(output,
    widgets, sizing_mode='stretch_both', height_policy='max')

).servable()