import os

import pandas as pd

import numpy as np

import pickle

import joblib

import re
import base64
import boto3

import string

import streamlit as st

import random

import tempfile

import logging

import torch

from transformers import pipeline
import streamlit.components.v1 as components  # Import Streamlit
from PIL import Image 
import requests 
from transformers import AutoModelForCausalLM 
from transformers import AutoProcessor 
from streamlit_chat import message

from sentence_transformers import util

from datetime import datetime

from langchain import HuggingFaceHub

from langchain.prompts import PromptTemplate

from langchain.chains import LLMChain

from langchain.llms import HuggingFacePipeline




from random import randrange


from .config import config




import datetime

now = datetime.date.today()

from datetime import datetime

 

 

#to find path of current working directory   

absolute_path = os.getcwd()

full_path = os.path.join(absolute_path, config['logging_foler'])

 

 

# Check whether the specified path exists or not

isExist = os.path.exists(full_path)

if not isExist:

   # Create a new directory because it does not exist

   os.makedirs(full_path)

 

file_name = str(full_path)+"qna_" + str(now) + ".log"

 

class ExcludeLogLevelsFilter(logging.Filter):

    def __init__(self, exclude_levels):

        self.exclude_levels = set(exclude_levels)


    def filter(self, record):

        return record.levelname not in self.exclude_levels

 

    

logger = logging.getLogger(__name__)

logger.setLevel(logging.INFO)  # Set the desired logging level, e.g., INFO, DEBUG, WARNING, ERROR, etc.

 

log_filename = 'custom_filename.log'  # Specify your custom filename here

# Create a file handler with the custom filename

file_handler = logging.FileHandler(file_name)

 

# Create a JSON formatter to format log records as JSON

json_formatter = logging.Formatter('{"timestamp": "%(asctime)s", "level": "%(levelname)s", "message": "%(message)s"}')

 

# Set the JSON formatter for the file handler

file_handler.setFormatter(json_formatter)

# Add the file handler to the logger

logger.addHandler(file_handler)

# Exclude log levels DEBUG, WARNING, and ERROR from being saved to the file

exclude_levels = ['DEBUG', 'WARNING', 'ERROR']

file_handler.addFilter(ExcludeLogLevelsFilter(exclude_levels))

 

 

 

class Qna:


    def __init__(self): 

        """

        load all the static files from s3

        """


        self.phi,self.phi_processor=self.load_generative()

        self.username=config['name']


 

    # @st.cache_resource
    def load_generative(self):

        model = AutoModelForCausalLM.from_pretrained(config["LLM_model"],device_map=config['LLM_gpu'],torch_dtype='auto', trust_remote_code=True, _attn_implementation='eager') # use _attn_implementation='eager' to disable flash attention
        processor = AutoProcessor.from_pretrained(config["LLM_model"], trust_remote_code=True) 
        return model, processor

    # Function to save uploaded file
    def save_uploaded_file(self,uploaded_file, save_directory):
        file_path = os.path.join(save_directory, uploaded_file.name)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        return file_path
   

    
    def chat_generative(self,question,model,file_path):

        """
        Answer a question based on the most similar context from the dataframe texts
        """

        
        
        if model=='Microsoft-Phi-Vision-128k':
            messages = [
                {"role": "user", "content": "<|image_1|>\n{}".format(question)}
            ]

            image = Image.open(file_path) 

            prompt = self.phi_processor.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

            inputs = self.phi_processor(prompt, [image], return_tensors="pt").to("cuda:0") 

            generation_args = { 
                "max_new_tokens": config['max_new_tokens'], 
                "temperature": config['LLM_temp'], 
                "do_sample": False, 
            } 

            generate_ids = self.phi.generate(**inputs, eos_token_id=self.phi_processor.tokenizer.eos_token_id, **generation_args) 

            # remove input tokens 
            generate_ids = generate_ids[:, inputs['input_ids'].shape[1]:]
            response = self.phi_processor.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0] 


            return response
        else:
            # Function to encode the image
            def encode_image(file_path):
                with open(file_path, "rb") as image_file:
                    return base64.b64encode(image_file.read()).decode('utf-8')

           

            # Getting the base64 string
            base64_image = encode_image(file_path)
            try:
                headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {config['api_key']}"
                }
            except:
                return "Please enter your Openapi key"

            payload = {
            "model": "gpt-4o",
            "messages": [
                {
                "role": "user",
                "content": [
                    {
                    "type": "text",
                    "text": question
                    },
                    {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{base64_image}"
                    }
                    }
                ]
                }
            ],
            "max_tokens": 300
            }

            response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)

            return response.json()

 

 

    # def chat_(self,index=None, question=None):

    #     return "Test Result"

 

    def generate_ans(self,user_input,model,file_path):

        print('generating answer...')
        

        result1 = self.chat_generative(question = user_input,model=model,file_path=file_path)

        logger.info("Answer generated by generative LLM algorithm : "+str(result1))

        # return st.session_state.bot.append(result1)

        return result1


 

    def main(self,query,model,file_path):

        try:

            logger.info("Question asked by client : "+str(query))

            logger.info("model choosen by User : "+str(model))

           

            answer = self.generate_ans(query,model=model,file_path=file_path)
           
            logger.info("Answer generated by bot : "+str(answer))

            return answer


        except Exception as e:  

            logger.error("Error inside main function of process file : "+str(e))

            return None