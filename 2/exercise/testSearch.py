import openai

import requests
import sys
from num2words import num2words

import numpy as np
from openai.embeddings_utils import get_embedding, cosine_similarity


API_KEY = "6c047dd3fd8e44bcbd2d6b94278c6251"
RESOURCE_ENDPOINT = 'https://lbopenai.openai.azure.com/'

openai.api_type = "azure"
openai.api_key = API_KEY
openai.api_base = 'https://lbopenai.openai.azure.com/'
openai.api_version = "2023-03-15-preview"

url = openai.api_base + "/openai/deployments?api-version=2023-03-15-preview" 

r = requests.get(url, headers={"api-key": API_KEY})

print(r.text)