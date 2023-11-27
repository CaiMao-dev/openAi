import MyMilvus
import uvicorn
from fastapi import FastAPI
import os
import pandas as pd
import re
import tiktoken
from openai.embeddings_utils import get_embedding, cosine_similarity
import openai

API_KEY = "6c047dd3fd8e44bcbd2d6b94278c6251"
RESOURCE_ENDPOINT = 'https://lbopenai.openai.azure.com/'

openai.api_type = "azure"
openai.api_key = API_KEY
openai.api_base = 'https://lbopenai.openai.azure.com/'
openai.api_version = "2023-03-15-preview"

url = openai.api_base + "/openai/deployments?api-version=2023-03-15-preview" 

df=pd.read_csv(os.path.join(os.getcwd(),'2/homework/bill_sum_data.csv')) 
df_bills = df[['text', 'summary', 'title']]
pd.options.mode.chained_assignment = None 

def normalize_text(s, sep_token = " \n "):
    s = re.sub(r'\s+',  ' ', s).strip()
    s = re.sub(r". ,","",s)
    s = s.replace("..",".")
    s = s.replace(". .",".")
    s = s.replace("\n", "")
    s = s.strip()
    return s
df_bills['text']= df_bills["text"].apply(lambda x : normalize_text(x))

tokenizer = tiktoken.get_encoding("cl100k_base")
df_bills['n_tokens'] = df_bills["text"].apply(lambda x: len(tokenizer.encode(x)))
df_bills = df_bills[df_bills.n_tokens<8192]
sample_encode = tokenizer.encode(df_bills.text[0]) 
decode = tokenizer.decode_tokens_bytes(sample_encode)
df_bills['ada_v2'] = df_bills["text"].apply(lambda x : get_embedding(x, engine = 'text-embedding-ada-002')) 

def get_v(v):
    return get_embedding(v, engine = 'text-embedding-ada-002');

_COLLECTION = "test";

global collection

def init():
    # create a connection
    MyMilvus.create_connection()

    # drop collection if the collection exists
    if MyMilvus.has_collection(_COLLECTION):
        MyMilvus.drop_collection(_COLLECTION)

    global collection
    # create collection
    collection = MyMilvus.create_collection(_COLLECTION)

    # alter ttl properties of collection level
    MyMilvus.set_properties(collection)

    for i in range(len(df_bills['text'])):
        MyMilvus.insert(collection, [[i],[df_bills['ada_v2'][i]], [df_bills['text'][i]]])

    collection.flush()

    # create index
    MyMilvus.create_index(collection)

# 初始化milvus客户端
init()
# 创建一个 FastAPI 应用
app = FastAPI();

@app.get("/analysed")
async def search(msg):
    msg =  "Can I get information on cable company tax revenue?"
    # load data to memory
    MyMilvus.load_collection(collection)
    # search
    results = MyMilvus.search(collection, get_v(msg))
    MyMilvus.release_collection(collection)
    return results[0][0].entity.get('content');

uvicorn.run(app, host="0.0.0.0", port=18001)