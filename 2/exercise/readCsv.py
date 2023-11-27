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

# This assumes that you have placed the bill_sum_data.csv in the same directory you are running Jupyter Notebooks
df=pd.read_csv(os.path.join(os.getcwd(),'2/homework/bill_sum_data.csv')) 
# print(df)

df_bills = df[['text', 'summary', 'title']]
# print(df_bills)

#https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#evaluation-order-matters
pd.options.mode.chained_assignment = None 

# s is input text
def normalize_text(s, sep_token = " \n "):
    s = re.sub(r'\s+',  ' ', s).strip()
    s = re.sub(r". ,","",s)
    # remove all instances of multiple spaces
    s = s.replace("..",".")
    s = s.replace(". .",".")
    s = s.replace("\n", "")
    s = s.strip()
    
    return s

df_bills['text']= df_bills["text"].apply(lambda x : normalize_text(x))

tokenizer = tiktoken.get_encoding("cl100k_base")
df_bills['n_tokens'] = df_bills["text"].apply(lambda x: len(tokenizer.encode(x)))
df_bills = df_bills[df_bills.n_tokens<8192]
# print(len(df_bills))

# print(df_bills)

sample_encode = tokenizer.encode(df_bills.text[0]) 
decode = tokenizer.decode_tokens_bytes(sample_encode)
# print(decode)
# print(len(decode))

# engine should be set to the deployment name you chose when you deployed the text-embedding-ada-002 (Version 2) model
df_bills['ada_v2'] = df_bills["text"].apply(lambda x : get_embedding(x, engine = 'text-embedding-ada-002')) 
# print(df_bills)
for i in df_bills["ada_v2"]:
    print(len(i))

# search through the reviews for a specific product
def search_docs(df, user_query, top_n=3, to_print=True):
    embedding = get_embedding(
        user_query,
        engine="text-embedding-ada-002" # engine should be set to the deployment name you chose when you deployed the text-embedding-ada-002 (Version 2) model
    )
    df["similarities"] = df.ada_v2.apply(lambda x: cosine_similarity(x, embedding))

    res = (
        df.sort_values("similarities", ascending=False)
        .head(top_n)
    )
    print(res)
    # if to_print:
    #     print(res)
    return res


res = search_docs(df_bills, "Can I get information on cable company tax revenue?", top_n=4)

print(res["summary"].iloc[0])