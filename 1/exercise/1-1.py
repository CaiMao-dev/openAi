import openai
import uvicorn
from fastapi import FastAPI

openai.api_key = '6c047dd3fd8e44bcbd2d6b94278c6251'
openai.api_base = 'https://lbopenai.openai.azure.com/'
openai.api_version = "2023-03-15-preview"
openai.api_type = "azure"

messages = [
    {"role": "system", "content": "You will be provided with a tweet, and your task is to classify its sentiment as positive, neutral, or negative, and you must answer between [positive, neutral, negative]"},
    {"role": "user", "content": "Today is a good weather"}
]

# 创建一个 FastAPI 应用
app = FastAPI()

@app.get("/analysed")
async def analysed(str):
    messages[1]["content"] = str
    resp = send_qa(messages)
    print(resp)

    return resp["choices"][0]["message"]["content"]

# 向 OpenAI 发送请求
def send_qa(messages):
    print(messages)
    return openai.ChatCompletion.create(
        engine="gpt35",
        messages=messages,
        max_tokens=1024
    )

uvicorn.run(app, host="0.0.0.0", port=18001)