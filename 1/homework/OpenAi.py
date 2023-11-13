import openai

openai.api_key = '6c047dd3fd8e44bcbd2d6b94278c6251'
openai.api_base = 'https://lbopenai.openai.azure.com/'
openai.api_version = "2023-03-15-preview"
openai.api_type = "azure"

# 向 OpenAI 发送请求
def send_qa(messages):
    print(f"向 OpenAI 发送请求: {messages}")
    return openai.ChatCompletion.create(
        engine="gpt35",
        messages=messages,
        max_tokens=1024,
        stream=True
    )