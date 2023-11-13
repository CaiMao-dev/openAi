import OpenAi
import ChartRecord
import uvicorn
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
import json


def send(userId, msg):
    messages = ChartRecord.process_chat_message(userId, {"role": "user", "content": f"{msg}"})
    response = OpenAi.send_qa(messages);
    return StreamingResponse(eventStream(response), media_type="text/event-stream")

# 以上代码获取chatgpt响应，
def eventStream(response):
    for item in response:
        json_data = item['choices'][0]['delta']
        res_data = json.dumps(item['choices'][0]['delta'], ensure_ascii=False)
        if json_data.__contains__('content'):
            yield 'data: {}\n\n'.format(res_data)

# 创建一个 FastAPI 应用
app = FastAPI()

@app.get("/analysed")
async def analysed(str):
    return send("caimao", str);

uvicorn.run(app, host="0.0.0.0", port=18001)



    