import json

import requests
from fastapi import FastAPI, Request

app = FastAPI()

CHATGLM_ENDPOINT="http://127.0.0.1:8000/"
EMBEDDING_PATH="embedding"
TOKEN = ""
dingtalk_webhook_url=""
onpremise_webhook_url=""


@app.post("/chatglm/embedding")
async def chatglm_prompt(request: Request):
    data = await request.json()
    headers = {
        "Content-Type": "application/json"
    }

    print(data)
    response = requests.post(CHATGLM_ENDPOINT + EMBEDDING_PATH, json=data, headers=headers)
    print(response)

    if response.status_code != 200:
        return {"message": "error happened"}

    return response.json()


@app.post("/chatglm/prompt")
async def chatglm_prompt(request: Request):
    data = await request.json()
    headers = {
        "Content-Type": "application/json"
    }

    print(data)
    response = requests.post(CHATGLM_ENDPOINT, json=data, headers=headers)
    print(response)

    if response.status_code != 200:
        return {"message": "error happened"}

    result = response.content.decode('utf-8')
    res = json.loads(result)

    return {"message": res["response"], "history": res["history"]}

@app.post("/chatglm/bot")
async def chatglm_bot(request: Request):
    data = await request.json()
    print(data)
    internalBody = {
        "prompt": data["text"]["content"]
    }

    print(internalBody)
    response = requests.post("http://127.0.0.1:8081/chatglm/prompt", json=internalBody, headers={"Content-Type": "application/json"})
    print(response)
    result = response.content.decode('utf-8')

    message = {
        "msgtype": "text",
        "text": {

        }
    }

    glm_result_json = json.loads(result)

    #message["text"]["content"] = "GPT Answer:" + glm_result_json["message"]
    message["text"]["content"] = glm_result_json["message"]
    message_json = json.dumps(message)
    response = requests.post(
        dingtalk_webhook_url,
        data=message_json,
        headers={"Content-Type": "application/json"}
    )
    print(response.content.decode('utf-8'))
    return {"message": response.status_code}

if __name__ == "__main__":
   import uvicorn
   uvicorn.run("api:app", host="0.0.0.0", port=8081, reload=True, workers=4)  # 启动端口为8081