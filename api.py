import json

import requests
from fastapi import FastAPI, Request
import json
from typing import List, Optional, Any

from fastapi import FastAPI, HTTPException, Request, status, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from sse_starlette.sse import EventSourceResponse

from context import context

app = FastAPI()

CHATGLM_ENDPOINT="http://127.0.0.1:8000/"
EMBEDDING_PATH="embedding"
STREAM_CHAT_PATH="stream_chat"
TOKEN = ""
dingtalk_webhook_url=""
onpremise_webhook_url=""

app = FastAPI()

class Message(BaseModel):
    role: str
    content: str


class ChatBody(BaseModel):
    messages: List[Message]
    model: str
    stream: Optional[bool] = False
    max_tokens: Optional[int]
    temperature: Optional[float]
    top_p: Optional[float]


class CompletionBody(BaseModel):
    prompt: str
    model: str
    stream: Optional[bool] = False
    max_tokens: Optional[int]
    temperature: Optional[float]
    top_p: Optional[float]


class EmbeddingsBody(BaseModel):
    # Python 3.8 does not support str | List[str]
    input: Any
    model: Optional[str]


def do_embedding_by_infer_api(query):
    data={"query":query}
    headers = {
        "Content-Type": "application/json"
    }
    response = requests.post(CHATGLM_ENDPOINT + EMBEDDING_PATH, json=data, headers=headers)
    if response.status_code != 200:
        return None
    return response.json()['vector']


def do_chat_by_infer_api(question, history, option:dict):
    if option is None:
        option = {}
    data = {"prompt": question, "history": history, "top_p": option.get("top_p"), "max_length": option.get("max_length"), "temperature": option.get("temperature")}
    headers = {
        "Content-Type": "application/json"
    }
    response = requests.post(CHATGLM_ENDPOINT, json=data, headers=headers)
    if response.status_code != 200:
        return ""
    result = response.content.decode('utf-8')
    res = json.loads(result)
    return res["response"]


async def do_chat_stream_by_infer_api(question, history, option:dict):
    data = {"prompt": question, "history": history, "top_p": option.get("top_p"), "max_length": option.get("max_length"), "temperature": option.get("temperature")}
    headers = {
        "Content-Type": "application/json"
    }
    return requests.post(CHATGLM_ENDPOINT + STREAM_CHAT_PATH, json=data, headers=headers)


@app.get("/")
def read_root():
    return {"Hello": "World!"}


@app.get("/v1/models")
def get_models():
    ret = {"data": [], "object": "list"}

    if context.model:
        ret['data'].append({
            "created": 1677610602,
            "id": "chatglm-6b",
            "object": "model",
            "owned_by": "zzd",
            "permission": [
                {
                    "created": 1680818747,
                    "id": "modelperm-fTUZTbzFp7uLLTeMSo9ks6oT",
                    "object": "model_permission",
                    "allow_create_engine": False,
                    "allow_sampling": True,
                    "allow_logprobs": True,
                    "allow_search_indices": False,
                    "allow_view": True,
                    "allow_fine_tuning": False,
                    "organization": "*",
                    "group": None,
                    "is_blocking": False
                }
            ],
            "root": "gpt-3.5-turbo",
            "parent": None,
        })
    if context.embeddings_model:
        ret['data'].append({
            "created": 1671217299,
            "id": "text-embedding-ada-002",
            "object": "model",
            "owned_by": "openai-internal",
            "permission": [
                {
                    "created": 1678892857,
                    "id": "modelperm-Dbv2FOgMdlDjO8py8vEjD5Mi",
                    "object": "model_permission",
                    "allow_create_engine": False,
                    "allow_sampling": True,
                    "allow_logprobs": True,
                    "allow_search_indices": True,
                    "allow_view": True,
                    "allow_fine_tuning": False,
                    "organization": "*",
                    "group": None,
                    "is_blocking": False
                }
            ],
            "root": "text-embedding-ada-002",
            "parent": ""
        })

    return ret


def generate_response(content: str, chat: bool = True):
    if chat:
        return {
            "id": "chatcmpl-77PZm95TtxE0oYLRx3cxa6HtIDI7s",
            "object": "chat.completion",
            "created": 1682000966,
            "model": "gpt-3.5-turbo-0301",
            "usage": {
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0,
            },
            "choices": [{
                "message": {"role": "assistant", "content": content},
                "finish_reason": "stop", "index": 0}
            ]
        }
    else:
        return {
            "id": "cmpl-uqkvlQyYK7bGYrRHQ0eXlWi7",
            "object": "text_completion",
            "created": 1589478378,
            "model": "text-davinci-003",
            "choices": [
                {
                "text": content,
                "index": 0,
                "logprobs": None,
                "finish_reason": "stop"
                }
            ],
            "usage": {
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0
            }
        }


def generate_stream_response_start():
    return {
        "id": "chatcmpl-77QWpn5cxFi9sVMw56DZReDiGKmcB",
        "object": "chat.completion.chunk", "created": 1682004627,
        "model": "gpt-3.5-turbo-0301",
        "choices": [{"delta": {"role": "assistant"}, "index": 0, "finish_reason": None}]
    }



def generate_stream_response(content: str, chat: bool = True):
    if chat:
        return {
            "id": "chatcmpl-77QWpn5cxFi9sVMw56DZReDiGKmcB",
            "object": "chat.completion.chunk",
            "created": 1682004627,
            "model": "gpt-3.5-turbo-0301",
            "choices": [{"delta": {"content": content}, "index": 0, "finish_reason": None}
                        ]}
    else:
        return {
            "id":"cmpl-7GfnvmcsDmmTVbPHmTBcNqlMtaEVj",
            "object":"text_completion",
            "created":1684208299,
            "choices":[
                {
                    "text": content,
                    "index": 0,
                    "logprobs": None,
                    "finish_reason": None,
                }
            ],
            "model": "text-davinci-003"
        }


def generate_stream_response_stop(chat: bool = True):
    if chat:
        return {"id": "chatcmpl-77QWpn5cxFi9sVMw56DZReDiGKmcB",
            "object": "chat.completion.chunk", "created": 1682004627,
            "model": "gpt-3.5-turbo-0301",
            "choices": [{"delta": {}, "index": 0, "finish_reason": "stop"}]
            }
    else:
        return {
            "id":"cmpl-7GfnvmcsDmmTVbPHmTBcNqlMtaEVj",
            "object":"text_completion",
            "created":1684208299,
            "choices":[
                {"text":"","index":0,"logprobs":None,"finish_reason":"stop"}],
            "model":"text-davinci-003",
        }

@app.post("/v1/embeddings")
async def embeddings(body: EmbeddingsBody, request: Request, background_tasks: BackgroundTasks):
    return do_embeddings(body, request, background_tasks)


def do_embeddings(body: EmbeddingsBody, request: Request, background_tasks: BackgroundTasks):
    # if request.headers.get("Authorization").split(" ")[1] not in context.tokens:
    #     raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Token is wrong!")

    # if not context.embeddings_model:
    #     raise HTTPException(status.HTTP_404_NOT_FOUND, "Embeddings model not found!")

    embeddings = do_embedding_by_infer_api(body.input)
    data = []
    if isinstance(body.input, str):
        data.append({
            "object": "embedding",
            "index": 0,
            "embedding":embeddings,
        })
    else:
        for i, single_input in enumerate(body.input):
            data.append({
                "object": "embedding",
                "index": i,
                "embedding": do_embedding_by_infer_api(emsingle_inputbed),
            })
    content = {
        "object": "list",
        "data": data,
        "model": "GanymedeNil/text2vec-large-chinese",
        "usage": {
            "prompt_tokens": 0,
            "total_tokens": 0
        }
    }
    return JSONResponse(status_code=200, content=content)


@app.post("/v1/engines/{engine}/embeddings")
async def engines_embeddings(engine: str, body: EmbeddingsBody, request: Request, background_tasks: BackgroundTasks):
    return do_embeddings(body, request, background_tasks)


@app.post("/v1/chat/completions")
async def chat_completions(body: ChatBody, request: Request, background_tasks: BackgroundTasks):
    # if request.headers.get("Authorization").split(" ")[1] not in context.tokens:
    #     raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Token is wrong!")

    # if not context.model:
    #     raise HTTPException(status.HTTP_404_NOT_FOUND, "LLM model not found!")
    question = body.messages[-1]
    if question.role == 'user':
        question = question.content
    else:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "No Question Found")

    history = []
    user_question = ''
    for message in body.messages:
        if message.role == 'system':
            history.append((message.content, "OK"))
        if message.role == 'user':
            user_question = message.content
        elif message.role == 'assistant':
            assistant_answer = message.content
            history.append((user_question, assistant_answer))

    print(f"question = {question}, history = {history}")

    if body.stream:
        return do_chat_stream_by_infer_api(question, history, {
            "temperature": body.temperature,
            "top_p": body.top_p,
            "max_tokens": body.max_tokens,
        })
    else:
        response = do_chat_by_infer_api(question, history, {
            "temperature": body.temperature,
            "top_p": body.top_p,
            "max_tokens": body.max_tokens,
        })
        return JSONResponse(content=generate_response(response))


@app.post("/v1/completions")
async def completions(body: CompletionBody, request: Request, background_tasks: BackgroundTasks):
    # if request.headers.get("Authorization").split(" ")[1] not in context.tokens:
    #     raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Token is wrong!")

    # if not context.model:
    #     raise HTTPException(status.HTTP_404_NOT_FOUND, "LLM model not found!")
    question = body.prompt

    print(f"question = {question}")

    if body.stream:
        return do_chat_stream_by_infer_api(question, history, {
            "temperature": body.temperature,
            "top_p": body.top_p,
            "max_tokens": body.max_tokens,
        })
    else:
        response = do_chat_by_infer_api(question, [], {
            "temperature": body.temperature,
            "top_p": body.top_p,
            "max_tokens": body.max_tokens,
        })
        return JSONResponse(content=generate_response(response, chat=False))


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