from fastapi import FastAPI, Request
from transformers import AutoTokenizer, AutoModel, HfArgumentParser
import uvicorn, json, datetime
import torch
from utils import ModelArguments, auto_configure_device_map, load_pretrained
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from model_config import *
from peft import PeftModel

DEVICE = "cuda"
DEVICE_ID = "0"
CUDA_DEVICE = f"{DEVICE}:{DEVICE_ID}" if DEVICE_ID else DEVICE


def torch_gc():
    if torch.cuda.is_available():
        with torch.cuda.device(CUDA_DEVICE):
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()


app = FastAPI()


@app.on_event("startup")
async def get_local_doc_qa():
    global model, tokenizer, embeddings
    torch_gc()
    parser = HfArgumentParser(ModelArguments)
    model_args, = parser.parse_args_into_dataclasses()
    model, tokenizer = load_pretrained(model_args)
    if torch.cuda.device_count() > 1:
        from accelerate import dispatch_model
        device_map = auto_configure_device_map(torch.cuda.device_count())
        model = dispatch_model(model, device_map)
    else:
        model = model.cuda()
    model.eval()

    embeddings = HuggingFaceEmbeddings(model_name=embedding_model_dict[EMBEDDING_MODEL],
                                                model_kwargs={'device': EMBEDDING_DEVICE})


@app.post("/")
async def create_item(request: Request):
    json_post_raw = await request.json()
    json_post = json.dumps(json_post_raw)
    json_post_list = json.loads(json_post)
    prompt = json_post_list.get('prompt')
    history = json_post_list.get('history')
    max_length = json_post_list.get('max_length')
    top_p = json_post_list.get('top_p')
    temperature = json_post_list.get('temperature')
    use_base = json_post_list.get('use_base')
    if use_base:
        with PeftModel.disable_adapter(model):
            response, history = model.chat(tokenizer,
                                        prompt,
                                        history=history,
                                        max_length=max_length if max_length else 2048,
                                        top_p=top_p if top_p else 0.7,
                                        temperature=temperature if temperature else 0.95)
    else:
        response, history = model.chat(tokenizer,
                                    prompt,
                                    history=history,
                                    max_length=max_length if max_length else 2048,
                                    top_p=top_p if top_p else 0.7,
                                    temperature=temperature if temperature else 0.95)

    now = datetime.datetime.now()
    time = now.strftime("%Y-%m-%d %H:%M:%S")
    answer = {
        "response": response,
        "history": history,
        "status": 200,
        "time": time
    }
    log = "[" + time + "] " + '", prompt:"' + prompt + '", response:"' + repr(response) + '"'
    print(log)
    torch_gc()
    return answer


@app.post("/stream_chat")
async def create_item(request: Request):
    json_post_raw = await request.json()
    json_post = json.dumps(json_post_raw)
    json_post_list = json.loads(json_post)
    prompt = json_post_list.get('prompt')
    history = json_post_list.get('history')
    max_length = json_post_list.get('max_length')
    top_p = json_post_list.get('top_p')
    temperature = json_post_list.get('temperature')
    async def eval_llm():
        first = True
        for response in model.stream_chat(tokenizer,
                                   prompt,
                                   history=history,
                                   max_length=max_length if max_length else 2048,
                                   top_p=top_p if top_p else 0.7,
                                   temperature=temperature if temperature else 0.95):
            if first:
                first = False
                yield json.dumps(generate_stream_response_start(),
                                ensure_ascii=False)
            yield json.dumps(generate_stream_response(response), ensure_ascii=False)
        yield json.dumps(generate_stream_response_stop(), ensure_ascii=False)
        yield "[DONE]"
    return EventSourceResponse(eval_llm(), ping=10000)
        

@app.post("/embedding")
async def create_item(request: Request):
    json_post_raw = await request.json()
    json_post = json.dumps(json_post_raw)
    json_post_list = json.loads(json_post)
    query = json_post_list.get('query')

    response = embeddings.embed_query(query)
    now = datetime.datetime.now()
    time = now.strftime("%Y-%m-%d %H:%M:%S")
    answer = {
        "vector": response,
        "status": 200,
        "time": time
    }
    log = "[" + time + "] " + '", query:"' + query + '", response:"' + repr(response) + '"'
    print(log)
    torch_gc()
    return answer


if __name__ == '__main__':
    uvicorn.run("infer:app", host='0.0.0.0', port=8000, reload=True, workers=1)