import pandas as pd
from together import Together
import openai
import time
import json
from tqdm import tqdm
import requests
import os
import requests
import copy
import time
from loguru import logger


def generate_openai(
    model,
    messages,
    max_tokens=2048,
    temperature=0.7,
):

    client = openai.OpenAI(
        api_key=os.environ.get("OPENAI_API_KEY"),
    )

    for sleep_time in [1, 2, 4, 8, 16, 32]:
        try:
            completion = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            output = completion.choices[0].message.content
            break

        except Exception as e:
            logger.error(e)
            logger.info(f"Retry in {sleep_time}s..")
            time.sleep(sleep_time)
    token_cost={'output_tokens':completion.usage.completion_tokens,
        'input_tokens':completion.usage.prompt_tokens,
        'total_tokens':completion.usage.total_tokens}
    output = output.strip()
    return output,token_cost


def single_model_answer(messages,model="gpt_4o",max_tokens=2048,temperature=0.7,mode='Together',
                        retry=5):
    t=0
    wait_time=[1,2,4,8,16,32,64]
    response=None
    token_cost={}
    while(t<retry):
        try:
            if mode=='Together':
                response,token_cost=generate_together(model=model,messages=messages,temperature=temperature,max_tokens=max_tokens)
            elif mode=='Openai':
                response,token_cost=generate_openai(model=model,messages=messages,temperature=temperature,max_tokens=max_tokens)
            else:
                print(f"{mode} 暂不支持。")
            break
        except Exception as e:
            wt=wait_time[t]
            time.sleep(wt)
        
    return response,token_cost




def inject_references_to_messages(
    messages,
    references,
):

    messages = copy.deepcopy(messages)
    system = f"""You have been provided with a set of responses from various models to the latest user query. Your task is to synthesize these responses into a single, high-quality response. It is crucial to critically evaluate the information provided in these responses, recognizing that some of it may be biased or incorrect. Your response should not simply replicate the given answers but should offer a refined, accurate, and comprehensive reply to the instruction. Ensure your response is well-structured, coherent, and adheres to the highest standards of accuracy and reliability.

Responses from models:"""

    for i, reference in enumerate(references):

        system += f"\n{i+1}. {reference}"

    if messages[0]["role"] == "system":

        messages[0]["content"] += "\n\n" + system

    else:

        messages = [{"role": "system", "content": system}] + messages
        

    return messages

def summary_references_to_messages(
    messages,
    references,
    residual_content,
    action
):
    messages=copy.deepcopy(messages)
    #读取txt文件
    with open('./src/prompt/moa/moa_residual_reference.txt','r') as f:
        reference_system=f.read()+'\n\nReference response:'
    with open('./src/prompt/moa/moa_residual_aggregation.txt','r') as f:
        summary_system=f.read()+'\n\nResponses from models:'
    #根据参考信息进行作答
    if action=='reference':
        for i,reference in enumerate(references):
            reference_system+=f'\n\n{i+1}. {reference}'
        reference_system+=f'\n\nResidual Content:{residual_content}'
        for message in messages:
            if message['role']=='system':
                message['content']=reference_system
        #print("reference")
    elif action=='summary':
        for i, reference in enumerate(references):
            summary_system += f"\n\n{i+1}. {reference}"
        summary_system+=f'\n\nResidual Content:{residual_content}'
        messages = [{"role": "system", "content": summary_system}] + messages
    return messages

def llm_api_batch(
    messages,
    model
):
    pass

def generate_with_references_v2(
    model,
    inputs,
    max_tokens=2048,
    temperature=0.7,
    is_residual=False,
    action=None,
    role=None,
    generate_fn=llm_api_batch):
    for input in inputs:
        if len(input['references']) > 0:
            if is_residual:
                messages= summary_references_to_messages(input['messages'],input['references'],input['residual_content'],action)
            else:
                messages =inject_references_to_messages(input['messages'],input['references'])
            input['messages']=messages
        if role != None and action != 'summary':
           input['messages']=inject_role_to_messages(input['messages'],role)
    outputs,tokens_costs=generate_fn(
        model_name=model,
        inputs=inputs,
        model_max_new_tokens=max_tokens,
        temperature=temperature
    )
    logger.info(f"{model}调用结束.")
    return outputs,tokens_costs


def inject_role_to_messages(
    messages,
    role,
):
    #找到message里面role为system的部分，并增加role提示词
    messages=copy.deepcopy(messages)
    exist_system=False
    for message in messages:
        if message['role']=='system':
            message['content']=message['content']+'\n\n'+f'Your Role Description:{role}'
            exist_system=True
            break
    if not exist_system:
        messages=[{"role": "system", "content": f'Your Role Description:{role}'}] + messages
    return messages

def generate_with_references(
    model,
    messages,
    references=[],
    max_tokens=2048,
    temperature=0.7,
    is_residual=False,
    action=None,
    role=None,
    residual_content=None,
    generate_fn=single_model_answer
):
    
    if len(references) > 0:
        if is_residual:
            messages = summary_references_to_messages(messages,references,residual_content,action)
        else:
            messages = inject_references_to_messages(messages, references)
    #print(messages)
    if role is not None:
        messages=inject_role_to_messages(messages,role)

    output=None
    while(output==None):
        output,token_cost=generate_fn(
        messages=messages,
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    return output,token_cost


def read_prompt(prompt_path):
    #读取txt文本文件内容
    with open(prompt_path, 'r') as file:
        content = file.read()
    return content
def read_json(json_path):       
    #读取json文件内容
    with open(json_path, 'r') as file:
        data = json.load(file)
    return data
    
def generate_together(
    model,
    messages,
    max_tokens=2048,
    temperature=0.7,
):
    output = None
    for sleep_time in [1, 2, 4, 8, 16, 32]:
        try:
            endpoint = "https://api.together.xyz/v1/chat/completions"
            res = requests.post(
                endpoint,
                json={
                    "model": model,
                    "max_tokens": max_tokens,
                    "temperature": (temperature if temperature > 1e-4 else 0),
                    "messages": messages,
                },
                headers={
                    "Authorization": f"Bearer {os.environ.get('TOGETHER_API_KEY')}",
                },
            )
            if "error" in res.json():
                logger.error(res.json())
                if res.json()["error"]["type"] == "invalid_request_error":
                    logger.info("Input + output is longer than max_position_id.")
                    return None

            output = res.json()["choices"][0]["message"]["content"]
            break
        except Exception as e:
            logger.error(e)
            logger.info(f"Retry in {sleep_time}s..")
            time.sleep(sleep_time)

    if output is None:

        return output

    output = output.strip()

    return output

