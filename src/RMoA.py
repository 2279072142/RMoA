import os
import sys
import json
from fire import Fire
from functools import partial
from typing import List
from loguru import logger
from src.llm_api import generate_with_references,generate_with_references_v2,single_model_answer,llm_api_batch
from src.utils import cosine_similarity,DEBUG
from FlagEmbedding import BGEM3FlagModel
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np
import random
import time
import re
from tqdm import tqdm
import ast
DEBUG=1

def get_json_object(response):
    json_object = {'Residual Answer': 'No', 'Residual Content': ''}
    # 对 LaTeX 反斜杠进行双重转义
    response = response.replace('\\', '\\\\')
    try:
        json_object = json.loads(response)
        # print("解析成功:", json_object)
    except json.JSONDecodeError as e:
        try:
            start = response.index('{')
            end = response.rindex('}') + 1
            trimmed_string = response[start:end]
            json_object = json.loads(trimmed_string)
            # print("修正后解析成功:", json_object)
        except (ValueError, json.JSONDecodeError) as e:
            print("解析失败:", e)
    return json_object

def extract_residuals(output):
    """
    从给定的输出中提取Residuals Detected状态和Residual Details内容。

    参数:
    - output (str): 要解析的文本输出。

    返回:
    - dict: 包含状态和Residual Details的字典。
      {
          'Residuals Detected': 'Yes' 或 'No',
          'Residual Details': 'Model 1: ...\nModel 2: ...' 或 空字符串
      }
    """
    result = {
        'Residuals Detected': None,
        'Residual Details': ""
    }

    # 模式匹配“Residuals Detected: Yes”或“Residuals Detected: No”
    status_pattern = r"Residuals Detected:\s*(Yes|No)"
    status_match = re.search(status_pattern, output, re.IGNORECASE)
    
    if status_match:
        status = status_match.group(1).strip().capitalize()
        result['Residuals Detected'] = status
    else:
        # 如果未找到状态行
        result['Residuals Detected'] = "Status not found"

    # 如果检测到Residuals，则提取Residual Details
    if result['Residuals Detected'] == "Yes":
        # 模式匹配所有以“Model <number>:”开头的行
        details_pattern = r"(Model\s*\d+:\s*.+)"
        # 使用re.MULTILINE和re.DOTALL以匹配跨行内容
        matches = re.findall(details_pattern, output, re.MULTILINE)
        
        # 将所有匹配的Residual Details合并为一个字符串，每个Detail占一行
        residual_details = "\n".join([match.strip() for match in matches])
        result['Residual Details'] = residual_details

    return result

def get_smoa_key(text):
    try:
        # 使用正则表达式提取'chosen responses'和'end debate'字段
        chosen_responses_match = re.search(r'"chosen responses": (\[.*?\])', text)
        end_debate_match = re.search(r'"end debate": (true|false)', text)

        # 转换提取的结果
        chosen_responses = json.loads(chosen_responses_match.group(1)) if chosen_responses_match else None
        end_debate = json.loads(end_debate_match.group(1)) if end_debate_match else None
    except:
        chosen_responses=None
        end_debate=None
    return chosen_responses,end_debate
def read_txt(file_path):
    with open(file_path, 'r') as file:
        data = file.read()
    return data

class RMoA():
    def __init__(self,model,temperature=0.7,max_tokens=2048,layer_nums=4,
                aggregation_model='gpt-4o-2024-08-06',
                evaluation_model='gpt-4o-2024-08-06',
                embedding_model='text-embedding-3-large',
                embedding_model_device='2',
                residual_model='gpt-4o-2024-08-06',
                issparse=False,
                residuals=False,
                k=2,
                start_round=0,
                dataset_type='alpaca_eval',
                add_role=False,
                cost_path=None,
                smoa=False   #使用SMOA的方法
                ):
        self.experts=model
        self.temperature=temperature
        self.max_tokens=max_tokens
        self.layer_nums=layer_nums
        self.aggregation_model=aggregation_model
        self.evaluate_model=evaluation_model
        if issparse and embedding_model=='bge':
            os.environ[ "CUDA_VISIBLE_DEVICES" ] = str(embedding_model_device)
            self.embedding_model='bge'
            self.local_emebedding_model=BGEM3FlagModel('./models/BAAI/bge-m3',  use_fp16=True,device='cuda')
        else:
            self.embedding_model=embedding_model
        self.residual_model=residual_model
        self.issparse=issparse
        self.k=k
        self.residuals=residuals
        self.start_round=start_round
        self.dataset_type=dataset_type
        self.add_role=add_role
        self.cost_path=cost_path
        if cost_path is not None and os.path.exists(self.cost_path):
            with open(self.cost_path,'r') as f:
                cost_data=json.load(f)
                self.input_cost=cost_data['input_token']
                self.output_cost=cost_data['output_token']
                self.input_ag_cost=cost_data['input_ag_token']
                self.output_ag_cost=cost_data['output_ag_token']
        else:
            self.input_cost=[0,0,0,0,0,0,0,0]
            self.output_cost=[0,0,0,0,0,0,0,0]
            self.input_ag_cost=[0,0,0,0,0,0,0,0]
            self.output_ag_cost=[0]*layer_nums
        self.init_prompt()
        self.init_message_output()
        self.smoa=smoa

    def init_prompt(self):
        self.residual_prompt=read_txt('./src/prompt/moa/moa_residual.txt')
        self.residual_aggregation_prompt=read_txt('./src/prompt/moa/moa_residual_aggregation.txt')
        if self.dataset_type=='math':
            with open('./src/prompt/tasks/math_role_prompt.json','r') as f:
                self.role_prompt_list=json.load(f)
        elif self.dataset_type=='crux':
            with open('./src/prompt/tasks/crux_role_prompt.json','r') as f:
                self.role_prompt_list=json.load(f)
        elif self.dataset_type=='alpaca_eval':
            with open('./src/prompt/tasks/alpaca_eval_role_prompt.json','r') as f:
                self.role_prompt_list=json.load(f)    
        elif self.dataset_type=='mmlu':
            with open('./src/prompt/tasks/mmlu_role_prompt.json','r') as f:
                self.role_prompt_list=json.load(f)    
 
    def init_message_output(self):
        experts_str=','.join([expert for expert in self.experts])
        info_r = "采用残差" if self.residuals else "不采用残差"
        info_pre = "从头开始" if self.start_round == 0 else f"从{self.start_round}层开始"
        if self.issparse:
            logger.info(f"{info_pre}初始化{self.layer_nums}层专家层,专家层模型包括{experts_str}。采用稀疏MOA架构,{info_r}。聚合模型为{self.aggregation_model},评估模型为{self.evaluate_model},embedding模型为{self.embedding_model},贪心策略寻找{self.k}个回复。")

        else:
            logger.info(f"{info_pre}初始化{self.layer_nums}层专家层,专家层模型包括{experts_str}。采用MOA架构,{info_r}。聚合模型为{self.aggregation_model},评估模型为{self.evaluate_model}。")
    def judge_and_end(self,query,references,layer):
        system=read_txt('./src/prompt/moa/smoa_judge_and_end.txt')
        cur_result=''
        for i, reference in enumerate(references):
            cur_result += f"\nModel {i} response:. {reference}"
        prompt=f'Question:{query}\nReferences responses:{cur_result}'
        messages=[{'role':'system','content':system},
                 {'role':'user','content':prompt}]
        response,token_cost=single_model_answer(messages=messages,model=self.aggregation_model)
        self.input_cost[layer]+=token_cost['input_tokens']
        self.output_cost[layer]+=token_cost['output_tokens']
        chosen_responses,end_debate=get_smoa_key(response)
        return chosen_responses,end_debate
    
    def judge_and_end_batch(self,inputs,layer):
        system=read_txt('./src/prompt/moa/smoa_judge_and_end.txt')
        tmp_messages=[]
        for input in inputs:
            tmp_messages.append(input['messages'])
        for input in inputs:
            cur_result=''
            for i, reference in enumerate(input['references']):
                cur_result += f"\nModel {i} response:. {reference}"
            
            prompt = f"Question:{input['instruction']}\nReferences responses:{cur_result}"

            messages=[{'role':'system','content':system},
                    {'role':'user','content':prompt}]
            input['messages']=messages
        responses,tokens_costs=llm_api_batch(self.residual_model,inputs,8192,self.temperature)
        for response,input in zip(responses,inputs):
            chosen_responses,end_debate=get_smoa_key(response)
            if chosen_responses==None:
                chosen_responses=range(len(self.experts))
            if end_debate==None:
                end_debate=False
            tmp_references=[]
            for i in chosen_responses:
                if i>=0 and i<len(input['references']):
                    tmp_references.append(input['references'][i])
            input['references']=tmp_references
            input['end']=end_debate

        for token_cost,input in zip(tokens_costs,inputs):
            self.input_cost[layer]+=token_cost['input_tokens']
            self.output_cost[layer]+=token_cost['output_tokens']

        for input,messages in zip(inputs,tmp_messages):
            input['messages']=messages
        return inputs
    def calculate_residual_v2_vllm(self,inputs,layer,is_pro=False):
        #残差连接
        logger.info(f"Round {layer+1}/{self.layer_nums} to calculate residuals")
        tmp_messages=[]

        #临时暂存信息
        for input in inputs:
            tmp_messages.append(input['messages'])
        #处理残差
        for input in inputs:
            cur_result=''
            pre_result=''
            for i, reference in enumerate(input['pre_references']):
                pre_result+= f"\n\nModel {i+1}. {reference}" 

            for i, reference in enumerate(input['references']):
                cur_result += f"\n\nModel {i+1}. {reference}"
 
            system=self.residual_prompt
            prompt=f'''Query:{input['instruction']}\nPrevious round of responses:{pre_result}\nThis round of multiple responses:{cur_result}'''
            messages=[
                {'role':'system','content':system},
                {'role':'user','content':prompt}
            ]
            input['messages']=messages
        #生成残差
        responses,tokens_costs=llm_api_batch(self.residual_model,inputs,2048,self.temperature)
        self.compute_tokens(tokens_costs=tokens_costs,layer=layer)
       
        retry=0
        #提前聚合残差并记录结果
        residual_answer='No'
        residual_content=''
        for input,response in zip(inputs,responses):
            while retry<2:
                try:
                    json_object=extract_residuals(response)
                    residual_answer=json_object['Residuals Detected']
                    residual_content=json_object['Residual Details']
                    break
                except:
                    time.sleep(3)
                    response=llm_api_batch(self.residual_model,[input],2048,self.temperature)[0]
                    retry+=1
            input['residual_content']=residual_content
            pre_result=''
            for i, reference in enumerate(input['pre_references']):
                pre_result+= f"\n\nModel {i+1}. {reference}" 
            system=self.residual_aggregation_prompt+f'''\nResponses from models:{pre_result}\n\nResidual Content:{residual_content}'''
            prompt=f'''Query:{input['instruction']}'''
            if residual_answer=='No':
                input['residual_flag']+=1
            else:
                input['residual_flag']=0
            messages=[
                {'role':'system','content':system},
                {'role':'user','content':prompt}
            ]
            input['messages']=messages
        #生成残差补全的结果
        responses,tokens_costs=llm_api_batch(self.aggregation_model,inputs,self.max_tokens,self.temperature)
        for tokens_cost in tokens_costs:
            self.input_ag_cost[layer]+=tokens_cost['input_tokens']
            self.output_ag_cost[layer]+=tokens_cost['output_tokens']
        #将inputs原来的messages补回来
        for input,messages in zip(inputs,tmp_messages):
            input['messages']=messages
        #将残差补全的结果填回inputs
        for input,response in zip(inputs,responses):
            input['pre_references']=input['references']
            input['cur_result']=response
            input['steps']=input['steps']+1
        return inputs

    def calculate_residual_v2(self,query,pre_references,cur_references,layer,Cache=True):
        #残差连接
        cur_result=""
        pre_result=""
        for i, reference in enumerate(cur_references):
            cur_result += f"\nModel {i+1}. {reference}"
        for i, reference in enumerate(pre_references):
            pre_result+= f"\nModel {i+1}. {reference}" 

        logger.info(f"Round {layer+1}/{self.layer_nums} to calculate residuals")
        system=self.residual_prompt
        prompt=f'''Query:{query}\nPrevious round of responses:{pre_result}\nThis round of multiple responses:{cur_result}'''
        messages=[
            {'role':'system','content':system},
            {'role':'user','content':prompt}
        ]
        is_residual='No'
        residual_content=''
        token_cost={'input_tokens':0,'output_tokens':0}
        while True:
            try:
                response,token_cost=single_model_answer(messages=messages,model=self.residual_model)
                json_object=extract_residuals(response)        
                is_residual=json_object['Residuals Detected']
                residual_content=json_object['Residual Details']
                break
            except Exception as error:
                print(error)
                time.sleep(3)
        self.input_cost[layer]+=token_cost['input_tokens']
        self.output_cost[layer]+=token_cost['output_tokens']       

        if Cache:
            system=self.residual_aggregation_prompt+f'''\nResponses from models:{pre_result}\n\nResidual Content:{residual_content}'''
            prompt=f'''Query:{query}'''
            messages=[
                {'role':'system','content':system},
                {'role':'user','content':prompt}
            ]
            if is_residual.lower()=='yes':
                is_residual='yes'
                logger.info(f"Round {layer+1}/{self.layer_nums} to merge residuals")
            else:
                is_residual='no'
                logger.info(f"Round {layer+1}/{self.layer_nums} has no residuals")
            response,token_cost=single_model_answer(messages=messages,model=self.residual_model)
            self.input_ag_cost[layer]+=token_cost['input_tokens']
            self.output_ag_cost[layer]+=token_cost['output_tokens']
            return residual_content,is_residual,response
        
        return residual_content,is_residual

    def calculate_diff_reference_vllm(self, inputs, k):
        all_references = []
        # 提取所有 references 并存储在 all_references 列表中
        for input in inputs:
            if input['residual_flag']==0:
                all_references.append(input['references'])
        all_embedding=[]
        # 计算所有 references 的嵌入
        if self.embedding_model=='bge':
            for item in tqdm(all_references):
                embeddings=self.local_emebedding_model.encode(item,batch_size=6)['dense_vecs']
                all_embedding.append(embeddings)
        
        # 使用 zip 函数将 inputs 和 all_embedding 组合在一起
        for input, embedding in zip(inputs, all_embedding):
            # 对每个 input 和对应的 embedding 调用 diff_reference 函数
            input['references'] = self.diff_reference(input['references'], embedding, k)
        return inputs

    def dataset_result_format(self,inputs):
        
        results=[]
        for input in inputs:
            if self.dataset_type=='alpaca_eval':
                if self.residuals:
                    results.append({'dataset':input['dataset'],'instruction':input['instruction'],
                                    'output':input['cur_result'],'generator':'ssr_moa','steps':input['steps']})
                else:
                    results.append({'dataset':input['dataset'],'instruction':input['instruction'],
                                                'output':input['cur_result'],'generator':'moa'})
            elif self.dataset_type=='gsm8k' or self.dataset_type=='math' or self.dataset_type=='mmlu_redux':
                if self.residuals:
                    results.append({'instruction':input['instruction'],'output':input['cur_result'],'ans':input['answer'],'generator':'ssr_moa','steps':input['steps']})
                else:
                    results.append({'instruction':input['instruction'],'output':input['cur_result'],'ans':input['answer'],'generator':'moa'})
            elif self.dataset_type=='crux':
                if self.residuals:
                    results.append({'instruction':input['instruction'],'output':input['cur_result'],'ans':input['answer'],'generator':'ssr_moa','steps':input['steps']})
                else:
                    results.append({'instruction':input['instruction'],'output':input['cur_result'],'ans':input['answer'],'generator':'moa'})
        return results


    def diff_reference(self, references, embeddings, k):
        """
        Selects k references by choosing references that maximize diversity based on embeddings using a greedy strategy.
        
        Parameters:
        - references (list): List of reference texts.
        - embeddings (list or np.ndarray): Corresponding list of embeddings for the references.
        - k (int): Number of references to select.
        
        Returns:
        - selected_references (list): List of k selected references.
        """
        num_texts = len(references)
        
        if k < 1:
            raise ValueError("k must be at least 1.")
        if k > num_texts:
            raise ValueError("k cannot exceed the number of available references.")

        # Compute the full similarity matrix
        similarity_matrix = np.zeros((num_texts, num_texts))
        for i in range(num_texts):
            for j in range(num_texts):
                similarity = cosine_similarity(embeddings[i], embeddings[j])
                similarity_matrix[i][j] = similarity
        
        # Initialize selected indices as empty
        selected_indices = []
        
        # Initialize remaining indices as all indices
        remaining_indices = list(range(num_texts))
        
        # Step 1: Select the first reference that has the lowest average similarity to all others
        avg_similarities = similarity_matrix.mean(axis=1)
        first_index = np.argmin(avg_similarities)
        selected_indices.append(first_index)
        remaining_indices.remove(first_index)
        
        # Step 2: Iteratively select the next reference that is least similar to the already selected ones
        while len(selected_indices) < k and remaining_indices:
            # For each candidate, find the maximum similarity to any selected reference
            max_similarities = []
            for idx in remaining_indices:
                sims_to_selected = similarity_matrix[idx][selected_indices]
                max_sim = sims_to_selected.max()
                max_similarities.append((idx, max_sim))
            
            # Select the candidate with the minimum of these maximum similarities
            candidate_index, _ = min(max_similarities, key=lambda x: x[1])
            
            # Add the selected candidate to the selected_indices
            selected_indices.append(candidate_index)
            
            # Remove the selected candidate from remaining_indices
            remaining_indices.remove(candidate_index)
        
        # Gather the selected references
        selected_references = [references[idx] for idx in selected_indices]
        
        return selected_references

    
    def message_add_role(self,messages,index):
        if self.add_role:
            
            for message in messages:
                if message['role']=='user':
                    message['content']=self.role_prompt_list[0][f'role{index}']+'\n'+message['content']
                    break
        return messages

    def message_add_role_vllm(self,inputs,index):
        if self.add_role:
            for input in inputs:
                for message in input['messages']:
                    if message['role']=='user':
                        message['content']=self.role_prompt_list[0][f'role{index}']+'\n'+message['content']
                        break
        return inputs

                

    def calculate_diff_reference(self,references,k):
        if self.embedding_model=='bge':
            embeddings=self.local_emebedding_model.encode(references,batch_size=6,max_length=2048)['dense_vecs']
        #贪心计算索引
        return self.diff_reference(references,embeddings,k)
    #计算token开销
    def compute_tokens(self,tokens_costs,layer):
        for tokens_cost in tokens_costs:
            self.input_cost[layer]+=tokens_cost['input_tokens']
            self.output_cost[layer]+=tokens_cost['output_tokens']

    def forward_vllm(self,inputs,workers=6,imm_store=None):
        results=[]
        inputs_done=[]
        inputs_tmp=[]
        #初始化残差标记表
        id=1
        for input in inputs:
            if 'id' not in input:
                input['id']=id
                id+=1
            input['pre_references']=[]
            input['cur_result']=[]
            input['references']=[]
            input['residual_flag']=0
            input['residual_content']=''
            input['messages']=[{"role": "user", "content": input['instruction']}]
            input['end']=False
        #residual_flag_list=[0]*len(inputs)
        for layer in range(self.layer_nums):
            if DEBUG:
                logger.info(
                f"Round {layer+1}/{self.layer_nums} to collecting reference responses."
            )
            #剔除早停的case
            inputs_tmp=[]
            for input in inputs:
                if input['residual_flag']!=0:
                    inputs_done.append(input)
                else:
                    inputs_tmp.append(input)
            inputs=inputs_tmp
            #只有当采用RMoA和SMoA时才有出现这种情况
            if len(inputs)==0:
                inputs_all=sorted(inputs+inputs_done,key=lambda x: x['id'])   
                result=self.dataset_result_format(inputs_all)
                results.append(result)
                if imm_store is not None:
                    #for result,path in zip(results,imm_store):
                    data = {
                        "input_token": self.input_cost,
                        "output_token": self.output_cost,
                        "input_ag_token":self.input_ag_cost,
                        "output_ag_token":self.output_ag_cost
                    }
                    # 写入JSON文件
                    with open(self.cost_path, 'w') as json_file:
                        json.dump(data, json_file, indent=4)

                    with open(imm_store[layer],'w') as f:
                        json.dump(result,f,indent=2)
                if layer==self.layer_nums-1:
                    return results
                continue
            #获取提议者的回答
            with ThreadPoolExecutor(max_workers=workers) as executor:
                future_to_expert = {
                    executor.submit(
                        generate_with_references_v2,
                        model=expert,
                        inputs=inputs,
                        role=self.role_prompt_list[0][f'role{index}'] if self.add_role else None,
                        temperature=self.temperature,
                        max_tokens=self.max_tokens,
                        is_residual=self.residuals,
                        action='reference'
                    ): expert
                    for expert,index in zip(self.experts,range(1,len(self.experts)+1))
                }
                #清空前一轮的参考回复
                for input in inputs:
                    input['references']=[]
                for future in as_completed(future_to_expert):
                    expert = future_to_expert[future]  # Get the expert associated with this future
                    outputs,tokens_costs = future.result()
                    #加入参考
                    for input_item,output in zip(inputs,outputs):
                        #服务于smoa的早停机制
                        if input_item['end']==False:
                            input_item['references'].append(output)
                    #计算token开销
                    self.compute_tokens(tokens_costs,layer)

                if layer < self.layer_nums - 1:
                    #批量计算差异度
                    if self.smoa:
                        logger.info(f"Round {layer+1}/{self.layer_nums} to judge and end")
                        inputs=self.judge_and_end_batch(inputs=inputs,layer=layer)
                    #若采用稀疏采样
                    if self.issparse:
                        logger.info(f"Round {layer+1}/{self.layer_nums} to caculate differences.")
                        inputs=self.calculate_diff_reference_vllm(inputs,k=self.k)
                    #若采用残差
                    if self.residuals:
                        logger.info(f"Round {layer+1}/{self.layer_nums} to calculate residuals.")
                        if layer==0:
                            #第一层采用原始moa
                            #生成总结结果 返回的是一个列表
                            outputs,tokens_costs=generate_with_references_v2(
                                model=self.aggregation_model,
                                inputs=inputs,
                                is_residual=self.residuals,
                                action='summary',
                                max_tokens=self.max_tokens,
                                temperature=self.temperature
                            )
                            for tokens_cost in tokens_costs:
                                #计算token成本
                                self.input_ag_cost[layer]+=tokens_cost['input_tokens']
                                self.output_ag_cost[layer]+=tokens_cost['output_tokens']
                            '''
                            1.记录当前轮的参考结果
                            2.记录当前轮的聚合结果
                            3.记录当前轮的steps
                            '''
                            for input,output in zip(inputs,outputs):
                                input['pre_references']=input['references']
                                input['cur_result']=output
                                input['steps']=1                          
                        else:
                            #后续层采用残差
                            inputs=self.calculate_residual_v2_vllm(inputs,layer=layer)
                        inputs_all=sorted(inputs+inputs_done,key=lambda x: x['id'])
                        
                        result=self.dataset_result_format(inputs_all)
                        results.append(result)
                        if imm_store is not None:
                            #for result,path in zip(results,imm_store):
                            data = {
                                "input_token": self.input_cost,
                                "output_token": self.output_cost,
                                "input_ag_token":self.input_ag_cost,
                                "output_ag_token":self.output_ag_cost
                            }
                            # 写入JSON文件
                            with open(self.cost_path, 'w') as json_file:
                                json.dump(data, json_file, indent=4)

                            with open(imm_store[layer],'w') as f:
                                json.dump(result,f,indent=2)
                    #否则采用原始moa
                    else:
                        #服务于原始moa
                        for input in inputs:
                            if input['end']==False:
                                input['pre_result']=input['cur_result']
                    #初始化各个moa参考答案
                    for input in inputs:
                        input['references']=[]
        #稀疏化
        if self.smoa:
            logger.info(f"Round {layer+1}/{self.layer_nums} to judge and end")
            inputs=self.judge_and_end_batch(inputs=inputs,layer=layer)

        if self.issparse:
            inputs=self.calculate_diff_reference_vllm(inputs,k=self.k)
            
        if self.residuals:
            inputs=self.calculate_residual_v2_vllm(inputs,layer=self.layer_nums-1)
        else:
            outputs,tokens_costs=generate_with_references_v2(
                model=self.aggregation_model,
                inputs=inputs,
                action='summary'
            )
            self.compute_tokens(tokens_costs,self.layer_nums-1)
            for input,output in zip(inputs,outputs):
                input['cur_result']=output
        
        inputs_all=sorted(inputs+inputs_done,key=lambda x: x['id'])
        result=self.dataset_result_format(inputs_all)
        if imm_store is not None:
            #for result,path in zip(results,imm_store):
            with open(imm_store[self.layer_nums-1],'w') as f:
                json.dump(result,f,indent=2)
            data = {
                "input_token": self.input_cost,
                "output_token": self.output_cost,
                "input_ag_token":self.input_ag_cost,
                "output_ag_token":self.output_ag_cost
            }
            # 写入JSON文件
            with open(self.cost_path, 'w') as json_file:
                json.dump(data, json_file, indent=4)
        results.append(result)
        if self.residuals:
            return results
        else:
            return result

    def forward(self,content,Cache=True,workers=1):
        outputs=[]
        output=''
        all_references=[]
        messages = [{"role": "user", "content": content}]
        references=[]
        end_debate=False
        if len(references) == 0 and len(self.experts) > 0:
            prev_references = []
            residual_flag=0
            residual_content=''
            step=0
            
            for layer in range(self.start_round,self.layer_nums):
                if end_debate:
                    logger.info(
                        f"Round {layer+1}/{self.layer_nums} escape"
                    )
                    all_references=prev_references
                    continue
                if residual_flag==1:
                    logger.info(
                        f"Round {layer+1}/{self.layer_nums} escape"
                    )
                    if Cache:
                        outputs.append(outputs[-1])
                    continue
                step+=1
                logger.info(
                    f"Round {layer+1}/{self.layer_nums} to collecting reference responses."
                )
                references = []
                with ThreadPoolExecutor(max_workers=workers) as executor:
                    future_to_expert = {
                        executor.submit(generate_with_references,
                                        model=expert,
                                        messages=messages,
                                        references=prev_references,
                                        temperature=random.choice([0.5, 0.7, 1.0]) if expert == 'gpt-4o-2024-08-06' or expert == 'claude3.5' else self.temperature,
                                        max_tokens=self.max_tokens,
                                        is_residual=self.residuals,
                                        residual_content=residual_content,
                                        action='reference',
                                        role=self.role_prompt_list[index - 1][f'role{index}'] if self.add_role else None
                                        ): expert
                        for expert, index in zip(self.experts, range(1, len(self.experts) + 1))
                    }

                    # 收集结果
                    references = []
                    for future in as_completed(future_to_expert):
                        expert = future_to_expert[future]
                        reference,token_cost= future.result()
                        references.append(reference)
                        self.input_cost[layer]+=token_cost['input_tokens']
                        self.output_cost[layer]+=token_cost['output_tokens']
                all_references.append(references)
                if layer < self.layer_nums - 1:
                    #baseline:smoa逻辑
                    if self.smoa:
                        logger.info(f"Round {layer+1}/{self.layer_nums} to judge and end")
                        chosen_responses,end_debate=self.judge_and_end(query=content,references=references,layer=layer)
                        tmp_references=[]
                        try:
                            for i in chosen_responses: 
                                i=int(i)
                                if i>=0 and i<len(references):
                                    tmp_references.append(references[i])
                        except:
                            #出现错误直接全给
                            tmp_references=references
                        references=tmp_references

                    if self.issparse:
                        logger.info(f"Round {layer+1}/{self.layer_nums} to caculate differences.")
                        references=self.calculate_diff_reference(references=references,k=self.k)

                    if self.residuals:
                        logger.info(f"Round {layer+1}/{self.layer_nums} to caculate residuals.")
                        if layer==0:
                            output,token_cost=generate_with_references(
                                model=self.aggregation_model,
                                messages=messages,
                                references=references,
                                is_residual=self.residuals,
                                residual_content=residual_content,
                                max_tokens=self.max_tokens,
                                temperature=self.temperature,
                                action='summary')
                            self.input_ag_cost[layer]+=token_cost['input_tokens']
                            self.output_ag_cost[layer]+=token_cost['output_tokens']
                        else:
                            if Cache:
                                residual_content,r_flag,output=self.calculate_residual_v2(query=content,pre_references=prev_references,cur_references=references,layer=layer,Cache=Cache)
                            else:
                                residual_content,r_flag=self.calculate_residual_v2(query=content,pre_references=prev_references,cur_references=references,layer=layer,Cache=Cache)                           
                            residual_flag=0 if r_flag=='yes' else residual_flag+1
                        if Cache:
                            outputs.append(output)
                    prev_references = references
                    references = []
        if residual_flag==1 and self.residuals:
            return outputs,all_references,step if Cache else output

        #只有当end_debate为False才再进行judge
        if self.smoa and end_debate==False:
            logger.info(f"Round {layer+1}/{self.layer_nums} to judge and end")
            chosen_responses,end_debate=self.judge_and_end(query=content,references=references,layer=layer)
            tmp_references=[]
            try:
                for i in chosen_responses: 
                    i=int(i)
                    if i>=0 and i<len(references):
                        tmp_references.append(references[i])
            except:
                #出现错误直接全给
                tmp_references=references
            #tmp_references = [references[i] for i in chosen_responses]
            references=tmp_references


        if self.issparse:
            references=self.calculate_diff_reference(references=references,k=self.k)
        if self.residuals:
            residual_content,r_flag,output=self.calculate_residual_v2(query=content,pre_references=prev_references,cur_references=references,layer=layer,Cache=True)
        else:
            output,token_cost = generate_with_references(
                model=self.aggregation_model,
                messages=messages,
                references=references,
                action='summary')
            self.input_cost[self.layer_nums-1]+=token_cost['input_tokens']
            self.output_cost[self.layer_nums-1]+=token_cost['output_tokens']
        outputs.append(output)
        data = {
                "input_token": self.input_cost,
                "output_token": self.output_cost,
                "input_ag_token":self.input_ag_cost,
                "output_ag_token":self.output_ag_cost
            }
        # 写入JSON文件
        if self.cost_path is not None:
            with open(self.cost_path, 'w') as json_file:
                json.dump(data, json_file, indent=4)
        if self.residuals and Cache:
            return outputs,all_references,step
        else:
            return output,all_references,step
        #eturn outputs,_ if self.issummary else output,all_references
