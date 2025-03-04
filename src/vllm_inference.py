import torch
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
import ray
import gc
import time
import re

class selfVllm:
    def __init__(self, model_path="", tp=4, device_ids=None):
        self.model_path = model_path
        self.tp = tp
        self.device_ids = device_ids if device_ids else [0]  # 默认使用 GPU 0
        self.load_model(model_path)
        self.model_name = model_path.split("/")[-1]
        
    def load_model(self, model_path):
        # 设置设备
        self.llm = LLM(
            model=model_path,
            tensor_parallel_size=self.tp,
            gpu_memory_utilization=0.9,
            trust_remote_code=True
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)


    def vllm_chat_messages(self, messages, kwargs={}):
        temperature = kwargs.get("temperature", 0.95)
        top_p = kwargs.get("top_p", 0.7)
        top_k = kwargs.get("top_k", 3)
        max_tokens = kwargs.get("max_tokens", 8192)

        sampling_params = SamplingParams(temperature=temperature, top_p=top_p, top_k=top_k, max_tokens=max_tokens)

        inputs_ = []
        for message in messages:
            text = self.tokenizer.apply_chat_template(
                message,
                tokenize=False,
                add_generation_prompt=True
            )
            inputs_.append(text)

        outputs = self.llm.generate(prompts=inputs_, sampling_params=sampling_params)
        
        outputs_=[output.outputs[0].text for output in outputs]
        return outputs_

    def clean_memory(self):
        '''
        用于清空显存，防止在一次启动中顺序部署多个模型报错
        :param llm:
        :return:
        '''
        llm=self.llm
        if llm:
            try:
                del llm.llm_engine.model_executor
                del llm
                try:
                    from vllm.model_executor.parallel_utils.parallel_state import destroy_model_parallel
                    destroy_model_parallel()
                except Exception as e:
                    print(e)
                # print(f'清空vllm的llm')
            except Exception as e:
                print(e)

        try:
            torch.cuda.empty_cache()
            # print(f'清空torch占用显存')
        except Exception as e:
            print(e)
        try:
            ray.shutdown()
            # print('stop ray')
        except Exception as e:
            print(e)
        try:
            gc.collect()
            # print(f'python垃圾回收')
        except Exception as e:
            print(e)

        print("Successfully delete the llm pipeline and free the GPU memory!")
