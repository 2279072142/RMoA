import os
import sys
import json
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '.', '.'))
if project_root not in sys.path:
    sys.path.append(project_root)
from src.RMoA import RMoA


if __name__=="__main__":
    
    experts='meta-llama/Meta-Llama-3-8B-Instruct-Turbo,meta-llama/Meta-Llama-3-8B-Instruct-Turbo,google/gemma-2-9b-it,google/gemma-2-9b-it'
    moa=RMoA(model=experts.split(','),
            issparse=True,
            embedding_model='bge',
            residual_model='meta-llama/Meta-Llama-3-8B-Instruct-Turbo',
            aggregation_model='meta-llama/Meta-Llama-3-8B-Instruct-Turbo',
            residuals=True,
            smoa=False,
            k=2,
            max_tokens=1024)
    question="Find the greatest integer less than $12.32^5$ without using a calculator."
    prompt=question
    response,all_references,steps=moa.forward(prompt,workers=4,Cache=False)

    