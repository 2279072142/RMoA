from transformers import AutoModel, AutoTokenizer

def download_model(model_name, save_directory):
    model = AutoModel.from_pretrained(model_name, cache_dir=save_directory)
    tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=save_directory)
    print(f"Model and tokenizer for {model_name} have been downloaded to {save_directory}")

# 模型名称和保存路径的列表
models_to_download = [
    ("BAAI/bge-m3", "./models/BAAI/bge-m3"),
    #("meta-llama/Llama-3.1-8B-Instruct", "./models/meta-llama/Llama3.1-8B-Instruct"),
    #("Qwen/Qwen2.5-7B-Instruct", "./models/Qwen/Qwen2.5-7B-Instruct"),
    #("google/gemma2-9b-instruct", "./models/google/gemma2-9b-instruct")
]

# 串行下载模型
for model_name, save_directory in models_to_download:
    download_model(model_name, save_directory)
