from transformers import AutoTokenizer, AutoModel
import os

model_name = "BAAI/bge-reranker-base"
local_dir = "/mnt/data/ckt77/IR_final/CFRAG/LLMs/bge-reranker-base"

os.makedirs(local_dir, exist_ok=True)

print(f"Downloading {model_name} to {local_dir}...")
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.save_pretrained(local_dir)

model = AutoModel.from_pretrained(model_name)
model.save_pretrained(local_dir)

print("Download complete!")