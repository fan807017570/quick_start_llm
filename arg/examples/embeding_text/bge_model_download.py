from transformers import AutoTokenizer, AutoModel

model_name = "BAAI/bge-base-en-v1.5"
local_path = "models/bge-base-en-v1.5"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

# save locally
tokenizer.save_pretrained(local_path)
model.save_pretrained(local_path)
