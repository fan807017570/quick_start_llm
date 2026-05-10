import torch
from transformers import AutoTokenizer, AutoModel

# Load a pretrained bi-encoder model
model_name = "sentence-transformers/all-MiniLM-L6-v2"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

sentences = [
    "The cat sits outside",
    "A dog plays in the garden",
    "Artificial intelligence is transforming technology"
]

# Tokenize
encoded_input = tokenizer(
    sentences,
    padding=True,
    truncation=True,
    return_tensors="pt"
)

# Forward pass
with torch.no_grad():
    model_output = model(**encoded_input)

# Mean pooling
token_embeddings = model_output.last_hidden_state
attention_mask = encoded_input["attention_mask"]

mask = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()

sentence_embeddings = torch.sum(token_embeddings * mask, 1) / torch.clamp(mask.sum(1), min=1e-9)

print(sentence_embeddings.shape)
print(sentence_embeddings)