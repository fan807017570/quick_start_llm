from sentence_transformers import SentenceTransformer

# Load pre-trained model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Your text
texts = ["Apple is red", "Banana is yellow"]

# Convert to vectors
vectors = model.encode(texts)

print(vectors)