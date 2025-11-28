from sentence_transformers import SentenceTransformer
import os

local_dir = "Qwen/Qwen3-Embedding-4B"
os.makedirs(local_dir, exist_ok=True)

print("Downloading model… this may take a moment.")

# Step 1: Load model normally (downloads to HF default cache)
model = SentenceTransformer('Qwen/Qwen3-Embedding-4B')

# Step 2: Save to your local directory
model.save(local_dir)

print(f"Model downloaded & saved locally at: {local_dir}")
