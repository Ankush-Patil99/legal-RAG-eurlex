from api.config import CONFIG, DEVICE

print("Project name:", CONFIG["project"]["name"])
print("Embedding model:", CONFIG["models"]["embedding_model"])
print("Top-k default:", CONFIG["retrieval"]["top_k_default"])
print("Device selected:", DEVICE)
