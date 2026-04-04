import numpy as np
import tensorflow as tf
import seaborn as sns
import matplotlib.pyplot as plt

from transformers import PreTrainedTokenizerFast
from sklearn.metrics.pairwise import cosine_similarity

from tensorflow.keras.models import load_model
from model.transformer import TinyLM

# 1. Setup Tokenizer
# Note: Ensure the path to your JSON is correct
tokenizer = PreTrainedTokenizerFast(tokenizer_file="tokenizers/tiny_lm_tokenizer.json")


def visualize_token_similarity(model, tokens_to_compare):
    """
    Extracts embeddings for specific tokens and plots a similarity heatmap.
    """
    # 2. Get Embedding Weights
    # Accessing the weight matrix directly from the layer
    weights = model.embedding.get_weights()[0]  # Shape: (vocab_size, d_model)

    # 3. Map tokens to IDs and extract vectors
    token_ids = []
    valid_tokens = []

    for token in tokens_to_compare:
        token_id = tokenizer.convert_tokens_to_ids(token)
        # Check if token exists in vocab (tokenizer returns unk_id or None if missing)
        if token_id is not None:
            token_ids.append(token_id)
            valid_tokens.append(token)
        else:
            print(f"Warning: Token '{token}' not found in tokenizer.")

    if not token_ids:
        print("No valid tokens found to visualize.")
        return

    # Extract specific vectors
    specific_embeddings = weights[token_ids]

    # 4. Calculate Cosine Similarity
    # Resulting matrix shape: (len(valid_tokens), len(valid_tokens))
    sim_matrix = cosine_similarity(specific_embeddings)

    # 5. Plotting
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        sim_matrix,
        annot=True,
        fmt=".2f",
        cmap="YlGnBu",
        xticklabels=valid_tokens,
        yticklabels=valid_tokens,
    )
    plt.title("Token Embedding Cosine Similarity")
    plt.xticks(rotation=45)
    plt.yticks(rotation=0)
    plt.show()


model = load_model("tlm.keras", compile=False)

# 2. Define tokens you are interested in
target_tokens = [
    "Enna",
    "you",
    "Yaru",
    "ShaAI",
    "Nee",
    "shaheen",
]

# 3. Run Visualization
visualize_token_similarity(model, target_tokens)
