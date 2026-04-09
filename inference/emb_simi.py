import os
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import tensorflow as tf
from tokenizers import Tokenizer
from transformers import PreTrainedTokenizerFast
from model.transformer import TinyLM


def plot_vocab_projection(model, tokenizer, start=1, end=101):
    """
    Plots a 2D scatter map of tokens from index 'start' to 'end'.
    """
    # 1. Extract Embeddings
    try:
        weights = model.get_layer("embedding").get_weights()[0]
    except:
        weights = model.layers[0].get_weights()[0]  # Fallback to first layer

    # 2. Slice the first 100 tokens
    indices = list(range(start, end))
    vectors = weights[indices]

    # Get the actual string representation for labels
    labels = [tokenizer.decode([i]) for i in indices]

    # 3. Dimensionality Reduction (PCA)
    pca = PCA(n_components=2)
    coords = pca.fit_transform(vectors)

    # 4. Plotting
    plt.figure(figsize=(15, 12))
    plt.scatter(coords[:, 0], coords[:, 1], alpha=0.6, c="red", edgecolors="k", s=80)

    # Annotate points with the token string
    for i, label in enumerate(labels):
        plt.annotate(
            label,
            (coords[i, 0], coords[i, 1]),
            fontsize=9,
            xytext=(3, 3),
            textcoords="offset points",
        )

    plt.title(f"2D Vector Map: Tokens {start} to {end - 1}")
    plt.grid(True, linestyle="--", alpha=0.5)

    # Save for Colab visibility
    save_path = "vocab_map.png"
    plt.savefig(save_path, bbox_inches="tight")
    print(f"Success: Vector map saved to {os.getcwd()}/{save_path}")


def visualize_token_vectors(model, tokens_to_compare, tokenizer):
    """
    Projects high-dimensional embeddings into 2D space and plots them as dots.
    """
    # 1. Get Embedding Weights
    try:
        if hasattr(model, "embedding"):
            weights = model.embedding.get_weights()[0]
        else:
            weights = next(
                l for l in model.layers if "embed" in l.name.lower()
            ).get_weights()[0]
    except (AttributeError, StopIteration):
        print("Error: Could not find embedding weights.")
        return

    # 2. Map tokens to IDs
    token_ids = []
    valid_labels = []

    for token in tokens_to_compare:
        token_id = tokenizer.convert_tokens_to_ids(token)
        if token_id is not None and token_id < weights.shape[0]:
            token_ids.append(token_id)
            valid_labels.append(token)

    if len(token_ids) < 2:
        print("Need at least 2 valid tokens to plot.")
        return

    # 3. Extract Vectors and Reduce Dimensions to 2D
    vectors = weights[token_ids]

    # PCA reduces [num_tokens, d_model] -> [num_tokens, 2]
    pca = PCA(n_components=2)
    coords = pca.fit_transform(vectors)

    # 4. Plotting
    plt.figure(figsize=(10, 8))
    plt.axhline(0, color="gray", linestyle="--", linewidth=0.5)
    plt.axvline(0, color="gray", linestyle="--", linewidth=0.5)

    # Plot the dots
    plt.scatter(coords[:, 0], coords[:, 1], c="royalblue", s=100, edgecolors="black")

    # Add labels to the dots
    for i, label in enumerate(valid_labels):
        plt.annotate(
            label,
            (coords[i, 0], coords[i, 1]),
            textcoords="offset points",
            xytext=(5, 5),
            ha="center",
            fontsize=12,
            fontweight="bold",
        )

    plt.title("2D Projection of Token Embeddings (PCA)")
    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    plt.grid(True, alpha=0.3)

    # Critical for Colab: Save if running as script, or show if running inline
    plt.savefig("vector_map.png")
    print("Vector map saved as vector_map.png")
    plt.show()


# --- Execution ---
tokenizer_raw = Tokenizer.from_file("tokenizer/tiny_lm_tokenizer.json")
tokenizer = PreTrainedTokenizerFast(
    tokenizer_object=tokenizer_raw, pad_token="[PAD]", unk_token="[UNK]"
)

model = tf.keras.models.load_model(
    "tlm.keras", custom_objects={"TinyLM": TinyLM}, compile=False
)

# Visualize 1 to 100
# plot_vocab_projection(model, tokenizer, start=1, end=101)
visualize_token_vectors(model, ["ShaAI", "AI", "nee", "shaheen", "ai"], tokenizer)
