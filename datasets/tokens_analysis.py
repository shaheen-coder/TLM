import pandas as pd
import glob
import os
from tokenizers import Tokenizer


def analyze_tokens(directory_path, tokenizer_path):
    # Load your custom local tokenizer
    tokenizer = Tokenizer.from_file(tokenizer_path)

    # Grab all CSV files in the directory
    csv_files = glob.glob(os.path.join(directory_path, "*.csv"))
    csv_files = [
        f for f in csv_files
        if os.path.basename(f) != "train_01.csv"
    ]

    all_prompt_counts = []
    all_response_counts = []

    for file in csv_files:
        df = pd.read_csv(file)

        # Tokenize and count lengths for both columns
        prompt_lengths = [len(tokenizer.encode(str(x)).ids) for x in df["prompt"]]
        response_lengths = [len(tokenizer.encode(str(x)).ids) for x in df["response"]]

        all_prompt_counts.extend(prompt_lengths)
        all_response_counts.extend(response_lengths)

    if not all_prompt_counts:
        print("No data found.")
        return

    # Calculate statistics
    stats = {
        "Prompt": {
            "Min": min(all_prompt_counts),
            "Avg": round(sum(all_prompt_counts) / len(all_prompt_counts), 2),
            "Max": max(all_prompt_counts),
        },
        "Response": {
            "Min": min(all_response_counts),
            "Avg": round(sum(all_response_counts) / len(all_response_counts), 2),
            "Max": max(all_response_counts),
        },
    }

    # Display results
    print(f"Analysis for {len(csv_files)} files:")
    print(pd.DataFrame(stats))


# Usage
analyze_tokens("datasets/", "tokenizer/tiny_lm_tokenizer.json")
