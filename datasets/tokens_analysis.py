import pandas as pd
import glob
import os
from transformers import PreTrainedTokenizerFast


def analyze_tokens(directory_path, tokenizer_path):
    # Load your custom local tokenizer
    tokenizer = PreTrainedTokenizerFast(tokenizer_file=tokenizer_path)

    # Grab all CSV files in the directory
    csv_files = glob.glob(os.path.join(directory_path, "*.csv"))

    all_prompt_counts = []
    all_response_counts = []

    for file in csv_files:
        df = pd.read_csv(file)

        # Tokenize and count lengths for both columns
        # Using list comprehension for speed
        prompt_lengths = [len(tokenizer.encode(str(x))) for x in df["prompt"]]
        response_lengths = [len(tokenizer.encode(str(x))) for x in df["response"]]

        all_prompt_counts.extend(prompt_lengths)
        all_response_counts.extend(response_lengths)

    if not all_prompt_counts:
        print("No data found.")
        return

    # Calculate statistics
    stats = {
        "Prompt": {"Min": min(all_prompt_counts), "Max": max(all_prompt_counts)},
        "Response": {"Min": min(all_response_counts), "Max": max(all_response_counts)},
    }

    # Display results
    print(f"Analysis for {len(csv_files)} files:")
    print(pd.DataFrame(stats))


# Usage
analyze_tokens("datasets/", "tokenizer/tiny_lm_tokenizer.json")
