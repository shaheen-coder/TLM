import pandas as pd
import numpy as np
from tokenizers import Tokenizer
from transformers import PreTrainedTokenizerFast
import glob
from tqdm import tqdm
import logging
from typing import Dict, List, Optional
import time
import json
from pathlib import Path


def setup_logging():
    """Setup basic logging configuration"""
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )
    return logging.getLogger(__name__)


def load_custom_tokenizer(tokenizer_path: str, config_path: Optional[str] = None):
    """
    Load custom trained tokenizer from JSON files

    Args:
        tokenizer_path: Path to tokenizer.json file
        config_path: Optional path to tokenizer_config.json

    Returns:
        Loaded tokenizer
    """
    try:
        # Method 1: Load using tokenizers library (recommended for custom tokenizers)
        tokenizer = Tokenizer.from_file(tokenizer_path)

        # Convert to Hugging Face Fast Tokenizer for better integration
        if config_path and Path(config_path).exists():
            with open(config_path, "r") as f:
                config = json.load(f)
            hf_tokenizer = PreTrainedTokenizerFast(tokenizer_object=tokenizer, **config)
        else:
            hf_tokenizer = PreTrainedTokenizerFast(
                tokenizer_object=tokenizer,
                unk_token="[UNK]",
                pad_token="[PAD]",
            )

        logger.info(f"Successfully loaded custom tokenizer from {tokenizer_path}")
        return hf_tokenizer

    except Exception as e:
        logger.error(f"Error loading tokenizer: {e}")
        raise


def load_custom_tokenizer_alternative(tokenizer_dir: str):
    """
    Alternative method: Load custom tokenizer from directory containing
    tokenizer.json and vocab files

    Args:
        tokenizer_dir: Directory containing tokenizer files
    """
    try:
        # If you have the full tokenizer directory with vocab files
        tokenizer = PreTrainedTokenizerFast.from_pretrained(tokenizer_dir)
        logger.info(
            f"Successfully loaded custom tokenizer from directory: {tokenizer_dir}"
        )
        return tokenizer
    except Exception as e:
        logger.error(f"Error loading tokenizer from directory: {e}")
        raise


def tokenize_texts(texts: List[str], tokenizer, batch_size: int = 32) -> List[int]:
    """
    Tokenize texts efficiently in batches

    Args:
        texts: List of texts to tokenize
        tokenizer: Hugging Face tokenizer
        batch_size: Batch size for tokenization

    Returns:
        List of tokenized sequence lengths
    """
    seq_lengths = []

    # Process in batches for efficiency
    for i in tqdm(
        range(0, len(texts), batch_size), desc="Tokenizing batches", leave=False
    ):
        batch_texts = texts[i : i + batch_size]

        # Tokenize with padding and truncation disabled to get original lengths
        encoded = tokenizer(
            batch_texts, truncation=False, padding=False, add_special_tokens=True
        )

        # Get sequence lengths
        batch_lengths = [len(ids) for ids in encoded["input_ids"]]
        seq_lengths.extend(batch_lengths)

    return seq_lengths


def read_csv_files(
    file_pattern: str, sample_size: Optional[int] = None
) -> pd.DataFrame:
    """
    Read multiple CSV files efficiently

    Args:
        file_pattern: Pattern to match CSV files (e.g., "data/*.csv")
        sample_size: Optional number of rows to sample (for quick testing)

    Returns:
        Combined DataFrame
    """
    csv_files = glob.glob(file_pattern)

    if not csv_files:
        logger.error(f"No CSV files found matching pattern: {file_pattern}")
        return pd.DataFrame()

    logger.info(f"Found {len(csv_files)} CSV files")

    # Read and combine all CSV files
    dfs = []
    for file in tqdm(csv_files, desc="Reading CSV files"):
        try:
            # Read CSV efficiently
            if sample_size:
                # Read only sample_size rows from each file
                df = pd.read_csv(file, nrows=sample_size)
            else:
                df = pd.read_csv(file)

            # Verify required columns exist
            if "prompt" not in df.columns or "response" not in df.columns:
                logger.warning(f"File {file} missing required columns. Skipping.")
                continue

            # Keep only required columns and drop NaN values
            df = df[["prompt", "response"]].dropna()
            dfs.append(df)

        except Exception as e:
            logger.error(f"Error reading {file}: {e}")

    if not dfs:
        logger.error("No valid CSV files found with required columns")
        return pd.DataFrame()

    combined_df = pd.concat(dfs, ignore_index=True)
    logger.info(f"Total rows loaded: {len(combined_df)}")

    return combined_df


def compute_statistics(seq_lengths: List[int], name: str) -> Dict[str, float]:
    """
    Compute comprehensive statistics for sequence lengths

    Args:
        seq_lengths: List of sequence lengths
        name: Name of the sequence type (prompt/response)

    Returns:
        Dictionary with statistics
    """
    if not seq_lengths:
        return {
            f"{name}_max": 0,
            f"{name}_avg": 0,
            f"{name}_min": 0,
            f"{name}_count": 0,
        }

    return {
        f"{name}_max": int(max(seq_lengths)),
        f"{name}_avg": float(np.mean(seq_lengths)),
        f"{name}_min": int(min(seq_lengths)),
        f"{name}_median": float(np.median(seq_lengths)),
        f"{name}_std": float(np.std(seq_lengths)),
        f"{name}_p95": float(np.percentile(seq_lengths, 95)),
        f"{name}_p99": float(np.percentile(seq_lengths, 99)),
        f"{name}_count": len(seq_lengths),
    }


def analyze_token_distribution(seq_lengths: List[int], name: str, bins: int = 50):
    """
    Analyze and print token length distribution

    Args:
        seq_lengths: List of sequence lengths
        name: Name of the sequence type
        bins: Number of bins for histogram
    """
    if not seq_lengths:
        return

    print(f"\n{name.upper()} LENGTH DISTRIBUTION:")
    print(f"  Length range: {min(seq_lengths)} - {max(seq_lengths)}")

    # Create histogram bins
    hist, bin_edges = np.histogram(seq_lengths, bins=bins)

    # Print distribution summary
    percentiles = [10, 25, 50, 75, 90, 95, 99]
    print("  Percentiles:")
    for p in percentiles:
        val = np.percentile(seq_lengths, p)
        print(f"    {p}th: {val:.0f}")


def main(
    csv_pattern: str,
    tokenizer_json_path: str,
    tokenizer_config_path: Optional[str] = None,
    batch_size: int = 32,
    sample_size: Optional[int] = None,
    save_results: bool = False,
):
    """
    Main function to process CSV files with custom tokenizer

    Args:
        csv_pattern: Pattern to match CSV files
        tokenizer_json_path: Path to tokenizer.json file
        tokenizer_config_path: Optional path to tokenizer_config.json
        batch_size: Batch size for tokenization
        sample_size: Optional number of rows to sample per file
        save_results: Whether to save statistics to file
    """
    start_time = time.time()

    # Step 1: Read CSV files
    logger.info("Step 1: Reading CSV files...")
    df = read_csv_files(csv_pattern, sample_size)
    if df.empty:
        logger.error("No data to process. Exiting.")
        return

    # Step 2: Load custom tokenizer
    logger.info("Step 2: Loading custom tokenizer...")
    tokenizer = load_custom_tokenizer(tokenizer_json_path, tokenizer_config_path)

    # Print tokenizer info
    logger.info(f"Tokenizer vocabulary size: {tokenizer.vocab_size}")

    # Step 3: Extract texts
    prompts = df["prompt"].astype(str).tolist()
    responses = df["response"].astype(str).tolist()

    logger.info(f"Processing {len(prompts):,} prompts and {len(responses):,} responses")

    # Step 4: Tokenize prompts
    logger.info("Step 3: Tokenizing prompts...")
    prompt_lengths = tokenize_texts(prompts, tokenizer, batch_size)

    # Step 5: Tokenize responses
    logger.info("Step 4: Tokenizing responses...")
    response_lengths = tokenize_texts(responses, tokenizer, batch_size)

    # Step 6: Compute statistics
    logger.info("Step 5: Computing statistics...")
    prompt_stats = compute_statistics(prompt_lengths, "prompt")
    response_stats = compute_statistics(response_lengths, "response")

    # Combine statistics
    all_stats = {**prompt_stats, **response_stats}
    all_stats["total_samples"] = len(df)
    all_stats["vocab_size"] = tokenizer.vocab_size

    # Step 7: Print results
    print("\n" + "=" * 60)
    print("CUSTOM TOKENIZER STATISTICS")
    print("=" * 60)
    print(f"\nTokenizer: {tokenizer_json_path}")
    print(f"Vocabulary size: {tokenizer.vocab_size:,}")
    print(f"Total samples processed: {len(df):,}")

    print(f"\n📝 PROMPT STATISTICS:")
    print(f"  ├─ Max sequence length: {all_stats['prompt_max']:,}")
    print(f"  ├─ Min sequence length: {all_stats['prompt_min']:,}")
    print(f"  ├─ Average sequence length: {all_stats['prompt_avg']:.2f}")
    print(f"  ├─ Median sequence length: {all_stats['prompt_median']:.2f}")
    print(f"  ├─ Std deviation: {all_stats['prompt_std']:.2f}")
    print(f"  ├─ 95th percentile: {all_stats['prompt_p95']:.0f}")
    print(f"  └─ 99th percentile: {all_stats['prompt_p99']:.0f}")

    print(f"\n💬 RESPONSE STATISTICS:")
    print(f"  ├─ Max sequence length: {all_stats['response_max']:,}")
    print(f"  ├─ Min sequence length: {all_stats['response_min']:,}")
    print(f"  ├─ Average sequence length: {all_stats['response_avg']:.2f}")
    print(f"  ├─ Median sequence length: {all_stats['response_median']:.2f}")
    print(f"  ├─ Std deviation: {all_stats['response_std']:.2f}")
    print(f"  ├─ 95th percentile: {all_stats['response_p95']:.0f}")
    print(f"  └─ 99th percentile: {all_stats['response_p99']:.0f}")

    # Show distribution
    analyze_token_distribution(prompt_lengths, "Prompt")
    analyze_token_distribution(response_lengths, "Response")

    elapsed_time = time.time() - start_time
    print(f"\n⏱️ Total processing time: {elapsed_time:.2f} seconds")

    # Save results if requested
    if save_results:
        save_statistics(all_stats, prompt_lengths, response_lengths)

    return all_stats, prompt_lengths, response_lengths


def save_statistics(
    stats: Dict,
    prompt_lengths: List[int],
    response_lengths: List[int],
    output_dir: str = "tokenizer_stats",
):
    """
    Save statistics and length distributions to files

    Args:
        stats: Dictionary with statistics
        prompt_lengths: List of prompt sequence lengths
        response_lengths: List of response sequence lengths
        output_dir: Output directory
    """
    import os
    import json
    from datetime import datetime

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Save statistics
    stats_file = os.path.join(output_dir, f"statistics_{timestamp}.json")
    with open(stats_file, "w") as f:
        json.dump(stats, f, indent=2)

    # Save length distributions
    lengths_data = {
        "prompt_lengths": prompt_lengths,
        "response_lengths": response_lengths,
    }
    lengths_file = os.path.join(output_dir, f"lengths_{timestamp}.json")
    with open(lengths_file, "w") as f:
        json.dump(lengths_data, f)

    # Save summary as text
    summary_file = os.path.join(output_dir, f"summary_{timestamp}.txt")
    with open(summary_file, "w") as f:
        f.write("TOKENIZATION STATISTICS SUMMARY\n")
        f.write("=" * 40 + "\n")
        for key, value in stats.items():
            f.write(f"{key}: {value}\n")

    logger.info(f"Results saved to {output_dir}")


def quick_analysis_mode(csv_pattern: str, tokenizer_json_path: str):
    """
    Quick analysis mode - samples data for faster processing

    Args:
        csv_pattern: Pattern to match CSV files
        tokenizer_json_path: Path to tokenizer.json file
    """
    logger.info("Running in QUICK ANALYSIS mode (sampling 1000 rows per file)")
    return main(
        csv_pattern=csv_pattern,
        tokenizer_json_path=tokenizer_json_path,
        batch_size=64,
        sample_size=1000,
        save_results=False,
    )


if __name__ == "__main__":
    # Setup logging
    logger = setup_logging()

    # Configuration - UPDATE THESE PATHS
    CSV_PATTERN = "datasets/*.csv"  # Path to your CSV files
    TOKENIZER_JSON_PATH = (
        "Tokenizer/tiny_lm_tokenizer.json"  # Path to your custom tokenizer.json
    )
    TOKENIZER_CONFIG_PATH = None

    # Choose mode
    QUICK_ANALYSIS = False  # Set to True for quick testing with sample data

    try:
        if QUICK_ANALYSIS:
            # Quick analysis with sampling
            stats, prompt_lengths, response_lengths = quick_analysis_mode(
                csv_pattern=CSV_PATTERN, tokenizer_json_path=TOKENIZER_JSON_PATH
            )
        else:
            # Full analysis
            stats, prompt_lengths, response_lengths = main(
                csv_pattern=CSV_PATTERN,
                tokenizer_json_path=TOKENIZER_JSON_PATH,
                tokenizer_config_path=TOKENIZER_CONFIG_PATH,
                batch_size=64,  # Adjust based on your memory
                sample_size=None,  # Process all data
                save_results=True,  # Save results to files
            )
    except FileNotFoundError as e:
        logger.error(f"File not found: {e}")
        logger.info("Please update the file paths in the configuration section")
    except Exception as e:
        logger.error(f"An error occurred: {e}")
