from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.trainers import BpeTrainer
from pathlib import Path


files_lst = [str(p) for p in Path("datasets").glob("*.csv")]
tokenizer = Tokenizer(BPE())

tokenizer.pre_tokenizer = Whitespace()

trainer = BpeTrainer(
    vocab_size=15000,
    min_frequency=1,
    special_tokens=["[PAD]", "[END]", "[PROMPT]", "[AI]", "[SEP]", "[START]", "[UNK]"],
)

tokenizer.train(files_lst, trainer)

tokenizer.save("tokenizer/tiny_lm_tokenizer.json")
