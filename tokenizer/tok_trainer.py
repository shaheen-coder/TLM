from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.trainers import BpeTrainer
from pathlib import Path


files_lst = [str(p) for p in Path("datasets").glob("*.csv")]
#f2 = [str(p) for p in Path("datasets").glob("*.txt")]
#files_lst.extend(f2)
tokenizer = Tokenizer(BPE(unk_token="[UNK]"))

tokenizer.pre_tokenizer = Whitespace()

trainer = BpeTrainer(
    vocab_size=10000,
    min_frequency=2,
    special_tokens=["[PAD]", "[END]", "[PROMPT]", "[AI]", "[SEP]", "[START]", "[UNK]"],
)

tokenizer.train(files_lst, trainer)

tokenizer.save("tokenizer/tiny_lm_tokenizer.json")
