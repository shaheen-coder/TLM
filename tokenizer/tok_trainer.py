from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.trainers import WordLevelTrainer
from pathlib import Path


files_lst = [str(p) for p in Path("datasets").glob("*.csv")]
tokenizer = Tokenizer(WordLevel(unk_token="[UNK]"))

tokenizer.pre_tokenizer = Whitespace()

trainer = WordLevelTrainer(
    special_tokens=["[PAD]", "[END]", "[PROMPT]", "[AI]", "[SEP]", "[START]", "[UNK]"],
)

tokenizer.train(files_lst, trainer)

tokenizer.save("tokenizer/tiny_lm_tokenizer.json")
