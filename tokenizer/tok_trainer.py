from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.trainers import BpeTrainer

files_lst = [
    "datasets/eng_convo.csv",
    "datasets/gen_claude.csv",
    "datasets/love_grok.csv",
    "datasets/grok.csv",
]
tokenizer = Tokenizer(BPE())

tokenizer.pre_tokenizer = Whitespace()

trainer = BpeTrainer(
    vocab_size=15000,
    min_frequency=1,
    special_tokens=["[START]", "[END]", "[PROMPT]", "[AI]", "[SEP]", "[PAD]", "[UNK]"],
)

tokenizer.train(files_lst, trainer)

tokenizer.save("Tokenizer/tiny_lm_tokenizer.json")
