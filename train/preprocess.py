from csv import DictReader
from pathlib import Path
from tokenizers import Tokenizer
from transformers import PreTrainedTokenizerFast
from typing import Tuple, List, Generator


class PreTokens:
    def __init__(
        self, dir_name: str, tokenizer_file: str = "tokenizers/tiny_lm_tokenizer.json"
    ) -> None:

        self.dir_path = Path(dir_name)
        self.tokenizer_path = Path(tokenizer_file)
        self._validate_paths()
        self.tokenizer = self._load_tokenizer()

    def _validate_paths(self) -> None:
        if not self.dir_path.exists():
            raise FileNotFoundError(f"Directory not found: {self.dir_path}")

        if not self.tokenizer_path.exists():
            raise FileNotFoundError(f"Tokenizer file not found: {self.tokenizer_path}")

    def _load_tokenizer(self) -> PreTrainedTokenizerFast:
        tokenize = Tokenizer.from_file(str(self.tokenizer_path))
        return PreTrainedTokenizerFast(
            tokenizer_object=tokenize, pad_token="[PAD]", unk_token="[UNK]"
        )

    def _list_csv_files(self) -> List[Path]:

        files = list(self.dir_path.glob("*.csv"))
        if not files:
            raise FileNotFoundError(f"No csv files found in `{self.dir_path}`")

        return files

    def _read_csv(self, file_path: Path) -> Generator[dict, None, None]:

        with file_path.open(newline="", encoding="utf-8") as file:
            reader = DictReader(file)

            for row in reader:
                yield row

    def _encode_tokens(self, prompt: str, response: str) -> Tuple[List[int], List[int]]:
        prompt_text = f"[PROMPT] {prompt}"
        response_text = f"[AI] {response}[END]"

        # Prompt → fixed 20 tokens
        prompt_ids = self.tokenizer(
            prompt_text,
            max_length=20,
            padding="max_length",
            truncation=True,
        )["input_ids"]

        # Response → fixed 50 tokens
        response_ids = self.tokenizer(
            response_text,
            max_length=50,
            padding="max_length",
            truncation=True,
        )["input_ids"]

        return prompt_ids, response_ids

    def get_tokens(self):
        for file_path in self._list_csv_files():
            for row in self._read_csv(file_path):
                try:
                    prompt_ids, response_ids = self._encode_tokens(
                        row["prompt"], row["response"]
                    )

                    yield prompt_ids, response_ids
                except ValueError:
                    continue
