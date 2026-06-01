from csv import DictReader
from pathlib import Path
from tokenizers import Tokenizer
from typing import List, Generator


class PreTokens:
    def __init__(
        self, dir_name: str, tokenizer_file: str = "tokenizers/tiny_lm_tokenizer.json"
    ) -> None:

        self.dir_path = Path(dir_name)
        self.val_dir_path = self.dir_path / "val/"
        self.tokenizer_path = Path(tokenizer_file)
        self._validate_paths()
        self.tokenizer = self._load_tokenizer()

    def _validate_paths(self) -> None:
        if not self.dir_path.exists():
            raise FileNotFoundError(f"Directory not found: {self.dir_path}")

        if not self.tokenizer_path.exists():
            raise FileNotFoundError(f"Tokenizer file not found: {self.tokenizer_path}")

    def _load_tokenizer(self) -> Tokenizer:
        tokenizer = Tokenizer.from_file(str(self.tokenizer_path))
        tokenizer.enable_padding(direction="right",pad_id=0,pad_token="[PAD]",length=55)
        return tokenizer

    def _list_csv_files(self, mode: str = "t") -> List[Path]:
        files = (
            list(self.dir_path.glob("*.csv"))
            if mode == "t"
            else list(self.val_dir_path.glob("*.csv"))
        )
        search_path = self.dir_path if mode == "t" else self.val_dir_path
        if not files:
            raise FileNotFoundError(f"No csv files found in `{search_path}`")

        return files

    def _read_csv(self, file_path: Path) -> Generator[dict, None, None]:

        with file_path.open(newline="", encoding="utf-8") as file:
            reader = DictReader(file)

            for row in reader:
                yield row

    def _encode_fine_tune_tokens(self, prompt: str, response: str) -> List[int]:
        prompt_text = f"[PROMPT] {prompt}"
        input_text = f"{prompt_text} [AI] {response}[END]"

        # Prompt → fixed 50 tokens
        prompt_ids = self.tokenizer.encode(input_text).ids

        return prompt_ids

    def get_fine_tune_tokens(self, mode: str = "t"):
        for file_path in self._list_csv_files(mode=mode):
            for row in self._read_csv(file_path):
                try:
                    prompt_ids = self._encode_fine_tune_tokens(row["prompt"], row["response"])

                    yield prompt_ids
                except ValueError:
                    continue

    def _encode_tokens(self, text : str ) -> List[int] :
        self.tokenizer.no_padding()
        return self.tokenizer.encode(text).ids

    def read_txt(self, filename : Path) -> Generator[List[int],None,None]:
        with filename.open() as txt_file:
            for line in txt_file:
                enc = self._encode_tokens(line.strip())
                yield enc
                
    def get_fd_tokens(self):
        files_path = list(self.dir_path.glob('*.txt'))
        for file in files_path:
            for tokens in self.read_txt(file):
                yield tokens

