from tokenizers import Tokenizer

tokenizer = Tokenizer.from_file("tokenizer/tiny_lm_tokenizer.json")

prompt = input("Enter your prompt : ")
ai = input("Enter the AI : ")

encoded = tokenizer.encode(prompt, ai)
print(f"tokens : {encoded.tokens}")
print(f"ids : {encoded.ids}")
