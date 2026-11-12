from tokenizer import Tokenizer

file_path = "TokenizerV1/the-verdict.txt"

Tokenizer0 = Tokenizer(file_path=file_path)

input_embeddings = Tokenizer0.dataloader_with_sliding_window()

print(input_embeddings)
