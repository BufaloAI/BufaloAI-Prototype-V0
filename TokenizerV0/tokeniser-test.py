from tokenizer_implementation import BasicTokenizer
import os
import re

with open(file="TokenizerV0/the-verdict.txt", mode="r", encoding="UTF-8") as F:

    raw_text = F.read()

preprocessed = re.split(r'([,.:;?_!"()\']|--|\s)', raw_text)
preprocessed = [item.strip() for item in preprocessed if item.strip()]

all_words = sorted(set(preprocessed))
all_words.extend(["<|endoftext|>", "<|unk|>"])
vocab = {token:integer for integer,token in enumerate(all_words)}

Tokenizer = BasicTokenizer(vocab)

Text = "Hello there! How is my mighty Gisburn?"

ids = Tokenizer.encode(Text)
print("\n Encoded Text:", ids)

ids = Tokenizer.decode(ids)
print("\n Decoded Text:", ids)